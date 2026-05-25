from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from simula_ctr.config import Settings, get_settings
from simula_ctr.features import FEATURE_COLUMNS
from simula_ctr.logging_config import configure_logging


logger = logging.getLogger(__name__)


def _sklearn_metrics() -> tuple[Any, Any, Any, Any]:
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required to compute evaluation metrics.") from exc
    return average_precision_score, brier_score_loss, log_loss, roc_auc_score


def _clip_probabilities(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions.astype(float), 1e-6, 1.0 - 1e-6)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    _, _, _, roc_auc_score = _sklearn_metrics()
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    average_precision_score, _, _, _ = _sklearn_metrics()
    return float(average_precision_score(y_true, y_score))


def compute_binary_metrics(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> dict[str, Any]:
    target = np.asarray(y_true, dtype=int)
    predictions = _clip_probabilities(np.asarray(y_score, dtype=float))
    _, brier_score_loss, log_loss, _ = _sklearn_metrics()
    return {
        "n": int(target.size),
        "clicks": int(target.sum()),
        "ctr": float(target.mean()) if target.size else None,
        "log_loss": float(log_loss(target, predictions, labels=[0, 1])) if target.size else None,
        "roc_auc": _safe_roc_auc(target, predictions),
        "pr_auc": _safe_pr_auc(target, predictions),
        "brier_score": float(brier_score_loss(target, predictions)) if target.size else None,
    }


def calibration_summary(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    target = np.asarray(y_true, dtype=int)
    predictions = _clip_probabilities(np.asarray(y_score, dtype=float))
    if target.size == 0:
        return []

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(predictions, bins, right=True)
    bin_ids = np.clip(bin_ids, 1, n_bins)
    rows: list[dict[str, Any]] = []
    for bin_id in range(1, n_bins + 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        lower = bins[bin_id - 1]
        upper = bins[bin_id]
        rows.append(
            {
                "bin": int(bin_id),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "count": int(mask.sum()),
                "mean_predicted_ctr": float(predictions[mask].mean()),
                "observed_ctr": float(target[mask].mean()),
            }
        )
    return rows


def ranking_metrics(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    k_values: tuple[int, ...] = (100, 1000, 10000),
) -> dict[str, Any]:
    target = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_score, dtype=float)
    if target.size == 0:
        return {}

    order = np.argsort(-predictions)
    overall_ctr = float(target.mean())
    metrics: dict[str, Any] = {"overall_ctr": overall_ctr}
    for k in k_values:
        effective_k = min(k, target.size)
        if effective_k <= 0:
            continue
        top_target = target[order[:effective_k]]
        ctr_at_k = float(top_target.mean())
        metrics[f"ctr_at_{k}"] = ctr_at_k
        metrics[f"precision_at_{k}"] = ctr_at_k
        metrics[f"lift_at_{k}"] = ctr_at_k / overall_ctr if overall_ctr > 0 else None
        metrics[f"effective_k_{k}"] = int(effective_k)
    return metrics


def calibration_error(calibration_rows: list[dict[str, Any]]) -> float | None:
    total = sum(int(row["count"]) for row in calibration_rows)
    if total <= 0:
        return None
    weighted_error = sum(
        int(row["count"]) * abs(float(row["mean_predicted_ctr"]) - float(row["observed_ctr"]))
        for row in calibration_rows
    )
    return float(weighted_error / total)


def add_evaluation_slice_columns(
    frame: pd.DataFrame,
    train_character_counts: pd.Series,
    cold_start_min_impressions: int,
) -> pd.DataFrame:
    out = frame.copy()
    turn = pd.to_numeric(out.get("conversation_turn", 0), errors="coerce").fillna(0)
    out["turn_stage"] = np.select(
        [turn <= 2, turn <= 8],
        ["early", "mid"],
        default="late",
    )
    character_ids = out.get("character_id", pd.Series("unknown", index=out.index)).astype(str)
    observed_counts = character_ids.map(train_character_counts).fillna(0)
    out["cold_start_character"] = np.where(
        observed_counts < cold_start_min_impressions,
        "cold_start_like",
        "graduated",
    )
    return out


def compute_slice_metrics(
    frame: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    min_count: int | None = None,
    max_values_per_slice: int = 25,
) -> dict[str, list[dict[str, Any]]]:
    scored = frame.copy()
    scored["_target"] = np.asarray(y_true, dtype=int)
    scored["_prediction"] = _clip_probabilities(np.asarray(y_score, dtype=float))
    if min_count is None:
        min_count = max(10, min(100, int(len(scored) * 0.01)))

    slice_columns = ["safety_tier", "creator_type", "turn_stage", "cold_start_character", "app_category"]
    output: dict[str, list[dict[str, Any]]] = {}
    for column in slice_columns:
        if column not in scored.columns:
            continue
        rows: list[dict[str, Any]] = []
        counts = scored[column].astype(str).value_counts(dropna=False)
        for value in counts.head(max_values_per_slice).index:
            subset = scored[scored[column].astype(str) == str(value)]
            if len(subset) < min_count:
                continue
            metrics = compute_binary_metrics(subset["_target"], subset["_prediction"])
            rows.append({"value": str(value), **metrics})
        output[column] = rows
    return output


def evaluate_split(
    frame: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    train_character_counts: pd.Series,
    cold_start_min_impressions: int,
) -> dict[str, Any]:
    slice_frame = add_evaluation_slice_columns(frame, train_character_counts, cold_start_min_impressions)
    calibration = calibration_summary(y_true, y_score)
    return {
        "metrics": compute_binary_metrics(y_true, y_score),
        "calibration": calibration,
        "calibration_error": calibration_error(calibration),
        "ranking": ranking_metrics(y_true, y_score),
        "slices": compute_slice_metrics(slice_frame, y_true, y_score),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_default(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_default(item) for item in value]
    if isinstance(value, tuple):
        return [_json_default(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return [_json_default(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat() if not pd.isna(value) else None
    if value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_default(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def extract_feature_importance(model: Any, feature_columns: list[str] | None = None) -> list[dict[str, Any]]:
    feature_columns = feature_columns or FEATURE_COLUMNS
    if not hasattr(model, "get_feature_importance"):
        return []
    importance_values = model.get_feature_importance()
    rows = [
        {
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in zip(feature_columns, importance_values)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def build_feature_importance_report(rows: list[dict[str, Any]], top_n: int = 30) -> str:
    lines = [
        "# Feature Importance Report",
        "",
        "CatBoost feature importance from the saved training artifact. Higher values indicate larger contribution to the fitted model under CatBoost's built-in importance calculation.",
        "",
        "| Rank | Feature | Importance |",
        "| ---: | --- | ---: |",
    ]
    if not rows:
        lines.append("| n/a | No feature importance available | n/a |")
    else:
        for rank, row in enumerate(rows[:top_n], start=1):
            lines.append(f"| {rank} | {row['feature']} | {_format_metric(row['importance'])} |")
    return "\n".join(lines) + "\n"


def write_feature_importance_outputs(settings: Settings, metadata: dict[str, Any]) -> tuple[Path | None, Path | None]:
    rows = metadata.get("feature_importance", [])
    if not rows:
        return None, None
    json_path = settings.reports_dir / "feature_importance.json"
    report_path = settings.reports_dir / "feature_importance.md"
    write_json(json_path, {"feature_importance": rows})
    report_path.write_text(build_feature_importance_report(rows), encoding="utf-8")
    return json_path, report_path


def build_calibration_report(evaluation: dict[str, Any]) -> str:
    lines = [
        "# Calibration Report",
        "",
        "Calibration compares predicted CTR with observed CTR in probability buckets. Lower expected calibration error is better.",
        "",
    ]
    for split_name in ["validation", "test"]:
        split = evaluation.get("splits", {}).get(split_name, {})
        calibration = split.get("calibration", [])
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                f"- Expected calibration error: {_format_metric(split.get('calibration_error'))}",
                "",
                "| Bin | Count | Mean Predicted CTR | Observed CTR |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        if not calibration:
            lines.append("| n/a | 0 | n/a | n/a |")
        else:
            for row in calibration:
                lines.append(
                    f"| {row['bin']} | {row['count']} | {_format_metric(row['mean_predicted_ctr'])} | {_format_metric(row['observed_ctr'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def write_calibration_outputs(settings: Settings, evaluation: dict[str, Any]) -> tuple[Path, Path]:
    payload = {
        split_name: {
            "expected_calibration_error": split.get("calibration_error"),
            "calibration": split.get("calibration", []),
        }
        for split_name, split in evaluation.get("splits", {}).items()
        if split_name in {"validation", "test"}
    }
    json_path = settings.reports_dir / "calibration_metrics.json"
    report_path = settings.reports_dir / "calibration_report.md"
    write_json(json_path, payload)
    report_path.write_text(build_calibration_report(evaluation), encoding="utf-8")
    return json_path, report_path


def build_model_report(evaluation: dict[str, Any], metadata: dict[str, Any]) -> str:
    split = evaluation.get("split", {})
    test_metrics = evaluation.get("splits", {}).get("test", {}).get("metrics", {})
    val_metrics = evaluation.get("splits", {}).get("validation", {}).get("metrics", {})
    baseline_test = evaluation.get("baseline", {}).get("test", {}).get("metrics", {})
    feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)
    categorical_columns = metadata.get("categorical_columns", [])
    params = metadata.get("training_config", {})

    lines = [
        "# Simula CTR Model Report",
        "",
        "## Summary",
        "",
        f"- Model: {metadata.get('model_name', 'catboost_ctr')}",
        f"- Version: {metadata.get('model_version', 'unknown')}",
        f"- Trained at: {metadata.get('trained_at', 'unknown')}",
        f"- Artifact: {metadata.get('artifact_path', 'unknown')}",
        "",
        "## Split Strategy",
        "",
        "Rows are sorted by parsed impression time and split chronologically. Older impressions train the model, the middle period is used for validation, and the latest period is held out for test.",
        "",
        f"- Train rows: {split.get('train_rows', 'n/a')}",
        f"- Validation rows: {split.get('validation_rows', 'n/a')}",
        f"- Test rows: {split.get('test_rows', 'n/a')}",
        f"- Train time range: {split.get('train_start', 'n/a')} to {split.get('train_end', 'n/a')}",
        f"- Validation time range: {split.get('validation_start', 'n/a')} to {split.get('validation_end', 'n/a')}",
        f"- Test time range: {split.get('test_start', 'n/a')} to {split.get('test_end', 'n/a')}",
        "",
        "## Feature Strategy",
        "",
        f"The model uses {len(feature_columns)} features: parsed time features, conversation/session features, joined character metadata, site/app/device categorical fields, and anonymized C-fields.",
        "",
        f"- Categorical feature count: {len(categorical_columns)}",
        f"- Numeric feature count: {len(feature_columns) - len(categorical_columns)}",
        "",
        "## Model Choice",
        "",
        "CatBoostClassifier is used because the ranking problem has many high-cardinality categorical fields. It can consume categorical columns directly while optimizing binary log loss for CTR probability estimates.",
        "",
        "## Evaluation Results",
        "",
        "| Split | Log Loss | ROC AUC | PR AUC | Brier | CTR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Validation | {_format_metric(val_metrics.get('log_loss'))} | {_format_metric(val_metrics.get('roc_auc'))} | {_format_metric(val_metrics.get('pr_auc'))} | {_format_metric(val_metrics.get('brier_score'))} | {_format_metric(val_metrics.get('ctr'))} |",
        f"| Test | {_format_metric(test_metrics.get('log_loss'))} | {_format_metric(test_metrics.get('roc_auc'))} | {_format_metric(test_metrics.get('pr_auc'))} | {_format_metric(test_metrics.get('brier_score'))} | {_format_metric(test_metrics.get('ctr'))} |",
        f"| Global CTR Baseline Test | {_format_metric(baseline_test.get('log_loss'))} | {_format_metric(baseline_test.get('roc_auc'))} | {_format_metric(baseline_test.get('pr_auc'))} | {_format_metric(baseline_test.get('brier_score'))} | {_format_metric(baseline_test.get('ctr'))} |",
        "",
        "## Ranking Metric",
        "",
    ]

    ranking = evaluation.get("splits", {}).get("test", {}).get("ranking", {})
    for key in ["ctr_at_100", "lift_at_100", "ctr_at_1000", "lift_at_1000", "ctr_at_10000", "lift_at_10000"]:
        if key in ranking:
            lines.append(f"- {key}: {_format_metric(ranking[key])}")

    lines.extend(
        [
            "",
            "## Additional Reports",
            "",
            "- `reports/feature_importance.md` ranks model features by CatBoost importance.",
            "- `reports/calibration_report.md` compares predicted CTR to observed CTR by probability bucket.",
            "- `reports/data_validation.md` summarizes input data quality checks.",
            "- `reports/latency_benchmark.md` reports local serving latency for predict and rank paths.",
            "- `reports/sample_ranked_output.json` contains an example `/rank` style response.",
            "",
            "## Training Parameters",
            "",
        ]
    )
    for key, value in sorted(params.items()):
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Trade-Offs",
            "",
            "This implementation favors chronological validation, direct categorical handling, and calibrated probability diagnostics over leaderboard-only optimization. CatBoost training is configurable through environment variables so local smoke runs and full offline jobs use the same code path.",
            "",
            "## Known Limitations",
            "",
            "- Probabilities are not post-calibrated yet; calibration is measured and can be improved with Platt or isotonic calibration on the validation period.",
            "- The current model uses only the provided impression and character tables. Online user history, campaign budget state, and richer creative metadata are not available yet.",
            "- Candidate ranking still needs online business signals such as live pacing, fatigue, and exploration uncertainty.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_evaluation_outputs(
    settings: Settings,
    evaluation: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = settings.reports_dir / "evaluation.json"
    report_path = settings.reports_dir / "model_report.md"
    write_json(evaluation_path, evaluation)
    report_path.write_text(build_model_report(evaluation, metadata), encoding="utf-8")
    write_feature_importance_outputs(settings, metadata)
    write_calibration_outputs(settings, evaluation)
    return evaluation_path, report_path


def main() -> int:
    configure_logging()
    settings = get_settings()
    metadata_path = settings.artifacts_dir / "model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    elif settings.model_artifact_path.exists():
        artifact = joblib.load(settings.model_artifact_path)
        metadata = artifact.get("metadata", {})
    else:
        logger.error("Model artifact not found: %s. Run make train first.", settings.model_artifact_path)
        return 1

    evaluation = metadata.get("evaluation")
    if not evaluation:
        logger.error("Stored model metadata does not contain an evaluation payload.")
        return 1

    evaluation_path, report_path = write_evaluation_outputs(settings, evaluation, metadata)
    logger.info("Wrote evaluation reports: %s, %s", evaluation_path, report_path)
    return 0
