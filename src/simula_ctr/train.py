from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from simula_ctr.config import Settings, get_settings
from simula_ctr.data import load_training_tables
from simula_ctr.evaluate import evaluate_split, extract_feature_importance, write_evaluation_outputs, write_json
from simula_ctr.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_features, select_feature_columns
from simula_ctr.logging_config import configure_logging
from simula_ctr.predict import ArtifactCtrPredictor
from simula_ctr.production_reports import write_latency_benchmark, write_sample_ranked_output
from simula_ctr.ranker import CampaignRanker
from simula_ctr.validation import validate_input_tables, write_data_validation_outputs


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    iterations: int = 250
    learning_rate: float = 0.08
    depth: int = 6
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 30
    verbose_eval: int = 50
    thread_count: int = -1
    max_rows: int | None = None


def load_training_config() -> TrainingConfig:
    max_rows_raw = os.environ.get("SIMULA_TRAIN_MAX_ROWS")
    return TrainingConfig(
        train_fraction=float(os.environ.get("SIMULA_TRAIN_FRACTION", "0.70")),
        validation_fraction=float(os.environ.get("SIMULA_VALIDATION_FRACTION", "0.15")),
        iterations=int(os.environ.get("SIMULA_CATBOOST_ITERATIONS", "250")),
        learning_rate=float(os.environ.get("SIMULA_CATBOOST_LEARNING_RATE", "0.08")),
        depth=int(os.environ.get("SIMULA_CATBOOST_DEPTH", "6")),
        l2_leaf_reg=float(os.environ.get("SIMULA_CATBOOST_L2_LEAF_REG", "3.0")),
        early_stopping_rounds=int(os.environ.get("SIMULA_CATBOOST_EARLY_STOPPING_ROUNDS", "30")),
        verbose_eval=int(os.environ.get("SIMULA_CATBOOST_VERBOSE_EVAL", "50")),
        thread_count=int(os.environ.get("SIMULA_CATBOOST_THREAD_COUNT", "-1")),
        max_rows=int(max_rows_raw) if max_rows_raw else None,
    )


def _validate_training_config(config: TrainingConfig) -> None:
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if config.train_fraction + config.validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave a non-empty test split")
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")


def _timestamp_range(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    timestamps = pd.to_datetime(frame["event_timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None, None
    return timestamps.min().isoformat(), timestamps.max().isoformat()


def sort_by_time(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_time_sort"] = pd.to_datetime(out["event_timestamp"], errors="coerce")
    out["_time_sort"] = out["_time_sort"].fillna(pd.Timestamp.max)
    return out.sort_values(["_time_sort", "hour"], kind="mergesort").drop(columns=["_time_sort"]).reset_index(drop=True)


def time_based_split(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sorted_frame = sort_by_time(frame)
    row_count = len(sorted_frame)
    if row_count < 3:
        raise ValueError("At least 3 rows are required for train/validation/test splitting")

    unique_times = sorted_frame["event_timestamp"].dropna().drop_duplicates().reset_index(drop=True)
    split_warnings: list[str] = []
    if len(unique_times) >= 3:
        train_boundary_index = min(max(0, int(len(unique_times) * train_fraction) - 1), len(unique_times) - 3)
        validation_boundary_index = min(
            max(train_boundary_index + 1, int(len(unique_times) * (train_fraction + validation_fraction)) - 1),
            len(unique_times) - 2,
        )
        train_boundary = unique_times.iloc[train_boundary_index]
        validation_boundary = unique_times.iloc[validation_boundary_index]
        timestamp = sorted_frame["event_timestamp"]
        invalid_timestamp = timestamp.isna()
        train_frame = sorted_frame[timestamp <= train_boundary].copy()
        validation_frame = sorted_frame[
            (timestamp > train_boundary)
            & (timestamp <= validation_boundary)
        ].copy()
        test_frame = sorted_frame[(timestamp > validation_boundary) | invalid_timestamp].copy()
        split_unit = "timestamp_bucket"
        if invalid_timestamp.any():
            split_warnings.append("Rows with invalid timestamps were assigned to the test split.")
    else:
        train_end = max(1, int(row_count * train_fraction))
        train_end = min(train_end, row_count - 2)
        validation_end = max(train_end + 1, int(row_count * (train_fraction + validation_fraction)))
        validation_end = min(validation_end, row_count - 1)
        train_frame = sorted_frame.iloc[:train_end].copy()
        validation_frame = sorted_frame.iloc[train_end:validation_end].copy()
        test_frame = sorted_frame.iloc[validation_end:].copy()
        split_unit = "row"
        split_warnings.append(
            "Fewer than three unique event timestamps were available; split fell back to row boundaries."
        )

    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError("Time split produced an empty split; provide more rows or adjust split fractions")

    train_start, train_stop = _timestamp_range(train_frame)
    validation_start, validation_stop = _timestamp_range(validation_frame)
    test_start, test_stop = _timestamp_range(test_frame)
    metadata = {
        "strategy": "time_based_chronological",
        "split_unit": split_unit,
        "sort_columns": ["event_timestamp", "hour"],
        "unique_timestamp_count": int(len(unique_times)),
        "warnings": split_warnings,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "total_rows": row_count,
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows": len(test_frame),
        "train_start": train_start,
        "train_end": train_stop,
        "validation_start": validation_start,
        "validation_end": validation_stop,
        "test_start": test_start,
        "test_end": test_stop,
    }
    return train_frame, validation_frame, test_frame, metadata


def _apply_max_rows(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    if max_rows < 3:
        raise ValueError("SIMULA_TRAIN_MAX_ROWS must be at least 3")
    return sort_by_time(frame).tail(max_rows).reset_index(drop=True)


def _target(frame: pd.DataFrame) -> pd.Series:
    if "click" not in frame.columns:
        raise ValueError("Training frame must contain click target")
    target = pd.to_numeric(frame["click"], errors="coerce")
    if target.isna().any():
        raise ValueError("Training target contains missing or non-numeric click values")
    return target.astype(int)


def _catboost_params(settings: Settings, config: TrainingConfig) -> dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": config.iterations,
        "learning_rate": config.learning_rate,
        "depth": config.depth,
        "l2_leaf_reg": config.l2_leaf_reg,
        "random_seed": settings.random_seed,
        "thread_count": config.thread_count,
        "allow_writing_files": False,
        "verbose": config.verbose_eval,
    }


def ensure_training_dependencies() -> None:
    try:
        import catboost  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CatBoost is not installed in this Python environment. Install project dependencies or run training in Docker."
        ) from exc


def train_catboost_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    settings: Settings,
    config: TrainingConfig,
) -> Any:
    from catboost import CatBoostClassifier, Pool

    train_pool = Pool(
        data=select_feature_columns(train_frame),
        label=_target(train_frame),
        cat_features=CATEGORICAL_COLUMNS,
    )
    validation_pool = Pool(
        data=select_feature_columns(validation_frame),
        label=_target(validation_frame),
        cat_features=CATEGORICAL_COLUMNS,
    )

    model = CatBoostClassifier(**_catboost_params(settings, config))
    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    return model


def predict_probabilities(model: Any, frame: pd.DataFrame) -> np.ndarray:
    predictions = model.predict_proba(select_feature_columns(frame))
    return np.asarray(predictions[:, 1], dtype=float)


def _baseline_predictions(train_target: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    global_ctr = float(train_target.mean())
    return np.full(len(frame), min(max(global_ctr, 1e-6), 1.0 - 1e-6), dtype=float)


def _metadata_path(settings: Settings) -> Path:
    return settings.artifacts_dir / "model_metadata.json"


def train_and_evaluate(settings: Settings | None = None, config: TrainingConfig | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    config = config or load_training_config()
    _validate_training_config(config)

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    ensure_training_dependencies()

    logger.info("Loading input tables")
    impressions, characters = load_training_tables(settings, validate=False)
    validation_report = validate_input_tables(impressions, characters)
    write_data_validation_outputs(settings, validation_report)
    if validation_report["status"] != "pass":
        raise ValueError("Input data validation failed. See reports/data_validation.md for details.")

    logger.info("Building feature frame")
    frame = build_features(impressions, characters)
    frame = _apply_max_rows(frame, config.max_rows)
    logger.info("Prepared %s rows with %s model features", len(frame), len(FEATURE_COLUMNS))

    train_frame, validation_frame, test_frame, split_metadata = time_based_split(
        frame,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    train_target = _target(train_frame)
    validation_target = _target(validation_frame)
    test_target = _target(test_frame)
    train_character_counts = train_frame["character_id"].astype(str).value_counts()

    logger.info(
        "Training CatBoost on %s rows, validating on %s rows, testing on %s rows",
        len(train_frame),
        len(validation_frame),
        len(test_frame),
    )
    model = train_catboost_model(train_frame, validation_frame, settings, config)

    logger.info("Scoring validation and test splits")
    validation_predictions = predict_probabilities(model, validation_frame)
    test_predictions = predict_probabilities(model, test_frame)
    train_predictions = predict_probabilities(model, train_frame)

    baseline_validation = _baseline_predictions(train_target, validation_frame)
    baseline_test = _baseline_predictions(train_target, test_frame)

    model_version = datetime.now(timezone.utc).strftime("ctr-%Y%m%dT%H%M%SZ")
    trained_at = datetime.now(timezone.utc).isoformat()

    evaluation = {
        "model_name": settings.model_name,
        "model_version": model_version,
        "trained_at": trained_at,
        "split": split_metadata,
        "features": {
            "feature_columns": FEATURE_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "numeric_columns": [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS],
        },
        "splits": {
            "train": evaluate_split(
                train_frame,
                train_target,
                train_predictions,
                train_character_counts,
                settings.cold_start_min_impressions,
            ),
            "validation": evaluate_split(
                validation_frame,
                validation_target,
                validation_predictions,
                train_character_counts,
                settings.cold_start_min_impressions,
            ),
            "test": evaluate_split(
                test_frame,
                test_target,
                test_predictions,
                train_character_counts,
                settings.cold_start_min_impressions,
            ),
        },
        "baseline": {
            "name": "global_ctr",
            "global_ctr": float(train_target.mean()),
            "validation": evaluate_split(
                validation_frame,
                validation_target,
                baseline_validation,
                train_character_counts,
                settings.cold_start_min_impressions,
            ),
            "test": evaluate_split(
                test_frame,
                test_target,
                baseline_test,
                train_character_counts,
                settings.cold_start_min_impressions,
            ),
        },
    }

    metadata = {
        "model_name": settings.model_name,
        "model_version": model_version,
        "trained_at": trained_at,
        "artifact_path": str(settings.model_artifact_path),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target_column": "click",
        "training_config": asdict(config),
        "catboost_params": _catboost_params(settings, config),
        "global_ctr": float(train_target.mean()),
        "feature_importance": extract_feature_importance(model, FEATURE_COLUMNS),
        "evaluation": evaluation,
    }
    artifact = {
        "model": model,
        "metadata": metadata,
    }

    logger.info("Saving model artifact to %s", settings.model_artifact_path)
    joblib.dump(artifact, settings.model_artifact_path)
    write_json(_metadata_path(settings), metadata)
    evaluation_path, report_path = write_evaluation_outputs(settings, evaluation, metadata)
    serving_predictor = ArtifactCtrPredictor(model=model, metadata=metadata)
    serving_ranker = CampaignRanker(serving_predictor)
    write_sample_ranked_output(settings, serving_ranker)
    write_latency_benchmark(
        settings,
        predictor=serving_predictor,
        ranker=serving_ranker,
        iterations=int(os.environ.get("SIMULA_LATENCY_BENCHMARK_ITERATIONS", "100")),
    )
    logger.info("Wrote evaluation outputs: %s, %s", evaluation_path, report_path)
    return metadata


def main() -> int:
    configure_logging()
    try:
        train_and_evaluate()
    except (RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0
