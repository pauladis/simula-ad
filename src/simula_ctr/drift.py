from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from simula_ctr.config import Settings, get_settings
from simula_ctr.data import load_training_tables
from simula_ctr.features import build_features
from simula_ctr.logging_config import configure_logging


logger = logging.getLogger(__name__)

DRIFT_FEATURES = ["safety_tier", "creator_type", "app_category", "site_category", "device_type"]
PERIOD_LABELS = ["early", "middle", "latest"]
EPSILON = 1e-9


def _distribution(values: pd.Series) -> dict[str, float]:
    counts = values.astype("string").fillna("unknown").value_counts(dropna=False)
    total = float(counts.sum())
    if total <= 0:
        return {}
    return {str(key): float(value / total) for key, value in counts.items()}


def population_stability_index(reference: dict[str, float], current: dict[str, float]) -> float:
    keys = set(reference) | set(current)
    return float(
        sum(
            (current.get(key, 0.0) - reference.get(key, 0.0))
            * math.log((current.get(key, 0.0) + EPSILON) / (reference.get(key, 0.0) + EPSILON))
            for key in keys
        )
    )


def kl_divergence(reference: dict[str, float], current: dict[str, float]) -> float:
    keys = set(reference) | set(current)
    return float(
        sum(
            (reference.get(key, 0.0) + EPSILON)
            * math.log((reference.get(key, 0.0) + EPSILON) / (current.get(key, 0.0) + EPSILON))
            for key in keys
        )
    )


def normalized_distribution_delta(reference: dict[str, float], current: dict[str, float]) -> float:
    keys = set(reference) | set(current)
    return float(0.5 * sum(abs(current.get(key, 0.0) - reference.get(key, 0.0)) for key in keys))


def _assign_periods(row_count: int) -> list[str]:
    if row_count <= 0:
        return []
    return [PERIOD_LABELS[min(2, int(index * len(PERIOD_LABELS) / row_count))] for index in range(row_count)]


def prepare_drift_frame(impressions: pd.DataFrame, characters: pd.DataFrame) -> pd.DataFrame:
    frame = build_features(impressions, characters)
    frame["click"] = pd.to_numeric(frame.get("click", 0), errors="coerce").fillna(0).clip(lower=0, upper=1)
    frame["_time_sort"] = pd.to_datetime(frame["event_timestamp"], errors="coerce").fillna(pd.Timestamp.max)
    frame = frame.sort_values(["_time_sort", "hour"], kind="mergesort").reset_index(drop=True)
    frame["period"] = _assign_periods(len(frame))
    timestamps = pd.to_datetime(frame["event_timestamp"], errors="coerce")
    frame["event_day"] = timestamps.dt.strftime("%Y-%m-%d").fillna("unknown")
    frame["event_hour_bucket"] = timestamps.dt.strftime("%Y-%m-%d %H:00").fillna("unknown")
    return frame


def _time_series(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    grouped = (
        frame.groupby(column, dropna=False)
        .agg(impressions=("click", "size"), ctr=("click", "mean"))
        .reset_index()
        .sort_values(column)
    )
    return [
        {column: str(row[column]), "impressions": int(row["impressions"]), "ctr": float(row["ctr"])}
        for _, row in grouped.iterrows()
    ]


def _period_summary(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    grouped = frame.groupby("period", sort=False).agg(impressions=("click", "size"), ctr=("click", "mean"))
    return {
        str(period): {"impressions": int(row["impressions"]), "ctr": float(row["ctr"])}
        for period, row in grouped.iterrows()
    }


def _concentration_by_period(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for period, group in frame.groupby("period", sort=False):
        shares = group["character_id"].astype("string").fillna("unknown").value_counts(normalize=True)
        output[str(period)] = {
            "top_character_share": float(shares.iloc[0]) if not shares.empty else 0.0,
            "top_10_character_share": float(shares.head(10).sum()) if not shares.empty else 0.0,
            "unique_characters": int(group["character_id"].nunique(dropna=False)),
        }
    return output


def _feature_shift_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    reference_frame = frame[frame["period"] == "early"]
    current_frame = frame[frame["period"] == "latest"]
    metrics: dict[str, dict[str, Any]] = {}
    for feature in DRIFT_FEATURES:
        reference = _distribution(reference_frame[feature]) if feature in reference_frame else {}
        current = _distribution(current_frame[feature]) if feature in current_frame else {}
        metrics[feature] = {
            "psi": population_stability_index(reference, current),
            "kl_divergence": kl_divergence(reference, current),
            "normalized_distribution_delta": normalized_distribution_delta(reference, current),
            "reference_top_values": dict(sorted(reference.items(), key=lambda item: item[1], reverse=True)[:5]),
            "current_top_values": dict(sorted(current.items(), key=lambda item: item[1], reverse=True)[:5]),
        }
    return metrics


def _distribution_by_period(frame: pd.DataFrame, feature: str) -> dict[str, dict[str, float]]:
    if feature not in frame.columns:
        return {}
    return {str(period): _distribution(group[feature]) for period, group in frame.groupby("period", sort=False)}


def build_drift_metrics(impressions: pd.DataFrame, characters: pd.DataFrame) -> dict[str, Any]:
    frame = prepare_drift_frame(impressions, characters)
    period_summary = _period_summary(frame)
    early_ctr = period_summary.get("early", {}).get("ctr", 0.0)
    latest_ctr = period_summary.get("latest", {}).get("ctr", 0.0)
    return {
        "status": "ok",
        "row_count": int(len(frame)),
        "periods": period_summary,
        "ctr_delta_latest_vs_early": float(latest_ctr - early_ctr),
        "ctr_over_time": _time_series(frame, "event_day"),
        "impressions_over_time": _time_series(frame, "event_hour_bucket")[:250],
        "top_character_concentration": _concentration_by_period(frame),
        "safety_tier_distribution": _distribution_by_period(frame, "safety_tier"),
        "app_category_distribution": _distribution_by_period(frame, "app_category"),
        "feature_shift": _feature_shift_metrics(frame),
        "adaptation_proposal": [
            "Weight recent CTR and calibration windows more heavily when drift is sustained.",
            "Use epsilon-greedy or Thompson Sampling exploration for under-observed candidates.",
            "Apply fatigue caps when dominant character cohorts concentrate traffic.",
            "Alert on high PSI/KL shifts before retraining or changing campaign allocation.",
        ],
    }


def build_drift_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Drift Report",
        "",
        f"- Status: {payload['status']}",
        f"- Rows analyzed: {payload['row_count']}",
        f"- Latest vs early CTR delta: {payload['ctr_delta_latest_vs_early']:.6f}",
        "",
        "## Period CTR",
        "",
        "| Period | Impressions | CTR |",
        "| --- | ---: | ---: |",
    ]
    for period in PERIOD_LABELS:
        summary = payload["periods"].get(period, {"impressions": 0, "ctr": 0.0})
        lines.append(f"| {period} | {summary['impressions']} | {summary['ctr']:.6f} |")

    lines.extend(["", "## Feature Shift", "", "| Feature | PSI | KL divergence | Distribution delta |", "| --- | ---: | ---: | ---: |"])
    for feature, metrics in payload["feature_shift"].items():
        lines.append(
            f"| {feature} | {metrics['psi']:.6f} | {metrics['kl_divergence']:.6f} | {metrics['normalized_distribution_delta']:.6f} |"
        )

    lines.extend(["", "## Character Concentration", "", "| Period | Top character share | Top 10 share | Unique characters |", "| --- | ---: | ---: | ---: |"])
    for period in PERIOD_LABELS:
        metrics = payload["top_character_concentration"].get(period, {})
        lines.append(
            f"| {period} | {metrics.get('top_character_share', 0.0):.6f} | {metrics.get('top_10_character_share', 0.0):.6f} | {metrics.get('unique_characters', 0)} |"
        )

    lines.extend(["", "## Adaptation Proposal", ""])
    for proposal in payload["adaptation_proposal"]:
        lines.append(f"- {proposal}")
    return "\n".join(lines) + "\n"


def write_drift_outputs(settings: Settings | None = None) -> tuple[Path, Path]:
    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    impressions, characters = load_training_tables(settings, validate=False)
    metrics = build_drift_metrics(impressions, characters)
    metrics_path = settings.reports_dir / "drift_metrics.json"
    report_path = settings.reports_dir / "drift_report.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_drift_report(metrics), encoding="utf-8")
    return report_path, metrics_path


def main() -> int:
    configure_logging()
    report_path, metrics_path = write_drift_outputs()
    logger.info("Wrote drift outputs: %s, %s", report_path, metrics_path)
    return 0
