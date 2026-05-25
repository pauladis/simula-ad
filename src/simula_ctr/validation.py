from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from simula_ctr.config import Settings
from simula_ctr.data import CHARACTER_REQUIRED_COLUMNS, IMPRESSION_REQUIRED_COLUMNS
from simula_ctr.features import parse_hour_series


def _missing_columns(frame: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [column for column in required_columns if column not in frame.columns]


def _numeric_negative_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce")
    return int((values < 0).sum())


def _null_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].isna().sum())


def validate_input_tables(impressions: pd.DataFrame, characters: pd.DataFrame) -> dict[str, Any]:
    impression_missing = _missing_columns(impressions, IMPRESSION_REQUIRED_COLUMNS)
    character_missing = _missing_columns(characters, CHARACTER_REQUIRED_COLUMNS)
    issues: list[dict[str, Any]] = []

    if impression_missing:
        issues.append(
            {
                "severity": "error",
                "table": "impressions",
                "code": "missing_required_columns",
                "details": impression_missing,
            }
        )
    if character_missing:
        issues.append(
            {
                "severity": "error",
                "table": "characters",
                "code": "missing_required_columns",
                "details": character_missing,
            }
        )

    if impressions.empty:
        issues.append({"severity": "error", "table": "impressions", "code": "empty_table", "details": []})
    if characters.empty:
        issues.append({"severity": "warning", "table": "characters", "code": "empty_table", "details": []})

    click_invalid_count = 0
    if "click" in impressions.columns:
        clicks = pd.to_numeric(impressions["click"], errors="coerce")
        click_invalid_count = int((clicks.isna() | ~clicks.isin([0, 1])).sum())
        if click_invalid_count:
            issues.append(
                {
                    "severity": "error",
                    "table": "impressions",
                    "code": "invalid_binary_target",
                    "details": {"column": "click", "invalid_count": click_invalid_count},
                }
            )

    invalid_hour_count = 0
    parsed_hours = pd.Series(pd.NaT, index=impressions.index, dtype="datetime64[ns]")
    if "hour" in impressions.columns:
        parsed_hours = parse_hour_series(impressions["hour"])
        invalid_hour_count = int(parsed_hours.isna().sum())
        if invalid_hour_count:
            issues.append(
                {
                    "severity": "warning",
                    "table": "impressions",
                    "code": "invalid_hour",
                    "details": {"invalid_count": invalid_hour_count},
                }
            )

    unknown_character_count = 0
    if "character_id" in impressions.columns and "character_id" in characters.columns:
        impression_character_ids = impressions["character_id"].astype("string").fillna("unknown")
        known_character_ids = set(characters["character_id"].astype("string").fillna("unknown"))
        unknown_character_count = int((~impression_character_ids.isin(known_character_ids)).sum())
        if unknown_character_count:
            issues.append(
                {
                    "severity": "warning",
                    "table": "impressions",
                    "code": "unjoined_character_ids",
                    "details": {"unjoined_count": unknown_character_count},
                }
            )

    duplicate_impression_id_count = 0
    if "id" in impressions.columns:
        duplicate_impression_id_count = int(impressions["id"].duplicated().sum())
        if duplicate_impression_id_count:
            issues.append(
                {
                    "severity": "warning",
                    "table": "impressions",
                    "code": "duplicate_ids",
                    "details": {"duplicate_count": duplicate_impression_id_count},
                }
            )

    duplicate_character_id_count = 0
    if "character_id" in characters.columns:
        duplicate_character_id_count = int(characters["character_id"].duplicated().sum())
        if duplicate_character_id_count:
            issues.append(
                {
                    "severity": "warning",
                    "table": "characters",
                    "code": "duplicate_ids",
                    "details": {"duplicate_count": duplicate_character_id_count},
                }
            )

    future_character_created_at_count = 0
    if (
        "character_id" in impressions.columns
        and "character_id" in characters.columns
        and "created_at" in characters.columns
        and "hour" in impressions.columns
    ):
        character_created_at_source = characters.drop_duplicates("character_id", keep="last").copy()
        character_created_at_source["character_id"] = character_created_at_source["character_id"].astype("string")
        character_created_at = pd.to_datetime(
            character_created_at_source.set_index("character_id")["created_at"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)
        impression_character_ids = impressions["character_id"].astype("string")
        created_at_by_impression = impression_character_ids.map(character_created_at)
        future_character_created_at_count = int((created_at_by_impression > parsed_hours).sum())
        if future_character_created_at_count:
            issues.append(
                {
                    "severity": "warning",
                    "table": "characters",
                    "code": "future_created_at",
                    "details": {"count": future_character_created_at_count},
                }
            )

    negative_counts = {
        "conversation_turn": _numeric_negative_count(impressions, "conversation_turn"),
        "session_msg_count": _numeric_negative_count(impressions, "session_msg_count"),
        "num_interactions": _numeric_negative_count(characters, "num_interactions"),
    }
    for column, count in negative_counts.items():
        if count:
            table = "characters" if column == "num_interactions" else "impressions"
            issues.append(
                {
                    "severity": "warning",
                    "table": table,
                    "code": "negative_numeric_values",
                    "details": {"column": column, "count": count},
                }
            )

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    return {
        "status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "warning_count": warning_count,
        "row_counts": {
            "impressions": int(len(impressions)),
            "characters": int(len(characters)),
        },
        "schema": {
            "missing_impression_columns": impression_missing,
            "missing_character_columns": character_missing,
        },
        "quality": {
            "invalid_click_count": click_invalid_count,
            "invalid_hour_count": invalid_hour_count,
            "unknown_character_count": unknown_character_count,
            "duplicate_impression_id_count": duplicate_impression_id_count,
            "duplicate_character_id_count": duplicate_character_id_count,
            "future_character_created_at_count": future_character_created_at_count,
            "negative_counts": negative_counts,
            "null_counts": {
                "hour": _null_count(impressions, "hour"),
                "character_id": _null_count(impressions, "character_id"),
                "safety_tier": _null_count(characters, "safety_tier"),
                "creator_type": _null_count(characters, "creator_type"),
            },
        },
        "issues": issues,
    }


def build_data_validation_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Data Validation Report",
        "",
        f"- Status: {payload['status']}",
        f"- Errors: {payload['error_count']}",
        f"- Warnings: {payload['warning_count']}",
        f"- Impression rows: {payload['row_counts']['impressions']}",
        f"- Character rows: {payload['row_counts']['characters']}",
        "",
        "## Quality Checks",
        "",
    ]
    quality = payload["quality"]
    for key, value in quality.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Issues", ""])
    if not payload["issues"]:
        lines.append("No blocking issues detected.")
    else:
        for issue in payload["issues"]:
            lines.append(f"- {issue['severity']} {issue['table']} {issue['code']}: {issue['details']}")
    return "\n".join(lines) + "\n"


def write_data_validation_outputs(settings: Settings, payload: dict[str, Any]) -> tuple[Path, Path]:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = settings.reports_dir / "data_validation.json"
    report_path = settings.reports_dir / "data_validation.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_data_validation_report(payload), encoding="utf-8")
    return json_path, report_path
