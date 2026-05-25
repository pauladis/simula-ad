from __future__ import annotations

from pathlib import Path

import pandas as pd

from simula_ctr.config import Settings, get_settings
from simula_ctr.features import C_FIELDS, build_features


IMPRESSION_REQUIRED_COLUMNS = [
    "id",
    "hour",
    "click",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    *C_FIELDS,
    "character_id",
    "conversation_turn",
    "session_msg_count",
]

CHARACTER_REQUIRED_COLUMNS = [
    "character_id",
    "character_name",
    "character_description",
    "safety_tier",
    "creator_type",
    "num_interactions",
    "created_at",
]

IMPRESSION_STRING_COLUMNS = [
    "id",
    "hour",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    *C_FIELDS,
    "character_id",
]

CHARACTER_STRING_COLUMNS = [
    "character_id",
    "character_name",
    "character_description",
    "safety_tier",
    "creator_type",
    "created_at",
]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return path


def validate_columns(frame: pd.DataFrame, required_columns: list[str], table_name: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _read_csv(path: Path, string_columns: list[str]) -> pd.DataFrame:
    dtypes = {column: "string" for column in string_columns}
    return pd.read_csv(require_file(path), dtype=dtypes)


def load_impressions(settings: Settings | None = None, validate: bool = True) -> pd.DataFrame:
    settings = settings or get_settings()
    frame = _read_csv(settings.impressions_path, IMPRESSION_STRING_COLUMNS)
    if validate:
        validate_columns(frame, IMPRESSION_REQUIRED_COLUMNS, "impressions.csv")
    for column in ["click", "conversation_turn", "session_msg_count"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_characters(settings: Settings | None = None, validate: bool = True) -> pd.DataFrame:
    settings = settings or get_settings()
    frame = _read_csv(settings.characters_path, CHARACTER_STRING_COLUMNS)
    if validate:
        validate_columns(frame, CHARACTER_REQUIRED_COLUMNS, "characters.csv")
    if "num_interactions" in frame.columns:
        frame["num_interactions"] = pd.to_numeric(frame["num_interactions"], errors="coerce")
    return frame


def load_training_tables(settings: Settings | None = None, validate: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or get_settings()
    return load_impressions(settings, validate=validate), load_characters(settings, validate=validate)


def load_feature_frame(settings: Settings | None = None) -> pd.DataFrame:
    impressions, characters = load_training_tables(settings)
    return build_features(impressions, characters)
