from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


MISSING_CATEGORY = "unknown"
MISSING_NUMERIC = 0.0

C_FIELDS = ["C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]

SITE_APP_DEVICE_CATEGORICAL_COLUMNS = [
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
]

CHARACTER_CATEGORICAL_COLUMNS = [
    "character_id",
    "safety_tier",
    "creator_type",
]

CHARACTER_RAW_COLUMNS = [
    "character_id",
    "character_name",
    "character_description",
    "safety_tier",
    "creator_type",
    "num_interactions",
    "created_at",
]

CATEGORICAL_COLUMNS = [
    *SITE_APP_DEVICE_CATEGORICAL_COLUMNS,
    *CHARACTER_CATEGORICAL_COLUMNS,
    *C_FIELDS,
]

NUMERIC_COLUMNS = [
    "conversation_turn",
    "session_msg_count",
    "turn_ratio",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "num_interactions",
    "num_interactions_log1p",
    "character_age_days",
]

FEATURE_COLUMNS = [*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]


def parse_hour(value: object) -> pd.Timestamp:
    """Parse Avazu-style yyMMddHH values and ISO-like timestamps."""
    if value is None or pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) == 8:
        return pd.to_datetime(text, format="%y%m%d%H", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def parse_hour_series(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    avazu_mask = text.str.fullmatch(r"\d{8}", na=False)
    if avazu_mask.any():
        parsed.loc[avazu_mask] = pd.to_datetime(text.loc[avazu_mask], format="%y%m%d%H", errors="coerce")

    fallback_mask = text.notna() & (text != "") & ~avazu_mask
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(text.loc[fallback_mask], errors="coerce")

    return parsed


def _to_naive_timestamp_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None)


def _column_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def add_time_features(frame: pd.DataFrame, hour_col: str = "hour") -> pd.DataFrame:
    out = frame.copy()
    if hour_col not in out.columns:
        out[hour_col] = pd.NA
    timestamps = parse_hour_series(out[hour_col])
    out["event_timestamp"] = timestamps
    out["hour_of_day"] = timestamps.dt.hour.fillna(-1).astype(int)
    out["day_of_week"] = timestamps.dt.dayofweek.fillna(-1).astype(int)
    out["is_weekend"] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
    return out


def add_turn_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["conversation_turn"] = pd.to_numeric(
        _column_or_default(out, "conversation_turn", MISSING_NUMERIC), errors="coerce"
    ).fillna(MISSING_NUMERIC)
    out["session_msg_count"] = pd.to_numeric(
        _column_or_default(out, "session_msg_count", MISSING_NUMERIC), errors="coerce"
    ).fillna(MISSING_NUMERIC)
    out["conversation_turn"] = out["conversation_turn"].clip(lower=0)
    out["session_msg_count"] = out["session_msg_count"].clip(lower=0)
    denominator = out["session_msg_count"].where(out["session_msg_count"] > 0)
    out["turn_ratio"] = (out["conversation_turn"] / denominator).fillna(MISSING_NUMERIC).clip(lower=0)
    return out


def _deduplicate_characters(characters: pd.DataFrame) -> pd.DataFrame:
    chars = characters.copy()
    if "character_id" not in chars.columns:
        chars["character_id"] = MISSING_CATEGORY
    chars["character_id"] = chars["character_id"].where(chars["character_id"].notna(), MISSING_CATEGORY).astype(str)
    return chars.drop_duplicates(subset=["character_id"], keep="last")


def join_character_features(impressions: pd.DataFrame, characters: pd.DataFrame) -> pd.DataFrame:
    out = impressions.copy()
    if "character_id" not in out.columns:
        out["character_id"] = MISSING_CATEGORY
    out["character_id"] = out["character_id"].where(out["character_id"].notna(), MISSING_CATEGORY).astype(str)

    chars = _deduplicate_characters(characters)
    for column in CHARACTER_RAW_COLUMNS:
        if column not in chars.columns:
            chars[column] = pd.NA

    joined = out.merge(
        chars[CHARACTER_RAW_COLUMNS],
        on="character_id",
        how="left",
        suffixes=("", "_character"),
    )

    for column in ["character_name", "character_description", "safety_tier", "creator_type", "num_interactions", "created_at"]:
        joined_column = f"{column}_character"
        if joined_column in joined.columns:
            if column in joined.columns:
                joined[column] = joined[joined_column].combine_first(joined[column])
            else:
                joined[column] = joined[joined_column]
            joined = joined.drop(columns=[joined_column])

    if "created_at" not in joined.columns:
        joined["created_at"] = pd.NaT
    joined["created_at"] = _to_naive_timestamp_series(joined["created_at"])

    if "event_timestamp" in out.columns:
        reference = _to_naive_timestamp_series(joined["event_timestamp"])
    elif "hour" in out.columns:
        reference = _to_naive_timestamp_series(parse_hour_series(out["hour"]))
    else:
        reference = pd.Series(pd.Timestamp.utcnow().tz_localize(None), index=joined.index)

    reference = reference.fillna(pd.Timestamp.utcnow().tz_localize(None))
    joined["character_age_days"] = (reference - joined["created_at"]).dt.days.fillna(MISSING_NUMERIC).clip(lower=0)
    return joined


def fill_missing_values(frame: pd.DataFrame, categorical_columns: Iterable[str] = CATEGORICAL_COLUMNS) -> pd.DataFrame:
    out = frame.copy()
    for column in categorical_columns:
        if column not in out.columns:
            out[column] = MISSING_CATEGORY
        out[column] = out[column].astype("string").fillna(MISSING_CATEGORY).str.strip()
        out[column] = out[column].replace(
            {"": MISSING_CATEGORY, "nan": MISSING_CATEGORY, "None": MISSING_CATEGORY, "<NA>": MISSING_CATEGORY}
        )
        out[column] = out[column].astype(str)

    for column in NUMERIC_COLUMNS:
        if column not in out.columns:
            out[column] = MISSING_NUMERIC
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(MISSING_NUMERIC)

    out["num_interactions"] = out["num_interactions"].clip(lower=0)
    out["character_age_days"] = out["character_age_days"].clip(lower=0)
    out["num_interactions_log1p"] = out["num_interactions"].map(lambda value: math.log1p(max(float(value), 0.0)))
    return out


def build_features(impressions: pd.DataFrame, characters: pd.DataFrame | None = None) -> pd.DataFrame:
    out = add_time_features(impressions)
    out = add_turn_features(out)
    if characters is not None:
        out = join_character_features(out, characters)
    else:
        out["safety_tier"] = _column_or_default(out, "safety_tier", MISSING_CATEGORY)
        out["creator_type"] = _column_or_default(out, "creator_type", MISSING_CATEGORY)
        out["num_interactions"] = _column_or_default(out, "num_interactions", MISSING_NUMERIC)
        out["character_age_days"] = _column_or_default(out, "character_age_days", MISSING_NUMERIC)
    return fill_missing_values(out)


def select_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature frame is missing required columns: {missing}")
    return frame[FEATURE_COLUMNS].copy()
