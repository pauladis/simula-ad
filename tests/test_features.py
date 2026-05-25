from __future__ import annotations

import math

import pandas as pd
import pytest

from simula_ctr.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    add_time_features,
    add_turn_features,
    build_features,
    fill_missing_values,
    join_character_features,
    parse_hour,
    select_feature_columns,
)


def test_parse_hour_avazu_style_timestamp() -> None:
    parsed = parse_hour("14102100")
    assert parsed.year == 2014
    assert parsed.month == 10
    assert parsed.day == 21
    assert parsed.hour == 0


def test_parse_hour_accepts_int_float_and_iso_values() -> None:
    assert parse_hour(14102113).hour == 13
    assert parse_hour(14102113.0).hour == 13
    assert parse_hour("2014-10-21 09:30:00").hour == 9
    assert pd.isna(parse_hour(""))
    assert pd.isna(parse_hour("not-a-timestamp"))


def test_add_time_features_extracts_calendar_fields_and_missing_timestamp() -> None:
    frame = pd.DataFrame({"hour": ["14102512", None]})

    features = add_time_features(frame)

    assert features.loc[0, "event_timestamp"] == pd.Timestamp("2014-10-25 12:00:00")
    assert features.loc[0, "hour_of_day"] == 12
    assert features.loc[0, "day_of_week"] == 5
    assert features.loc[0, "is_weekend"] == 1
    assert pd.isna(features.loc[1, "event_timestamp"])
    assert features.loc[1, "hour_of_day"] == -1
    assert features.loc[1, "day_of_week"] == -1
    assert features.loc[1, "is_weekend"] == 0


def test_add_turn_features_handles_strings_missing_zero_denominator_and_negative_values() -> None:
    frame = pd.DataFrame(
        [
            {"conversation_turn": "2", "session_msg_count": "4"},
            {"conversation_turn": 3, "session_msg_count": 0},
            {"conversation_turn": None, "session_msg_count": None},
            {"conversation_turn": -2, "session_msg_count": -3},
        ]
    )

    features = add_turn_features(frame)

    assert features.loc[0, "conversation_turn"] == 2
    assert features.loc[0, "session_msg_count"] == 4
    assert features.loc[0, "turn_ratio"] == 0.5
    assert features.loc[1, "turn_ratio"] == 0
    assert features.loc[2, "turn_ratio"] == 0
    assert features.loc[3, "conversation_turn"] == 0
    assert features.loc[3, "session_msg_count"] == 0


def test_join_character_features_adds_metadata_age_and_preserves_impression_rows() -> None:
    impressions = add_time_features(
        pd.DataFrame(
            [
                {"hour": "14102100", "character_id": "char-1"},
                {"hour": "14102100", "character_id": "char-missing"},
            ]
        )
    )
    characters = pd.DataFrame(
        [
            {
                "character_id": "char-1",
                "character_name": "older-copy",
                "character_description": "old",
                "safety_tier": "mature",
                "creator_type": "studio",
                "num_interactions": 10,
                "created_at": "2014-10-10",
            },
            {
                "character_id": "char-1",
                "character_name": "newer-copy",
                "character_description": "new",
                "safety_tier": "sfw",
                "creator_type": "community",
                "num_interactions": 20,
                "created_at": "2014-10-01",
            },
        ]
    )

    joined = join_character_features(impressions, characters)

    assert len(joined) == 2
    assert joined.loc[0, "character_name"] == "newer-copy"
    assert joined.loc[0, "safety_tier"] == "sfw"
    assert joined.loc[0, "creator_type"] == "community"
    assert joined.loc[0, "num_interactions"] == 20
    assert joined.loc[0, "character_age_days"] == 20
    assert pd.isna(joined.loc[1, "safety_tier"])
    assert joined.loc[1, "character_age_days"] == 0


def test_fill_missing_values_creates_required_categoricals_and_numeric_defaults() -> None:
    frame = pd.DataFrame(
        [
            {
                "site_id": None,
                "app_category": "",
                "conversation_turn": "bad",
                "session_msg_count": None,
                "num_interactions": -5,
            }
        ]
    )

    features = fill_missing_values(frame)

    for column in CATEGORICAL_COLUMNS:
        assert column in features.columns
        assert isinstance(features.loc[0, column], str)
    assert features.loc[0, "site_id"] == "unknown"
    assert features.loc[0, "app_category"] == "unknown"
    assert features.loc[0, "conversation_turn"] == 0
    assert features.loc[0, "session_msg_count"] == 0
    assert features.loc[0, "num_interactions"] == 0
    assert features.loc[0, "num_interactions_log1p"] == 0


def test_build_features_turn_ratio_character_join_and_missing_values() -> None:
    impressions = pd.DataFrame(
        [
            {
                "hour": "14102112",
                "site_id": None,
                "app_category": "game",
                "character_id": "char-1",
                "conversation_turn": 2,
                "session_msg_count": 4,
            }
        ]
    )
    characters = pd.DataFrame(
        [
            {
                "character_id": "char-1",
                "safety_tier": "sfw",
                "creator_type": "community",
                "num_interactions": 9,
                "created_at": "2014-10-01",
            }
        ]
    )

    features = build_features(impressions, characters)

    assert features.loc[0, "turn_ratio"] == 0.5
    assert features.loc[0, "safety_tier"] == "sfw"
    assert features.loc[0, "creator_type"] == "community"
    assert features.loc[0, "site_id"] == "unknown"
    assert features.loc[0, "num_interactions_log1p"] == pytest.approx(math.log1p(9))
    assert features.loc[0, "character_age_days"] == 20


def test_build_features_without_character_table_uses_safe_defaults() -> None:
    features = build_features(
        pd.DataFrame(
            [
                {
                    "hour": "14102112",
                    "character_id": "new-character",
                    "conversation_turn": 1,
                    "session_msg_count": 2,
                }
            ]
        )
    )

    assert features.loc[0, "safety_tier"] == "unknown"
    assert features.loc[0, "creator_type"] == "unknown"
    assert features.loc[0, "num_interactions"] == 0
    assert features.loc[0, "character_age_days"] == 0
    assert features.loc[0, "turn_ratio"] == 0.5


def test_select_feature_columns_returns_ordered_model_features() -> None:
    features = build_features(pd.DataFrame([{"hour": "14102112"}]))

    selected = select_feature_columns(features)

    assert selected.columns.tolist() == FEATURE_COLUMNS


def test_select_feature_columns_raises_for_missing_features() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        select_feature_columns(pd.DataFrame({"hour_of_day": [1]}))
