from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from simula_ctr.config import Settings
from simula_ctr.data import load_characters, load_feature_frame, load_impressions, load_training_tables


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        impressions_path=tmp_path / "impressions.csv",
        characters_path=tmp_path / "characters.csv",
        model_artifact_path=tmp_path / "artifacts" / "ctr_model.joblib",
        model_name="test-model",
        model_version="test-version",
        random_seed=42,
        fallback_ctr=0.05,
        cold_start_min_impressions=500,
    )


def _write_valid_inputs(settings: Settings) -> None:
    pd.DataFrame(
        [
            {
                "id": "10000724729988544911",
                "hour": "14102100",
                "click": "1",
                "banner_pos": "0",
                "site_id": "site",
                "site_domain": "domain",
                "site_category": "site-cat",
                "app_id": "app",
                "app_domain": "app-domain",
                "app_category": "app-cat",
                "device_id": "device",
                "device_ip": "ip",
                "device_model": "model",
                "device_type": "1",
                "device_conn_type": "0",
                "C1": "1005",
                "C14": "15708",
                "C15": "320",
                "C16": "50",
                "C17": "1722",
                "C18": "0",
                "C19": "35",
                "C20": "100083",
                "C21": "79",
                "character_id": "char-1",
                "conversation_turn": "2",
                "session_msg_count": "4",
            }
        ]
    ).to_csv(settings.impressions_path, index=False)
    pd.DataFrame(
        [
            {
                "character_id": "char-1",
                "character_name": "guide",
                "character_description": "Helpful guide.",
                "safety_tier": "sfw",
                "creator_type": "community",
                "num_interactions": "12",
                "created_at": "2014-10-01",
            }
        ]
    ).to_csv(settings.characters_path, index=False)


def test_load_impressions_validates_schema_and_preserves_categorical_identifiers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_valid_inputs(settings)

    frame = load_impressions(settings)

    assert frame.loc[0, "id"] == "10000724729988544911"
    assert frame.loc[0, "C14"] == "15708"
    assert frame.loc[0, "click"] == 1
    assert frame.loc[0, "conversation_turn"] == 2
    assert frame.loc[0, "session_msg_count"] == 4


def test_load_characters_validates_schema_and_numeric_interactions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_valid_inputs(settings)

    frame = load_characters(settings)

    assert frame.loc[0, "character_id"] == "char-1"
    assert frame.loc[0, "num_interactions"] == 12


def test_load_training_tables_returns_both_tables(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_valid_inputs(settings)

    impressions, characters = load_training_tables(settings)

    assert len(impressions) == 1
    assert len(characters) == 1


def test_load_feature_frame_builds_joined_feature_table(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_valid_inputs(settings)

    features = load_feature_frame(settings)

    assert features.loc[0, "event_timestamp"] == pd.Timestamp("2014-10-21 00:00:00")
    assert features.loc[0, "turn_ratio"] == 0.5
    assert features.loc[0, "safety_tier"] == "sfw"
    assert features.loc[0, "character_age_days"] == 20


def test_load_impressions_raises_for_missing_required_columns(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    pd.DataFrame([{"id": "1", "hour": "14102100"}]).to_csv(settings.impressions_path, index=False)

    with pytest.raises(ValueError, match="impressions.csv is missing required columns"):
        load_impressions(settings)


def test_load_characters_raises_for_missing_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(FileNotFoundError, match="Required data file not found"):
        load_characters(settings)
