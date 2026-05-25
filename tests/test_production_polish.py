from __future__ import annotations

from pathlib import Path

import pandas as pd

from simula_ctr.config import Settings
from simula_ctr.predict import FallbackCtrPredictor, ResilientCtrPredictor, load_predictor
from simula_ctr.production_reports import run_latency_benchmark, write_sample_ranked_output
from simula_ctr.validation import validate_input_tables, write_data_validation_outputs


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        impressions_path=tmp_path / "data" / "impressions.csv",
        characters_path=tmp_path / "data" / "characters.csv",
        model_artifact_path=tmp_path / "artifacts" / "ctr_model.joblib",
        model_name="test-model",
        model_version="test-version",
        random_seed=42,
        fallback_ctr=0.05,
        cold_start_min_impressions=500,
    )


def test_data_validation_reports_invalid_target_and_unjoined_character(tmp_path: Path) -> None:
    impressions = pd.DataFrame(
        [
            {
                "id": "1",
                "hour": "bad-hour",
                "click": 2,
                "character_id": "missing-character",
                "conversation_turn": -1,
                "session_msg_count": 2,
            }
        ]
    )
    characters = pd.DataFrame(
        [
            {
                "character_id": "char-1",
                "character_name": "guide",
                "character_description": "Helpful.",
                "safety_tier": "sfw",
                "creator_type": "community",
                "num_interactions": 10,
                "created_at": "2014-10-01",
            }
        ]
    )

    report = validate_input_tables(impressions, characters)

    assert report["status"] == "fail"
    assert report["quality"]["invalid_click_count"] == 1
    assert report["quality"]["invalid_hour_count"] == 1
    assert report["quality"]["unknown_character_count"] == 1

    settings = _settings(tmp_path)
    json_path, report_path = write_data_validation_outputs(settings, report)
    assert json_path.exists()
    assert report_path.exists()


def test_data_validation_flags_future_character_creation_with_mixed_id_types() -> None:
    impressions = pd.DataFrame(
        [
            {
                "id": "1",
                "hour": 14102100,
                "click": 0,
                "character_id": "101",
                "conversation_turn": 1,
                "session_msg_count": 2,
            }
        ]
    )
    characters = pd.DataFrame(
        [
            {
                "character_id": 101,
                "character_name": "guide",
                "character_description": "Helpful.",
                "safety_tier": "sfw",
                "creator_type": "community",
                "num_interactions": 10,
                "created_at": "2014-10-22",
            }
        ]
    )

    report = validate_input_tables(impressions, characters)

    assert report["quality"]["future_character_created_at_count"] == 1
    assert any(issue["code"] == "future_created_at" for issue in report["issues"])


def test_missing_model_artifact_uses_fallback_predictor(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.artifacts_dir.mkdir(parents=True)

    predictor = load_predictor(settings)
    prediction = predictor.predict_one({"hour": 14102100, "app_category": "game"})

    assert prediction.predicted_ctr > 0
    assert "fallback global CTR prior" in prediction.reasons


def test_resilient_predictor_falls_back_when_primary_inference_fails() -> None:
    class BrokenPrimary:
        def predict_one(self, context: dict) -> None:
            raise RuntimeError("primary failed")

        def predict_batch(self, contexts: list[dict]) -> None:
            raise RuntimeError("primary failed")

    predictor = ResilientCtrPredictor(primary=BrokenPrimary(), fallback=FallbackCtrPredictor())

    prediction = predictor.predict_one({"hour": 14102100, "app_category": "game"})

    assert prediction.predicted_ctr > 0
    assert prediction.reasons[0] == "primary model unavailable: RuntimeError"


def test_sample_ranked_output_and_latency_benchmark_use_fallback(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.reports_dir.mkdir(parents=True)

    sample_path = write_sample_ranked_output(settings)
    benchmark = run_latency_benchmark(iterations=2, warmup=0)

    assert sample_path.exists()
    assert benchmark["predict"]["count"] == 2
    assert benchmark["rank"]["p99_ms"] >= 0
