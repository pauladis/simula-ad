from __future__ import annotations

import pandas as pd
import pytest

from simula_ctr.drift import (
    build_drift_metrics,
    normalized_distribution_delta,
    population_stability_index,
)


def test_distribution_shift_metrics_increase_when_mix_changes() -> None:
    reference = {"a": 0.8, "b": 0.2}
    current = {"a": 0.2, "b": 0.8}

    assert population_stability_index(reference, current) > 0
    assert normalized_distribution_delta(reference, current) == pytest.approx(0.6)


def test_build_drift_metrics_reports_ctr_delta_and_feature_shift() -> None:
    impressions = pd.DataFrame(
        [
            {"id": "1", "hour": 14102100, "click": 0, "character_id": "c1", "app_category": "game", "site_category": "chat", "device_type": 1},
            {"id": "2", "hour": 14102101, "click": 0, "character_id": "c1", "app_category": "game", "site_category": "chat", "device_type": 1},
            {"id": "3", "hour": 14102200, "click": 1, "character_id": "c2", "app_category": "social", "site_category": "chat", "device_type": 4},
            {"id": "4", "hour": 14102201, "click": 1, "character_id": "c2", "app_category": "social", "site_category": "chat", "device_type": 4},
            {"id": "5", "hour": 14102300, "click": 1, "character_id": "c2", "app_category": "social", "site_category": "chat", "device_type": 4},
            {"id": "6", "hour": 14102301, "click": 1, "character_id": "c2", "app_category": "social", "site_category": "chat", "device_type": 4},
        ]
    )
    characters = pd.DataFrame(
        [
            {"character_id": "c1", "character_name": "one", "character_description": "", "safety_tier": "sfw", "creator_type": "studio", "num_interactions": 10, "created_at": "2014-10-01"},
            {"character_id": "c2", "character_name": "two", "character_description": "", "safety_tier": "mature", "creator_type": "community", "num_interactions": 20, "created_at": "2014-10-01"},
        ]
    )

    metrics = build_drift_metrics(impressions, characters)

    assert metrics["status"] == "ok"
    assert metrics["periods"]["early"]["ctr"] == 0.0
    assert metrics["periods"]["latest"]["ctr"] == 1.0
    assert metrics["ctr_delta_latest_vs_early"] == 1.0
    assert metrics["feature_shift"]["safety_tier"]["psi"] > 0
