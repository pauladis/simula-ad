from __future__ import annotations

import pandas as pd

from simula_ctr.train import time_based_split


def test_time_based_split_keeps_timestamp_buckets_together() -> None:
    frame = pd.DataFrame(
        [
            {"event_timestamp": pd.Timestamp("2014-10-21 00:00:00"), "hour": "14102100", "click": 0},
            {"event_timestamp": pd.Timestamp("2014-10-21 00:00:00"), "hour": "14102100", "click": 1},
            {"event_timestamp": pd.Timestamp("2014-10-21 01:00:00"), "hour": "14102101", "click": 0},
            {"event_timestamp": pd.Timestamp("2014-10-21 01:00:00"), "hour": "14102101", "click": 1},
            {"event_timestamp": pd.Timestamp("2014-10-21 02:00:00"), "hour": "14102102", "click": 0},
            {"event_timestamp": pd.Timestamp("2014-10-21 02:00:00"), "hour": "14102102", "click": 1},
            {"event_timestamp": pd.Timestamp("2014-10-21 03:00:00"), "hour": "14102103", "click": 0},
        ]
    )

    train, validation, test, metadata = time_based_split(frame, train_fraction=0.5, validation_fraction=0.25)

    assert metadata["split_unit"] == "timestamp_bucket"
    assert set(train["event_timestamp"]).isdisjoint(set(validation["event_timestamp"]))
    assert set(train["event_timestamp"]).isdisjoint(set(test["event_timestamp"]))
    assert set(validation["event_timestamp"]).isdisjoint(set(test["event_timestamp"]))


def test_time_based_split_falls_back_when_timestamp_buckets_are_too_coarse() -> None:
    frame = pd.DataFrame(
        [
            {"event_timestamp": pd.Timestamp("2014-10-21 00:00:00"), "hour": "14102100", "click": 0},
            {"event_timestamp": pd.Timestamp("2014-10-21 00:00:00"), "hour": "14102100", "click": 1},
            {"event_timestamp": pd.Timestamp("2014-10-21 00:00:00"), "hour": "14102100", "click": 0},
        ]
    )

    train, validation, test, metadata = time_based_split(frame, train_fraction=0.7, validation_fraction=0.15)

    assert len(train) == 1
    assert len(validation) == 1
    assert len(test) == 1
    assert metadata["split_unit"] == "row"
    assert metadata["warnings"]
