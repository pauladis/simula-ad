from __future__ import annotations

import pytest
from fastapi import HTTPException

from simula_ctr.api import app, batch_predict, health, rank
from simula_ctr.predict import to_plain_dict
from simula_ctr.schemas import CandidateAd, ImpressionContext, RankRequest


def test_health_endpoint() -> None:
    response = health()

    assert to_plain_dict(response) == {"status": "ok"}


def test_rank_endpoint_applies_safety_gate() -> None:
    response = rank(
        RankRequest(
            context=ImpressionContext(hour=14102100, safety_tier="sfw", app_category="game"),
            candidates=[
                CandidateAd(candidate_id="safe", safety_tier="sfw", fatigue_count=0, pacing_ratio=1.0),
                CandidateAd(candidate_id="mature", safety_tier="mature", fatigue_count=0, pacing_ratio=1.0),
            ],
        )
    )

    payload = to_plain_dict(response)
    assert [candidate["candidate_id"] for candidate in payload["ranked_candidates"]] == ["safe"]


def test_rank_endpoint_rejects_empty_candidates() -> None:
    with pytest.raises(HTTPException) as exc_info:
        rank(RankRequest(context=ImpressionContext(hour=14102100, safety_tier="sfw"), candidates=[]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"]["code"] == "empty_candidates"


def test_batch_predict_rejects_empty_batch() -> None:
    with pytest.raises(HTTPException) as exc_info:
        batch_predict([])

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"]["code"] == "empty_batch"


def test_openapi_contains_rank_request_schema() -> None:
    """Verify /rank endpoint is documented with a request body schema."""
    schema = app.openapi()
    request_body = schema["paths"]["/rank"]["post"]["requestBody"]

    media_type = request_body["content"]["application/json"]
    # Check that schema reference exists (examples handling varies by FastAPI/Pydantic version)
    assert "schema" in media_type
    assert "$ref" in media_type["schema"] or "properties" in media_type["schema"]
