from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 compatibility
    ConfigDict = None  # type: ignore[assignment]


SafetyTier = Literal["sfw", "suggestive", "mature", "unknown"]

EXAMPLE_IMPRESSION_CONTEXT = {
    "hour": 14102112,
    "banner_pos": 0,
    "site_id": "8fda644b",
    "site_domain": "25d4cfcd",
    "site_category": "f028772b",
    "app_id": "ecad2386",
    "app_domain": "7801e8d9",
    "app_category": "07d7df22",
    "device_id": "a99f214a",
    "device_ip": "b264c159",
    "device_model": "be6db1d7",
    "device_type": 1,
    "device_conn_type": 0,
    "C1": 1005,
    "C14": 15708,
    "C15": 320,
    "C16": 50,
    "C17": 1722,
    "C18": 0,
    "C19": 35,
    "C20": 100083,
    "C21": 79,
    "character_id": "char-123",
    "conversation_turn": 2,
    "session_msg_count": 8,
    "safety_tier": "sfw",
    "creator_type": "community",
    "num_interactions": 350,
}

EXAMPLE_CANDIDATES = [
    {
        "candidate_id": "campaign-safe-1",
        "campaign_id": "camp-001",
        "advertiser_id": "adv-001",
        "banner_pos": 0,
        "C1": 1005,
        "C14": 15708,
        "C15": 320,
        "C16": 50,
        "C17": 1722,
        "C18": 0,
        "C19": 35,
        "C20": 100083,
        "C21": 79,
        "bid": 1.2,
        "safety_tier": "sfw",
        "fatigue_count": 1,
        "pacing_ratio": 0.92,
    },
    {
        "candidate_id": "campaign-suggestive-1",
        "campaign_id": "camp-002",
        "advertiser_id": "adv-002",
        "banner_pos": 0,
        "C1": 1005,
        "C14": 20362,
        "C15": 320,
        "C16": 50,
        "C17": 2333,
        "C18": 0,
        "C19": 39,
        "C20": -1,
        "C21": 157,
        "bid": 0.9,
        "safety_tier": "suggestive",
        "fatigue_count": 0,
        "pacing_ratio": 0.75,
    },
]

EXAMPLE_RANK_REQUEST = {
    "context": EXAMPLE_IMPRESSION_CONTEXT,
    "candidates": EXAMPLE_CANDIDATES,
}

EXAMPLE_PREDICTION_RESPONSE = {
    "predicted_ctr": 0.1234,
    "model_version": "ctr-20260525T020257Z",
    "reasons": ["CatBoost CTR model", "online context features"],
}

EXAMPLE_RANK_RESPONSE = {
    "ranked_candidates": [
        {
            "candidate_id": "campaign-safe-1",
            "rank": 1,
            "predicted_ctr": 0.123,
            "final_score": 0.119,
            "reasons": ["CatBoost CTR model", "pacing on target", "fatigue within cap", "safe tier match"],
        }
    ]
}

EXAMPLE_ERROR_RESPONSE = {
    "error": {
        "code": "validation_error",
        "message": "Request validation failed.",
        "details": [{"loc": ["body", "candidates"], "msg": "field required"}],
    }
}


class ApiBaseModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(protected_namespaces=())
    else:
        class Config:
            protected_namespaces = ()


class FlexibleApiModel(ApiBaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow", protected_namespaces=())
    else:
        class Config:
            extra = "allow"
            protected_namespaces = ()


class ImpressionContext(FlexibleApiModel):
    if ConfigDict is not None:
        model_config = ConfigDict(
            extra="allow",
            protected_namespaces=(),
            json_schema_extra={"examples": [EXAMPLE_IMPRESSION_CONTEXT]},
        )
    else:
        class Config:
            extra = "allow"
            protected_namespaces = ()
            schema_extra = {"example": EXAMPLE_IMPRESSION_CONTEXT}

    id: str | None = None
    hour: int | str | None = None
    banner_pos: int | str | None = None
    site_id: str | None = None
    site_domain: str | None = None
    site_category: str | None = None
    app_id: str | None = None
    app_domain: str | None = None
    app_category: str | None = None
    device_id: str | None = None
    device_ip: str | None = None
    device_model: str | None = None
    device_type: int | str | None = None
    device_conn_type: int | str | None = None
    C1: int | str | None = None
    C14: int | str | None = None
    C15: int | str | None = None
    C16: int | str | None = None
    C17: int | str | None = None
    C18: int | str | None = None
    C19: int | str | None = None
    C20: int | str | None = None
    C21: int | str | None = None
    character_id: str | None = None
    conversation_turn: int | float | None = 0
    session_msg_count: int | float | None = 0
    safety_tier: SafetyTier | None = "unknown"
    creator_type: str | None = "unknown"
    num_interactions: int | float | None = 0
    character_description: str | None = None

class CandidateAd(FlexibleApiModel):
    if ConfigDict is not None:
        model_config = ConfigDict(
            extra="allow",
            protected_namespaces=(),
            json_schema_extra={"examples": EXAMPLE_CANDIDATES},
        )
    else:
        class Config:
            extra = "allow"
            protected_namespaces = ()
            schema_extra = {"example": EXAMPLE_CANDIDATES[0]}

    banner_pos: int | str | None = None
    C1: int | str | None = None
    C14: int | str | None = None
    C15: int | str | None = None
    C16: int | str | None = None
    C17: int | str | None = None
    C18: int | str | None = None
    C19: int | str | None = None
    C20: int | str | None = None
    C21: int | str | None = None
    candidate_id: str | None = None
    advertiser_id: str | None = None
    campaign_id: str | None = None
    bid: float = Field(default=1.0, ge=0.0)
    safety_tier: SafetyTier | None = "unknown"
    fatigue_count: int = Field(default=0, ge=0)
    pacing_ratio: float = Field(default=1.0, ge=0.0)

class PredictionResponse(ApiBaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(protected_namespaces=(), json_schema_extra={"examples": [EXAMPLE_PREDICTION_RESPONSE]})
    else:
        class Config:
            protected_namespaces = ()
            schema_extra = {"example": EXAMPLE_PREDICTION_RESPONSE}

    predicted_ctr: float = Field(ge=0.0, le=1.0)
    model_version: str
    reasons: list[str]


class RankedCandidate(ApiBaseModel):
    candidate_id: str
    rank: int
    predicted_ctr: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0)
    reasons: list[str]


class RankRequest(ApiBaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(protected_namespaces=(), json_schema_extra={"examples": [EXAMPLE_RANK_REQUEST]})
    else:
        class Config:
            protected_namespaces = ()
            schema_extra = {"example": EXAMPLE_RANK_REQUEST}

    context: ImpressionContext
    candidates: list[CandidateAd]


class RankResponse(ApiBaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(protected_namespaces=(), json_schema_extra={"examples": [EXAMPLE_RANK_RESPONSE]})
    else:
        class Config:
            protected_namespaces = ()
            schema_extra = {"example": EXAMPLE_RANK_RESPONSE}

    ranked_candidates: list[RankedCandidate]


class ModelInfoResponse(ApiBaseModel):
    model_name: str
    model_version: str
    trained_at: str | None
    features: list[str]
    metrics: dict[str, Any]


class HealthResponse(ApiBaseModel):
    status: str


class ErrorResponse(ApiBaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(protected_namespaces=(), json_schema_extra={"examples": [EXAMPLE_ERROR_RESPONSE]})
    else:
        class Config:
            protected_namespaces = ()
            schema_extra = {"example": EXAMPLE_ERROR_RESPONSE}

    error: dict[str, Any]
