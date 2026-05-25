from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from simula_ctr.config import get_settings
from simula_ctr.logging_config import configure_logging
from simula_ctr.predict import load_predictor
from simula_ctr.ranker import CampaignRanker
from simula_ctr.schemas import (
    HealthResponse,
    ImpressionContext,
    ModelInfoResponse,
    PredictionResponse,
    RankRequest,
    RankResponse,
    EXAMPLE_ERROR_RESPONSE,
    EXAMPLE_IMPRESSION_CONTEXT,
    EXAMPLE_RANK_REQUEST,
    ErrorResponse,
)


configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
predictor = load_predictor(settings)
ranker = CampaignRanker(predictor)

app = FastAPI(
    title="Simula CTR Ranking API",
    version="0.1.0",
    description=(
        "Production-oriented serving API for contextual ad CTR prediction and campaign ranking. "
        "The service uses the saved CatBoost artifact when available and falls back to safe CTR priors."
    ),
)


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Bad request", "content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
    413: {"model": ErrorResponse, "description": "Request too large", "content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
    422: {"model": ErrorResponse, "description": "Validation error", "content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
    503: {"model": ErrorResponse, "description": "Serving error", "content": {"application/json": {"example": EXAMPLE_ERROR_RESPONSE}}},
}


def _error_payload(code: str, message: str, details: Any | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or []}}


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next: Any) -> Response:
    request_id = request.headers.get("x-request-id", str(uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{duration_ms:.3f}"
    logger.info(
        "api_request path=%s method=%s status=%s duration_ms=%.3f request_id=%s",
        request.url.path,
        request.method,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_error_payload("validation_error", "Request validation failed.", exc.errors()),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail), "details": []}
    if "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload("http_error", str(detail.get("message", "HTTP error.")), detail.get("details", [])),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error")
    return JSONResponse(
        status_code=503,
        content=_error_payload("serving_error", "CTR serving failed.", [{"type": exc.__class__.__name__}]),
    )


def _serving_error(exc: Exception) -> HTTPException:
    logger.exception("Serving operation failed")
    return HTTPException(
        status_code=503,
        detail=_error_payload("serving_error", "CTR serving failed.", [{"type": exc.__class__.__name__}]),
    )


def _validate_batch_size(contexts: list[ImpressionContext]) -> None:
    if not contexts:
        raise HTTPException(status_code=400, detail=_error_payload("empty_batch", "Batch predict requires at least one context."))
    if len(contexts) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=_error_payload(
                "batch_too_large",
                f"Batch size exceeds configured maximum of {settings.max_batch_size}.",
                [{"max_batch_size": settings.max_batch_size, "received": len(contexts)}],
            ),
        )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ModelInfoResponse, tags=["ops"], responses=ERROR_RESPONSES)
def ready() -> ModelInfoResponse:
    try:
        return predictor.model_info()
    except Exception as exc:
        raise _serving_error(exc) from exc


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"], responses=ERROR_RESPONSES)
def model_info() -> ModelInfoResponse:
    try:
        return predictor.model_info()
    except Exception as exc:
        raise _serving_error(exc) from exc


@app.post("/predict", response_model=PredictionResponse, tags=["ctr"], responses=ERROR_RESPONSES)
def predict(
    context: ImpressionContext = Body(
        ...,
        examples={
            "standard": {
                "summary": "Single contextual ad opportunity",
                "value": EXAMPLE_IMPRESSION_CONTEXT,
            }
        },
    )
) -> PredictionResponse:
    try:
        return predictor.predict_one(context)
    except Exception as exc:
        raise _serving_error(exc) from exc


@app.post("/rank", response_model=RankResponse, tags=["ranking"], responses=ERROR_RESPONSES)
def rank(
    request: RankRequest = Body(
        ...,
        examples={
            "standard": {
                "summary": "Rank candidate campaigns for one opportunity",
                "value": EXAMPLE_RANK_REQUEST,
            }
        },
    )
) -> RankResponse:
    if not request.candidates:
        raise HTTPException(status_code=400, detail=_error_payload("empty_candidates", "Rank requires at least one candidate."))
    if len(request.candidates) > settings.max_rank_candidates:
        raise HTTPException(
            status_code=413,
            detail=_error_payload(
                "too_many_candidates",
                f"Candidate count exceeds configured maximum of {settings.max_rank_candidates}.",
                [{"max_rank_candidates": settings.max_rank_candidates, "received": len(request.candidates)}],
            ),
        )
    try:
        return RankResponse(ranked_candidates=ranker.rank(request.context, request.candidates))
    except Exception as exc:
        raise _serving_error(exc) from exc


@app.post("/batch-predict", response_model=list[PredictionResponse], tags=["ctr"], responses=ERROR_RESPONSES)
def batch_predict(
    contexts: list[ImpressionContext] = Body(
        ...,
        examples={
            "small_batch": {
                "summary": "Two impression contexts",
                "value": [EXAMPLE_IMPRESSION_CONTEXT, {**EXAMPLE_IMPRESSION_CONTEXT, "id": "impression-2", "conversation_turn": 6}],
            }
        },
    )
) -> list[PredictionResponse]:
    _validate_batch_size(contexts)
    try:
        return predictor.predict_batch(contexts)
    except Exception as exc:
        raise _serving_error(exc) from exc
