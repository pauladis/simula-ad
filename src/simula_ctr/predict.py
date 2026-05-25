from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import joblib
import pandas as pd

from simula_ctr.config import Settings, get_settings
from simula_ctr.features import FEATURE_COLUMNS, build_features, select_feature_columns
from simula_ctr.schemas import ImpressionContext, ModelInfoResponse, PredictionResponse


logger = logging.getLogger(__name__)


def to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    return dict(value)


def _clamp_probability(value: float) -> float:
    return min(max(value, 0.0001), 0.9999)


@dataclass
class FallbackCtrPredictor:
    global_ctr: float = 0.05
    model_version: str = "untrained-0.1.0"
    model_name: str = "global_ctr_fallback"
    fallback_reason: str | None = None

    def predict_one(self, context: ImpressionContext | dict[str, Any]) -> PredictionResponse:
        payload = to_plain_dict(context)
        score = self.global_ctr
        reasons = ["fallback global CTR prior"]
        if self.fallback_reason:
            reasons.append(self.fallback_reason)

        app_category = str(payload.get("app_category") or "unknown")
        site_category = str(payload.get("site_category") or "unknown")
        if app_category != "unknown":
            score *= 1.05
            reasons.append("app category available")
        if site_category != "unknown":
            score *= 1.03
            reasons.append("site category available")

        turn = float(payload.get("conversation_turn") or 0)
        session_count = float(payload.get("session_msg_count") or 0)
        if session_count > 0 and turn / session_count <= 0.25:
            score *= 1.02
            reasons.append("early session context")
        elif session_count > 0:
            score *= 0.98
            reasons.append("later session context")

        return PredictionResponse(
            predicted_ctr=round(_clamp_probability(score), 6),
            model_version=self.model_version,
            reasons=reasons,
        )

    def predict_batch(self, contexts: list[ImpressionContext]) -> list[PredictionResponse]:
        return [self.predict_one(context) for context in contexts]

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_name=self.model_name,
            model_version=self.model_version,
            trained_at=None,
            features=FEATURE_COLUMNS,
            metrics={"serving_mode": "fallback", "fallback_reason": self.fallback_reason},
        )


@dataclass
class ArtifactCtrPredictor:
    model: Any
    metadata: dict[str, Any]

    def predict_one(self, context: ImpressionContext | dict[str, Any]) -> PredictionResponse:
        payload = to_plain_dict(context)
        frame = build_features(pd.DataFrame([payload]))
        features = select_feature_columns(frame)
        predicted_ctr = float(self.model.predict_proba(features)[0, 1])
        return PredictionResponse(
            predicted_ctr=round(_clamp_probability(predicted_ctr), 6),
            model_version=str(self.metadata.get("model_version", "unknown")),
            reasons=["CatBoost CTR model", "online context features"],
        )

    def predict_batch(self, contexts: list[ImpressionContext]) -> list[PredictionResponse]:
        if not contexts:
            return []
        payloads = [to_plain_dict(context) for context in contexts]
        frame = build_features(pd.DataFrame(payloads))
        features = select_feature_columns(frame)
        predictions = self.model.predict_proba(features)[:, 1]
        model_version = str(self.metadata.get("model_version", "unknown"))
        return [
            PredictionResponse(
                predicted_ctr=round(_clamp_probability(float(prediction)), 6),
                model_version=model_version,
                reasons=["CatBoost CTR model", "online context features"],
            )
            for prediction in predictions
        ]

    def model_info(self) -> ModelInfoResponse:
        evaluation = self.metadata.get("evaluation", {})
        test_metrics = evaluation.get("splits", {}).get("test", {}).get("metrics", {})
        return ModelInfoResponse(
            model_name=str(self.metadata.get("model_name", "catboost_ctr")),
            model_version=str(self.metadata.get("model_version", "unknown")),
            trained_at=self.metadata.get("trained_at"),
            features=list(self.metadata.get("feature_columns", FEATURE_COLUMNS)),
            metrics=test_metrics,
        )


@dataclass
class ResilientCtrPredictor:
    primary: ArtifactCtrPredictor
    fallback: FallbackCtrPredictor

    def predict_one(self, context: ImpressionContext | dict[str, Any]) -> PredictionResponse:
        try:
            return self.primary.predict_one(context)
        except Exception as exc:
            logger.exception("Primary model inference failed; using fallback predictor.")
            fallback = self.fallback.predict_one(context)
            fallback.reasons.insert(0, f"primary model unavailable: {exc.__class__.__name__}")
            return fallback

    def predict_batch(self, contexts: list[ImpressionContext]) -> list[PredictionResponse]:
        try:
            return self.primary.predict_batch(contexts)
        except Exception as exc:
            logger.exception("Primary model batch inference failed; using fallback predictor.")
            predictions = self.fallback.predict_batch(contexts)
            for prediction in predictions:
                prediction.reasons.insert(0, f"primary model unavailable: {exc.__class__.__name__}")
            return predictions

    def model_info(self) -> ModelInfoResponse:
        info = self.primary.model_info()
        info.metrics = {**info.metrics, "serving_mode": "primary_with_fallback"}
        return info


def load_predictor(settings: Settings | None = None) -> FallbackCtrPredictor | ResilientCtrPredictor:
    settings = settings or get_settings()
    fallback = FallbackCtrPredictor(
        global_ctr=settings.fallback_ctr,
        model_version=settings.model_version,
    )
    if settings.model_artifact_path.exists():
        try:
            artifact = joblib.load(settings.model_artifact_path)
            model = artifact["model"]
            metadata = artifact.get("metadata", {})
            return ResilientCtrPredictor(
                primary=ArtifactCtrPredictor(model=model, metadata=metadata),
                fallback=FallbackCtrPredictor(
                    global_ctr=float(metadata.get("global_ctr", settings.fallback_ctr)),
                    model_version=str(metadata.get("model_version", settings.model_version)),
                    fallback_reason="primary model load succeeded but inference failed",
                ),
            )
        except Exception as exc:
            logger.warning("Could not load model artifact %s; using fallback predictor: %s", settings.model_artifact_path, exc)
            fallback.fallback_reason = "model artifact unavailable or incompatible"
            return fallback
    logger.info("Model artifact not found at %s; using fallback predictor.", settings.model_artifact_path)
    fallback.fallback_reason = "model artifact not found"
    return fallback
