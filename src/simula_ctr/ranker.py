from __future__ import annotations

from typing import Any

from simula_ctr.features import C_FIELDS
from simula_ctr.predict import FallbackCtrPredictor, to_plain_dict
from simula_ctr.schemas import CandidateAd, ImpressionContext, RankedCandidate


AD_FEATURE_FIELDS = ["banner_pos", *C_FIELDS]


def _normalize_tier(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def _context_blocks_candidate(context_tier: str, candidate_tier: str) -> bool:
    return context_tier == "sfw" and candidate_tier == "mature"


def _pacing_multiplier(pacing_ratio: float) -> tuple[float, str]:
    if pacing_ratio > 1.0:
        return max(0.5, 1.0 / pacing_ratio), "pacing reduced overspending campaign"
    if 0.0 <= pacing_ratio < 0.8:
        return 1.05, "pacing boosted underspending campaign"
    return 1.0, "pacing on target"


def _fatigue_penalty(fatigue_count: int) -> tuple[float, str]:
    if fatigue_count <= 3:
        return 1.0, "fatigue within cap"
    penalty = max(0.3, 1.0 - ((fatigue_count - 3) * 0.12))
    return penalty, "fatigue penalty applied"


def _exploration_bonus(fatigue_count: int, predicted_ctr: float) -> tuple[float, str]:
    if fatigue_count == 0 and predicted_ctr < 0.08:
        return 1.03, "under-explored candidate bonus"
    return 1.0, "no exploration bonus"


class CampaignRanker:
    def __init__(self, predictor: Any | None = None) -> None:
        self.predictor = predictor or FallbackCtrPredictor()

    def rank(self, context: ImpressionContext, candidates: list[CandidateAd]) -> list[RankedCandidate]:
        context_payload = to_plain_dict(context)
        context_tier = _normalize_tier(context_payload.get("safety_tier"))
        score_inputs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        for index, candidate in enumerate(candidates):
            candidate_payload = to_plain_dict(candidate)
            candidate_tier = _normalize_tier(candidate_payload.get("safety_tier"))
            candidate_id = str(candidate_payload.get("candidate_id") or f"candidate-{index + 1}")

            if _context_blocks_candidate(context_tier, candidate_tier):
                continue

            merged_context = dict(context_payload)
            for field in AD_FEATURE_FIELDS:
                if candidate_payload.get(field) is not None:
                    merged_context[field] = candidate_payload[field]

            score_inputs.append((candidate_id, candidate_payload, merged_context))

        predictions = self.predictor.predict_batch([merged_context for _, _, merged_context in score_inputs])
        scored: list[RankedCandidate] = []
        for (candidate_id, candidate_payload, _), prediction in zip(score_inputs, predictions):
            candidate_tier = _normalize_tier(candidate_payload.get("safety_tier"))
            predicted_ctr = prediction.predicted_ctr

            safety_multiplier = 1.05 if candidate_tier == context_tier and candidate_tier != "unknown" else 1.0
            pacing, pacing_reason = _pacing_multiplier(float(candidate_payload.get("pacing_ratio") or 1.0))
            fatigue, fatigue_reason = _fatigue_penalty(int(candidate_payload.get("fatigue_count") or 0))
            exploration, exploration_reason = _exploration_bonus(int(candidate_payload.get("fatigue_count") or 0), predicted_ctr)
            final_score = predicted_ctr * safety_multiplier * pacing * fatigue * exploration

            reasons = [*prediction.reasons, pacing_reason, fatigue_reason, exploration_reason]
            if safety_multiplier > 1.0:
                reasons.append("safe tier match")

            scored.append(
                RankedCandidate(
                    candidate_id=candidate_id,
                    rank=0,
                    predicted_ctr=predicted_ctr,
                    final_score=round(final_score, 8),
                    reasons=reasons,
                )
            )

        scored.sort(key=lambda item: item.final_score, reverse=True)
        for rank, candidate in enumerate(scored, start=1):
            candidate.rank = rank
        return scored
