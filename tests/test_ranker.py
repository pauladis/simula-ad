from __future__ import annotations

from simula_ctr.ranker import CampaignRanker
from simula_ctr.schemas import PredictionResponse
from simula_ctr.schemas import CandidateAd, ImpressionContext


def test_ranking_orders_by_business_adjusted_score() -> None:
    context = ImpressionContext(
        hour=14102100,
        app_category="game",
        site_category="chat",
        safety_tier="sfw",
        conversation_turn=1,
        session_msg_count=5,
    )
    candidates = [
        CandidateAd(candidate_id="fatigued", safety_tier="sfw", fatigue_count=9, pacing_ratio=1.0),
        CandidateAd(candidate_id="fresh", safety_tier="sfw", fatigue_count=0, pacing_ratio=1.0),
    ]

    ranked = CampaignRanker().rank(context, candidates)

    assert [candidate.candidate_id for candidate in ranked] == ["fresh", "fatigued"]
    assert ranked[0].rank == 1
    assert ranked[0].final_score > ranked[1].final_score


def test_safety_gate_blocks_mature_candidate_in_sfw_context() -> None:
    context = ImpressionContext(hour=14102100, safety_tier="sfw")
    candidates = [
        CandidateAd(candidate_id="safe", safety_tier="sfw"),
        CandidateAd(candidate_id="mature", safety_tier="mature"),
    ]

    ranked = CampaignRanker().rank(context, candidates)

    assert [candidate.candidate_id for candidate in ranked] == ["safe"]


def test_ranker_batches_candidate_scoring() -> None:
    class CountingPredictor:
        def __init__(self) -> None:
            self.batch_calls = 0

        def predict_batch(self, contexts: list[dict]) -> list[PredictionResponse]:
            self.batch_calls += 1
            return [
                PredictionResponse(predicted_ctr=0.1 + index * 0.01, model_version="test", reasons=["batch"])
                for index, _ in enumerate(contexts)
            ]

    predictor = CountingPredictor()
    context = ImpressionContext(hour=14102100, safety_tier="sfw")
    candidates = [
        CandidateAd(candidate_id="a", safety_tier="sfw"),
        CandidateAd(candidate_id="b", safety_tier="sfw"),
        CandidateAd(candidate_id="blocked", safety_tier="mature"),
    ]

    ranked = CampaignRanker(predictor).rank(context, candidates)

    assert predictor.batch_calls == 1
    assert [candidate.candidate_id for candidate in ranked] == ["b", "a"]
