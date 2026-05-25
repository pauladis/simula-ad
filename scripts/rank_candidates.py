from __future__ import annotations

import json

from simula_ctr.ranker import CampaignRanker
from simula_ctr.predict import to_plain_dict
from simula_ctr.schemas import CandidateAd, ImpressionContext


def main() -> int:
    context = ImpressionContext(
        hour=14102100,
        app_category="07d7df22",
        site_category="28905ebd",
        safety_tier="sfw",
        conversation_turn=1,
        session_msg_count=4,
    )
    candidates = [
        CandidateAd(candidate_id="safe-a", C14=15708, safety_tier="sfw", fatigue_count=0, pacing_ratio=0.9),
        CandidateAd(candidate_id="mature-b", C14=20362, safety_tier="mature", fatigue_count=0, pacing_ratio=0.9),
    ]
    ranked = CampaignRanker().rank(context, candidates)
    print(json.dumps([to_plain_dict(candidate) for candidate in ranked], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
