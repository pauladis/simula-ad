from __future__ import annotations

import json

from simula_ctr.predict import load_predictor, to_plain_dict
from simula_ctr.schemas import ImpressionContext


def main() -> int:
    predictor = load_predictor()
    example = ImpressionContext(
        hour=14102100,
        app_category="07d7df22",
        site_category="28905ebd",
        safety_tier="sfw",
        conversation_turn=1,
        session_msg_count=4,
    )
    predictions = predictor.predict_batch([example])
    print(json.dumps([to_plain_dict(prediction) for prediction in predictions], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
