# Recording Script

This project is framed as an ads ranking system, not just a binary classifier. The model estimates click probability, but the product outcome is a ranked set of campaign candidates after safety, pacing, fatigue, and exploration rules are applied.

The data represents contextual ad opportunities inside AI companion, RPG, and chat surfaces. Each impression combines app or site metadata, device metadata, anonymized ad fields, character persona fields, and conversation state. The target is `P(click = 1)`.

The offline split should be time based. We sort by `hour`, train on older impressions, validate on a middle period, and test on the latest period. That is closer to production than a random split because the serving system always predicts future traffic.

Feature engineering starts with reliable reusable transformations: parsed timestamps, hour of day, day of week, weekend flag, conversation turn ratio, character metadata joins, character age, log interaction count, and consistent missing-value handling. High-cardinality categorical fields are preserved for CatBoost.

CatBoost is the planned main model because it handles categorical-heavy data well without forcing fragile manual encodings. We still keep a global CTR or grouped CTR baseline because it is useful for cold start, fallback behavior, and calibration sanity checks.

Evaluation should include log loss, ROC AUC, PR AUC, Brier score, calibration summaries, ranking metrics such as CTR@K, and slices by safety tier, creator type, conversation stage, cold-start characters, and app category. Calibration matters because the probability feeds business rules downstream.

The ranking layer multiplies predicted CTR by safety, pacing, fatigue, and exploration adjustments. Mature candidates are blocked in SFW contexts. Overexposed candidates are penalized. Overspending campaigns are reduced. Under-explored candidates can get a small controlled bonus when confidence is low.

Cold start is handled explicitly. New characters rely on safety tier, creator type, description signals, app and site context, and interaction counts if available. New users or devices rely on contextual features like app category, site category, device type, hour, and conversation turn. Entities graduate after enough impressions and a stable CTR estimate.

Drift monitoring should track CTR over time, impression volume, character mix, safety-tier distribution, app-category distribution, and feature distribution shift using PSI, KL divergence, and period CTR deltas. Adaptation options include recent CTR weighting, epsilon-greedy or Thompson Sampling exploration, and fatigue caps for dominant cohorts.

For production serving under 50ms p99, the API should preload the model, avoid online joins, precompute character features, cache hot features in Redis, batch candidate scoring, and log events asynchronously. Next steps include real conversion labels, user history, campaign budget data, creative metadata, landing page quality, online A/B tests, a feature store, consistency checks, calibrated monitoring, contextual bandits, and ANN retrieval for candidate generation.
