# Simula CTR Ranking

Production-oriented CTR prediction and campaign ranking system for Simula-style AI companion, RPG, and chat surfaces.

## How To Run

```bash
make build
make test
make train
make evaluate
make benchmark
make api
```

The API is served at `http://localhost:8000`. Docker Compose is also supported:
The swagger can be found at `http://localhost:8000/docs`

```bash
docker compose up --build
```

## Data Assumptions

Expected inputs live in `data/impressions.csv` and `data/characters.csv`. Impression rows contain ad opportunity, host app/site, device, anonymized creative fields, character id, and conversation state. Character rows contain persona metadata, safety tier, creator type, interaction counts, and creation time

## Modeling Approach

The main model is a `CatBoostClassifier` because the impression data has many high-cardinality categorical fields. Training uses a global CTR baseline for comparison and serving keeps a fallback prior for cold start, incident response, and model artifact failures.

## Feature Strategy

Implemented feature helpers cover Avazu-style hour parsing, hour of day, day of week, weekend flag, turn ratio, character joins, character age, log interaction count, categorical missing-value handling, and reusable feature column definitions

## Split Strategy

Training must use a time-based split sorted by `hour`: older impressions for training, the middle period for validation, and the latest period for test. This mirrors production, where the model predicts future traffic rather than randomly sampled historical rows.

## Metrics And Reports

Training writes `artifacts/ctr_model.joblib`, `artifacts/model_metadata.json`, `reports/evaluation.json`, and `reports/model_report.md`. It also writes `reports/feature_importance.md`, `reports/calibration_report.md`, `reports/data_validation.md`, `reports/sample_ranked_output.json`, and `reports/latency_benchmark.md`.

Evaluation includes log loss, ROC AUC, PR AUC, Brier score, calibration summaries, expected calibration error, CTR@K/Precision@K, lift@K, and slices for safety tier, creator type, turn stage, cold-start-like characters, and app category.

## Ranking Logic

The API ranks candidates by:

```txt
final_score =
  predicted_ctr
  * safety_multiplier
  * pacing_multiplier
  * fatigue_penalty
  * exploration_bonus
```

The current ranker blocks mature candidates in SFW contexts, penalizes high fatigue, reduces overspending campaigns, lightly boosts underspending campaigns, and adds a small exploration bonus for fresh low-confidence candidates.

## Cold Start Handling

New characters should use safety tier, creator type, description features, app/site context, and available interaction counts. New users or devices should rely on contextual features such as app category, site category, device type, hour, and conversation turn. An entity graduates from cold start once it reaches at least 500 impressions and its CTR confidence interval is sufficiently narrow.

## Drift Handling

`scripts/drift_report.py` writes `reports/drift_metrics.json` and `reports/drift_report.md`. It measures CTR over time, impression volume, top-character concentration, safety-tier distribution, app-category distribution, PSI, KL divergence, normalized distribution deltas, and CTR deltas between early and latest traffic. Adaptation options include recent CTR weighting, epsilon-greedy or Thompson Sampling exploration, and fatigue caps for dominant cohorts.

## Production Architecture

```txt
Ad Opportunity
  ↓
Context + Candidate Ads
  ↓
Feature Builder
  ↓
CTR Model
  ↓
Business Rules / Safety / Pacing / Fatigue
  ↓
Ranked Campaigns
  ↓
Event Logging
  ↓
Offline Training + Drift Monitoring
```

For sub-50ms p99 serving, preload the model in memory, avoid online joins, precompute character features, cache hot character features in Redis, score candidates in batches, log events asynchronously, and keep inference lightweight.

## API

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /rank`
- `POST /batch-predict`

## Next Steps

Future improvements include real conversion labels, user history features, campaign budget data, creative metadata, a landing page quality model, online A/B testing, a feature store, online/offline feature consistency checks, calibrated model monitoring, contextual bandits for exploration, and approximate nearest neighbor retrieval for candidate generation.
