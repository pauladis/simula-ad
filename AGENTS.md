# AGENTS.md

## Project goal

Build a production-quality CTR prediction and campaign ranking system for Simula contextual ads.

The system must support:

- batch model training
- batch prediction
- ranked candidate ads
- FastAPI serving API
- drift analysis
- Dockerized execution
- clean README and recording notes

Prioritize production architecture, code quality, reproducibility, and clear reasoning over leaderboard-only optimization.

## Domain context

Simula serves contextual ads inside AI companion apps, RPGs, and chat platforms.

Each ad impression is tied to:

- character persona
- conversation state
- host app/site metadata
- device metadata
- anonymized categorical ad/creative signals

The goal is to predict:

```txt
P(click = 1)
```
Then use that probability to rank candidate campaigns.

This should be treated as an ads ranking system, not just a binary classifier.


## Input data

Expected files:

data/impressions.csv
data/characters.csv


impressions.csv contains:

id
hour
click
banner_pos
site_id
site_domain
site_category
app_id
app_domain
app_category
device_id
device_ip
device_model
device_type
device_conn_type
C1
C14-C21
character_id
conversation_turn
session_msg_count

characters.csv contains:

character_id
character_name
character_description
safety_tier
creator_type
num_interactions
created_at
Required repo structure

Create this structure:

simula-ad/
  AGENTS.md
  README.md
  Dockerfile
  docker-compose.yml
  pyproject.toml
  Makefile

  data/
    .gitkeep

  artifacts/
    .gitkeep

  reports/
    .gitkeep

  src/
    simula_ctr/
      __init__.py
      config.py
      schemas.py
      data.py
      features.py
      train.py
      evaluate.py
      predict.py
      ranker.py
      drift.py
      api.py
      logging_config.py

  scripts/
    train_model.py
    batch_predict.py
    rank_candidates.py
    drift_report.py

  tests/
    test_features.py
    test_ranker.py
    test_api.py
Technology choices

Use:

Python 3.11+
FastAPI
Pydantic
Pandas or Polars
scikit-learn
CatBoost as the main model
LightGBM or sklearn baseline if useful
joblib for artifacts
pytest
uvicorn
Docker and docker-compose

CatBoost is preferred because the dataset contains many high-cardinality categorical features.

Modeling requirements

Implement at least:

Baseline model
global CTR
optionally grouped CTR by app/category/character/safety tier
Main model
CatBoostClassifier
binary target: click
output calibrated probability where possible
Time-based split
Do not use random split as the primary split
Sort by hour
Train on older impressions
Validate on middle period
Test on latest period

Reason: production predicts future behavior.

Feature engineering

Implement clean, reusable feature code.

Required features:

parsed timestamp from hour
hour of day
day of week
weekend flag
conversation_turn
session_msg_count
turn_ratio = conversation_turn / session_msg_count
joined character features:
safety_tier
creator_type
num_interactions
character_age_days
log-transformed num_interactions
all site/app/device categorical fields
C1 and C14-C21 categorical fields
missing value handling

Optional advanced features:

character_description TF-IDF or sentence embedding
character popularity bucket
early/mid/late session bucket
mature/suggestive/sfw interaction with app category
frequency encoding for high-cardinality categorical fields

Avoid target leakage.

If implementing target encoding, it must be time-safe or fold-safe.

Evaluation requirements

Produce metrics in reports/evaluation.json.

Include:

log loss
ROC AUC
PR AUC
Brier score
calibration summary
Precision@K or CTR@K style ranking metric
slice metrics by:
safety_tier
creator_type
early vs late conversation_turn
cold-start-like characters
app_category

Also produce human-readable markdown:

reports/model_report.md

Explain:

split strategy
features used
model choice
evaluation results
trade-offs
known limitations
Ranking requirements

Create a ranking module that accepts:

one impression context
N candidate ads

Each candidate ad contains:

banner_pos
C1
C14-C21
optional candidate_id
optional advertiser_id
optional campaign_id
optional bid
optional safety_tier
optional fatigue_count
optional pacing_ratio

Return a ranked list with:

candidate_id
predicted_ctr
final_score
rank
reasons

Use:

final_score =
  predicted_ctr
  * safety_multiplier
  * pacing_multiplier
  * fatigue_penalty
  * exploration_bonus

Default rules:

safety gate:
do not allow mature candidate into SFW context
fatigue penalty:
reduce score if candidate was shown too many times
pacing:
reduce score if campaign is overspending
exploration:
small bonus for under-explored candidates when uncertainty is high

When model confidence is low, ranking should rely more on safe priors and contextual averages.

Cold start requirements

Implement explicit cold-start logic.

For new character:

use safety_tier
creator_type
character_description
app/site context
num_interactions if available
fallback to character safety-tier/app-category averages

For new user/device:

rely on contextual features:
app_category
site_category
device_type
hour
conversation_turn

Graduation rule:

entity graduates from cold start when it has enough impressions and stable CTR estimate

Example:

= 500 impressions

CTR confidence interval width below threshold

Add this explanation to README.

Drift requirements

Create scripts/drift_report.py.

Analyze temporal shifts using only provided data:

CTR over time
impressions over time
character mix over time
top character concentration
safety_tier distribution over time
app_category distribution over time
feature distribution shift

Implement simple drift metrics:

Population Stability Index, if reasonable
KL divergence or normalized distribution delta
CTR delta by period

Output:

reports/drift_report.md
reports/drift_metrics.json

Add an adaptation proposal:

recent CTR weighting
epsilon-greedy or Thompson Sampling exploration
fatigue caps for dominant character cohorts
API requirements

Implement FastAPI app in:

src/simula_ctr/api.py

Required endpoints:

GET /health
GET /model-info
POST /predict
POST /rank
POST /batch-predict
/health

Returns:

{
  "status": "ok"
}
/model-info

Returns:

{
  "model_name": "...",
  "model_version": "...",
  "trained_at": "...",
  "features": [...],
  "metrics": {...}
}
/predict

Input: one impression context.

Output:

{
  "predicted_ctr": 0.1234,
  "model_version": "...",
  "reasons": [...]
}
/rank

Input:

{
  "context": {...},
  "candidates": [...]
}

Output:

{
  "ranked_candidates": [
    {
      "candidate_id": "...",
      "rank": 1,
      "predicted_ctr": 0.123,
      "final_score": 0.118,
      "reasons": ["high contextual relevance", "safe tier match"]
    }
  ]
}
/batch-predict

Input: list of impression contexts.

Output: list of predictions.

Performance target

Design for:

<50ms p99 online latency

Implementation can be local/simple, but README must explain production strategy:

preload model in memory
avoid online joins
precompute character features
cache hot character features
batch candidate scoring
use Redis for feature cache
async event logging
keep model inference lightweight
Docker requirements

Everything must run in Docker.

Required commands:

make build
make train
make evaluate
make api
make test
make batch-predict
make drift

docker-compose.yml should support API startup.

Example:

docker compose up --build

API should be available at:

http://localhost:8000
Code quality requirements

Follow:

typed functions where practical
modular design
no giant notebook-only solution
no hardcoded absolute paths
clear config in config.py
reproducible random seeds
readable logs
graceful errors for missing model/data
tests for feature engineering, ranking, and API

Use clean architecture:

data loading → feature engineering → model training → evaluation → prediction → ranking → API

Do not mix API code with training code.

Testing requirements

Add pytest tests for:

timestamp parsing
turn_ratio calculation
character join
missing values
ranking order
safety gate
API health endpoint
API rank endpoint
README requirements

README must include:

project overview
how to run with Docker
data assumptions
modeling approach
feature strategy
split strategy
metrics
ranking logic
cold-start handling
drift handling
production architecture
trade-offs
next steps

Include this architecture diagram:

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


Recording script requirements

Create:

reports/recording_script.md

It should be a 3-5 minute explanation covering:

Problem framing
Why this is a ranking system, not just classification
Time-based split
Feature engineering
Model choice
Evaluation
Ranking layer
Cold start
Drift
Production serving under 50ms
Next steps

Tone should be senior/staff-level, concise, and practical.

Next steps section

Mention future improvements:

real conversion labels
user history features
campaign budget data
creative metadata
landing page quality model
online A/B testing
feature store
online/offline feature consistency checks
calibrated model monitoring
contextual bandits for exploration
approximate nearest neighbor retrieval for candidate generation
Important principles

Do not optimize only for leaderboard metrics.

Optimize for:

realistic production behavior
ranking quality
calibration
cold start resilience
drift awareness
latency
maintainability
explainability