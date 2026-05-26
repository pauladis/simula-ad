# Simula CTR Model Report

## Summary

- Model: catboost_ctr
- Version: ctr-20260525T033100Z
- Trained at: 2026-05-25T03:31:00.715137+00:00
- Artifact: /app/artifacts/ctr_model.joblib

## Split Strategy

Rows are sorted by parsed impression time and split chronologically. Older impressions train the model, the middle period is used for validation, and the latest period is held out for test.

- Train rows: 680553
- Validation rows: 176676
- Test rows: 142771
- Train time range: 2014-10-21T00:00:00 to 2014-10-27T10:00:00
- Validation time range: 2014-10-27T11:00:00 to 2014-10-28T19:00:00
- Test time range: 2014-10-28T20:00:00 to 2014-10-30T05:00:00

## Feature Strategy

The model uses 33 features: parsed time features, conversation/session features, joined character metadata, site/app/device categorical fields, and anonymized C-fields.

- Categorical feature count: 24
- Numeric feature count: 9

## Model Choice

CatBoostClassifier is used because the ranking problem has many high-cardinality categorical fields. It can consume categorical columns directly while optimizing binary log loss for CTR probability estimates.

## Evaluation Results

| Split | Log Loss | ROC AUC | PR AUC | Brier | CTR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.408065 | 0.739554 | 0.370555 | 0.128016 | 0.173968 |
| Test | 0.407965 | 0.728095 | 0.352203 | 0.127329 | 0.170056 |
| Global CTR Baseline Test | 0.456662 | 0.500000 | 0.170056 | 0.141339 | 0.170056 |

## Ranking Metric

- ctr_at_100: 0.780000
- lift_at_100: 4.586737
- ctr_at_1000: 0.663000
- lift_at_1000: 3.898726
- ctr_at_10000: 0.442300
- lift_at_10000: 2.600915

## Additional Reports

- `reports/feature_importance.md` ranks model features by CatBoost importance.
- `reports/calibration_report.md` compares predicted CTR to observed CTR by probability bucket.
- `reports/data_validation.md` summarizes input data quality checks.
- `reports/latency_benchmark.md` reports local serving latency for predict and rank paths.
- `reports/sample_ranked_output.json` contains an example `/rank` style response.

## Training Parameters

- depth: 6
- early_stopping_rounds: 30
- iterations: 250
- l2_leaf_reg: 3.0
- learning_rate: 0.08
- max_rows: None
- thread_count: -1
- train_fraction: 0.7
- validation_fraction: 0.15
- verbose_eval: 50

## Trade-Offs

This implementation favors chronological validation, direct categorical handling, and calibrated probability diagnostics over leaderboard-only optimization. CatBoost training is configurable through environment variables so local smoke runs and full offline jobs use the same code path.

## Known Limitations

- Probabilities are not post-calibrated yet; calibration is measured and can be improved with Platt or isotonic calibration on the validation period.
- The current model uses only the provided impression and character tables. Online user history, campaign budget state, and richer creative metadata are not available yet.
- Candidate ranking still needs online business signals such as live pacing, fatigue, and exploration uncertainty.
