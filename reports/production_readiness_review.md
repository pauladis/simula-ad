# Production Readiness Review

## Staff-Level Findings

1. Time split leakage risk: the previous chronological split could cut inside a single timestamp bucket, which lets train, validation, and test share the same hour. The splitter now keeps whole timestamp buckets together when at least three buckets are available and records warnings when it must fall back.

2. Serving scalability risk: ranking previously called the predictor once per candidate. The ranker now batches candidate scoring through `predict_batch`, keeping the ranking layer closer to production candidate scoring.

3. Online/offline feature consistency risk: the training path joins character features offline, while online serving expects the context to already contain those features. This is acceptable for the exercise but must be formalized in production with a feature service or precomputed character feature cache.

4. Potential aggregate leakage risk: `num_interactions` may be a point-in-time feature or may be an all-time aggregate. Production training must guarantee it is available as of impression time. The current data contract flags future character creation dates but cannot prove aggregate point-in-time correctness from the provided CSVs.

5. Observability gap: API request timing and request IDs were missing. The API now adds `x-request-id`, `x-process-time-ms`, and structured request logs. Production still needs metrics export, tracing, alerting, and event logging.

6. API contract gap: candidate and batch sizes were previously unbounded. The API now rejects empty batches, empty candidate sets, oversized batches, and oversized candidate lists with structured error payloads.

7. Docker production gap: the image is functional but still optimized for an interview project rather than a minimal production runtime. A production build should separate train/test/dev dependencies from serving dependencies and publish immutable model artifacts separately.

8. Evaluation gap: offline metrics include probability and simple ranking metrics, but do not yet evaluate auction value, budget pacing quality, counterfactual bias, user-level holdouts, campaign-level fairness, or online A/B performance.

## Improvements Applied

- Timestamp-bucket-aware time split.
- Batched candidate scoring in ranking.
- API request observability middleware.
- Readiness endpoint.
- Candidate count and batch size limits.
- Data validation warning for future character creation dates.
- Structured API error responses and OpenAPI examples.
- Feature importance, calibration, data validation, sample ranking, and latency reports.
- Drift metrics and drift markdown report with PSI, KL divergence, concentration, and CTR deltas.

## Remaining Production Work

- Replace local CSV/Pandas training with scalable data processing for larger histories.
- Add point-in-time feature generation and offline/online feature parity tests.
- Add model registry semantics, artifact checksums, and deploy rollback controls.
- Add Prometheus/OpenTelemetry metrics and structured event logging.
- Add calibrated model monitoring, drift alerts, and auto-retraining policies.
- Add online experimentation and counterfactual evaluation before optimizing revenue.
