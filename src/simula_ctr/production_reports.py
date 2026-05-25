from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from simula_ctr.config import Settings, get_settings
from simula_ctr.evaluate import write_json
from simula_ctr.predict import FallbackCtrPredictor, load_predictor, to_plain_dict
from simula_ctr.ranker import CampaignRanker
from simula_ctr.schemas import (
    CandidateAd,
    EXAMPLE_CANDIDATES,
    EXAMPLE_IMPRESSION_CONTEXT,
    ImpressionContext,
)


def sample_context() -> ImpressionContext:
    return ImpressionContext(**EXAMPLE_IMPRESSION_CONTEXT)


def sample_candidates() -> list[CandidateAd]:
    return [CandidateAd(**candidate) for candidate in EXAMPLE_CANDIDATES]


def write_sample_ranked_output(
    settings: Settings | None = None,
    ranker: CampaignRanker | None = None,
) -> Path:
    settings = settings or get_settings()
    predictor = None if ranker is not None else load_predictor(settings)
    ranker = ranker or CampaignRanker(predictor)
    context = sample_context()
    candidates = sample_candidates()
    ranked = ranker.rank(context, candidates)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "context": to_plain_dict(context),
        "candidates": [to_plain_dict(candidate) for candidate in candidates],
        "ranked_candidates": [to_plain_dict(candidate) for candidate in ranked],
    }
    path = settings.reports_dir / "sample_ranked_output.json"
    write_json(path, payload)
    return path


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_ms": mean(values) if values else 0.0,
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "max_ms": max(values) if values else 0.0,
    }


def run_latency_benchmark(
    predictor: Any | None = None,
    ranker: CampaignRanker | None = None,
    iterations: int = 200,
    warmup: int = 20,
) -> dict[str, Any]:
    predictor = predictor or FallbackCtrPredictor()
    ranker = ranker or CampaignRanker(predictor)
    context = sample_context()
    candidates = sample_candidates()

    for _ in range(max(warmup, 0)):
        predictor.predict_one(context)
        ranker.rank(context, candidates)

    predict_ms: list[float] = []
    rank_ms: list[float] = []
    for _ in range(max(iterations, 1)):
        start = time.perf_counter()
        predictor.predict_one(context)
        predict_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        ranker.rank(context, candidates)
        rank_ms.append((time.perf_counter() - start) * 1000.0)

    model_info = predictor.model_info()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "iterations": max(iterations, 1),
        "warmup": max(warmup, 0),
        "model_name": model_info.model_name,
        "model_version": model_info.model_version,
        "candidate_count": len(candidates),
        "predict": _summarize_ms(predict_ms),
        "rank": _summarize_ms(rank_ms),
        "target": {"p99_ms": 50.0},
    }


def build_latency_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Latency Benchmark",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Model: {payload['model_name']} ({payload['model_version']})",
        f"- Iterations: {payload['iterations']}",
        f"- Candidate count: {payload['candidate_count']}",
        f"- Target p99: {payload['target']['p99_ms']} ms",
        "",
        "| Path | Mean ms | P50 ms | P95 ms | P99 ms | Max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for path_name in ["predict", "rank"]:
        metrics = payload[path_name]
        lines.append(
            f"| {path_name} | {metrics['mean_ms']:.4f} | {metrics['p50_ms']:.4f} | {metrics['p95_ms']:.4f} | {metrics['p99_ms']:.4f} | {metrics['max_ms']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_latency_benchmark(
    settings: Settings | None = None,
    predictor: Any | None = None,
    ranker: CampaignRanker | None = None,
    iterations: int | None = None,
    warmup: int | None = None,
) -> tuple[Path, Path]:
    settings = settings or get_settings()
    if predictor is None:
        predictor = load_predictor(settings)
    ranker = ranker or CampaignRanker(predictor)
    iterations = iterations or int(os.environ.get("SIMULA_LATENCY_BENCHMARK_ITERATIONS", "200"))
    warmup = warmup if warmup is not None else int(os.environ.get("SIMULA_LATENCY_BENCHMARK_WARMUP", "20"))
    payload = run_latency_benchmark(predictor=predictor, ranker=ranker, iterations=iterations, warmup=warmup)
    json_path = settings.reports_dir / "latency_benchmark.json"
    report_path = settings.reports_dir / "latency_benchmark.md"
    write_json(json_path, payload)
    report_path.write_text(build_latency_report(payload), encoding="utf-8")
    return json_path, report_path
