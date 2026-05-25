from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    impressions_path: Path
    characters_path: Path
    model_artifact_path: Path
    model_name: str
    model_version: str
    random_seed: int
    fallback_ctr: float
    cold_start_min_impressions: int
    max_batch_size: int = 1000
    max_rank_candidates: int = 200


def get_settings() -> Settings:
    data_dir = _path_from_env("SIMULA_DATA_DIR", PROJECT_ROOT / "data")
    artifacts_dir = _path_from_env("SIMULA_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
    reports_dir = _path_from_env("SIMULA_REPORTS_DIR", PROJECT_ROOT / "reports")
    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        impressions_path=data_dir / "impressions.csv",
        characters_path=data_dir / "characters.csv",
        model_artifact_path=artifacts_dir / "ctr_model.joblib",
        model_name=os.environ.get("SIMULA_MODEL_NAME", "catboost_ctr"),
        model_version=os.environ.get("SIMULA_MODEL_VERSION", "untrained-0.1.0"),
        random_seed=int(os.environ.get("SIMULA_RANDOM_SEED", "42")),
        fallback_ctr=float(os.environ.get("SIMULA_FALLBACK_CTR", "0.05")),
        cold_start_min_impressions=int(os.environ.get("SIMULA_COLD_START_MIN_IMPRESSIONS", "500")),
        max_batch_size=int(os.environ.get("SIMULA_MAX_BATCH_SIZE", "1000")),
        max_rank_candidates=int(os.environ.get("SIMULA_MAX_RANK_CANDIDATES", "200")),
    )
