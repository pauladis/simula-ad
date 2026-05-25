from __future__ import annotations

from simula_ctr.logging_config import configure_logging
from simula_ctr.production_reports import write_latency_benchmark, write_sample_ranked_output


def main() -> int:
    configure_logging()
    write_sample_ranked_output()
    write_latency_benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
