from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert numpy / pandas / nan to JSON-serializable scalars."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, (int, str, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return str(value)


@dataclass
class SampleResult:
    indices: dict[str, int]
    labels: dict[str, Any]
    kind: str
    equal: bool
    values: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class Report:
    task: str
    seed: int
    n_samples: int
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    n_failures: int = 0
    passed: bool = False
    samples: list[SampleResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def record(self, result: SampleResult) -> None:
        self.samples.append(result)
        if not result.equal:
            self.n_failures += 1

    def finalize(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.passed = self.n_failures == 0

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **{k: v for k, v in asdict(self).items() if k != "samples"},
            "samples": [_json_safe(asdict(s)) for s in self.samples],
        }
        p.write_text(json.dumps(payload, indent=2, default=str))
        log.info("Wrote report to %s (%d samples, %d failures)",
                 p, len(self.samples), self.n_failures)

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"[{self.task}] {status}: {len(self.samples)} comparisons, "
                f"{self.n_failures} failures")
