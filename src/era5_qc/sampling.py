from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Point:
    t: int
    y: int
    x: int


@dataclass(frozen=True)
class TimestepSample:
    t: int
    variables: tuple[str, ...]


def sample_points(n: int, seed: int, *, n_time: int, n_lat: int, n_lon: int) -> list[Point]:
    rng = np.random.default_rng(seed)
    ts = rng.integers(0, n_time, size=n)
    ys = rng.integers(0, n_lat, size=n)
    xs = rng.integers(0, n_lon, size=n)
    return [Point(int(t), int(y), int(x)) for t, y, x in zip(ts, ys, xs)]


def sample_timestep_variables(
    n_steps: int,
    n_vars: int,
    seed: int,
    *,
    available_vars: Sequence[str],
    n_time: int,
) -> list[TimestepSample]:
    if n_vars > len(available_vars):
        raise ValueError(
            f"n_vars={n_vars} exceeds available variable count {len(available_vars)}"
        )
    rng = np.random.default_rng(seed)
    ts = rng.integers(0, n_time, size=n_steps)
    samples = []
    for t in ts:
        chosen = rng.choice(available_vars, size=n_vars, replace=False)
        samples.append(TimestepSample(t=int(t), variables=tuple(sorted(str(v) for v in chosen))))
    return samples
