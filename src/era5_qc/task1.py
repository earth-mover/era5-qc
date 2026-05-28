from __future__ import annotations

import logging

import xarray as xr
from tqdm import tqdm

from .dataset import DEFAULT_SPATIAL_GROUP, DEFAULT_TEMPORAL_GROUP, open_repo
from .report import Report, SampleResult
from .sampling import sample_points

log = logging.getLogger(__name__)


def run_task1(
    *,
    n_points: int,
    seed: int,
    report_path: str,
    repo_name: str,
    branch: str = "main",
    spatial_group: str = DEFAULT_SPATIAL_GROUP,
    temporal_group: str = DEFAULT_TEMPORAL_GROUP,
) -> bool:
    log.info("Task 1: sampling %d points (seed=%d)", n_points, seed)
    repo = open_repo(repo_name, branch=branch,
                     spatial_group=spatial_group, temporal_group=temporal_group)
    variables = repo.variables
    points = sample_points(n_points, seed,
                           n_time=repo.n_time, n_lat=repo.n_lat, n_lon=repo.n_lon)

    times = repo.spatial.valid_time.values
    lats = repo.spatial.latitude.values
    lons = repo.spatial.longitude.values

    report = Report(task="task1_spatial_vs_temporal", seed=seed, n_samples=n_points)
    report.metadata = {
        "repo": repo_name,
        "branch": branch,
        "spatial_group": spatial_group,
        "temporal_group": temporal_group,
        "variables": variables,
        "has_status_subgroup": bool(repo.spatial_status is not None
                                    and repo.temporal_status is not None),
        "comparison": "xr.testing.assert_equal per point on identical .sel(...) indexers",
    }

    if repo.spatial_status is None or repo.temporal_status is None:
        log.info("status subgroup missing on one or both groups; skipping status comparison")

    for p in tqdm(points, desc="task1 points", unit="pt"):
        time_label = times[p.t]
        lat_label = float(lats[p.y])
        lon_label = float(lons[p.x])
        indexer = {"valid_time": time_label, "latitude": lat_label, "longitude": lon_label}
        status_indexer = {"valid_time": time_label}
        labels = {"time": str(time_label), "latitude": lat_label, "longitude": lon_label}
        indices = {"t": p.t, "y": p.y, "x": p.x}

        a = repo.spatial.sel(**indexer).load()
        b = repo.temporal.sel(**indexer).load()
        try:
            xr.testing.assert_equal(a, b)
            data_eq, data_err = True, None
        except AssertionError as e:
            data_eq, data_err = False, str(e)
        data_values = {
            var: {"spatial": float(a[var].item()), "temporal": float(b[var].item())}
            for var in variables
        }
        report.record(SampleResult(
            indices=indices, labels=labels, kind="data",
            equal=data_eq, values=data_values, error=data_err,
        ))

        if repo.spatial_status is not None and repo.temporal_status is not None:
            sa = repo.spatial_status.sel(**status_indexer).load()
            sb = repo.temporal_status.sel(**status_indexer).load()
            try:
                xr.testing.assert_equal(sa, sb)
                stat_eq, stat_err = True, None
            except AssertionError as e:
                stat_eq, stat_err = False, str(e)
            status_values = {
                var: {"spatial": int(sa[var].item()), "temporal": int(sb[var].item())}
                for var in variables if var in sa.data_vars and var in sb.data_vars
            }
            report.record(SampleResult(
                indices=indices, labels={"time": str(time_label)}, kind="status",
                equal=stat_eq, values=status_values, error=stat_err,
            ))

    report.finalize()
    report.write(report_path)
    log.info(report.summary_line())
    if not report.passed:
        log.error("Task 1 FAILED: %d mismatches out of %d comparisons",
                  report.n_failures, len(report.samples))
        for s in report.samples:
            if not s.equal:
                log.error("  MISMATCH kind=%s t=%d y=%d x=%d:\n%s",
                          s.kind, s.indices["t"], s.indices["y"], s.indices["x"], s.error)
    return report.passed
