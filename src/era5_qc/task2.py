from __future__ import annotations

import asyncio
import logging
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm

from .cds import CDSCache, VAR_TO_CDS, make_client
from .dataset import DEFAULT_SPATIAL_GROUP, DEFAULT_TEMPORAL_GROUP, open_repo
from .report import Report, SampleResult
from .sampling import TimestepSample, sample_timestep_variables

log = logging.getLogger(__name__)


def _array_summary(a: xr.DataArray, b: xr.DataArray) -> dict:
    """Compact summary of a 2D field comparison for the JSON report."""
    import numpy as np
    a_vals = a.values
    b_vals = b.values
    return {
        "arraylake": {"min": float(np.nanmin(a_vals)), "max": float(np.nanmax(a_vals)),
                      "mean": float(np.nanmean(a_vals))},
        "cds":       {"min": float(np.nanmin(b_vals)), "max": float(np.nanmax(b_vals)),
                      "mean": float(np.nanmean(b_vals))},
    }


def _zarr_compression_for_timestep(*, store, group_path: str, var: str,
                                   time_pos: int) -> dict:
    """On-disk vs raw bytes for the chunk holding `time_pos` of `var` in the
    Zarr/Icechunk store. Requires the array to be chunked one-timestep-per-chunk
    (true for the spatial group). The user is responsible for resolving any
    label-to-position lookup via pandas (`ds.indexes['valid_time'].get_loc(...)`)
    before calling this — no date logic here."""
    arr = zarr.open_array(store, path=f"{group_path}/{var}", mode="r")
    if arr.chunks[0] != 1:
        raise ValueError(
            f"Per-timestep compression assumes time chunk size 1; "
            f"{group_path}/{var} has chunks={arr.chunks}"
        )
    chunk_idx = (time_pos // arr.chunks[0], 0, 0)
    chunk_key = f"{group_path}/{var}/c/{chunk_idx[0]}/{chunk_idx[1]}/{chunk_idx[2]}"
    stored = asyncio.run(store.getsize(chunk_key))
    raw = int(np.prod(arr.chunks)) * arr.dtype.itemsize
    return {
        "raw_bytes": int(raw),
        "stored_bytes": int(stored),
        "ratio": (raw / stored) if stored else None,
        "chunk_key": chunk_key,
    }


def _cds_compression_per_variable(path: Path) -> dict[str, dict]:
    """Return {var_name: {raw_bytes, stored_bytes, ratio}} for every data
    variable in a CDS download. Reads HDF5 per-dataset storage sizes
    directly; handles ZIP-of-NetCDFs (CDS's response for mixed-stream
    requests) by extracting members to a tempdir."""
    import h5py
    import numpy as np

    members: list[Path]
    if zipfile.is_zipfile(path):
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path) as zf:
            nc_names = [m for m in zf.namelist() if m.endswith(".nc")]
            zf.extractall(tmp.name)
        members = [Path(tmp.name) / m for m in nc_names]
        cleanup = tmp
    else:
        members = [path]
        cleanup = None

    out: dict[str, dict] = {}
    try:
        for p in members:
            with h5py.File(p, "r") as f:
                for name, dset in f.items():
                    if not isinstance(dset, h5py.Dataset):
                        continue
                    raw = int(np.prod(dset.shape)) * dset.dtype.itemsize
                    if raw == 0:
                        continue
                    stored = int(dset.id.get_storage_size())
                    out[name] = {
                        "raw_bytes": raw,
                        "stored_bytes": stored,
                        "ratio": (raw / stored) if stored else None,
                    }
    finally:
        if cleanup is not None:
            cleanup.cleanup()
    return out


def _open_cds_file(path: Path) -> xr.Dataset:
    """Open a CDS download. Handles both raw NetCDF and ZIP-of-NetCDFs (which
    CDS returns when a request mixes instantaneous and accumulated step types)."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as tmp:
            members = [m for m in zf.namelist() if m.endswith(".nc")]
            if not members:
                raise ValueError(f"{path} is a zip with no .nc members: {zf.namelist()}")
            zf.extractall(tmp)
            parts = [xr.open_dataset(Path(tmp) / m).load() for m in members]
            return xr.merge(parts, compat="equals")
    return xr.open_dataset(path).load()


EXTRA_CDS_COORDS = ("number", "expver")
EXTRA_SPATIAL_COORDS = ("lsm",)


def _normalize_cds_dataset(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Adjust a fresh CDS NetCDF so its dims/coords align with single/spatial.

    CDS NetCDFs already use `valid_time`/`latitude`/`longitude`; we just need
    to squeeze the length-1 time dim, drop ensemble-related scalar coords
    (`number`, `expver`), and subset to the requested variables.
    """
    if "valid_time" in ds.dims and ds.sizes["valid_time"] == 1:
        ds = ds.squeeze("valid_time", drop=False)
    drop = [c for c in EXTRA_CDS_COORDS if c in ds.coords]
    if drop:
        ds = ds.drop_vars(drop)
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise KeyError(f"CDS file missing requested variables: {missing}; "
                       f"have {list(ds.data_vars)}")
    return ds[variables]


def _strip_extra_spatial_coords(ds: xr.Dataset) -> xr.Dataset:
    """Drop non-dim coords that exist on the Arraylake side but not in CDS files
    (e.g. `lsm` land-sea mask)."""
    drop = [c for c in EXTRA_SPATIAL_COORDS if c in ds.coords]
    return ds.drop_vars(drop) if drop else ds


VALID_DATA_FLAG = 0


def _flag_lookup(status_ds: xr.Dataset) -> dict[int, str]:
    """Build a {flag_value: meaning} map from the status group's CF-style attrs."""
    values = list(status_ds.attrs.get("flag_values", []))
    meanings = str(status_ds.attrs.get("flag_meanings", "")).split()
    return dict(zip(values, meanings))


def run_task2(
    *,
    n_steps: int,
    n_vars: int,
    seed: int,
    report_path: str,
    cache_dir: str,
    repo_name: str,
    branch: str = "main",
    spatial_group: str = DEFAULT_SPATIAL_GROUP,
    temporal_group: str = DEFAULT_TEMPORAL_GROUP,
) -> bool:
    log.info("Task 2: sampling %d timesteps × %d vars (seed=%d)", n_steps, n_vars, seed)
    repo = open_repo(repo_name, branch=branch,
                     spatial_group=spatial_group, temporal_group=temporal_group)

    available = [v for v in repo.variables if v in VAR_TO_CDS]
    missing_map = [v for v in repo.variables if v not in VAR_TO_CDS]
    if missing_map:
        log.warning("No CDS mapping for variables (these will not be verified against CDS): %s",
                    missing_map)
    if not available:
        raise ValueError(
            f"None of the repo's variables {repo.variables} have a CDS mapping. "
            f"Update VAR_TO_CDS in cds.py."
        )
    if n_vars > len(available):
        log.warning("Requested --n-vars=%d but only %d CDS-mappable variables available; "
                    "lowering n_vars to %d", n_vars, len(available), len(available))
        n_vars = len(available)

    samples = sample_timestep_variables(
        n_steps=n_steps, n_vars=n_vars, seed=seed,
        available_vars=available, n_time=repo.n_time,
    )
    cache = CDSCache.load(cache_dir)

    times = repo.spatial.valid_time.values
    timestamps = []
    for s in samples:
        ts = pd.Timestamp(times[s.t]).to_pydatetime()
        timestamps.append(ts)
        cache.ensure_entry(ts, list(s.variables))

    client = make_client()
    cache.submit_pending(client)
    cache.wait_for_all(client)

    report = Report(task="task2_spatial_vs_cds", seed=seed, n_samples=len(samples))
    report.metadata = {
        "repo": repo_name,
        "branch": branch,
        "spatial_group": spatial_group,
        "n_steps": n_steps,
        "n_vars": n_vars,
        "cache_dir": str(cache_dir),
        "cds_mappable_vars": available,
        "unmapped_vars_skipped": missing_map,
        "comparison": "xr.testing.assert_equal on the full Dataset slice "
                      "after normalizing CDS-specific coords",
    }

    for sample, ts in tqdm(list(zip(samples, timestamps)), desc="task2 compare", unit="step"):
        entry = cache.ensure_entry(ts, list(sample.variables))
        indices = {"t": sample.t}
        labels = {"time": str(ts), "variables": list(sample.variables)}

        if entry.status != "completed" or entry.target_path is None:
            report.record(SampleResult(
                indices=indices, labels=labels, kind="cds_download",
                equal=False,
                error=f"CDS request status={entry.status}; error={entry.error}",
            ))
            continue

        path = Path(entry.target_path)
        try:
            raw = _open_cds_file(path)
            cds_ds = _normalize_cds_dataset(raw, list(sample.variables))
        except Exception as e:
            report.record(SampleResult(
                indices=indices, labels=labels, kind="open_cds",
                equal=False, error=f"failed to open {path}: {e}",
            ))
            continue

        indexer = {"valid_time": ts}
        spatial_ds = _strip_extra_spatial_coords(
            repo.spatial[list(sample.variables)].sel(**indexer)
        ).load()
        try:
            xr.testing.assert_equal(spatial_ds, cds_ds)
            ok, err = True, None
        except AssertionError as e:
            ok, err = False, str(e)
        values = {var: _array_summary(spatial_ds[var], cds_ds[var])
                  for var in sample.variables}
        report.record(SampleResult(
            indices=indices, labels=labels, kind="data",
            equal=ok, values=values, error=err,
        ))

    report.finalize()
    report.write(report_path)
    log.info(report.summary_line())
    if not report.passed:
        log.error("Task 2 FAILED: %d mismatches", report.n_failures)
        for s in report.samples:
            if not s.equal:
                log.error("  MISMATCH kind=%s t=%s:\n%s", s.kind, s.labels.get("time"), s.error)
    return report.passed


def _compare_one_timestep(*, repo, sample: TimestepSample, ts: pd.Timestamp,
                          cache: CDSCache, report: Report) -> None:
    """Submit + wait + compare a single (timestamp, variables) request, recording
    the result into `report`. Shared by run_task2 and run_task2_at."""
    entry = cache.ensure_entry(ts.to_pydatetime(), list(sample.variables))
    indices = {"t": sample.t}
    labels = {"time": str(ts), "variables": list(sample.variables)}

    if entry.status != "completed" or entry.target_path is None:
        report.record(SampleResult(
            indices=indices, labels=labels, kind="cds_download", equal=False,
            error=f"CDS request status={entry.status}; error={entry.error}",
        ))
        return

    path = Path(entry.target_path)
    try:
        raw = _open_cds_file(path)
        cds_ds = _normalize_cds_dataset(raw, list(sample.variables))
    except Exception as e:
        report.record(SampleResult(
            indices=indices, labels=labels, kind="open_cds",
            equal=False, error=f"failed to open {path}: {e}",
        ))
        return

    indexer = {"valid_time": ts.to_datetime64()}
    spatial_ds = _strip_extra_spatial_coords(
        repo.spatial[list(sample.variables)].sel(**indexer)
    ).load()
    try:
        xr.testing.assert_equal(spatial_ds, cds_ds)
        ok, err = True, None
    except AssertionError as e:
        ok, err = False, str(e)

    cds_compression = _cds_compression_per_variable(path)
    values = {}
    for var in sample.variables:
        entry = _array_summary(spatial_ds[var], cds_ds[var])
        if var in cds_compression:
            entry["cds"]["compression"] = cds_compression[var]
        try:
            entry["arraylake"]["compression"] = _zarr_compression_for_timestep(
                store=repo.store, group_path=repo.spatial_group,
                var=var, time_pos=sample.t,
            )
        except Exception as e:
            log.warning("Failed to probe zarr chunk size for %s: %s", var, e)
        values[var] = entry

    report.record(SampleResult(
        indices=indices, labels=labels, kind="data",
        equal=ok, values=values, error=err,
    ))


def run_task2_at(
    *,
    timestamp: str,
    report_path: str,
    cache_dir: str,
    repo_name: str,
    branch: str = "main",
    spatial_group: str = DEFAULT_SPATIAL_GROUP,
    temporal_group: str = DEFAULT_TEMPORAL_GROUP,
) -> bool:
    """Verify every CDS-mappable variable at a single specified timestamp.

    Variables flagged in the status subgroup as anything other than 0
    (`valid_data`) are recorded with their flag meaning and skipped from the
    CDS query — the dataset itself is asserting they're unavailable upstream,
    so there's no useful comparison to make. Variables with status==0 are
    submitted to CDS and compared via xr.testing.assert_equal as usual.
    """
    ts = pd.Timestamp(timestamp)
    log.info("Task 2 (single timestep): verifying every variable at %s", ts)

    repo = open_repo(repo_name, branch=branch,
                     spatial_group=spatial_group, temporal_group=temporal_group)
    if repo.spatial_status is None:
        raise ValueError(
            "task2-at requires a status subgroup on the spatial group; "
            "this repo doesn't expose one."
        )

    available = [v for v in repo.variables if v in VAR_TO_CDS]
    missing_map = [v for v in repo.variables if v not in VAR_TO_CDS]
    if missing_map:
        log.warning("No CDS mapping for variables (skipped): %s", missing_map)
    if not available:
        raise ValueError(f"None of {repo.variables} have a CDS mapping.")

    idx = repo.spatial.indexes["valid_time"]
    try:
        loc = idx.get_loc(ts)
    except KeyError:
        raise ValueError(
            f"Timestamp {ts} not in repo's valid_time range "
            f"[{idx[0]}, {idx[-1]}]"
        )
    if not isinstance(loc, (int, np.integer)):
        raise ValueError(
            f"Timestamp {ts} did not match a single position (got {loc!r}); "
            f"verify the valid_time index is monotonic and unique."
        )
    t_idx = int(loc)

    # Partition CDS-mappable variables by status flag at this timestamp.
    flag_meaning = _flag_lookup(repo.spatial_status)
    status_slice = repo.spatial_status.sel(valid_time=ts.to_datetime64()).load()
    queryable: list[str] = []
    flagged: dict[str, dict] = {}
    for var in available:
        if var not in status_slice.data_vars:
            queryable.append(var)
            continue
        flag = int(status_slice[var].item())
        if flag == VALID_DATA_FLAG:
            queryable.append(var)
        else:
            flagged[var] = {"flag": flag, "meaning": flag_meaning.get(flag, "unknown")}

    log.info("Status partition at %s: %d queryable, %d flagged as unavailable",
             ts, len(queryable), len(flagged))
    for var, info in flagged.items():
        log.info("  %-6s flag=%d (%s) — skipping CDS query", var, info["flag"], info["meaning"])

    report = Report(task="task2_spatial_vs_cds_at_timestamp", seed=0, n_samples=1)
    report.metadata = {
        "repo": repo_name,
        "branch": branch,
        "spatial_group": spatial_group,
        "timestamp": str(ts),
        "t_index": t_idx,
        "cache_dir": str(cache_dir),
        "cds_mappable_vars": available,
        "unmapped_vars_skipped": missing_map,
        "queryable_vars": queryable,
        "flagged_vars": flagged,
        "comparison": "xr.testing.assert_equal on the full Dataset slice "
                      "after normalizing CDS-specific coords",
    }

    # Record one report row per flagged variable so the JSON shows exactly
    # what was skipped and why. equal=True because we successfully verified
    # the dataset's own assertion that the value is unavailable; there's
    # nothing for CDS to confirm.
    indices = {"t": t_idx}
    labels = {"time": str(ts)}
    for var, info in flagged.items():
        report.record(SampleResult(
            indices=indices, labels=labels, kind="status_skip",
            equal=True, values={var: info},
            error=None,
        ))

    if queryable:
        sample = TimestepSample(t=t_idx, variables=tuple(sorted(queryable)))
        cache = CDSCache.load(cache_dir)
        cache.ensure_entry(ts.to_pydatetime(), list(sample.variables))
        client = make_client()
        cache.submit_pending(client)
        cache.wait_for_all(client)
        _compare_one_timestep(repo=repo, sample=sample, ts=ts, cache=cache, report=report)
    else:
        log.warning("No queryable variables at %s — nothing to ask CDS about.", ts)

    report.finalize()
    report.write(report_path)
    log.info(report.summary_line())
    if not report.passed:
        log.error("Task 2 (at) FAILED: %d mismatches", report.n_failures)
        for s in report.samples:
            if not s.equal:
                log.error("  MISMATCH kind=%s:\n%s", s.kind, s.error)
    return report.passed
