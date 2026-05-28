from __future__ import annotations

import logging
from dataclasses import dataclass

import xarray as xr

DEFAULT_SPATIAL_GROUP = "single/spatial"
DEFAULT_TEMPORAL_GROUP = "single/temporal"
STATUS_SUBGROUP = "status"

log = logging.getLogger(__name__)


@dataclass
class OpenedRepo:
    spatial: xr.Dataset
    temporal: xr.Dataset
    spatial_status: xr.Dataset | None
    temporal_status: xr.Dataset | None
    variables: list[str]
    repo_name: str
    spatial_group: str
    temporal_group: str
    branch: str
    store: object  # IcechunkStore; kept for low-level chunk-size probing

    @property
    def n_time(self) -> int:
        return self.spatial.sizes["valid_time"]

    @property
    def n_lat(self) -> int:
        return self.spatial.sizes["latitude"]

    @property
    def n_lon(self) -> int:
        return self.spatial.sizes["longitude"]


def _try_open_status(store, parent_group: str) -> xr.Dataset | None:
    path = f"{parent_group}/{STATUS_SUBGROUP}"
    try:
        return xr.open_zarr(store, group=path, consolidated=False, chunks=None)
    except Exception as e:
        log.info("No status subgroup at %s (%s)", path, e)
        return None


def open_repo(
    repo_name: str,
    *,
    branch: str = "main",
    spatial_group: str = DEFAULT_SPATIAL_GROUP,
    temporal_group: str = DEFAULT_TEMPORAL_GROUP,
) -> OpenedRepo:
    import arraylake

    log.info("Opening Arraylake repo %s on branch %s (spatial=%s, temporal=%s)",
             repo_name, branch, spatial_group, temporal_group)
    repo = arraylake.Client().get_repo(repo_name)
    store = repo.readonly_session(branch=branch).store

    spatial = xr.open_zarr(store, group=spatial_group, consolidated=False, chunks=None)
    temporal = xr.open_zarr(store, group=temporal_group, consolidated=False, chunks=None)
    spatial_status = _try_open_status(store, spatial_group)
    temporal_status = _try_open_status(store, temporal_group)

    sp_vars = set(spatial.data_vars)
    tp_vars = set(temporal.data_vars)
    common = sp_vars & tp_vars
    only_spatial = sp_vars - tp_vars
    only_temporal = tp_vars - sp_vars
    if only_spatial or only_temporal:
        log.warning(
            "Variable mismatch between groups — only-spatial=%s only-temporal=%s; "
            "verification will use intersection (%d vars)",
            sorted(only_spatial), sorted(only_temporal), len(common),
        )
    variables = sorted(common)
    if not variables:
        raise ValueError(f"No common variables between {spatial_group} and {temporal_group}")

    log.info("Repo opened: %d variables, dims valid_time=%d latitude=%d longitude=%d, "
             "time %s -> %s",
             len(variables), spatial.sizes["valid_time"], spatial.sizes["latitude"],
             spatial.sizes["longitude"],
             spatial.valid_time.values[0], spatial.valid_time.values[-1])

    return OpenedRepo(
        spatial=spatial,
        temporal=temporal,
        spatial_status=spatial_status,
        temporal_status=temporal_status,
        variables=variables,
        repo_name=repo_name,
        spatial_group=spatial_group,
        temporal_group=temporal_group,
        branch=branch,
        store=store,
    )
