# Earthmover ERA5 Dataset QC Tool

Independent QC verifier for ERA5 hourly reanalysis stored in Arraylake
repositories. Point it at any repo that follows the expected group
layout with `--repo ORG/REPO`.

The verifier randomly samples the dataset and proves correctness against
two independent sources:

- **Task 1** — internal consistency: the same data must appear under the
  `single/spatial` group (chunked one-map-per-timestep, for spatial
  analysis) and the `single/temporal` group (chunked one-tile-per-year,
  for timeseries).
- **Task 2** — upstream truth: at random timesteps, the dataset must
  match what the Copernicus Climate Data Store (CDS) returns for the
  same time and variables.

Both tasks compare full xarray Datasets via `xarray.testing.assert_equal`
on identical `.sel(...)` indexers, so equivalence covers values, dtypes,
coordinates, and dimensions — not just numeric closeness.

## What gets verified

| Aspect | Task 1 | Task 2 |
|---|---|---|
| Sample shape | 200 random `(time, lat, lon)` points | 20 random timesteps × 5 random variables each |
| Data variables | All variables discovered in the repo (intersection of `spatial` and `temporal` groups) | Random subset per timestep, drawn from variables the verifier knows how to request from CDS |
| Per-timestep QA flags | Yes — `status/<var>` is compared at every point | No |
| Reference source | The other group in the same repo | Fresh CDS NetCDF for that hour |
| Equivalence test | `xr.testing.assert_equal(spatial.sel(t,y,x), temporal.sel(t,y,x))` | `xr.testing.assert_equal(spatial.sel(t)[vars], cds_dataset[vars])` |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) for
environment management. If you don't have `uv` installed:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the project root:

```sh
uv venv                       # create .venv/ with the matching Python
source .venv/bin/activate     # activate (use .venv/bin/activate.fish etc. for other shells)
uv pip install -e .           # install era5-qc and its dependencies
```

The project declares four top-level dependencies — `arraylake`,
`xarray[io]`, `cdsapi`, and `pcodec` — which transitively pull in
everything else (`icechunk`, `zarr`, `numpy`, `pandas`, `netCDF4`,
`click`, `tqdm`, etc.). `pcodec` is required because the dataset's
arrays are stored with the PCodec compressor and zarr needs the
matching codec to decode them. After install, the `era5-qc` console
script is on PATH inside the venv.

### Arraylake prerequisites (both tasks)

The user running the tool must already be authenticated to Arraylake
with at least read access to the repo under test. The CLI uses
`arraylake.Client()`, which reads its config from the same locations
as the `al` CLI — log in with `al auth login` if you haven't already.

### CDS prerequisites (Task 2 only)

1. A CDS personal access token configured in `~/.cdsapirc`. Sign up at
   <https://cds.climate.copernicus.eu/> and follow
   <https://cds.climate.copernicus.eu/how-to-api> for the token format.
2. The dataset licence "Licence to use Copernicus Products" must be
   accepted on the CDS account that owns the token. If not, the first
   request will fail with HTTP 403 and a link to the manage-licences
   page:
   <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences>
   Tick the checkbox once and it stays accepted for that account.

## Running the tests

`--repo ORG/REPO` is required. Substitute the Arraylake path to the
ERA5 repo you want to verify.

```sh
era5-qc task1 --repo ORG/REPO                     # full 200-point spatial vs temporal check
era5-qc task2 --repo ORG/REPO                     # full 20-timestep × 5-var CDS check
era5-qc all   --repo ORG/REPO                     # run both, write two reports
```

Useful smaller invocations while iterating:

```sh
era5-qc task1 --repo ORG/REPO -n 5                    # quick smoke (~1 minute)
era5-qc task2 --repo ORG/REPO --n-steps 1 --n-vars 1  # one-CDS-request smoke (~30s if queue is empty)
```

To investigate a specific timestamp — every CDS-mappable variable in
one request, with status-flagged variables (e.g. ERA5 accumulated
fluxes during the first hours of 1940-01-01) automatically skipped
from the CDS query:

```sh
era5-qc task2-at --repo ORG/REPO --timestamp 1940-01-01T00:00
```

Common options:

| Flag | Default | Meaning |
|---|---|---|
| `--repo ORG/REPO` | (required) | Which Arraylake repo to verify. |
| `--branch NAME` | `main` | Which Arraylake branch to verify. |
| `--spatial-group PATH` | `single/spatial` | Zarr group within the repo holding the per-timestep-chunked layout. |
| `--temporal-group PATH` | `single/temporal` | Zarr group within the repo holding the per-tile-chunked layout. |
| `--seed N` | `42` | Seed the RNG. Same seed → same sample set. Override to draw a fresh sample. |
| `--report PATH` | `reports/<task>.json` | Where to write the JSON report. |
| `--cache-dir PATH` (task2) | `cache/cds` | Where to persist CDS state + downloaded NetCDFs. |
| `-v / --verbose` | off | Switch logging to DEBUG. |

Exit code is `0` if every comparison passed and non-zero otherwise.

### What the verifier discovers at startup

At startup the verifier:

1. Opens the spatial and temporal groups, intersects their data
   variables, and uses the intersection as the verification set (with a
   warning if either group has extra variables).
2. Probes for a `status/` subgroup under each parent group; task1
   silently skips the status comparison if it doesn't exist.
3. For task2, intersects the discovered variables with the built-in
   ERA5 CDS variable map (`cds.VAR_TO_CDS`). Variables without a
   mapping are logged once and skipped from CDS verification — task1
   still covers them.

If your repo uses a different group layout (e.g., bare `spatial` and
`temporal` paths instead of nested `single/spatial`,
`single/temporal`), pass the right paths with `--spatial-group` and
`--temporal-group`.

### Expected wall time

- **Task 1**: ~30 minutes for 200 points. Each point triggers small chunk
  reads from both groups (temporal chunks are `(8736, 12, 12)` so each
  random point pulls a ~5 MB tile per variable).
- **Task 2**: dominated by the CDS queue. With an empty queue, ~1 minute
  per request × 20 requests ≈ 20 minutes. Under heavy queue load,
  CDS jobs can take hours; the cache is idempotent, so killing and
  resuming is safe.

## How Task 2's CDS pipeline works

State lives in `cache/cds/state.json`; downloaded NetCDFs are at
`cache/cds/<request_hash>.nc`. The flow is:

1. **Submit**: each sampled timestep becomes one CDS request bundling
   all chosen variables. Submit is non-blocking — `request_id` is
   persisted immediately.
2. **Poll**: exponential backoff from 30s up to 5 min. The poller
   surfaces CDS's own status messages.
3. **Download**: on `successful`, the NetCDF is streamed to disk.
4. **Compare**: each cached file is opened with xarray, normalized,
   and compared to the corresponding spatial selection.

If a request fails or its file disappears, the next run will resubmit
it; previously-completed requests are skipped. To force a clean re-run,
delete `cache/cds/state.json` and the `.nc` files.

### CDS multi-stream caveat

ERA5 single-levels variables come from two GRIB streams (instantaneous
for `t2m`, `u10`, ... and accumulated for `ssr`, `ssrd`, `tisr`, ...).
When a request mixes them, CDS returns a **ZIP of two NetCDFs** even
though the request asks for `download_format=unarchived`. The verifier
detects this and merges the parts transparently, so the cached files
keep their `.nc` extension but may be either format internally.

## Reading the JSON reports

Every report has the same top-level shape:

```json
{
  "task": "task1_spatial_vs_temporal",
  "seed": 42,
  "n_samples": 200,
  "started_at": "2026-05-28T15:29:49+00:00",
  "finished_at": "2026-05-28T16:00:30+00:00",
  "n_failures": 0,
  "passed": true,
  "metadata": { ... task-specific configuration ... },
  "samples": [ ... one entry per comparison ... ]
}
```

### Task 1 sample shape

Two rows per sampled point — one `kind: "data"` (14 data variables) and
one `kind: "status"` (the per-timestep QA flags). The `values` field
shows the actual scalar values that were compared.

```json
{
  "indices": {"t": 67587, "y": 471, "x": 623},
  "labels":  {"time": "1947-09-17 03:00:00", "latitude": -27.75, "longitude": 155.75},
  "kind": "data",
  "equal": true,
  "values": {
    "t2m":  {"spatial": 292.388,   "temporal": 292.388},
    "u10":  {"spatial": -1.089,    "temporal": -1.089},
    "...":  {"...": "..."}
  },
  "error": null
}
```

### Task 2 sample shape

One row per sampled timestep with all chosen variables in `values`.
Because each comparison covers a full 2D field (721 × 1440 grid), the
report records min/max/mean per side instead of individual values.

```json
{
  "indices": {"t": 67587},
  "labels":  {"time": "1947-09-17 03:00:00"},
  "kind": "data",
  "equal": true,
  "values": {
    "u10": {
      "arraylake": {"min": -21.560, "max": 18.946, "mean": 0.0166},
      "cds":       {"min": -21.560, "max": 18.946, "mean": 0.0166}
    },
    "...": "..."
  },
  "error": null
}
```

### When a comparison fails

`equal` is `false` and `error` carries the `xr.testing.assert_equal`
diff (mismatched values, coords, or dims). To reproduce a single
failure, re-run with the same `--seed` and inspect the offending
entry in the JSON.

## Project layout

```
src/era5_qc/
├── cli.py             # click entry: era5-qc task1|task2|all
├── dataset.py         # opens the Arraylake repo and its groups
├── sampling.py        # seeded RNG: sample_points / sample_timestep_variables
├── task1.py           # spatial vs temporal verifier
├── task2.py           # spatial vs CDS verifier
├── cds.py             # CDS submit/poll/cache (Submit-poll-cache resilience)
├── report.py          # JSON report writer + tqdm-safe summary
└── logging_config.py  # tqdm-compatible logging setup

reports/               # JSON outputs (gitignored)
cache/cds/             # CDS state + downloaded NetCDFs (gitignored)
```

## Notes on the dataset

- ERA5 single-level variables, hourly, on a 0.25° regular lat/lon grid
  (721 × 1440), float32, PCodec-compressed. The variable set and time
  range vary by subscription; both are discovered at runtime from the
  spatial group.
- The default group layout is `single/spatial` and `single/temporal`,
  each carrying `valid_time`, `latitude`, and `longitude` coordinate
  variables that xarray loads directly. Override the group paths with
  `--spatial-group` / `--temporal-group` if your repo differs.

## Extending the CDS variable map

ERA5 short codes (`t2m`, `d2m`, `u10`, ...) map to CDS long names
(`2m_temperature`, ...) via the `VAR_TO_CDS` dict in
[src/era5_qc/cds.py](src/era5_qc/cds.py). If your subscription
includes variables that aren't already in the map, task2 will log a
warning and skip them. Add the missing mappings to that dict; the
canonical short→long name list lives in the CDS catalogue page for
`reanalysis-era5-single-levels`.

## License

Released under the Apache License, Version 2.0. See [LICENSE](LICENSE).
