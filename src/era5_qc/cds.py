from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

CDS_DATASET = "reanalysis-era5-single-levels"

VAR_TO_CDS = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "u100": "100m_u_component_of_wind",
    "v100": "100m_v_component_of_wind",
    "tcc": "total_cloud_cover",
    "tisr": "toa_incident_solar_radiation",
    "ssr": "surface_net_solar_radiation",
    "ssrd": "surface_solar_radiation_downwards",
    "fdir": "total_sky_direct_solar_radiation_at_surface",
    "fsr": "forecast_surface_roughness",
    "stl1": "soil_temperature_level_1",
    "stl4": "soil_temperature_level_4",
}


def request_hash(timestamp: datetime, variables: Iterable[str]) -> str:
    key = f"{timestamp.isoformat()}|{','.join(sorted(variables))}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_request(timestamp: datetime, variables: Iterable[str]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": [VAR_TO_CDS[v] for v in sorted(variables)],
        "year": [f"{timestamp.year:04d}"],
        "month": [f"{timestamp.month:02d}"],
        "day": [f"{timestamp.day:02d}"],
        "time": [timestamp.strftime("%H:%M")],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


@dataclass
class CDSEntry:
    request_hash: str
    timestamp_iso: str
    variables: list[str]
    request_id: str | None = None
    status: str = "pending"  # pending|submitted|completed|failed
    target_path: str | None = None
    submitted_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


@dataclass
class CDSCache:
    cache_dir: Path
    state: dict[str, CDSEntry] = field(default_factory=dict)

    @property
    def state_file(self) -> Path:
        return self.cache_dir / "state.json"

    @classmethod
    def load(cls, cache_dir: str | Path) -> "CDSCache":
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        c = cls(cache_dir=cache_dir)
        if c.state_file.exists():
            raw = json.loads(c.state_file.read_text())
            c.state = {h: CDSEntry(**rec) for h, rec in raw.items()}
        return c

    def save(self) -> None:
        payload = {h: rec.__dict__ for h, rec in self.state.items()}
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.state_file)

    def ensure_entry(self, timestamp: datetime, variables: list[str]) -> CDSEntry:
        h = request_hash(timestamp, variables)
        if h not in self.state:
            self.state[h] = CDSEntry(
                request_hash=h,
                timestamp_iso=timestamp.isoformat(),
                variables=sorted(variables),
            )
            self.save()
        return self.state[h]

    def path_for(self, entry: CDSEntry) -> Path:
        return self.cache_dir / f"{entry.request_hash}.nc"

    def submit_pending(self, client) -> None:
        for entry in self.state.values():
            if entry.status in ("submitted", "completed"):
                # Check that completed files still exist on disk
                if entry.status == "completed":
                    if entry.target_path and Path(entry.target_path).exists():
                        continue
                    log.warning("Cache hash %s marked completed but file missing; resubmitting",
                                entry.request_hash)
                    entry.status = "pending"
                    entry.request_id = None
                else:
                    continue
            ts = datetime.fromisoformat(entry.timestamp_iso)
            req = build_request(ts, entry.variables)
            log.info("Submitting CDS request %s (%s, %d vars)",
                     entry.request_hash, entry.timestamp_iso, len(entry.variables))
            remote = client.submit(collection_id=CDS_DATASET, request=req)
            entry.request_id = remote.request_id
            entry.status = "submitted"
            entry.submitted_at = datetime.now(timezone.utc).isoformat()
            self.save()

    def _refresh_one(self, client, entry: CDSEntry) -> None:
        remote = client.get_remote(entry.request_id)
        # Force a status refresh from server
        try:
            remote.update()
        except Exception as e:
            log.warning("update() failed for %s: %s", entry.request_hash, e)
        s = remote.status
        log.debug("Request %s status=%s", entry.request_hash, s)
        if s == "successful" or remote.results_ready:
            target = self.path_for(entry)
            log.info("Downloading %s -> %s", entry.request_hash, target)
            remote.download(str(target))
            entry.status = "completed"
            entry.target_path = str(target)
            entry.completed_at = datetime.now(timezone.utc).isoformat()
            self.save()
        elif s == "failed":
            entry.status = "failed"
            entry.error = str(remote.error) if remote.error else "unknown"
            self.save()

    def wait_for_all(self, client, *, sleep_start: float = 30.0,
                     sleep_max: float = 300.0) -> None:
        from tqdm import tqdm

        pending = [e for e in self.state.values()
                   if e.status not in ("completed", "failed")]
        if not pending:
            log.info("No pending CDS requests; everything is already cached.")
            return

        sleep = sleep_start
        pbar = tqdm(total=len(pending), desc="CDS requests", unit="req")
        done = 0
        while True:
            still_pending = []
            for entry in pending:
                if entry.status in ("completed", "failed"):
                    continue
                try:
                    self._refresh_one(client, entry)
                except Exception as e:
                    log.warning("Polling %s raised %s; will retry", entry.request_hash, e)
                if entry.status in ("completed", "failed"):
                    done += 1
                    pbar.update(1)
                else:
                    still_pending.append(entry)
            pending = still_pending
            if not pending:
                break
            log.info("Waiting %.0fs before next CDS poll (%d still pending)", sleep, len(pending))
            time.sleep(sleep)
            sleep = min(sleep * 1.5, sleep_max)
        pbar.close()


def make_client():
    from cdsapi.api import get_url_key_verify
    from ecmwf import datastores

    url, key, verify = get_url_key_verify(None, None, None)
    return datastores.Client(url=url, key=key, verify=bool(verify))
