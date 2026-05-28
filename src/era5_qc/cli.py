from __future__ import annotations

import sys

import click

from . import __version__
from .dataset import DEFAULT_SPATIAL_GROUP, DEFAULT_TEMPORAL_GROUP


def _repo_options(f):
    """Shared repo/group/branch click options."""
    f = click.option("--repo", "repo_name", required=True,
                     help="Arraylake repo to verify, formatted as ORG/REPO.")(f)
    f = click.option("--branch", default="main", show_default=True,
                     help="Branch name to verify.")(f)
    f = click.option("--spatial-group", default=DEFAULT_SPATIAL_GROUP, show_default=True,
                     help="Zarr group path within the repo holding the "
                          "spatial-chunked layout.")(f)
    f = click.option("--temporal-group", default=DEFAULT_TEMPORAL_GROUP, show_default=True,
                     help="Zarr group path within the repo holding the "
                          "temporal-chunked layout.")(f)
    return f


@click.group()
@click.version_option(__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG-level logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Independent QC verifier for ERA5 Arraylake repos (spatial vs temporal vs CDS)."""
    from .logging_config import setup_logging

    setup_logging(verbose=verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.option("-n", "--n", "n_points", default=200, show_default=True, type=int,
              help="Number of random (time, lat, lon) points to sample.")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--report", "report_path", default="reports/task1.json", show_default=True,
              type=click.Path(dir_okay=False))
@_repo_options
def task1(n_points: int, seed: int, report_path: str, repo_name: str, branch: str,
          spatial_group: str, temporal_group: str) -> None:
    """Verify spatial vs temporal group consistency at random points."""
    from .task1 import run_task1

    ok = run_task1(
        n_points=n_points, seed=seed, report_path=report_path,
        repo_name=repo_name, branch=branch,
        spatial_group=spatial_group, temporal_group=temporal_group,
    )
    sys.exit(0 if ok else 1)


@main.command()
@click.option("--n-steps", default=20, show_default=True, type=int,
              help="Number of random timesteps to verify against CDS.")
@click.option("--n-vars", default=5, show_default=True, type=int,
              help="Number of random variables to verify per timestep.")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--report", "report_path", default="reports/task2.json", show_default=True,
              type=click.Path(dir_okay=False))
@click.option("--cache-dir", default="cache/cds", show_default=True,
              type=click.Path(file_okay=False))
@_repo_options
def task2(n_steps: int, n_vars: int, seed: int, report_path: str, cache_dir: str,
          repo_name: str, branch: str, spatial_group: str, temporal_group: str) -> None:
    """Verify spatial group against fresh CDS downloads at random (timestep, vars)."""
    from .task2 import run_task2

    ok = run_task2(
        n_steps=n_steps, n_vars=n_vars, seed=seed, report_path=report_path,
        cache_dir=cache_dir, repo_name=repo_name, branch=branch,
        spatial_group=spatial_group, temporal_group=temporal_group,
    )
    sys.exit(0 if ok else 1)


@main.command(name="task2-at")
@click.option("--timestamp", required=True,
              help="ISO 8601 timestamp to verify (e.g., 1940-01-01T00:00).")
@click.option("--report", "report_path", default="reports/task2_at.json",
              show_default=True, type=click.Path(dir_okay=False))
@click.option("--cache-dir", default="cache/cds", show_default=True,
              type=click.Path(file_okay=False))
@_repo_options
def task2_at(timestamp: str, report_path: str, cache_dir: str,
             repo_name: str, branch: str, spatial_group: str, temporal_group: str) -> None:
    """Verify every CDS-mappable variable at a single specified timestamp.

    Use this to investigate individual timesteps where you suspect a data
    quirk — e.g., to confirm that the first hours of 1940-01-01 contain
    NaN accumulated fluxes upstream in CDS, not just in the Arraylake repo.
    """
    from .task2 import run_task2_at

    ok = run_task2_at(
        timestamp=timestamp, report_path=report_path, cache_dir=cache_dir,
        repo_name=repo_name, branch=branch,
        spatial_group=spatial_group, temporal_group=temporal_group,
    )
    sys.exit(0 if ok else 1)


@main.command(name="all")
@click.pass_context
def run_all(ctx: click.Context) -> None:
    """Run task1 and task2 with default options."""
    ctx.invoke(task1)
    ctx.invoke(task2)


if __name__ == "__main__":
    main()
