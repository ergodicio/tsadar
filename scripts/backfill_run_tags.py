"""
Backfills the canonical tags from ergodicio/tsadar#115 onto runs that predate
them, reading the values back out of the flattened params those runs already
logged.

Runs written before the tags landed carry everything the tags hold, but only
inside ``misc.log_mlflow``'s flattened param dump -- ``data.shotnum``,
``data.load_ele_spec``, ``other.username``. Params are searchable, so a reader
*could* fall back to them, but the fallback has to live in every downstream
client forever. Backfilling instead means the browser's run table can query one
stable surface.

Two things this deliberately does not do:

- It does not invent a terminal ``status``. A historical run that died mid-stage
  is genuinely ambiguous from params alone, and writing "failed" onto a run that
  may have finished fine would be worse than leaving it unknown. The MLflow
  lifecycle status (FINISHED/FAILED/KILLED) already covers that case for readers
  who need it.
- It does not touch ``tsadar.version``: the version that produced a historical
  run was never recorded anywhere, and guessing it from the run date would put a
  wrong-but-confident value on the one tag whose whole point is provenance.

Usage::

    python scripts/backfill_run_tags.py --experiment my-experiment --dry-run
    python scripts/backfill_run_tags.py --experiment my-experiment
    python scripts/backfill_run_tags.py --all-experiments
"""

import argparse
import sys

import mlflow
from mlflow.tracking import MlflowClient

from tsadar.utils import misc


def tags_from_params(params: dict) -> dict:
    """
    Reconstructs the canonical tags from a historical run's flattened params.

    Args:
        params: the run's ``data.params`` mapping, flattened with a dot reducer

    Returns:
        tags: the canonical tags derivable from those params, omitting any that
            cannot be determined
    """

    # Rebuild just enough of the nested config for the shared helpers, so the
    # backfilled values cannot drift from what a live run would tag.
    config = {
        "data": {
            "shotnum": params.get("data.shotnum"),
            # Params come back as strings; "False" is truthy, so compare textually.
            "load_ele_spec": str(params.get("data.load_ele_spec", "")).casefold() == "true",
            "load_ion_spec": str(params.get("data.load_ion_spec", "")).casefold() == "true",
        },
        "other": {"username": params.get("other.username")},
    }

    tags = misc.canonical_tags(config, mode="fit")

    # mode was never logged as a param, and "fit" is only the default of
    # canonical_tags -- drop it rather than assert a mode we cannot know.
    tags.pop(f"{misc.TAG_NAMESPACE}.mode", None)
    tags.pop(f"{misc.TAG_NAMESPACE}.version", None)

    # "none" here means neither load switch was present in the params, which is
    # absence of evidence rather than a run that loaded no spectra.
    if not any(k.startswith("data.load_") for k in params):
        tags.pop(f"{misc.TAG_NAMESPACE}.data", None)

    return tags


def backfill(experiment_names, dry_run: bool = False) -> int:
    """
    Adds missing canonical tags to every run in the given experiments.

    Existing tags are never overwritten: a run tagged by a live tsadar is the
    authority on itself, and this script's params-derived values are strictly
    the weaker source.

    Args:
        experiment_names: experiment names to walk, or None for all of them
        dry_run: when True, report what would change without writing

    Returns:
        the number of runs that were (or would be) updated
    """

    client = MlflowClient()

    if experiment_names:
        experiments = []
        for name in experiment_names:
            exp = client.get_experiment_by_name(name)
            if exp is None:
                print(f"no such experiment: {name!r}", file=sys.stderr)
                return -1
            experiments.append(exp)
    else:
        experiments = client.search_experiments()

    updated = 0
    for exp in experiments:
        for run in mlflow.search_runs(
            experiment_ids=[exp.experiment_id], output_format="list", max_results=50_000
        ):
            existing = run.data.tags or {}
            wanted = tags_from_params(run.data.params or {})
            missing = {k: v for k, v in wanted.items() if k not in existing}
            if not missing:
                continue

            print(f"{exp.name}/{run.info.run_id}: {missing}")
            if not dry_run:
                for key, value in missing.items():
                    client.set_tag(run.info.run_id, key, value)
            updated += 1

    verb = "would update" if dry_run else "updated"
    print(f"{verb} {updated} run(s)")

    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", dest="experiments", help="experiment name; repeatable")
    parser.add_argument("--all-experiments", action="store_true", help="walk every experiment")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing them")
    args = parser.parse_args()

    if not args.experiments and not args.all_experiments:
        parser.error("pass --experiment NAME (repeatable) or --all-experiments")

    return 0 if backfill(args.experiments, dry_run=args.dry_run) >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
