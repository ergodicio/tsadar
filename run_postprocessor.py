import argparse, os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
from jax import config

config.update("jax_enable_x64", True)

import yaml

from tsadar.postprocess_runner import run_postprocess_local, run_postprocess_remote


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay postprocess() on an already-completed fit")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="local directory containing defaults.yaml, inputs.yaml, and fitted_weights.eqx")
    group.add_argument("--run", help="mlflow run id, or a run URL (e.g. from continuum.ergodic.io)")
    parser.add_argument(
        "--overrides",
        help=(
            "Path to a small YAML stub deck (same nesting as inputs.yaml) with just the postprocessing "
            "keys to change (e.g. plotting ranges), deep-merged on top of the original fit's saved "
            "config in memory. Leaves the saved deck itself untouched. Do not use this to override "
            "fields the fit reconstruction depends on (data.lineouts, optimizer.batch_size, "
            "parameters.*.active, etc.) -- those must match what fitted_weights.eqx was saved against."
        ),
    )

    args = parser.parse_args()
    overrides = None
    if args.overrides:
        with open(args.overrides, "r") as fi:
            overrides = yaml.safe_load(fi)

    if args.dir is not None:
        run_postprocess_local(args.dir, overrides=overrides)
    else:
        run_postprocess_remote(args.run, overrides=overrides)
