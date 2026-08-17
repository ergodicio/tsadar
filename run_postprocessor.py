import argparse, os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
from jax import config

config.update("jax_enable_x64", True)

from tsadar.postprocess_runner import run_postprocess_local, run_postprocess_remote


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay postprocess() on an already-completed fit")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="local directory containing defaults.yaml, inputs.yaml, and fitted_weights.eqx")
    group.add_argument("--run", help="mlflow run id, or a run URL (e.g. from continuum.ergodic.io)")

    args = parser.parse_args()

    if args.dir is not None:
        run_postprocess_local(args.dir)
    else:
        run_postprocess_remote(args.run)
