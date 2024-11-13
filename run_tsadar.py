import argparse, os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
from jax import config

config.update("jax_enable_x64", True)
#config.update("jax_disable_jit", True)

from tsadar.runner import run, run_job
from tsadar.misc.utils import export_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSADAR")
    parser.add_argument("--cfg", help="enter path to cfg")
    parser.add_argument("--run_id", help="enter run_id to continue")
    parser.add_argument("--mode", help="forward or fit")

    args = parser.parse_args()

    if args.run_id is not None:
        run_job(args.run_id, args.mode, nested=None)
        run_id = args.run_id
    else:
        run_id = run(args.cfg, mode=args.mode)

    if "MLFLOW_EXPORT" in os.environ:
        export_run(run_id)
