"""Queues run_mcmc_postprocessor.py as a Slurm batch job, mirroring queue_tsadar.py's cluster submission
for the fit/forward entry point (run_tsadar.py).

Unlike queue_tsadar.py, this never creates the mlflow run up front: run_mcmc_postprocess_local/remote
(invoked by run_mcmc_postprocessor.py once the job actually starts) create their own new run against the
completed fit's saved artifacts, so there is nothing to pre-create here. The only thing this script needs
before submitting is which Slurm partition to use, auto-detected the same way queue_tsadar.py does -- from
the "machine" field of the target's config -- read locally for --dir or downloaded (config only, not
fitted_weights.eqx) for --run.
"""
import argparse, os, tempfile, time

os.environ["JAX_PLATFORMS"] = "cpu"

import mlflow

from tsadar.postprocess_runner import _extract_run_id, _load_merged_config
from tsadar.utils import misc

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def _resolve_machine(args) -> str:
    """Determines which Slurm partition to submit to, the same way queue_tsadar.py does: by reading the
    "machine" field out of the target's config (top-level in inputs.yaml, so it survives
    merge_defaults_and_inputs unchanged). For --run, downloads just the config artifact(s) -- never
    fitted_weights.eqx, which is only needed once the job actually runs -- to a scratch temp dir."""
    if args.dir is not None:
        config = _load_merged_config(args.dir)
    else:
        run_id = _extract_run_id(args.run)
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            try:
                mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.yaml", dst_path=td)
            except Exception:
                for fname in ["defaults.yaml", "inputs.yaml"]:
                    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=fname, dst_path=td)
            config = _load_merged_config(td)

    if args.overrides:
        import yaml

        with open(args.overrides, "r") as fi:
            overrides = yaml.safe_load(fi)
        config = misc.merge_defaults_and_inputs(config, overrides)

    machine = config.get("machine")
    if not machine:
        raise ValueError(
            f"Could not find a top-level 'machine' field (cpu/gpu) in the config for "
            f"{'--dir ' + args.dir if args.dir is not None else '--run ' + args.run}."
        )
    return machine


def _queue_run_(machine: str, args):
    if "cpu" in machine:
        base_job_file = os.environ["CPU_BASE_JOB_FILE"]
    elif "gpu" in machine:
        base_job_file = os.environ["GPU_BASE_JOB_FILE"]
    else:
        raise NotImplementedError

    run_cmd = "srun python run_mcmc_postprocessor.py"
    if args.dir is not None:
        run_cmd += f" --dir {args.dir}"
    else:
        run_cmd += f" --run {args.run}"
    if args.overrides:
        run_cmd += f" --overrides {args.overrides}"

    with open(base_job_file, "r") as fh:
        base_job = fh.read()

    with open(os.path.join(os.getcwd(), "new_job.sh"), "w") as job_file:
        job_file.write(base_job + "\n")
        job_file.writelines(run_cmd)

    os.system("sbatch new_job.sh")
    time.sleep(0.1)
    os.system("sqs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Queue the MCMC uncertainty postprocessor as a Slurm job")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="local directory containing defaults.yaml, inputs.yaml, and fitted_weights.eqx")
    group.add_argument("--run", help="mlflow run id, or a run URL (e.g. from continuum.ergodic.io)")
    parser.add_argument(
        "--overrides",
        help=(
            "Path to a small YAML stub deck (same nesting as inputs.yaml) with just the postprocessing "
            "keys to change (e.g. config['other']['mcmc']), deep-merged on top of the original fit's "
            "saved config. Passed straight through to run_mcmc_postprocessor.py on the compute node, so "
            "the path must also resolve there (e.g. a repo-relative path, not a local absolute one)."
        ),
    )
    args = parser.parse_args()

    os.system("uv sync --extra gpu,hdf")

    machine = _resolve_machine(args)
    _queue_run_(machine, args)
