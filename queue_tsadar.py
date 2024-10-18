import argparse, os, time

from tsadar.runner import load_and_make_folders

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def _queue_run_(machine, mode, run_id):
    if "cpu" in machine:
        base_job_file = os.environ["CPU_BASE_JOB_FILE"]
    elif "gpu" in machine:
        base_job_file = os.environ["GPU_BASE_JOB_FILE"]
    else:
        raise NotImplementedError

    with open(base_job_file, "r") as fh:
        base_job = fh.read()

    with open(os.path.join(os.getcwd(), "new_job.sh"), "w") as job_file:
        job_file.write(base_job + "\n")
        job_file.writelines(f"srun python run_tsadar.py --mode {mode} --run_id {run_id}")

    os.system(f"sbatch new_job.sh")
    time.sleep(0.1)
    os.system("sqs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSADAR")
    parser.add_argument("--cfg", help="enter path to cfg")
    parser.add_argument("--mode", help="forward or fit")
    args = parser.parse_args()

    run_id, config = load_and_make_folders(args.cfg)
    _queue_run_(config["inputs"]["machine"], args.mode, run_id)
