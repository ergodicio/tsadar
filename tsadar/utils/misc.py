import os, mlflow, flatten_dict, boto3, botocore, shutil, time, tempfile
from urllib.parse import urlparse
from functools import partial


def log_mlflow(cfg, which="params", step=0):
    """
    Logs the parameters form the input deck in the parameters section of MLFlow.


    Args:
        cfg: input dictionary

    Returns:

    """
    flattened_dict = flatten_dict.flatten(cfg, reducer="dot")  # dict(flatdict.FlatDict(cfg, delimiter="."))
    num_entries = len(flattened_dict.keys())

    if which == "params":
        log_func = mlflow.log_params
    elif which == "metrics":
        log_func = partial(mlflow.log_metrics, step=step)
    else:
        raise ValueError("which must be either 'params' or 'metrics'")

    if num_entries > 100:
        num_batches = num_entries % 100
        fl_list = list(flattened_dict.items())
        for i in range(num_batches):
            end_ind = min((i + 1) * 100, num_entries)
            trunc_dict = {k: v for k, v in fl_list[i * 100 : end_ind]}
            log_func(trunc_dict)
    else:
        log_func(flattened_dict)


def update(base_dict, new_dict):
    """
    Combines 2 dictionaries overwriting common fields


    Args:
        base_dict: dictionary to be modified
        new_dict: dictionary containing new or additional values to be inserted

    Returns:
        combined_dict: combined dictionary with the updated values

    """
    combined_dict = {}
    for k, v in new_dict.items():
        combined_dict[k] = base_dict[k]
        if isinstance(v, dict):
            combined_dict[k] = update(base_dict[k], v)
        else:
            combined_dict[k] = new_dict[k]

    return combined_dict


def upload_dir_to_s3(local_directory: str, bucket: str, destination: str, run_id: str, prefix="ingest", step=0):
    """
    Uploads the contents of a local directory to an S3 bucket, preserving the directory structure.
    After uploading all files, creates a marker file indicating completion and uploads it to the bucket.

    Args:    
        local_directory (str): Path to the local directory to upload.
        bucket (str): Name of the S3 bucket to upload to.
        destination (str): S3 key prefix (folder path) where files will be uploaded.
        run_id (str): Identifier for the current run, used in the marker filename.
        prefix (str, optional): Prefix for the marker filename. Defaults to "ingest".
        step (int, optional): Step number for the marker filename. Defaults to 0.
    Returns:    
        None
    """
    client = boto3.client("s3")

    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)
            client.upload_file(local_path, bucket, s3_path)

    filename = f"{prefix}-{run_id}-{step}.txt"
    filepath = os.path.join(local_directory, filename)

    with open(filepath, "w") as fi:
        fi.write("ready")

    client.upload_file(filepath, bucket, filename)


def export_run(run_id, prefix="ingest", step=0):
    """
    Exports an MLflow run and uploads its artifacts to an S3 bucket.
    Args:
        run_id (str): The unique identifier of the MLflow run to export.
        prefix (str, optional): Prefix to use when uploading to S3. Defaults to "ingest".
        step (int, optional): Step number or identifier for the upload process. Defaults to 0.
    Side Effects:
        - Exports the specified MLflow run to a temporary directory.
        - Uploads the exported run directory to the specified S3 bucket and path.
        - Prints the time taken for export and upload operations.
    Environment Variables:
        BASE_TEMPDIR: If set, used as the base directory for the temporary export directory.
    Raises:
        Any exceptions raised by MLflow or S3 upload operations will propagate.
    """

    t0 = time.time()
    from mlflow_export_import.run.export_run import RunExporter

    run_exp = RunExporter(mlflow_client=mlflow.MlflowClient())
    with tempfile.TemporaryDirectory(dir=os.getenv("BASE_TEMPDIR")) as td2:
        run_exp.export_run(run_id, td2)
        print(f"Export took {round(time.time() - t0, 2)} s")
        t0 = time.time()
        upload_dir_to_s3(td2, "remote-mlflow-staging", f"artifacts/{run_id}", run_id, prefix=prefix, step=step)
    print(f"Uploading took {round(time.time() - t0, 2)} s")


def get_cfg(artifact_uri, temp_path):
    """
    Downloads configuration files from the specified artifact URI to a temporary path. Allows configuration files to be locked at queue time.
    Parameters:
        artifact_uri (str): The URI of the artifact containing the configuration files.
        temp_path (str): The temporary directory path where the files will be downloaded.
    Returns:
        None
    Note:
        This function currently downloads 'defaults.yaml' and 'inputs.yaml' files but does not load or return their contents.
    """

    dest_file_path = download_file("defaults.yaml", artifact_uri, temp_path)
    dest_file_path = download_file("inputs.yaml", artifact_uri, temp_path)
    # with open(dest_file_path, "r") as file:
    #     cfg = yaml.safe_load(file)

    # return cfg


def download_file(fname, artifact_uri, destination_path):
    """
    Downloads a file from an MLflow artifact URI to a specified local destination.
    Supports downloading from both S3 and local file system artifact URIs.
    Args:
        fname (str): The name of the file to download.
        artifact_uri (str): The MLflow artifact URI indicating the storage location.
        destination_path (str): The local directory path where the file should be saved.
    Returns:
        str or None: The full local path to the downloaded file if successful, otherwise None.
    Raises:
        None: Any exceptions are handled internally and None is returned on failure.
    """
    
    file_uri = mlflow.get_artifact_uri(fname)
    dest_file_path = os.path.join(destination_path, fname)

    if "s3" in artifact_uri:
        s3 = boto3.client("s3")
        out = urlparse(file_uri, allow_fragments=False)
        bucket_name = out.netloc
        rest_of_path = out.path
        try:
            s3.download_file(bucket_name, rest_of_path[1:], dest_file_path)
        except botocore.exceptions.ClientError as e:
            return None
    else:
        if "file" in artifact_uri:
            file_uri = file_uri[7:]
        if os.path.exists(file_uri):
            shutil.copyfile(file_uri, dest_file_path)
        else:
            return None

    return dest_file_path


#: Tag namespace for tsadar's canonical, queryable run metadata. Everything under
#: this prefix is a *tag* rather than a param so that downstream readers (the
#: Thomson analysis browser in ergodicio/tsadar-app) can filter with MLflow
#: ``search_runs`` filter strings without paging through the hundreds of
#: flattened config params that :func:`log_mlflow` writes.
TAG_NAMESPACE = "tsadar"

#: MLflow caps tag values at 5000 characters. Stay well under it: these are meant
#: to be read in a table cell, not parsed.
MAX_TAG_LEN = 500

#: The ``status`` tag's terminal values. The intermediate values ("preprocessing",
#: "minimizing", "postprocessing", "plotting", "done plotting") are written by
#: :mod:`tsadar.inverse.fitter` and :mod:`tsadar.inverse.postprocess` as the fit
#: progresses; these three bracket them. A run whose ``status`` is not terminal
#: and whose MLflow lifecycle status is not RUNNING died without unwinding.
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED)


def _truncate(value: str) -> str:
    """Clips a tag value to :data:`MAX_TAG_LEN`, marking that it was clipped."""

    text = str(value)
    if len(text) <= MAX_TAG_LEN:
        return text

    return text[: MAX_TAG_LEN - 3] + "..."


def get_version() -> str:
    """
    Returns the installed tsadar version, or "unknown" when the package metadata
    is unavailable (e.g. running from a source tree that was never installed).
    """

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("tsadar")
    except PackageNotFoundError:
        return "unknown"


def _data_kind(config) -> str:
    """
    Which spectra a run loads: "epw", "iaw", "both", or "none".

    Derived from the ``load_ele_spec`` / ``load_ion_spec`` switches rather than
    from the filenames, since those are what the fit actually branches on.
    """

    data = (config.get("data") or {}) if isinstance(config, dict) else {}
    ele = bool(data.get("load_ele_spec", False))
    ion = bool(data.get("load_ion_spec", False))

    if ele and ion:
        return "both"
    elif ele:
        return "epw"
    elif ion:
        return "iaw"
    else:
        return "none"


def _shotnum(config) -> str:
    """
    The shot number as a tag value. ``data.shotnum`` may be a list (a multi-shot
    fit, see ``fitter.load_data_for_fitting``), in which case the shots are
    comma-joined so a filter on a single shot can still match with ``LIKE``.
    """

    shotnum = (config.get("data") or {}).get("shotnum") if isinstance(config, dict) else None
    if shotnum is None:
        return ""
    if isinstance(shotnum, (list, tuple)):
        return ",".join(str(s) for s in shotnum)

    return str(shotnum)


def canonical_tags(config, mode: str = "fit") -> dict:
    """
    Builds the canonical tag set describing a run: the handful of fields worth
    filtering a run table on. See ergodicio/tsadar#115.

    Every lookup is defensive because this runs on configs from three different
    entry points (NERSC decks, app-submitted decks, forward-only decks) which do
    not all carry the same keys. A field that cannot be determined is omitted
    rather than tagged with a placeholder, so an absent tag means "unknown"
    instead of "known to be empty".

    Args:
        config: configuration dictionary
        mode: the run mode -- "fit", "forward", "series" or "interactive"

    Returns:
        tags: dict of tag name to string value, ready for ``mlflow.set_tags``
    """

    tags = {
        f"{TAG_NAMESPACE}.version": get_version(),
        f"{TAG_NAMESPACE}.mode": str(mode).casefold(),
        f"{TAG_NAMESPACE}.data": _data_kind(config),
    }

    shotnum = _shotnum(config)
    if shotnum:
        tags[f"{TAG_NAMESPACE}.shotnum"] = shotnum

    username = (config.get("other") or {}).get("username") if isinstance(config, dict) else None
    if username:
        tags[f"{TAG_NAMESPACE}.user"] = str(username)

    return {k: _truncate(v) for k, v in tags.items()}


def format_error(exc: BaseException) -> str:
    """
    Renders an exception as a short, single-line tag value: the exception type
    plus its message. The full traceback stays in the job log -- this exists so a
    run table can show *why* a run is failed without opening it.
    """

    message = " ".join(str(exc).split())
    rendered = f"{type(exc).__name__}: {message}" if message else type(exc).__name__

    return _truncate(rendered)
