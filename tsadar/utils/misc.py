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
