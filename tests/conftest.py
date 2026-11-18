"""Session-wide test setup."""

import os
import tempfile

import matplotlib
import mlflow

matplotlib.use("Agg")

if not os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("TSADAR_TEST_TRACKING"):
    os.environ["TSADAR_TEST_TRACKING"] = "1"
    mlflow.set_tracking_uri(f"sqlite:///{tempfile.mkdtemp(prefix='tsadar-tests-')}/mlflow.db")
