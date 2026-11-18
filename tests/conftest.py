"""Session-wide test setup."""

import os
import tempfile

import matplotlib
import mlflow

matplotlib.use("Agg")

if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(f"sqlite:///{tempfile.mkdtemp(prefix='tsadar-tests-')}/mlflow.db")
