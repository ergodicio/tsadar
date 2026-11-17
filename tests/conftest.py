"""Session-wide test setup."""

import os
import tempfile

import matplotlib
import mlflow

# Tests only write figures to files. On Windows the default interactive backend is TkAgg, and
# MCMC postprocessing draws from worker threads: Tk then fails with "main thread is not in main
# loop" and the run slows from ~25 s to many minutes. CI on Linux already defaults to Agg.
matplotlib.use("Agg")

# Runs made by the tests go to a throwaway store, not ./mlruns, unless a tracking server is configured.
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(f"sqlite:///{tempfile.mkdtemp(prefix='tsadar-tests-')}/mlflow.db")
