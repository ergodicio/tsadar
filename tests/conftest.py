"""Session-wide test setup."""

import matplotlib

# Tests only write figures to files. On Windows the default interactive backend is TkAgg, and
# MCMC postprocessing draws from worker threads: Tk then fails with "main thread is not in main
# loop" and the run slows from ~25 s to many minutes. CI on Linux already defaults to Agg.
matplotlib.use("Agg")
