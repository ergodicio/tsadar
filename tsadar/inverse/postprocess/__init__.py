"""postprocess: alternate uncertainty/artifact-cataloguing postprocessors run after a fit completes.

`.laplace` is the original postprocessor (Hessian/Laplace-approximation uncertainty, refitting, plots).
`.mcmc_postprocess` is the alternate Metropolis-Hastings MCMC uncertainty postprocessor. The public names
below are re-exported from `.laplace` so existing call sites (`from . import postprocess;
postprocess.postprocess(...)`) keep working unchanged now that this is a package rather than a module.
"""
from .laplace import (
    postprocess,
    get_sigmas,
    recalculate_with_chosen_weights,
    refit_bad_fits,
    process_data,
    process_angular_data,
)

__all__ = [
    "postprocess",
    "get_sigmas",
    "recalculate_with_chosen_weights",
    "refit_bad_fits",
    "process_data",
    "process_angular_data",
]
