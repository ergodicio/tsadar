"""One small real fit of shot 101675, shared by the MCMC and Laplace unit tests."""

import mlflow
import numpy as np
import pytest
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tsadar.data import prepare
from tsadar.inverse.loops import build_batch, one_d_loop

from .shared import base_config


@pytest.fixture(scope="session")
def fitted_fixture():
    """A fitted point (10 epochs, fe inactive), not necessarily a good fit: the tests only need one."""
    cfg = base_config()
    mlflow.set_experiment("test-inverse-units")
    with mlflow.start_run():
        all_data, sa, all_axes = prepare.prepare_data(cfg, cfg["data"]["shotnum"])
        sample_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))
        num_batches = len(sample_indices) // cfg["optimizer"]["batch_size"] or 1
        fitted_weights, _, loss_fn = one_d_loop(cfg, all_data, sa, sample_indices, num_batches)
    batch = build_batch(
        all_data, sample_indices[: cfg["optimizer"]["batch_size"]], cfg["data"]["background"]["bg_subtract"]
    )
    return {
        "config": cfg,
        "all_data": all_data,
        "all_axes": all_axes,
        "sa": sa,
        "loss_fn": loss_fn,
        "fitted_weights": fitted_weights,
        "batch": batch,
        "sample_indices": sample_indices,
    }
