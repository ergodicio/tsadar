import os

import equinox as eqx
import jax
from jax import numpy as jnp
import pytest

from tsadar.inverse import loops
from tsadar.inverse.loops import (
    NonFiniteOptimizationError,
    _1d_optax_loop_,
    advance_refinement_shape,
    angular_multiple_optax,
    apply_ang_res_unit,
    multirun_angular_optax,
    validate_active_leaves,
)


class _FakeDistribution(eqx.Module):
    pass


class _FakeElectron(eqx.Module):
    normed_Te: jax.Array
    distribution_functions: _FakeDistribution


class _FakeParams(eqx.Module):
    electron: _FakeElectron


class _QuadraticLoss:
    """A deterministic angular stand-in whose first SGD update overshoots."""

    _loss_ = None

    def vg_loss(self, diff_params, static_params, batch):
        del static_params, batch
        value = (diff_params.electron.normed_Te - 1.0) ** 2
        grad = jax.grad(lambda x: (x - 1.0) ** 2)(diff_params.electron.normed_Te)
        grad_tree = eqx.tree_at(lambda tree: tree.electron.normed_Te, diff_params, grad)
        return (value, None), grad_tree


def _fake_angular_config(**overrides):
    optimizer = {
        "method": "sgd",
        "param_method": "sgd",
        "learning_rate_init": 3.0,
        "learning_rate_final": 3.0,
        "param_learning_rate": 3.0,
        "num_epochs": 3,
        "patience": 1,
        "min_delta": 0.0,
        "save_state": False,
        "save_state_freq": 1,
        "validate_active_leaves": False,
        "seed": 7,
    }
    optimizer.update(overrides)
    return {
        "optimizer": optimizer,
        "parameters": {"electron": {"fe": {"active": False}}},
    }


@pytest.fixture
def isolated_optimizer(monkeypatch):
    monkeypatch.setattr(loops.mlflow, "set_tags", lambda *args, **kwargs: None)
    monkeypatch.setattr(loops.mlflow, "set_tag", lambda *args, **kwargs: None)
    monkeypatch.setattr(loops.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(loops, "get_filter_spec", lambda cfg, params: jax.tree.map(eqx.is_array, params))


def test_angular_overshoot_restores_evaluated_checkpoint(tmp_path, isolated_optimizer):
    initial = _FakeParams(_FakeElectron(jnp.asarray(0.0), _FakeDistribution()))
    loss_fn = _QuadraticLoss()

    best_weights, best_loss, total_steps, _, exit_cond = angular_multiple_optax(
        _fake_angular_config(), None, loss_fn, {}, previous_weights=initial
    )

    # SGD(lr=3) moves 0 -> 6, worsening loss from 1 to 25. The initial state is a
    # real checkpoint, so the overshoot must not be paired with its pre-update loss.
    assert best_loss == pytest.approx(1.0)
    assert float(best_weights.electron.normed_Te) == pytest.approx(0.0)
    assert total_steps == 1
    assert exit_cond.startswith("No improvement")

    checkpoint_path = os.path.join(tmp_path, "fitted_weights.eqx")
    eqx.tree_serialise_leaves(checkpoint_path, best_weights)
    restored = eqx.tree_deserialise_leaves(checkpoint_path, initial)
    restored_loss = float((restored.electron.normed_Te - 1.0) ** 2)
    assert restored_loss == pytest.approx(best_loss)


def test_1d_optax_overshoot_keeps_loss_and_weights_on_same_iterate(isolated_optimizer):
    class Progress:
        @staticmethod
        def set_description(description):
            del description

    initial = _FakeParams(_FakeElectron(jnp.asarray(0.0), _FakeDistribution()))
    best_loss, best_weights = _1d_optax_loop_(
        _fake_angular_config(), _QuadraticLoss(), initial, {}, Progress()
    )

    assert best_loss == pytest.approx(1.0)
    assert float(best_weights.electron.normed_Te) == pytest.approx(0.0)


def test_angular_nonfinite_initial_loss_fails_structurally(isolated_optimizer):
    class NonFiniteLoss(_QuadraticLoss):
        def vg_loss(self, diff_params, static_params, batch):
            (_, aux), grad = super().vg_loss(diff_params, static_params, batch)
            return (jnp.asarray(jnp.nan), aux), grad

    initial = _FakeParams(_FakeElectron(jnp.asarray(0.0), _FakeDistribution()))
    with pytest.raises(NonFiniteOptimizationError, match="Nonfinite loss") as exc_info:
        angular_multiple_optax(
            _fake_angular_config(), None, NonFiniteLoss(), {}, previous_weights=initial
        )
    assert exc_info.value.quantity == "loss"
    assert exc_info.value.step == 0


def test_angular_continuation_returns_global_best_stage(monkeypatch, isolated_optimizer):
    loss_functions = []

    class FakeLossFunction:
        def __init__(self, config, sa, batch):
            del config, sa, batch
            loss_functions.append(self)

    stage_results = [(0.0, 1.0), (2.0, 4.0)]

    def fake_stage(config, sa, loss_fn, actual_data, previous_weights, previous_epoch, stage):
        del config, sa, actual_data, previous_weights
        value, loss = stage_results[stage]
        weights = _FakeParams(_FakeElectron(jnp.asarray(value), _FakeDistribution()))
        return weights, loss, previous_epoch + 1, loss_fn, "test"

    monkeypatch.setattr(loops, "LossFunction", FakeLossFunction)
    monkeypatch.setattr(loops, "angular_multiple_optax", fake_stage)
    monkeypatch.setattr(loops, "refine_angular_weights", lambda config, weights: weights)

    config = {
        "optimizer": {"batch_size": 6, "num_mins": 2, "method": "sgd", "param_method": "sgd"},
        "data": {"lineouts": {"start": 0, "end": 1}, "shotnum": 1},
        "other": {"ang_res_unit": 1},
        "parameters": {"electron": {"fe": {"active": False}}},
    }
    data = {
        "e_data": jnp.ones((1, 1)),
        "e_amps": jnp.ones((1, 1)),
        "i_data": jnp.ones((1, 1)),
        "i_amps": jnp.ones((1, 1)),
        "noiseE": jnp.zeros((1, 1)),
        "noiseI": jnp.zeros((1, 1)),
    }

    weights, loss, loss_fn = multirun_angular_optax(config, data, None)

    assert loss == pytest.approx(1.0)
    assert float(weights.electron.normed_Te) == pytest.approx(0.0)
    assert loss_fn is loss_functions[0]
    assert config["optimizer"]["checkpoint_refinements"] == 0


def test_active_leaf_validation_rejects_zero_forward_sensitivity(isolated_optimizer):
    initial = _FakeParams(_FakeElectron(jnp.asarray(0.0), _FakeDistribution()))
    spec = jax.tree.map(eqx.is_array, initial)
    diff_params, static_params = eqx.partition(initial, spec)
    config = _fake_angular_config(validate_active_leaves=True)
    config["parameters"] = {
        "electron": {
            "Te": {"active": True},
            "fe": {"active": False},
        }
    }

    class InsensitiveLoss:
        multiplex_ang = False
        cfg = config

        @staticmethod
        def ts_diag(weights, batch):
            del weights, batch
            zeros = jnp.zeros(2)
            return zeros, zeros, zeros, zeros

    with pytest.raises(ValueError, match="electron.Te: no forward sensitivity"):
        validate_active_leaves(config, initial, diff_params, static_params, InsensitiveLoss(), {})


def test_apply_ang_res_unit():
    config = {"data": {"lineouts": {"start": 100, "end": 205}}, "other": {"ang_res_unit": 10}}

    apply_ang_res_unit(config)

    assert config["data"]["lineouts"]["start"] == 10
    assert config["data"]["lineouts"]["end"] == 20


def test_advance_refinement_shape():
    # matches multirun_angular_optax's nvx *= refine_factor, window.len = window.len * refine_factor + 1
    config = {
        "parameters": {"electron": {"fe": {"dim": 1, "nvx": 64, "params": {"window": {"len": 5}}}}},
        "optimizer": {"refine_factor": 2},
    }

    advance_refinement_shape(config)

    fe = config["parameters"]["electron"]["fe"]
    assert fe["nvx"] == 128
    assert fe["params"]["window"]["len"] == 11


def test_advance_refinement_shape_multiple_steps_matches_manual_iteration():
    # regression guard for postprocess_runner, which calls this num_mins - 1 times to replay the shape
    # multirun_angular_optax's refinement loop would have reached, without redoing the fit
    config = {
        "parameters": {"electron": {"fe": {"dim": 1, "nvx": 32, "params": {"window": {"len": 3}}}}},
        "optimizer": {"refine_factor": 2},
    }

    for _ in range(3):
        advance_refinement_shape(config)

    fe = config["parameters"]["electron"]["fe"]
    assert fe["nvx"] == 32 * 2**3
    assert fe["params"]["window"]["len"] == 31  # 3 -> 7 -> 15 -> 31


def test_advance_refinement_shape_rejects_non_1d():
    config = {"parameters": {"electron": {"fe": {"dim": 2}}}, "optimizer": {"refine_factor": 2}}

    with pytest.raises(ValueError):
        advance_refinement_shape(config)
