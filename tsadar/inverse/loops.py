"""The per-batch optimization loops that drive fitting: batch construction, the 1D (scipy/optax) and
angular (optax) training loops, and unbatching the resulting per-batch fitted parameters."""
import copy
from dataclasses import dataclass
from functools import partial
from collections import defaultdict
from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
import equinox as eqx
import scipy.optimize as spopt
from tsadar.inverse.loss_function import LossFunction
from tsadar.core.modules.distribution_functions.base import DLM1V

import mlflow
import numpy as np
import pickle
import jax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree
from tqdm import trange
import optax
import optimistix as optx


from typing import Any, Dict, List, Tuple


class NonFiniteOptimizationError(FloatingPointError):
    """A structured failure raised when an optimizer produces a nonfinite quantity."""

    def __init__(self, quantity: str, *, step: int, stage: int):
        self.quantity = quantity
        self.step = step
        self.stage = stage
        super().__init__(f"Nonfinite {quantity} at optimizer stage {stage}, step {step}.")


@dataclass(frozen=True)
class OptimizationCheckpoint:
    """An objective value and the exact parameter tree on which it was evaluated."""

    loss: float
    weights: Any
    step: int
    stage: int


def _numeric_leaves(tree):
    """Return the numeric array leaves of a possibly-None Equinox pytree."""
    return [
        jnp.asarray(leaf)
        for leaf in jtu.tree_leaves(tree, is_leaf=lambda leaf: leaf is None)
        if eqx.is_array(leaf)
        or (np.isscalar(leaf) and not isinstance(leaf, (str, bytes, bool, type(None))))
    ]


def _require_finite(tree, quantity: str, *, step: int, stage: int) -> None:
    """Fail at the first optimizer boundary where a nonfinite value appears."""
    if any(not bool(jnp.all(jnp.isfinite(leaf))) for leaf in _numeric_leaves(tree)):
        raise NonFiniteOptimizationError(quantity, step=step, stage=stage)


def _gradient_norm(grad) -> float:
    leaves = _numeric_leaves(grad)
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)))


def _stopping_options(config: Dict) -> Tuple[int, float]:
    """Read standard patience/minimum-delta controls, retaining the old numerical defaults."""
    patience = int(config["optimizer"].get("patience", 500))
    min_delta = float(config["optimizer"].get("min_delta", 1e-8))
    return patience, min_delta


def _runtime_parameterization(ts_params) -> str:
    distribution = ts_params.electron.distribution_functions
    if isinstance(distribution, list) and distribution:
        distribution = distribution[0]
    return type(distribution).__name__


def _log_optimizer_runtime(
    config: Dict, ts_params, *, stage: int, partitioned: bool = False
) -> None:
    """Record the instantiated algorithms and EDF class rather than inferred labels."""
    method = config["optimizer"]["method"]
    param_method = config["optimizer"].get("param_method")
    actual_optimizer = (
        f"dist={method}, macro={param_method}" if partitioned else method
    )
    mlflow.set_tags(
        {
            "optimizer.actual": actual_optimizer,
            "optimizer.edf_parameterization": _runtime_parameterization(ts_params),
            "optimizer.stage_kind": "continuation/refinement" if partitioned else "batch",
            "optimizer.seed": str(config["optimizer"].get("seed", 0)),
            "optimizer.stage": str(stage),
        }
    )


def _log_optimizer_step(
    *, current_loss: float, checkpoint: OptimizationCheckpoint, learning_rate: float,
    grad, step: int, stage: int, seed: int,
) -> None:
    mlflow.log_metrics(
        {
            "epoch loss": current_loss,
            "optimizer.current_loss": current_loss,
            "optimizer.best_loss": checkpoint.loss,
            "optimizer.best_step": float(checkpoint.step),
            "optimizer.learning_rate": learning_rate,
            "optimizer.gradient_norm": _gradient_norm(grad),
            "optimizer.stage": float(stage),
            "optimizer.seed": float(seed),
        },
        step=step,
    )


def _parameter_only_config(parameters: Dict, active_species: str, active_key: str) -> Dict:
    selected = copy.deepcopy(parameters)
    for species, species_params in selected.items():
        for key, param in species_params.items():
            if isinstance(param, dict) and "active" in param:
                param["active"] = species == active_species and key == active_key
    return selected


def _sensitivity_direction(active):
    """Build a deterministic nonuniform tangent that avoids normalization null directions."""
    if active.size == 1:
        return jnp.ones_like(active)
    return jnp.linspace(1.0, 2.0, active.size, dtype=active.dtype).reshape(active.shape)


def _forward_outputs(loss_fn, weights, batch):
    """Return detector spectra used to validate forward sensitivity in the fit geometry."""
    data_config = loss_fn.cfg.get("data", {})
    fit_electron = data_config.get("fit_EPWb", True) or data_config.get("fit_EPWr", True)
    fit_ion = data_config.get("fit_IAW", True)
    if loss_fn.multiplex_ang:
        first = loss_fn.ts_diag(weights, batch["b1"])
        rotated = eqx.tree_at(lambda tree: tree.electron.dist_rot, weights, loss_fn.cfg["data"]["shot_rot"])
        second = loss_fn.ts_diag(rotated, batch["b2"])
        outputs = []
        if fit_electron:
            outputs.extend((first[0], second[0]))
        if fit_ion:
            outputs.append(first[1])
        return tuple(outputs)
    result = loss_fn.ts_diag(weights, batch)
    outputs = []
    if fit_electron:
        outputs.append(result[0])
    if fit_ion:
        outputs.append(result[1])
    return tuple(outputs)


def validate_active_leaves(config: Dict, ts_params, diff_params, static_params, loss_fn, batch) -> None:
    """Validate that every active deck parameter is finite, differentiable, and observable.

    Each active parameter gets its own filter specification and deterministic JVP.
    This distinguishes an objective gradient that happens to be zero at a good fit from a
    parameter whose detector-space forward model has no sensitivity in the fitted geometry.
    All problems are collected into one actionable error rather than failing on the first
    misspelled/static leaf.
    """
    issues = []
    directions = []
    sensitivity_tol = float(config["optimizer"].get("sensitivity_tol", 0.0))
    for species, species_params in config["parameters"].items():
        for key, param in species_params.items():
            if not isinstance(param, dict) or not param.get("active", False):
                continue

            name = f"{species}.{key}"
            try:
                parameter_spec = get_filter_spec(
                    _parameter_only_config(config["parameters"], species, key), ts_params
                )
                parameter_diff, _ = eqx.partition(ts_params, parameter_spec)
            except Exception as exc:
                issues.append(f"{name}: absent or static ({type(exc).__name__}: {exc})")
                continue

            leaves = _numeric_leaves(parameter_diff)
            if not leaves:
                issues.append(f"{name}: absent from the differentiable pytree")
                continue
            if any(not bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves):
                issues.append(f"{name}: nonfinite initial value")
                continue

            direction = jtu.tree_map(
                lambda active, selected: (
                    None
                    if active is None
                    else _sensitivity_direction(active)
                    if selected is not None
                    else jnp.zeros_like(active)
                ),
                diff_params,
                parameter_diff,
                is_leaf=lambda leaf: leaf is None,
            )
            directions.append((name, direction))

    if directions:
        try:
            _, linearized_forward = jax.linearize(
                lambda params: _forward_outputs(
                    loss_fn, eqx.combine(params, static_params), batch
                ),
                diff_params,
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            issues.extend(
                f"{name}: forward-sensitivity validation failed ({detail})"
                for name, _ in directions
            )
            directions = []

    for name, direction in directions:
        try:
            tangent = linearized_forward(direction)
            sensitivity = _gradient_norm(tangent)
        except Exception as exc:
            issues.append(
                f"{name}: forward-sensitivity validation failed "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        if not np.isfinite(sensitivity):
            issues.append(f"{name}: nonfinite forward sensitivity")
        elif sensitivity <= sensitivity_tol:
            issues.append(
                f"{name}: no forward sensitivity in validation geometry "
                f"(norm={sensitivity:.3e}, tolerance={sensitivity_tol:.3e})"
            )

    if issues:
        mlflow.set_tag("optimizer.active_leaf_validation", "failed")
        raise ValueError("Active parameter validation failed:\n- " + "\n- ".join(issues))
    mlflow.set_tag("optimizer.active_leaf_validation", "passed")


def build_batch(all_data: Dict, inds, background_subtract: bool) -> Dict:
    """
    Slices a batch of lineouts out of `all_data`, optionally subtracting the background noise from the data
    (leaving the model to add it back in) instead of passing the noise through as a separate fit term.

    Args:
        all_data (Dict): Dictionary containing all input data arrays required for fitting.
        inds: Indices (or index array) selecting which lineouts to include in the batch.
        background_subtract (bool): Whether to subtract noise from the data here rather than fit it separately.
    Returns:
        Dict: Batch dictionary with e_data/i_data, amplitudes, and noise terms for the selected indices.
    """
    return {
        "e_data": all_data["e_data"][inds] - all_data["noiseE"][inds] if background_subtract else all_data["e_data"][inds],
        "e_amps": all_data["e_amps"][inds],
        "i_data": all_data["i_data"][inds] - all_data["noiseI"][inds] if background_subtract else all_data["i_data"][inds],
        "i_amps": all_data["i_amps"][inds],
        "noise_e": all_data["noiseE"][inds] if not background_subtract else 0.0,
        "noise_i": all_data["noiseI"][inds] if not background_subtract else 0.0,
    }


def build_angular_batch(config: Dict, all_data: Dict) -> Dict:
    """
    Builds the angular-data batch (or dict of two batches, for multiplexed/rotated shots) sliced to the
    configured lineout start:end range. Assumes any unit conversion of the lineout start/end (e.g. by
    ang_res_unit) has already been applied to `config` by the caller.

    Args:
        config (Dict): Configuration dictionary containing the lineout start/end and shot settings.
        all_data (Dict): Dictionary containing all input data arrays required for fitting.
    Returns:
        Dict: `batch1` directly for a single shot, or {"b1": batch1, "b2": batch2} for rotated multi-shot data.
    """
    start, end = config["data"]["lineouts"]["start"], config["data"]["lineouts"]["end"]
    batch1 = {
        "e_data": all_data["e_data"][start:end, :],
        "e_amps": all_data["e_amps"][start:end, :],
        "i_data": all_data["i_data"],
        "i_amps": all_data["i_amps"],
        "noise_e": all_data["noiseE"][start:end, :],
        "noise_i": all_data["noiseI"][start:end, :],
    }
    if isinstance(config["data"]["shotnum"], list):
        batch2 = {
            "e_data": all_data["e_data_rot"][start:end, :],
            "e_amps": all_data["e_amps_rot"][start:end, :],
            "noise_e": all_data["noiseE_rot"][start:end, :],
            "i_data": all_data["i_data"],
            "i_amps": all_data["i_amps"],
            "noise_i": all_data["noiseI"][start:end, :],
        }
        return {"b1": batch1, "b2": batch2}
    return batch1


def _1d_scipy_loop_(
    config: Dict, loss_fn: LossFunction, previous_weights: np.ndarray, batch: Dict
) -> Tuple[float, Dict]:
    """
    Runs a 1D optimization loop using SciPy's minimize function for inverse Thomson scattering.
    Args:
        config (Dict): Configuration dictionary containing optimizer and parameter settings.
        loss_fn (LossFunction): Loss function object with methods for evaluating the loss and its gradient.
        previous_weights (np.ndarray): Previous weights to initialize the optimizer, or None to use default initialization.
        batch (Dict): Batch of data to be used in the loss function.
    Returns:
        Tuple[float, Dict]: A tuple containing the best loss value and the corresponding optimized weights.
    """
    
    _activate = True
    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"], activate=_activate)
    else:
        ts_params = previous_weights

    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    init_weights, loss_fn.unravel_weights = ravel_pytree(diff_params)

    res = spopt.minimize(
        loss_fn.vg_loss if config["optimizer"]["grad_method"] == "AD" else loss_fn.loss,
        init_weights,
        args=(static_params, batch),
        method=config["optimizer"]["method"],
        jac=True if config["optimizer"]["grad_method"] == "AD" else False,
        bounds=None if _activate else ((0, 1) for _ in range(len(init_weights))),
        options={"disp": True, "maxiter": config["optimizer"]["num_epochs"]},
    )

    best_loss = res["fun"]
    best_weights = eqx.combine(loss_fn.unravel_weights(res["x"]), static_params)

    return best_loss, best_weights


def _1d_lm_loop_(
    config: Dict, loss_fn: LossFunction, previous_weights, batch: Dict
) -> Tuple[float, Dict]:
    """
    Runs a 1D optimization loop using Levenberg-Marquardt least squares (optimistix).

    LM minimises ``sum(loss_fn.residuals(...)**2)``, which equals the l2 training loss, by
    exploiting the residual/Jacobian structure. On the 1D inverse problem it typically needs
    far fewer iterations than L-BFGS for the same fit. Requires ``loss_method == "l2"``.

    Args:
        config (Dict): Configuration dictionary containing optimizer and parameter settings.
        loss_fn (LossFunction): Loss function object exposing a ``residuals`` method.
        previous_weights: Weights to initialize from, or None to build fresh parameters.
        batch (Dict): Batch of data to be used in the loss function.
    Returns:
        Tuple[float, Dict]: The best loss value (sum of squared residuals) and the optimized weights.
    """
    if config["optimizer"]["loss_method"] != "l2":
        raise ValueError("optimizer method 'lsq'/'lm' requires loss_method 'l2' (least squares).")

    if previous_weights is None:
        ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"], activate=True)
    else:
        ts_params = previous_weights

    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))

    # loss_fn.residuals returns (residual, aux); optimistix wants just the residual vector.
    # LM converges quadratically near the optimum, so a loose tolerance reaches the same fit in
    # fewer iterations. Configurable via optimizer.lm_tol.
    tol = config["optimizer"].get("lm_tol", 1e-3)
    solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
    sol = optx.least_squares(
        lambda dp, args: loss_fn.residuals(dp, static_params, batch)[0],
        solver,
        diff_params,
        max_steps=config["optimizer"]["num_epochs"],
        throw=False,
    )

    best_weights = eqx.combine(sol.value, static_params)
    residual, _ = loss_fn.residuals(sol.value, static_params, batch)
    best_loss = float(jnp.sum(residual**2))

    return best_loss, best_weights


def _1d_optax_loop_(
    config: Dict, loss_fn: LossFunction, previous_weights: np.ndarray, batch: Dict, tbatch, stage: int = 0,
) -> Tuple[float, Dict]:
    """
    Runs a 1D optimization loop using the Adam optimizer for a specified number of epochs.
    Args:
        config (Dict): Configuration dictionary containing optimizer and parameter settings.
        loss_fn (LossFunction): Loss function object with a `vg_loss` method for computing loss and gradients.
        previous_weights (np.ndarray): Previous weights to initialize the model parameters. If None, initializes new parameters.
        batch (Dict): Batch of data to be used for optimization.
        tbatch: Progress bar or tracker object for displaying epoch progress.
    Returns:
        Tuple[float, Dict]: A tuple containing the best loss achieved and the corresponding model weights.
    """

    minimizer = getattr(optax, config["optimizer"]["method"])
    # schedule = optax.schedules.cosine_decay_schedule(config["optimizer"]["learning_rate"], 100, alpha = 0.00001)
    # solver = minimizer(schedule)
    opt = minimizer(None if config["optimizer"]["method"]=='lbfgs' else config["optimizer"]["learning_rate_init"])

    #ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False, activate=True)
    #diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    #opt_state = solver.init(diff_params)

    
    #opt = optax.adam(config["optimizer"]["learning_rate"])
    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"], activate=True)
    else:
        ts_params = previous_weights
    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    opt_state = opt.init(diff_params)

    _log_optimizer_runtime(config, ts_params, stage=stage)
    (raw_loss, aux), grad = loss_fn.vg_loss(diff_params, static_params, batch)
    _require_finite(raw_loss, "loss", step=0, stage=stage)
    _require_finite(grad, "gradient", step=0, stage=stage)
    if config["optimizer"].get("validate_active_leaves", True):
        validate_active_leaves(config, ts_params, diff_params, static_params, loss_fn, batch)

    current_loss = float(raw_loss)
    checkpoint = OptimizationCheckpoint(
        loss=current_loss,
        weights=eqx.combine(diff_params, static_params),
        step=0,
        stage=stage,
    )
    patience, min_delta = _stopping_options(config)
    wait = 0
    patience_reference_loss = checkpoint.loss
    learning_rate = float(config["optimizer"].get("learning_rate_init", 0.0))
    seed = int(config["optimizer"].get("seed", 0))
    _log_optimizer_step(
        current_loss=current_loss,
        checkpoint=checkpoint,
        learning_rate=learning_rate,
        grad=grad,
        step=0,
        stage=stage,
        seed=seed,
    )

    for i_epoch in range(config["optimizer"]["num_epochs"]):
        tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {current_loss:.2e}")
        updates, opt_state = opt.update(
            grad,
            opt_state,
            diff_params,
            value=raw_loss,
            grad=grad,
            value_fn=loss_fn._loss_,
        )
        step = i_epoch + 1
        _require_finite(updates, "update", step=step, stage=stage)
        candidate_params = eqx.apply_updates(diff_params, updates)
        _require_finite(candidate_params, "parameters", step=step, stage=stage)

        (candidate_raw_loss, aux), candidate_grad = loss_fn.vg_loss(candidate_params, static_params, batch)
        _require_finite(candidate_raw_loss, "loss", step=step, stage=stage)
        _require_finite(candidate_grad, "gradient", step=step, stage=stage)
        candidate_loss = float(candidate_raw_loss)

        improvement = checkpoint.loss - candidate_loss
        if improvement > 0.0:
            checkpoint = OptimizationCheckpoint(
                loss=candidate_loss,
                weights=eqx.combine(candidate_params, static_params),
                step=step,
                stage=stage,
            )
        if patience_reference_loss - candidate_loss > min_delta:
            patience_reference_loss = candidate_loss
            wait = 0
        else:
            wait += 1

        _log_optimizer_step(
            current_loss=candidate_loss,
            checkpoint=checkpoint,
            learning_rate=learning_rate,
            grad=candidate_grad,
            step=step,
            stage=stage,
            seed=seed,
        )
        diff_params, raw_loss, grad, current_loss = (
            candidate_params,
            candidate_raw_loss,
            candidate_grad,
            candidate_loss,
        )
        if patience > 0 and wait >= patience:
            break

    return checkpoint.loss, checkpoint.weights


def one_d_loop(
    config: Dict, all_data: Dict, sa: Tuple, batch_indices: np.ndarray, num_batches: int, previous_weights=None,
) -> Tuple[List, float, LossFunction]:
    """
    Higher level wrapper form minimization of 1D fits, preparing data and dispatching to the appropriate optimizer.
    This function prepares batches of data and fits model parameters using either the ADAM optimizer or a SciPy optimizer,
    depending on the configuration. It supports sequential optimization by passing weights between batches if enabled.
        
    Args:    
        config (Dict): Configuration dictionary containing optimizer settings and batch size.
        all_data (Dict): Dictionary containing all input data arrays required for fitting.
        sa (Tuple): Scattering angles and weights used to calculate k-smea r corrections.
        batch_indices (np.ndarray): Array of indices specifying how to split data into batches.
        num_batches (int): Number of batches to process.
        previous_weights (np.ndarray, optional): Weights to initialize the optimizer. If None, initializes new parameters.
    Returns:
        all_weights (List): List of weights from each batch.
        overall_loss (float): Overall accumulated loss across all batches.
        loss_fn (LossFunction): The final LossFunction instance used for fitting.
    Notes: 
        - The function uses a progress bar to display the fitting progress for each batch.
        - The function logs metrics to MLflow for tracking the fitting process.

    """
    sample = {k: v[: config["optimizer"]["batch_size"]] for k, v in all_data.items()}
    sample = {
        "noise_e": all_data["noiseE"][: config["optimizer"]["batch_size"]],
        "noise_i": all_data["noiseI"][: config["optimizer"]["batch_size"]],
    } | sample
    loss_fn = LossFunction(config, sa, sample)

    batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
    all_weights = []
    overall_loss = 0.0
    previous_batch = None
    background_subtract = config["data"]["background"]["bg_subtract"]
    with trange(num_batches, unit="batch") as tbatch:
        for i_batch in tbatch:
            previous_batch = previous_weights[i_batch] if previous_weights is not None else previous_batch
            inds = batch_indices[i_batch]
            batch = build_batch(all_data, inds, background_subtract)

            if config["optimizer"]["method"] == "l-bfgs-b":  # Stochastic Gradient Descent
                # not sure why this is needed but something needs to be reset, either the weights or the bounds
                loss_fn = LossFunction(config, sa, batch)
                best_loss, best_weights = _1d_scipy_loop_(config, loss_fn, previous_batch, batch)
            elif config["optimizer"]["method"] in ("lsq", "lm"):  # Levenberg-Marquardt least squares
                best_loss, best_weights = _1d_lm_loop_(config, loss_fn, previous_batch, batch)
            else:
                best_loss, best_weights = _1d_optax_loop_(
                    config, loss_fn, previous_batch, batch, tbatch, stage=i_batch
                )
                

            all_weights.append(best_weights)
            mlflow.log_metrics({"batch loss": float(best_loss)}, step=i_batch)
            overall_loss += best_loss

            # ugly
            if "sequential" in config["optimizer"]:
                if config["optimizer"]["sequential"]:
                    previous_batch = best_weights
                    # if config["optimizer"]["method"] == "adam":
                    #     previous_weights = best_weights
                    # else:
                    #     previous_weights, _ = ravel_pytree(best_weights)

    return all_weights, overall_loss, loss_fn


def unbatch_fitted_params(config: Dict, fitted_weights: List) -> Tuple[Dict, int]:
    """
    Flattens the per-batch fitted-weight objects returned by `one_d_loop` into a single dict of
    concatenated parameter arrays.

    Args:
        config (Dict): Configuration dictionary containing the parameter settings.
        fitted_weights (List): List of per-batch fitted weight objects, each exposing `get_fitted_params`.
    Returns:
        Tuple[Dict, int]: The unbatched parameters and the number of active fitted parameters.
    """
    all_params = {k: defaultdict(list) for k in config["parameters"].keys()}
    num_params = 0
    for _fw in fitted_weights:
        batch_fitted_params, num_params = _fw.get_fitted_params(config["parameters"])
        for k in batch_fitted_params.keys():
            for k2 in batch_fitted_params[k].keys():
                all_params[k][k2].append(batch_fitted_params[k][k2])

    for k in all_params.keys():
        for k2 in all_params[k].keys():
            all_params[k][k2] = np.concatenate(all_params[k][k2])

    return all_params, num_params


def apply_ang_res_unit(config: Dict) -> None:
    """
    Converts the configured lineout start/end (in raw pixel units) to ang_res_unit-binned pixel units,
    matching the resolution reduction prepare_data applies to angular data before batching. Mutates
    config in place, and is not idempotent -- calling it twice on the same config divides twice.

    Factored out of multirun_angular_optax so that anything reconstructing the batch a saved angular
    checkpoint was fit against (e.g. postprocess_runner, which starts from a freshly-loaded config that
    never went through multirun_angular_optax) applies the exact same conversion.

    Args:
        config (Dict): Configuration dictionary; config["data"]["lineouts"]["start"]/["end"] are read and
            overwritten, using config["other"]["ang_res_unit"].
    """
    config["data"]["lineouts"]["start"] = int(config["data"]["lineouts"]["start"] / config["other"]["ang_res_unit"])
    config["data"]["lineouts"]["end"] = int(config["data"]["lineouts"]["end"] / config["other"]["ang_res_unit"])


def advance_refinement_shape(config: Dict) -> None:
    """
    Mutates config["parameters"]["electron"]["fe"]["nvx"] and (for 'arbitrary' distributions) its
    smoothing window length in place, to the shape multirun_angular_optax's next minimization pass would
    use -- one step of the same nvx *= refine_factor, window.len = window.len * refine_factor + 1
    progression applied between minimizations there.

    Reconstructing a ThomsonParams skeleton to deserialize a saved checkpoint only
    needs the checkpoint's shape, not the intermediate numeric values. The standalone
    postprocessor therefore calls this once per refinement recorded in the checkpoint
    metadata (falling back to ``num_mins - 1`` for older runs).

    Args:
        config (Dict): Configuration dictionary; config["parameters"]["electron"]["fe"] is mutated using
            config["optimizer"]["refine_factor"]. Only supports dim == 1, matching multirun_angular_optax.

    Raises:
        ValueError: if the electron distribution function is not 1D, since refinement (there and here) is
            only defined for 1D EDFs.
    """
    if config["parameters"]["electron"]["fe"]["dim"] != 1:
        raise ValueError("Multiple minimizations are only enabled for 1D edfs")

    fe_config = config["parameters"]["electron"]["fe"]
    refine_factor = config["optimizer"]["refine_factor"]
    fe_config["nvx"] = fe_config["nvx"] * refine_factor
    fe_config["params"]["window"]["len"] = fe_config["params"]["window"]["len"] * refine_factor + 1


def refine_angular_weights(config: Dict, previous_weights):
    """Refine an angular continuation checkpoint onto the next stage's EDF grid."""
    advance_refinement_shape(config)
    new_vx = np.linspace(
        previous_weights.electron.distribution_functions.vx[0],
        previous_weights.electron.distribution_functions.vx[-1],
        config["parameters"]["electron"]["fe"]["nvx"],
    )
    if config["parameters"]["electron"]["fe"]["type"] == "arbitrary":
        old_distribution = previous_weights.electron.distribution_functions
        fenorm = np.sum(old_distribution.fval) * (old_distribution.vx[1] - old_distribution.vx[0])
        refined_fe = np.interp(new_vx, old_distribution.vx, old_distribution.fval)
        refined_fe = fenorm * refined_fe / np.sum(refined_fe) / (new_vx[1] - new_vx[0])
        previous_weights = eqx.tree_at(
            lambda tree: tree.electron.distribution_functions.fval,
            previous_weights,
            refined_fe,
        )
    elif config["parameters"]["electron"]["fe"]["type"] == "dlm":
        distconfigs = config["parameters"]["electron"]["fe"]
        current_m = previous_weights.electron.distribution_functions.get_unnormed_params()
        distconfigs["params"]["m"]["val"] = current_m["m"]
        previous_weights = eqx.tree_at(
            lambda tree: tree.electron.distribution_functions,
            previous_weights,
            DLM1V(distconfigs, True),
        )

    return eqx.tree_at(
        lambda tree: tree.electron.distribution_functions.vx,
        previous_weights,
        new_vx,
    )


def multirun_angular_optax(
    config: Dict, all_data: Dict, sa: Tuple,
) -> Tuple[List, float, LossFunction]:
    """
    Higher level wrapper for angular Thomson scattering data optimization using Optax.

        
    Args:    
        config (Dict): Configuration dictionary containing optimizer settings and batch size.
        all_data (Dict): Dictionary containing all input data arrays required for fitting.
        sa (Tuple): Scattering angles and weights used to calculate k-smea r corrections.
        batch_indices (np.ndarray): Array of indices specifying how to split data into batches.
        num_batches (int): Number of batches to process.
        previous_weights (np.ndarray, optional): Weights to initialize the optimizer. If None, initializes new parameters.
    Returns:
        all_weights (List): List of weights from each batch.
        overall_loss (float): Overall accumulated loss across all batches.
        loss_fn (LossFunction): The final LossFunction instance used for fitting.
    Notes:
        ``num_mins`` means sequential continuation/refinement stages, not independent
        random restarts. Each stage starts from the preceding stage's best checkpoint;
        when requested, the EDF grid is refined between stages. The true best evaluated
        checkpoint across all stages is returned.

    """
    config["optimizer"]["batch_size"] = 1
    apply_ang_res_unit(config)
    actual_data = build_angular_batch(config, all_data)
    batch1 = actual_data["b1"] if isinstance(config["data"]["shotnum"], list) else actual_data

    previous_weights = None
    total_epochs = 0
    global_checkpoint = None
    global_loss_fn = None
    global_config = None

    # Run sequential continuation/refinement stages. This is deliberately not a set of
    # independent random restarts; ``previous_weights`` seeds the next stage.
    for i_min in range(config["optimizer"]["num_mins"]):
        loss_fn = LossFunction(config, sa, batch1)
        previous_weights, overall_loss, total_epochs, loss_fn, exit_cond = angular_multiple_optax(
            config,
            sa,
            loss_fn,
            actual_data,
            previous_weights,
            total_epochs,
            stage=i_min,
        )
        mlflow.set_tag(f"exit cond {i_min}", exit_cond)
        mlflow.log_metrics({"min loss": float(overall_loss)}, step=i_min)
        if global_checkpoint is None or overall_loss < global_checkpoint.loss:
            global_checkpoint = OptimizationCheckpoint(
                loss=float(overall_loss),
                weights=previous_weights,
                step=total_epochs,
                stage=i_min,
            )
            global_loss_fn = loss_fn
            global_config = copy.deepcopy(config)
        if i_min < config["optimizer"]["num_mins"]-1:
            previous_weights = refine_angular_weights(config, previous_weights)

    # Keep immediate postprocessing consistent with the returned checkpoint even when
    # the global best came from an earlier refinement shape. Standalone restoration uses
    # checkpoint_refinements from the companion metadata artifact written by fitter.py.
    config.clear()
    config.update(global_config)
    config["optimizer"]["checkpoint_refinements"] = global_checkpoint.stage
    _log_optimizer_runtime(
        config, global_checkpoint.weights, stage=global_checkpoint.stage, partitioned=True
    )
    return global_checkpoint.weights, global_checkpoint.loss, global_loss_fn


def angular_multiple_optax(
    config, sa, loss_fn, actual_data, previous_weights=None, previous_epoch=0, stage=0
):
    """
    This performs an fitting routines from the optax packages, different minimizers have different requirements for updating steps
    Performs parameter optimization using Optax minimizers for angular Thomson scattering data.
    This function sets up and runs a fitting routine using the Optax optimization library, applying the specified minimizer to fit model parameters to experimental data. It handles data batching, optimizer initialization, training loop with early stopping, and logging of metrics and optimizer state.
        
    Args:    
        config (dict): Configuration dictionary built from the input decks, specifying optimizer, data, and parameter settings.
        all_data (dict): Dictionary containing datasets, amplitudes, and backgrounds as constructed by the prepare.py code.
        sa (dict): Dictionary of the scattering angles and their relative weights.
    Returns:
        best_weights (dict): Best parameter weights as returned by the minimizer.
        best_loss (float): Best value of the fit metric found by the minimizer.
        ts_instance (LossFunction): Instance of the LossFunction object used for minimization.
    Notes:
        - Supports early stopping based on loss improvement or degradation.
        - Logs training metrics and optimizer state using mlflow.
        - Handles both single and multiple shot number data configurations for rotated repeats of data.

    """
    # minimizer = getattr(optax, config["optimizer"]["method"])
    # schedule = optax.schedules.cosine_decay_schedule(config["optimizer"]["learning_rate"], 100, alpha = 0.00001)
    # solver = minimizer(schedule)
    # solver = minimizer(config["optimizer"]["learning_rate"])
    if previous_epoch is None:
        previous_epoch = 0

    minimizer = getattr(optax, config["optimizer"]["method"])
    param_minimizer = getattr(optax, config["optimizer"]["param_method"])
    schedule = optax.schedules.cosine_decay_schedule(
        config["optimizer"]["learning_rate_init"],
        max(1, int(round(0.75 * config["optimizer"]["num_epochs"]))),
        alpha=config["optimizer"]["learning_rate_final"] / config["optimizer"]["learning_rate_init"],
    )

    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False, activate=True)
    else:
        ts_params = previous_weights
    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    
    solver = optax.partition({"macro": param_minimizer(config["optimizer"]["param_learning_rate"]), "dist": minimizer(schedule)}, partial(label, cfg_params=config["parameters"]))
    opt_state = solver.init(diff_params)

    _log_optimizer_runtime(config, ts_params, stage=stage, partitioned=True)
    (raw_loss, aux), grad = loss_fn.vg_loss(diff_params, static_params, actual_data)
    _require_finite(raw_loss, "loss", step=previous_epoch, stage=stage)
    _require_finite(grad, "gradient", step=previous_epoch, stage=stage)
    if config["optimizer"].get("validate_active_leaves", True):
        validate_active_leaves(config, ts_params, diff_params, static_params, loss_fn, actual_data)

    current_loss = float(raw_loss)
    checkpoint = OptimizationCheckpoint(
        loss=current_loss,
        weights=eqx.combine(diff_params, static_params),
        step=previous_epoch,
        stage=stage,
    )
    patience, min_delta = _stopping_options(config)
    wait = 0
    patience_reference_loss = checkpoint.loss
    seed = int(config["optimizer"].get("seed", 0))
    _log_optimizer_step(
        current_loss=current_loss,
        checkpoint=checkpoint,
        learning_rate=float(schedule(0)),
        grad=grad,
        step=previous_epoch,
        stage=stage,
        seed=seed,
    )

    # Start the train loop from a valid, evaluated checkpoint.
    state_weights = {}
    updates_completed = 0
    exit_cond = "Reached epoch limit"
    for i_epoch in (pbar := trange(config["optimizer"]["num_epochs"])):
        updates, opt_state = solver.update(grad, opt_state, diff_params)
        global_step = previous_epoch + i_epoch + 1
        _require_finite(updates, "update", step=global_step, stage=stage)
        candidate_params = eqx.apply_updates(diff_params, updates)
        _require_finite(candidate_params, "parameters", step=global_step, stage=stage)

        (candidate_raw_loss, aux), candidate_grad = loss_fn.vg_loss(
            candidate_params, static_params, actual_data
        )
        _require_finite(candidate_raw_loss, "loss", step=global_step, stage=stage)
        _require_finite(candidate_grad, "gradient", step=global_step, stage=stage)
        candidate_loss = float(candidate_raw_loss)

        improvement = checkpoint.loss - candidate_loss
        if improvement > 0.0:
            checkpoint = OptimizationCheckpoint(
                loss=candidate_loss,
                weights=eqx.combine(candidate_params, static_params),
                step=global_step,
                stage=stage,
            )
        if patience_reference_loss - candidate_loss > min_delta:
            patience_reference_loss = candidate_loss
            wait = 0
        else:
            wait += 1
        learning_rate = float(schedule(i_epoch))
        pbar.set_description(
            f"Loss {candidate_loss:.2e}, Best {checkpoint.loss:.2e}, Learning rate {learning_rate:.2e}"
        )

        if config["optimizer"]["save_state"]:
            if global_step % config["optimizer"]["save_state_freq"] == 0:
                state_weights[global_step] = checkpoint.weights.get_unnormed_params()

        _log_optimizer_step(
            current_loss=candidate_loss,
            checkpoint=checkpoint,
            learning_rate=learning_rate,
            grad=candidate_grad,
            step=global_step,
            stage=stage,
            seed=seed,
        )
        diff_params, raw_loss, grad, current_loss = (
            candidate_params,
            candidate_raw_loss,
            candidate_grad,
            candidate_loss,
        )
        updates_completed = i_epoch + 1
        if patience > 0 and wait >= patience:
            exit_cond = f"No improvement >= {min_delta:g} for {patience} steps"
            break
        
    if config["optimizer"]["save_state"]:
        with open("state_weights.txt", "wb") as file:
            file.write(pickle.dumps(state_weights))

        mlflow.log_artifact("state_weights.txt")
    return (
        checkpoint.weights,
        checkpoint.loss,
        previous_epoch + updates_completed,
        loss_fn,
        exit_cond,
    )

def label(diff_params, cfg_params):
    """
    Builds the partition-label pytree used by angular_multiple_optax's optax.partition solver: a pytree
    with the same structure as diff_params where every leaf is labeled "macro" or "dist", selecting which
    of the two sub-optimizers (param_method for "macro", method for "dist") is applied to that leaf.

    By default every leaf is "macro". If the electron distribution function is active, its leaves are
    relabeled "dist" via get_distribution_filter_spec -- except normed_m (the DLM shape parameter, present
    for DLM-type distributions), which stays "macro" alongside the other scalar plasma parameters rather
    than being treated as part of the distribution function.

    Args:
        diff_params: the differentiable parameter pytree (matching the structure to be labeled).
        cfg_params (Dict): config["parameters"], used to check whether the electron distribution function
            is active.

    Returns:
        A pytree matching diff_params' structure, with each leaf replaced by the string "macro" or "dist".
    """
    from jax import tree_util as jtu
    from tsadar.core.modules.distribution_functions.base import get_distribution_filter_spec
    label_spec = jtu.tree_map(lambda _: "macro", diff_params)

    if cfg_params["electron"]["fe"]["active"]:
        label_spec = get_distribution_filter_spec(label_spec, dist_params=cfg_params["electron"]["fe"], replace="dist")
        if "normed_m" in dir(label_spec.electron.distribution_functions):
            label_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.normed_m, label_spec, replace="macro"
                )
    return label_spec
