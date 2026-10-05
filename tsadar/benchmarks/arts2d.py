"""ISS140: production ARTS2D recovery with withheld samples and EDF-space SVD.

Run ``python -m tsadar.benchmarks.arts2d --preset smoke --output <new-directory>``.
The full preset is a multi-seed GPU experiment; see the benchmark documentation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.inverse.loss_function import LossFunction
from tsadar.inverse.recovery_benchmark import (
    bootstrap_mean_interval,
    config_hash,
    detector_visible_null_basis,
    edf_moments,
    make_detector_mask,
    noise_whitened_residuals,
    poisson_read_observation,
    project_gain_nuisance,
    residual_rms,
    subspace_edf_errors,
    synthetic_truth_families,
)

MODEL_NAMES = ("arbitrary", "nn-sh", "coefficient-sh", "mora-yahi")
TRUTH_NAMES = ("maxwellian", "elliptic", "supergaussian", "skew_tail")


@dataclass(frozen=True)
class BenchmarkSpec:
    """All experiment choices; resolved values are persisted before execution."""

    models: tuple[str, ...] = MODEL_NAMES
    truths: tuple[str, ...] = TRUTH_NAMES
    seeds: tuple[int, ...] = tuple(range(10))
    peak_counts: tuple[float, ...] = (1000.0, 10000.0)
    angles_deg: tuple[float, ...] = (
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
    )
    heldout_angles: tuple[int, ...] = (1, 4, 7)
    heldout_wedges: tuple[tuple[int, int], ...] = ((22, 26),)
    wavelength_bounds_nm: tuple[float, float] = (470.0, 515.0)
    detector_bins: int = 64
    nvx: int = 128
    truth_refinement: int = 2
    nvr: int = 64
    nvz: int = 64
    harmonic_degree: int = 2
    n_beta: int = 512
    root_scan_panels: int = 2048
    integration_panels: int = 128
    regular_order: int = 8
    root_order: int = 32
    max_roots: int = 8
    steps: int = 500
    learning_rate: float = 0.01
    metric_interval: int = 25
    initialization_seed: int = 173
    truth_rotation_deg: float = 27.0
    true_gain: float = 1.15
    background_counts: float = 10.0
    read_noise_counts: float = 3.0
    svd_rtol: float = 1e-6
    svd_atol: float = 1.0
    jacobian_batch_size: int = 4
    bootstrap_resamples: int = 2000

    @classmethod
    def smoke(cls):
        return cls(
            truths=("elliptic",),
            seeds=(0,),
            peak_counts=(1000.0,),
            angles_deg=(45.0, 60.0, 75.0),
            heldout_angles=(1,),
            heldout_wedges=((3, 4),),
            wavelength_bounds_nm=(470.0, 515.0),
            detector_bins=8,
            nvx=32,
            nvr=16,
            nvz=32,
            harmonic_degree=1,
            n_beta=16,
            root_scan_panels=128,
            integration_panels=16,
            regular_order=4,
            root_order=8,
            max_roots=4,
            steps=2,
            metric_interval=1,
            jacobian_batch_size=1,
        )

    def validate(self):
        integer_fields = (
            "detector_bins",
            "nvx",
            "truth_refinement",
            "nvr",
            "nvz",
            "harmonic_degree",
            "n_beta",
            "root_scan_panels",
            "integration_panels",
            "regular_order",
            "root_order",
            "max_roots",
            "steps",
            "metric_interval",
            "initialization_seed",
            "jacobian_batch_size",
            "bootstrap_resamples",
        )
        integer_values = [getattr(self, name) for name in integer_fields]
        integer_values.extend(self.seeds)
        integer_values.extend(self.heldout_angles)
        integer_values.extend(edge for wedge in self.heldout_wedges for edge in wedge)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in integer_values
        ):
            raise ValueError(
                "grid sizes, indices, seeds and iteration controls must be integers"
            )
        for values, allowed, name in (
            (self.models, MODEL_NAMES, "models"),
            (self.truths, TRUTH_NAMES, "truths"),
        ):
            if (
                not values
                or len(set(values)) != len(values)
                or not set(values) <= set(allowed)
            ):
                raise ValueError(f"{name} must be unique members of {allowed}")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or min(self.seeds) < 0
        ):
            raise ValueError("seeds must be unique nonnegative integers")
        positive = (
            self.detector_bins,
            self.nvx,
            self.nvr,
            self.nvz,
            self.n_beta,
            self.root_scan_panels,
            self.integration_panels,
            self.regular_order,
            self.max_roots,
            self.steps,
            self.learning_rate,
            self.metric_interval,
            self.true_gain,
            self.jacobian_batch_size,
            self.bootstrap_resamples,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError(
                "resolutions, step budget, gain, and learning rate must be positive"
            )
        if self.nvx < 8 or self.nvr < 12 or self.nvz < 16 or self.truth_refinement < 2:
            raise ValueError(
                "require nvx >= 8, nvr >= 12 (nonzero Hann window), nvz >= 16 and truth_refinement >= 2"
            )
        if self.root_order < 2 or self.root_order % 2 or self.harmonic_degree < 1:
            raise ValueError(
                "root_order must be even and harmonic_degree must be positive"
            )
        if (
            len(self.wavelength_bounds_nm) != 2
            or not 0
            < self.wavelength_bounds_nm[0]
            < self.wavelength_bounds_nm[1]
            < 526.5
        ):
            raise ValueError(
                "this benchmark requires an ordered blue-wing interval below 526.5 nm"
            )
        angles = np.asarray(self.angles_deg)
        if (
            angles.size < 3
            or not np.all(np.isfinite(angles))
            or np.any(np.diff(angles) <= 0)
            or np.any((angles <= 0) | (angles >= 180))
        ):
            raise ValueError(
                "at least three distinct increasing scattering angles inside (0,180) are required"
            )
        if (
            not self.peak_counts
            or len(set(self.peak_counts)) != len(self.peak_counts)
            or not all(np.isfinite(v) and v > 0 for v in self.peak_counts)
        ):
            raise ValueError("peak photon levels must be unique, finite, and positive")
        if (
            not np.isfinite(self.background_counts + self.read_noise_counts)
            or min(self.background_counts, self.read_noise_counts) < 0
        ):
            raise ValueError("noise controls must be finite and nonnegative")
        if min(self.svd_rtol, self.svd_atol) < 0 or not np.isfinite(
            self.svd_rtol + self.svd_atol
        ):
            raise ValueError("SVD tolerances must be finite and nonnegative")
        if not np.isfinite(self.truth_rotation_deg) or self.initialization_seed < 0:
            raise ValueError("invalid truth rotation or initialization seed")
        make_detector_mask(
            len(angles),
            self.detector_bins,
            heldout_angles=self.heldout_angles,
            heldout_wedges=self.heldout_wedges,
        )


def _parameter(value, lower=0.0, upper=1.0, **kwargs):
    return {"active": False, "val": value, "lb": lower, "ub": upper, **kwargs}


def make_config(spec, *, model="arbitrary", refined=False):
    """Self-contained production deck, independent of tests and shot calibrations."""
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model}")
    factor = spec.truth_refinement if refined else 1
    fe = {
        "active": True,
        "dim": 2,
        "nvx": spec.nvx * factor,
        "type": "arbitrary" if model == "arbitrary" else "sphericalharmonic",
        "params": {
            "init_m": 2.0,
            "learn_log": True,
            "flm_type": {
                "arbitrary": "dlm",
                "nn-sh": "nn",
                "coefficient-sh": "arbitrary",
                "mora-yahi": "mora-yahi",
            }[model],
            "nvr": spec.nvr,
            "nvz": spec.nvz,
            "Nl": 1 if model == "mora-yahi" else spec.harmonic_degree,
            "dtx": 0.0,
            "dty": 0.0,
        },
    }
    lower, upper = spec.wavelength_bounds_nm
    edges = np.linspace(lower, upper, spec.detector_bins + 1)
    return {
        "parameters": {
            "electron": {
                "fe": fe,
                "Te": _parameter(0.6, 0.01, 2.0),
                "ne": _parameter(0.2, 0.001, 1.0),
            },
            "general": {
                "amp1": _parameter(1.0, 0.01, 4.0),
                "amp2": _parameter(1.0, 0.01, 4.0),
                "amp3": _parameter(1.0, 0.01, 4.0),
                "lam": _parameter(526.5, 523.0, 529.0),
                "ud": _parameter(0.0, -100.0, 100.0, angle=0.0),
                "Te_gradient": _parameter(0.0, 0.0, 10.0, num_grad_points=1),
                "ne_gradient": _parameter(0.0, 0.0, 10.0, num_grad_points=1),
            },
            "ion-1": {
                "A": _parameter(1.0),
                "Z": _parameter(1.0, 0.5, 2.0),
                "Ti": _parameter(0.08, 0.001, 1.0),
                "fract": _parameter(1.0),
                "Va": _parameter(0.0, -100.0, 100.0, angle=0.0),
            },
        },
        "data": {
            "load_ele_spec": True,
            "load_ion_spec": False,
            "fit_EPWb": True,
            "fit_EPWr": False,
            "fit_IAW": False,
            "ele_lam_shift": 0.0,
            "shotnum": 0,
            "lineouts": {"start": 0, "end": len(spec.angles_deg)},
            "fit_rng": {
                "blue_min": lower,
                "blue_max": upper,
                "red_min": 530.0,
                "red_max": 700.0,
            },
        },
        "other": {
            "extraoptions": {"spectype": "angular_full"},
            "CCDsize": [spec.detector_bins, len(spec.angles_deg)],
            "npts": spec.detector_bins,
            "lamrangE": [lower, upper],
            "lamrangI": [524.0, 529.0],
            "n_beta": spec.n_beta * factor,
            "ang_res_unit": 1,
            "lam_res_unit": 1,
            "iawfilter": [False, 0, 0, 0],
            "iawoff": False,
            "detector_specs": {
                "norm": 0,
                "electron_wavelength_edges": edges.tolist(),
                "electron_wavelength_centers": ((edges[:-1] + edges[1:]) / 2).tolist(),
                "widIRF": {"spect_FWHM_ele": 1.3, "ang_FWHM_ele": 1.0},
            },
            "resonance_quadrature": {
                "enabled": True,
                "root_scan_panels": spec.root_scan_panels * factor,
                "integration_panels": spec.integration_panels * factor,
                "regular_order": spec.regular_order,
                "root_order": spec.root_order,
                "max_roots": spec.max_roots,
                "neighbor_panels": 1,
                "bisection_iterations": 48,
                "tail_sigma": 6.0,
                "scan_phase": 0.0,
                "map_batch_size": 1,
            },
        },
        "optimizer": {
            "method": "adam",
            "loss_method": "l2",
            "y_norm": False,
            "x_norm": False,
            "angular_objective": {
                "noise": {"model": "measured_variance"},
                "gain": {"mode": "global", "prior_strength": 0.0, "smoothness": 0.0},
                "robust": {"kind": "gaussian"},
            },
        },
    }


def initialize_model(config, seed):
    """All models start at the same Maxwellian; only EDF leaves are active."""
    params = ThomsonParams(
        config["parameters"], num_params=1, batch=False, activate=True
    )
    distribution = params.electron.distribution_functions
    if config["parameters"]["electron"]["fe"]["type"] == "sphericalharmonic":
        for degree in range(1, distribution.Nl + 1):
            for order in range(degree + 1):
                radial = distribution.flm[degree][order]
                if distribution.flm_type == "nn":
                    key = jax.random.fold_in(
                        jax.random.PRNGKey(seed), degree * 100 + order
                    )
                    mag_key, sign_key = jax.random.split(key)
                    magnitude = eqx.nn.MLP(
                        1, 1, 32, 3, final_activation=jax.nn.relu, key=mag_key
                    )
                    sign = eqx.nn.MLP(
                        1, 1, 32, 3, final_activation=jnp.tanh, key=sign_key
                    )
                    sign = eqx.tree_at(
                        lambda m: (m.layers[-1].weight, m.layers[-1].bias),
                        sign,
                        (
                            jnp.zeros_like(sign.layers[-1].weight),
                            jnp.zeros_like(sign.layers[-1].bias),
                        ),
                    )
                    radial = eqx.tree_at(
                        lambda r: (r.flm_mag, r.flm_sign), radial, (magnitude, sign)
                    )
                elif distribution.flm_type == "arbitrary":
                    # Zero sign gives the same isotropic EDF; O(1) magnitude avoids
                    # imposing a 1e-5 coefficient learning-rate penalty at startup.
                    radial = eqx.tree_at(
                        lambda r: r.flm_mag, radial, jnp.full_like(radial.flm_mag, -4.0)
                    )
                distribution = eqx.tree_at(
                    lambda d: d.flm[degree][order], distribution, radial
                )
        params = eqx.tree_at(
            lambda p: p.electron.distribution_functions, params, distribution
        )
    diff, static = eqx.partition(params, get_filter_spec(config["parameters"], params))
    active_leaves = [
        {
            "path": jax.tree_util.keystr(path),
            "shape": list(value.shape),
            "size": int(value.size),
        }
        for path, value in jax.tree_util.tree_flatten_with_path(diff)[0]
        if eqx.is_inexact_array(value)
    ]
    return params, diff, static, active_leaves


def _blank_batch(spec):
    shape = (len(spec.angles_deg), spec.detector_bins)
    return {
        "e_data": jnp.ones(shape),
        "i_data": jnp.zeros((1, 1)),
        "noise_e": jnp.zeros(shape),
        "noise_i": jnp.zeros((1, 1)),
        "e_amps": jnp.ones((shape[0], 1)),
        "i_amps": jnp.ones(1),
    }


def build_forward(config, geometry, batch):
    """Expose the actual detector/IRF path as a differentiable function of an EDF."""
    diagnostic = ThomsonScatteringDiagnostic(config, geometry)
    template = ThomsonParams(config["parameters"], num_params=1, batch=False)()
    velocity = template["electron"]["v"]

    def forward(edf):
        physical = dict(template)
        physical["electron"] = dict(template["electron"], fe=edf)
        return diagnostic(lambda: physical, batch)[0]

    return jax.jit(forward), np.asarray(velocity)


def detector_jacobian(
    forward, truth, sigma, train_mask, *, cell_area, gain, batch_size
):
    """Bound reverse-mode memory by processing a few detector cotangents at a time."""
    indices = np.flatnonzero(train_mask)
    sigma_train = jnp.asarray(sigma.ravel()[indices])
    q0 = jnp.asarray(truth).ravel() * np.sqrt(cell_area)

    def measurement(q):
        edf = q.reshape(truth.shape) / np.sqrt(cell_area)
        edf = edf / (
            jnp.sum(edf) * cell_area
        )  # density constraint on the common tangent
        return (gain * forward(edf)).ravel()[indices] / sigma_train

    _, pullback = jax.vjp(measurement, q0)
    pull_rows = jax.jit(jax.vmap(lambda row: pullback(row)[0]))
    rows = []
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        cotangents = jax.nn.one_hot(
            jnp.arange(start, stop), len(indices), dtype=q0.dtype
        )
        rows.append(np.asarray(pull_rows(cotangents)))
    jacobian = np.concatenate(rows)
    # Restrict the common Euclidean q coordinates to the orthogonal fixed-density
    # tangent. Normalization's derivative alone has a truth-shaped gauge direction;
    # its right singular vectors need not conserve mass. For equal-area cells the
    # mass direction is constant, so J @ (I - 11^T/N) subtracts each row's mean.
    jacobian = jacobian - jacobian.mean(axis=1, keepdims=True)
    whitened_signal = np.asarray(forward(jnp.asarray(truth))).ravel()[
        indices
    ] / np.asarray(sigma_train)
    return project_gain_nuisance(jacobian, whitened_signal)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _git_provenance():
    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
    }


def _metrics(edf, fitted_counts, data, sigma, masks, truth, velocity, decomposition):
    metrics = {
        f"{name}_whitened_rms": residual_rms(
            noise_whitened_residuals(data, fitted_counts, sigma, mask)
        )
        for name, mask in masks.items()
        if mask.any()
    }
    metrics.update(
        subspace_edf_errors(
            truth,
            edf,
            decomposition["visible_basis"],
            cell_area=(velocity[1] - velocity[0]) ** 2,
        )
    )
    moments = edf_moments(edf, velocity, velocity)
    truth_moments = edf_moments(truth, velocity, velocity)
    metrics.update({f"moment_{key}": value for key, value in moments.items()})
    metrics.update(
        {
            f"moment_error_{key}": value - truth_moments[key]
            for key, value in moments.items()
        }
    )
    return metrics


def fit_model(
    spec, config, forward, loss, batch, masks, truth, velocity, decomposition, output
):
    """Fixed budget; select best only among evaluated training-objective iterates."""
    initial, diff, static, active_leaves = initialize_model(
        config, spec.initialization_seed
    )
    initial_edf = np.asarray(initial.electron.distribution_functions())
    expected_initial = synthetic_truth_families(velocity, velocity)["maxwellian"]
    if not np.allclose(initial_edf, expected_initial, rtol=1e-6, atol=1e-10):
        raise ValueError(
            "parameterization does not satisfy the common Maxwellian initialization policy"
        )
    centers = jnp.asarray(
        config["other"]["detector_specs"]["electron_wavelength_centers"]
    )

    def objective(trainable):
        params = eqx.combine(trainable, static)
        edf = params.electron.distribution_functions()
        signal = forward(edf)
        # The production likelihood expects the known background in its theory.
        total, _, predicted, _, diagnostics = loss._angular_data_objective(
            batch, signal + batch["noise_e"], centers
        )
        # Production loss deliberately excludes invalid pixels. A numerical model
        # failure must instead invalidate this controlled benchmark, including at
        # held-out pixels. Require finite, nonnegative raw photon signals before
        # background addition or sanitized predictions can conceal a failure.
        valid_signal = jnp.all(jnp.isfinite(signal) & (signal >= 0))
        total = jnp.where(valid_signal, total, jnp.nan)
        return total, (edf, predicted, diagnostics["profiled_gains"][0])

    evaluate = eqx.filter_jit(objective)
    value_grad = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))
    optimizer = optax.chain(
        optax.clip_by_global_norm(100.0), optax.adam(spec.learning_rate)
    )
    state = optimizer.init(diff)
    sigma = np.sqrt(np.asarray(batch["e_variance"]))
    data = np.asarray(batch["e_data"])
    best_value, best_step, best_diff, best_aux = np.inf, None, None, None
    history = []
    start = time.perf_counter()
    first_evaluation_seconds = None
    for step in range(spec.steps + 1):
        if step < spec.steps:
            (value, aux), gradient = value_grad(diff)
            if not all(
                np.all(np.isfinite(np.asarray(leaf)))
                for leaf in jax.tree_util.tree_leaves(gradient)
            ):
                raise FloatingPointError(f"nonfinite gradient at step {step}")
        else:
            value, aux = evaluate(diff)
        value = float(value)
        if not np.isfinite(value) or not all(
            np.all(np.isfinite(np.asarray(a))) for a in aux
        ):
            raise FloatingPointError(
                f"invalid production benchmark evaluation at step {step}: "
                "outputs must be finite and raw photon signals nonnegative"
            )
        if first_evaluation_seconds is None:
            first_evaluation_seconds = time.perf_counter() - start
        if value < best_value:
            best_value, best_step, best_diff, best_aux = value, step, diff, aux
        if step % spec.metric_interval == 0 or step == spec.steps:
            metrics = _metrics(
                np.asarray(aux[0]),
                np.asarray(aux[1]),
                data,
                sigma,
                masks,
                truth,
                velocity,
                decomposition,
            )
            history.append(
                {
                    "step": step,
                    "train_objective": value,
                    "gain": float(aux[2]),
                    **metrics,
                }
            )
        if step < spec.steps:
            updates, state = optimizer.update(gradient, state, diff)
            diff = eqx.apply_updates(diff, updates)
    elapsed = time.perf_counter() - start
    best_metrics = _metrics(
        np.asarray(best_aux[0]),
        np.asarray(best_aux[1]),
        data,
        sigma,
        masks,
        truth,
        velocity,
        decomposition,
    )
    result = {
        "best_step": best_step,
        "best_train_objective": best_value,
        "best_gain": float(best_aux[2]),
        "best": best_metrics,
        "initial": history[0],
        "final": history[-1],
        "gradient_evaluations": spec.steps,
        "evaluated_states": spec.steps + 1,
        "wall_seconds": elapsed,
        "first_evaluation_seconds": first_evaluation_seconds,
    }
    output.mkdir()
    _write_json(output / "metrics.json", result)
    _write_json(output / "history.json", history)
    _write_json(output / "fit_config.json", config)
    np.savez_compressed(
        output / "recovery.npz",
        initial_edf=initial_edf,
        best_edf=np.asarray(best_aux[0]),
        final_edf=np.asarray(aux[0]),
        best_prediction=np.asarray(best_aux[1]),
        final_prediction=np.asarray(aux[1]),
        best_whitened_residual=(data - np.asarray(best_aux[1])) / sigma,
    )
    eqx.tree_serialise_leaves(
        output / "best_weights.eqx", eqx.combine(best_diff, static)
    )
    actual = initial.electron.distribution_functions
    model_info = {
        "class": f"{type(actual).__module__}.{type(actual).__qualname__}",
        "flm_type": getattr(actual, "flm_type", None),
        "active_leaves": active_leaves,
        "active_parameter_count": sum(leaf["size"] for leaf in active_leaves),
        "initialization": "Maxwellian m=2; zero harmonic sign/head; seeded hidden NN weights",
        "initialization_seed": spec.initialization_seed,
        "budget": {
            "policy": "fixed gradient evaluations",
            "steps": spec.steps,
            "learning_rate": spec.learning_rate,
            "optimizer": "Adam with global norm clip 100",
            "restart_count": 0,
        },
    }
    return result, model_info, history


def _log_mlflow(experiment, name, output, shared, metadata, result, history):
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(
            {
                "benchmark": "ISS140",
                "fit_model": metadata["fit_model"],
                "truth_family": metadata["truth_family"],
                "git_sha": metadata["git_sha"],
                "config_sha256": metadata["config_sha256"],
            }
        )
        mlflow.log_params(
            {
                "seed": metadata["seed"],
                "peak_counts": metadata["peak_counts"],
                "actual_class": metadata["class"],
                "flm_type": metadata["flm_type"],
                "active_parameter_count": metadata["active_parameter_count"],
            }
        )
        for row in history:
            mlflow.log_metrics(
                {k: float(v) for k, v in row.items() if k != "step"}, step=row["step"]
            )
        mlflow.log_metrics({f"best_{k}": float(v) for k, v in result["best"].items()})
        mlflow.log_metrics(
            {
                "best_step": result["best_step"],
                "best_train_objective": result["best_train_objective"],
                "wall_seconds": result["wall_seconds"],
            }
        )
        mlflow.log_artifacts(str(output))
        # Preserve valid relative links in the downloaded MLflow artifact layout.
        mlflow.log_dict(
            {
                **metadata,
                "shared_case": "shared/case.npz",
                "subspaces": "shared/subspaces.npz",
                "observations": "shared/observations.npy",
            },
            "provenance.json",
        )
        mlflow.log_artifact(str(shared / "case.npz"), artifact_path="shared")
        mlflow.log_artifact(str(shared / "subspaces.npz"), artifact_path="shared")
        mlflow.log_artifact(str(shared / "case.json"), artifact_path="shared")
        mlflow.log_artifact(
            str(output.parent / "observations.npy"), artifact_path="shared"
        )
        for filename in (
            "benchmark.json",
            "forward_config.json",
            "truth_forward_config.json",
        ):
            mlflow.log_artifact(str(shared.parent / filename), artifact_path="shared")
        return run.info.run_id


def summarize_runs(records, spec):
    """Aggregate all seeds, keeping truths/noise levels distinct and differences paired."""
    groups = []
    for truth in spec.truths:
        for photons in spec.peak_counts:
            selected = [
                r
                for r in records
                if r["truth_family"] == truth and r["peak_counts"] == photons
            ]
            expected_pairs = {
                (model, seed) for model in spec.models for seed in spec.seeds
            }
            actual_pairs = [(run["fit_model"], run["seed"]) for run in selected]
            if (
                len(actual_pairs) != len(expected_pairs)
                or set(actual_pairs) != expected_pairs
            ):
                raise ValueError(
                    "seed summaries require exactly one completed run per model and common seed"
                )
            by_model = {
                model: {r["seed"]: r for r in selected if r["fit_model"] == model}
                for model in spec.models
            }
            for model, seed_runs in by_model.items():
                metrics = {
                    name: bootstrap_mean_interval(
                        [seed_runs[s]["best"][name] for s in spec.seeds],
                        resamples=spec.bootstrap_resamples,
                    )
                    for name in seed_runs[spec.seeds[0]]["best"]
                }
                paired = {}
                if "arbitrary" in by_model and model != "arbitrary":
                    paired = {
                        name: bootstrap_mean_interval(
                            [
                                seed_runs[s]["best"][name]
                                - by_model["arbitrary"][s]["best"][name]
                                for s in spec.seeds
                            ],
                            resamples=spec.bootstrap_resamples,
                        )
                        for name in metrics
                    }
                groups.append(
                    {
                        "truth_family": truth,
                        "peak_counts": photons,
                        "fit_model": model,
                        "metrics": metrics,
                        "paired_difference_from_arbitrary": paired,
                    }
                )
    return {
        "interval_method": "seed-level percentile bootstrap of mean, 95%; paired common-seed differences",
        "bootstrap_resamples": spec.bootstrap_resamples,
        "bootstrap_seed": 0,
        "groups": groups,
    }


def _run_benchmark(spec, output, *, mlflow_experiment=None):
    """Execute within the new result directory owned by run_benchmark."""
    provenance = {
        **_git_provenance(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "jax_version": jax.__version__,
        "jaxlib_version": version("jaxlib"),
        "scipy_version": version("scipy"),
        "numpy_version": np.__version__,
        "equinox_version": version("equinox"),
        "optax_version": version("optax"),
        "devices": [str(device) for device in jax.devices()],
        "dtype": "float64",
    }
    resolved = asdict(spec)
    _write_json(
        output / "benchmark.json",
        {"spec": resolved, "spec_sha256": config_hash(resolved), **provenance},
    )
    geometry = {
        "sa": np.asarray(spec.angles_deg),
        "angAxis": np.asarray(spec.angles_deg),
        "weights": np.eye(len(spec.angles_deg)),
    }
    blank = _blank_batch(spec)
    config, truth_config = make_config(spec), make_config(spec, refined=True)
    forward, velocity = build_forward(config, geometry, blank)
    truth_forward, truth_velocity = build_forward(truth_config, geometry, blank)
    area = float((velocity[1] - velocity[0]) ** 2)
    _write_json(output / "forward_config.json", config)
    _write_json(output / "truth_forward_config.json", truth_config)
    train_mask = make_detector_mask(
        len(spec.angles_deg),
        spec.detector_bins,
        heldout_angles=spec.heldout_angles,
        heldout_wedges=spec.heldout_wedges,
    )
    angle_mask = np.zeros_like(train_mask)
    angle_mask[list(spec.heldout_angles)] = True
    masks = {
        "train": train_mask,
        "heldout": ~train_mask,
        "heldout_angle": angle_mask,
        "heldout_wedge": ~train_mask & ~angle_mask,
    }
    coarse_truths = synthetic_truth_families(
        velocity, velocity, rotation_deg=spec.truth_rotation_deg
    )
    fine_truths = synthetic_truth_families(
        truth_velocity, truth_velocity, rotation_deg=spec.truth_rotation_deg
    )
    records = []
    for truth_name in spec.truths:
        print(
            f"Generating {truth_name} on {len(truth_velocity)} vs {len(velocity)} velocity cells",
            flush=True,
        )
        coarse_truth, fine_truth = coarse_truths[truth_name], fine_truths[truth_name]
        reference = np.asarray(truth_forward(jnp.asarray(fine_truth)))
        coarse_reference = np.asarray(forward(jnp.asarray(coarse_truth)))
        if (
            not np.all(np.isfinite(reference))
            or np.any(reference < 0)
            or reference.max() <= 0
        ):
            raise FloatingPointError(
                "invalid refined truth spectrum; inspect the quadrature resolution"
            )
        if not np.all(np.isfinite(coarse_reference)) or np.any(coarse_reference < 0):
            raise FloatingPointError("invalid inversion-grid truth spectrum")
        for photons in spec.peak_counts:
            shared = output / f"{truth_name}-counts-{photons:.17g}"
            shared.mkdir()
            exposure = photons / reference.max()
            count_forward = jax.jit(lambda f: exposure * forward(f))
            clean_signal = spec.true_gain * exposure * reference
            _, variance = poisson_read_observation(
                clean_signal,
                seed=0,
                background=spec.background_counts,
                read_noise=spec.read_noise_counts,
            )
            sigma = np.sqrt(variance)
            print(
                f"Computing common detector Jacobian for {truth_name}, peak counts={photons:g}",
                flush=True,
            )
            jacobian = detector_jacobian(
                count_forward,
                coarse_truth,
                sigma,
                train_mask,
                cell_area=area,
                gain=spec.true_gain,
                batch_size=spec.jacobian_batch_size,
            )
            decomposition = detector_visible_null_basis(
                jacobian, rtol=spec.svd_rtol, atol=spec.svd_atol
            )
            np.savez_compressed(
                shared / "subspaces.npz", jacobian=jacobian, **decomposition
            )
            np.savez_compressed(
                shared / "case.npz",
                velocity=velocity,
                truth_velocity=truth_velocity,
                truth_edf=coarse_truth,
                refined_truth_edf=fine_truth,
                angles_deg=spec.angles_deg,
                detector_edges_nm=config["other"]["detector_specs"][
                    "electron_wavelength_edges"
                ],
                clean_signal=clean_signal,
                background=spec.background_counts,
                variance=variance,
                coarse_truth_signal=spec.true_gain * exposure * coarse_reference,
                exposure=exposure,
                **masks,
            )
            coarse_bias = (
                spec.true_gain * exposure * (coarse_reference - reference)
            ) / sigma
            case_info = {
                "exposure": float(exposure),
                "true_gain": spec.true_gain,
                "truth_grid_discrepancy_whitened_rms": residual_rms(coarse_bias),
                "truth_grid_discrepancy_below_noise": residual_rms(coarse_bias) < 1.0,
                "svd_rank": decomposition["rank"],
                "numerical_rank": decomposition["numerical_rank"],
                "svd_threshold": decomposition["threshold"],
                "edf_coordinates": "q=sqrt(cell_area)*f",
                "linearization": "truth sampled on inversion grid; orthogonal fixed-density tangent; global gain projected out",
                "truth_moments": edf_moments(coarse_truth, velocity, velocity),
                "refined_truth_moments": edf_moments(
                    fine_truth, truth_velocity, truth_velocity
                ),
            }
            _write_json(shared / "case.json", case_info)
            for seed in spec.seeds:
                seed_dir = shared / f"seed-{seed}"
                seed_dir.mkdir()
                data, _ = poisson_read_observation(
                    clean_signal,
                    seed=seed,
                    background=spec.background_counts,
                    read_noise=spec.read_noise_counts,
                )
                np.save(seed_dir / "observations.npy", data)
                batch = dict(
                    blank,
                    e_data=jnp.asarray(data),
                    noise_e=jnp.full_like(blank["e_data"], spec.background_counts),
                    e_variance=jnp.asarray(variance),
                    e_mask=jnp.asarray(train_mask),
                )
                loss = LossFunction(config, geometry, batch)
                for model in spec.models:
                    print(
                        f"Fitting {model}: truth={truth_name}, photons={photons:g}, seed={seed}",
                        flush=True,
                    )
                    fit_config = make_config(spec, model=model)
                    run_dir = seed_dir / model
                    result, model_info, history = fit_model(
                        spec,
                        fit_config,
                        count_forward,
                        loss,
                        batch,
                        masks,
                        coarse_truth,
                        velocity,
                        decomposition,
                        run_dir,
                    )
                    identity = {
                        "truth_family": truth_name,
                        "fit_model": model,
                        "seed": seed,
                        "peak_counts": photons,
                    }
                    metadata = {
                        **identity,
                        **provenance,
                        **model_info,
                        "config_sha256": config_hash(
                            {"benchmark": resolved, "fit": fit_config, "case": identity}
                        ),
                        "train_mask": train_mask.tolist(),
                        "angles_deg": list(spec.angles_deg),
                        "forward_resolution": config["other"],
                        "truth_forward_resolution": truth_config["other"],
                        "truth_nvx": len(truth_velocity),
                        "fit_nvx": len(velocity),
                        "shared_case": "../../case.npz",
                        "subspaces": "../../subspaces.npz",
                        "observations": "../observations.npy",
                        "case_diagnostics": case_info,
                    }
                    _write_json(run_dir / "provenance.json", metadata)
                    run_id = None
                    if mlflow_experiment:
                        run_id = _log_mlflow(
                            mlflow_experiment,
                            f"iss140-{truth_name}-{model}-seed{seed}-{photons:g}",
                            run_dir,
                            shared,
                            metadata,
                            result,
                            history,
                        )
                        _write_json(run_dir / "mlflow.json", {"run_id": run_id})
                    records.append(
                        {
                            **identity,
                            **result,
                            "directory": str(run_dir.relative_to(output)),
                            "mlflow_run_id": run_id,
                        }
                    )
                    _write_json(output / "runs.json", records)
                    print(
                        f"Completed {model}: train={result['best']['train_whitened_rms']:.4g}, "
                        f"heldout={result['best']['heldout_whitened_rms']:.4g}, best step={result['best_step']}",
                        flush=True,
                    )
                # Each seed closes over a different data set. Release its compiled
                # fit graphs so a many-seed campaign does not accumulate all of them.
                jax.clear_caches()
    summary = summarize_runs(records, spec)
    _write_json(output / "summary.json", summary)
    if mlflow_experiment:
        import mlflow

        with mlflow.start_run(run_name="iss140-summary") as run:
            mlflow.set_tags(
                {
                    "benchmark": "ISS140",
                    "kind": "seed-summary",
                    "git_sha": provenance["git_sha"],
                }
            )
            for filename in ("summary.json", "runs.json", "benchmark.json"):
                mlflow.log_artifact(str(output / filename))
            _write_json(output / "mlflow-summary.json", {"run_id": run.info.run_id})
    return records, summary


def run_benchmark(spec, output, *, mlflow_experiment=None):
    """Run a complete comparison, preserving completed fits and explicit status."""
    spec.validate()
    jax.config.update("jax_enable_x64", True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    status = {"state": "running", "started_at_utc": started}
    _write_json(output / "status.json", status)
    try:
        result = _run_benchmark(spec, output, mlflow_experiment=mlflow_experiment)
    except BaseException as error:
        _write_json(
            output / "status.json",
            {
                **status,
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    _write_json(
        output / "status.json",
        {
            **status,
            "state": "complete",
            "completed_fits": len(result[0]),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON object overriding the preset BenchmarkSpec fields",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new result directory (must not exist)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        help="optional MLflow experiment; uses configured tracking URI",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="explicitly allow the full preset on CPU",
    )
    args = parser.parse_args()
    values = asdict(
        BenchmarkSpec.smoke() if args.preset == "smoke" else BenchmarkSpec()
    )
    if args.config:
        values.update(json.loads(args.config.read_text()))
    spec = BenchmarkSpec(**values)
    spec.validate()
    if (
        args.preset == "full"
        and not args.allow_cpu
        and not any(device.platform == "gpu" for device in jax.devices())
    ):
        parser.error(
            "the full benchmark requires a GPU; use --allow-cpu only for an intentional CPU run"
        )
    run_benchmark(spec, args.output, mlflow_experiment=args.mlflow_experiment)


if __name__ == "__main__":
    main()
