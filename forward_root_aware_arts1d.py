"""Full-image forward calculation of a finished 1D-fe ARTS fit, with and without the root-aware quadrature.

Reloads a fit (merged decks + fitted_weights.eqx) and evaluates the electron spectrum over every angular
row of the detector, not only the fitted lineout window, through both model paths:

  legacy:     fixed wavelength grid, spectral blur, bin average
  root-aware: other.resonance_quadrature, detector-bin integration with the spectral IRF folded in

Whichever of the two the fit itself used is checked against the fit's own forward model in the window.
The decks and weights must come from a fit made on the current velocity grid.
"""

import argparse, os, copy, json

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import yaml, mlflow
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsadar.utils import misc
from tsadar.postprocess_runner import _reconstruct_fit_state
from tsadar.inverse.loops import build_angular_batch
from tsadar.core.instrument import irf
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic, _bin_average


def full_image(ts_diag, params, n_bins, ang_step):
    """ThomsonScatteringDiagnostic.__call__ for angular data without the lineout slice or the noise."""
    binned = ts_diag.model.electron_spectrum_is_detector_binned
    modlE, _, lam_axis, _ = ts_diag.model(params)
    lam_axis, theory = irf.add_ATS_IRF(ts_diag.ats_irf, lam_axis, modlE, params, apply_spectral_blur=not binned)
    if not binned:
        lam_step = round(theory.shape[1] / n_bins)
        theory = _bin_average(theory, lam_step, axis=1)
        lam_axis = _bin_average(lam_axis, lam_step, axis=0)
    theory = _bin_average(theory, ang_step, axis=0)
    lam0 = jnp.squeeze(jnp.asarray(params["general"]["lam"]))
    amp1 = jnp.squeeze(jnp.asarray(params["general"]["amp1"]))
    amp2 = jnp.squeeze(jnp.asarray(params["general"]["amp2"]))
    return jnp.where(lam_axis < lam0, amp1 * theory, amp2 * theory), lam_axis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="folder with the run's defaults.yaml and inputs.yaml")
    parser.add_argument("--weights", required=True, help="path to the run's fitted_weights.eqx")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--tracking", default=None, help="local mlflow folder, defaults to <out>/mlruns_local")
    parser.add_argument(
        "--eval_rows", nargs="*", default=[], help="extra binned-row ranges to score, as start:end (e.g. 2:100)"
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    decks = {}
    for name in ("defaults", "inputs"):
        with open(os.path.join(args.cfg, f"{name}.yaml"), "r") as fi:
            decks[name] = yaml.safe_load(fi)
    config = misc.merge_defaults_and_inputs(decks["defaults"], decks["inputs"])

    # keep this calculation out of the tracking server; the loader only needs some active run
    tracking = os.path.abspath(args.tracking or os.path.join(args.out, "mlruns_local"))
    mlflow.set_tracking_uri("file:///" + tracking.replace("\\", "/").lstrip("/"))
    with mlflow.start_run(run_name="full-image forward"):
        state = _reconstruct_fit_state(config, args.weights)
    cfg, loss_fn, sa = state.config, state.loss_fn, state.sa
    params = state.fitted_weights()
    start, end = cfg["data"]["lineouts"]["start"], cfg["data"]["lineouts"]["end"]
    ang_step = int(cfg["other"]["ang_res_unit"])
    n_rows, n_bins = state.all_data["e_data"].shape
    vx = np.squeeze(np.asarray(params["electron"]["v"]))
    print(f"prepared detector image: {n_rows} angle rows x {n_bins} wavelength bins; fit window rows {start}:{end}")
    print(f"velocity grid +/-{vx.max():.2f} vTe, {vx.size} points")

    fit_diag = loss_fn.ts_diag
    fit_is_root_aware = bool(fit_diag.model.electron_spectrum_is_detector_binned)
    other_cfg = copy.deepcopy(cfg)
    other_cfg["other"].setdefault("resonance_quadrature", {})["enabled"] = not fit_is_root_aware
    other_diag = ThomsonScatteringDiagnostic(other_cfg, sa)
    diags = {"root_aware" if fit_is_root_aware else "legacy": fit_diag}
    diags["legacy" if fit_is_root_aware else "root_aware"] = other_diag
    fit_name = "root_aware" if fit_is_root_aware else "legacy"
    print(f"the fit used the {fit_name} path")

    images = {name: full_image(diag, params, n_bins, ang_step) for name, diag in diags.items()}

    window_batch = build_angular_batch(cfg, state.all_data)
    reference, _, _, _ = fit_diag(state.fitted_weights, window_batch)
    reference = reference - jnp.asarray(window_batch["noise_e"])
    check = float(jnp.max(jnp.abs(images[fit_name][0][start:end] - reference)) / jnp.max(jnp.abs(reference)))
    print(f"self-check, full image vs the fit's own forward model in the window: max rel diff {check:.2e}")

    _, _, _, diagnostics = diags["root_aware"].model.detector_integrated_electron_spectrum(params)
    root_count = np.asarray(diagnostics.root_count)
    flags = {k: int(np.asarray(getattr(diagnostics, k)).sum()) for k in ("root_overflow", "nonfinite", "zero_width")}
    print(f"roots per angle: min {int(root_count.min())} max {int(root_count.max())}; angles flagged {flags}")

    # compare with data: profile the detector gains exactly as the fit objective does
    full_cfg = copy.deepcopy({"data": cfg["data"]})
    full_cfg["data"]["lineouts"]["start"], full_cfg["data"]["lineouts"]["end"] = 0, n_rows
    full_batch = build_angular_batch(full_cfg, state.all_data)
    background = jnp.broadcast_to(jnp.asarray(full_batch["noise_e"]), (n_rows, n_bins))

    def rows_only(batch, a, b):
        return {k: (v[a:b] if getattr(v, "shape", ())[:1] == (n_rows,) else v) for k, v in batch.items()}

    def window_only(batch):
        return rows_only(batch, start, end)

    eval_ranges = [tuple(int(x) for x in item.split(":")) for item in args.eval_rows]

    summary = {
        "fit_path": fit_name,
        "self_check_max_rel_diff": check,
        "fit_window_rows": [int(start), int(end)],
        "velocity_grid": [float(vx.max()), int(vx.size)],
        "roots_per_angle_min_max": [int(root_count.min()), int(root_count.max())],
        "angles_flagged": flags,
    }
    fitted = {}
    for name, (theory, lam_axis) in images.items():
        total_theory = theory + background
        full_obj, _, fitted_full, full_terms, _ = loss_fn._angular_data_objective(full_batch, total_theory, lam_axis)
        win_obj, _, _, win_terms, _ = loss_fn._angular_data_objective(
            window_only(full_batch), total_theory[start:end], lam_axis
        )
        fitted[name] = np.where(np.isfinite(np.asarray(theory)), np.asarray(fitted_full), np.nan)
        extra = {}
        for a, b in eval_ranges:
            obj, _, _, terms, _ = loss_fn._angular_data_objective(rows_only(full_batch, a, b), total_theory[a:b], lam_axis)
            extra[f"{a}:{b}"] = {"objective": float(obj)} | {k: float(v) for k, v in terms.items()}
        summary[name] = {
            "window_objective": float(win_obj),
            "window_data_term": float(win_terms["data"]),
            "full_image_objective": float(full_obj),
            "full_image_data_term": float(full_terms["data"]),
            "rows": extra,
        }
    relative = np.abs(np.asarray(images["root_aware"][0]) - np.asarray(images["legacy"][0])) / np.nanmax(
        np.abs(np.asarray(images["legacy"][0]))
    )
    summary["root_aware_vs_legacy_max_diff_over_legacy_max"] = float(np.nanmax(relative))
    summary["root_aware_vs_legacy_same_in_window"] = float(np.nanmax(relative[start:end]))
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, "summary.json"), "w") as fo:
        json.dump(summary, fo, indent=2)

    angle_axis = np.squeeze(np.asarray(state.all_axes["epw_x"]))
    lam_out = np.squeeze(np.asarray(images["root_aware"][1]))
    data = np.asarray(full_batch["e_data"])
    arrays = {
        "data": data,
        "background": np.asarray(background),
        "legacy_raw": np.asarray(images["legacy"][0]),
        "root_aware_raw": np.asarray(images["root_aware"][0]),
        "legacy_fit": fitted["legacy"],
        "root_aware_fit": fitted["root_aware"],
    }
    coords = {"angle": angle_axis, "wavelength": lam_out}
    ds = xr.Dataset({k: xr.DataArray(v, dims=("angle", "wavelength"), coords=coords) for k, v in arrays.items()})
    ds.attrs["fit_window_rows"] = [int(start), int(end)]
    ds.attrs["fit_path"] = fit_name
    ds.to_netcdf(os.path.join(args.out, "full_image_forward.nc"))

    vmax = np.nanpercentile(data, 99.5)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True, sharey=True)
    panels = (
        ("Data", data),
        ("Root-aware forward", fitted["root_aware"]),
        ("Legacy forward", fitted["legacy"]),
        ("Root-aware - legacy", fitted["root_aware"] - fitted["legacy"]),
    )
    for a, (title, image) in zip(ax, panels):
        diff = title.startswith("Root-aware -")
        lim = np.nanmax(np.abs(image)) if diff else vmax
        pc = a.pcolormesh(
            angle_axis, lam_out, image.T, shading="nearest", cmap="RdBu_r" if diff else "viridis",
            vmin=-lim if diff else 0, vmax=lim,
        )
        for row in (start, end - 1):
            a.axvline(angle_axis[row], color="k" if diff else "w", ls="--", lw=1)
        a.set_title(title)
        a.set_xlabel("Scattering angle (degrees)")
        fig.colorbar(pc, ax=a)
    ax[0].set_ylabel("Wavelength (nm)")
    fig.savefig(os.path.join(args.out, "full_image.png"), dpi=130)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
