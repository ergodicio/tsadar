"""Compare ARTS recovery fits of the synthetic images against each other and against the ground truth.

For every dataset x fe model, each fit set (e.g. the 2025 fits and today's baseline) is read from
``<root>/<key>_<model>/`` holding the run's ``csv/learned_parameters.csv``, ``csv/learned_dist.csv``
(absent for DLM fits) and ``inputs.yaml``/``defaults.yaml``. The truth EDF is the generating run's
``csv/learned_dist.csv``.

Every EDF is reduced to 1D marginals (a 2D fe ``[vy, vx]`` is integrated over vy and, separately, over
vx; the truth is isotropic, so the difference between those two marginals is spurious anisotropy),
normalized to unit area and rescaled to <v^2> = 1 so fits on different velocity grids (+-6 vTe in 2025,
+-8 now) and with different Te compare by shape. Reported per fit:

  Te, ne (and m for DLM) against truth, and Te_eff = Te * <v^2> (the width/temperature trade-off),
  the 4th standardized moment <v^4>/<v^2>^2 (3 for a Maxwellian),
  median fe / truth in |v| bands 0-1, 1-2, 2-3, 3-4 (shape-normalized units),
  L2 of log10 fe against truth for |v| < 3,
  for 2D fe: L2 of log10 between the vx and vy marginals, and the in-plane moments of edf_moments.

Writes ``summary.csv``, ``summary.md`` and fe overlay figures (linear/log, against v and |v|) to --out.
"""

import argparse, os

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = {"M": ("Maxwellian", "M_H", 2.0), "S": ("SG4", "SG4_H", 4.0)}
MODELS = ("dlm", "arb1D", "arb2D", "arbSH", "nnSH", "mySH")
TRUTH = {"Te": 1.0, "ne": 0.4}
BANDS = ((0, 1), (1, 2), (2, 3), (3, 4))


def load_truth(truth_root, name):
    dist = pd.read_csv(os.path.join(truth_root, f"{name}_learned_dist.csv"), index_col=0)
    return dist["vx"].to_numpy(), dist["fe"].to_numpy()


def dlm_fe(fit_dir, m):
    """Tabulated DLM projection at the fitted m, on the fit's own velocity grid."""
    from tsadar.utils import misc
    from tsadar.core.modules.distribution_functions.base import DLM1V

    decks = {k: yaml.safe_load(open(os.path.join(fit_dir, f"{k}.yaml"))) for k in ("defaults", "inputs")}
    dist_cfg = misc.merge_defaults_and_inputs(decks["defaults"], decks["inputs"])["parameters"]["electron"]["fe"]
    dist_cfg["params"]["m"]["val"] = float(m)
    dist = DLM1V(dist_cfg, activate=False)
    return np.asarray(dist.vx), np.asarray(dist())


def load_fit(fit_dir):
    """Returns (params dict, 1D marginals {label: (v, f)}, 2D fe or None, v of the 2D fe)."""
    params = pd.read_csv(os.path.join(fit_dir, "csv", "learned_parameters.csv"), index_col=0).iloc[0]
    out = {k: float(params[f"{k}_electron"]) for k in ("Te", "ne", "m") if f"{k}_electron" in params}
    dist_path = os.path.join(fit_dir, "csv", "learned_dist.csv")
    if not os.path.exists(dist_path):
        v, f = dlm_fe(fit_dir, out["m"])
        return out, {"vx": (v, f)}, None, None
    dist = pd.read_csv(dist_path, index_col=0)
    if list(dist.columns) == ["fe", "vx"]:
        return out, {"vx": (dist["vx"].to_numpy(), dist["fe"].to_numpy())}, None, None
    v = dist.index.to_numpy(dtype=float)
    fe2 = dist.to_numpy(dtype=float)  # [vy, vx]
    dv = v[1] - v[0]
    return out, {"vx": (v, fe2.sum(0) * dv), "vy": (v, fe2.sum(1) * dv)}, fe2, v


def standardize(v, f):
    """Unit area, zero mean, <v^2> = 1. Returns (v_s, f_s, <v^2>, 4th standardized moment)."""
    f = np.clip(np.asarray(f, float), 0, None)
    dv = v[1] - v[0]
    f = f / (f.sum() * dv)
    mean = (v * f).sum() * dv
    var = ((v - mean) ** 2 * f).sum() * dv
    kurt = ((v - mean) ** 4 * f).sum() * dv / var**2
    s = np.sqrt(var)
    return (v - mean) / s, f * s, var, kurt


def shape_errors(v, f, vt, ft):
    """Band ratios and log-L2 of a standardized fe (v, f) against the standardized truth (vt, ft)."""
    truth_on_v = np.interp(v, vt, ft, left=np.nan, right=np.nan)
    ratio = f / truth_on_v
    bands = {f"ratio_{lo}-{hi}": float(np.nanmedian(ratio[(np.abs(v) >= lo) & (np.abs(v) < hi)])) for lo, hi in BANDS}
    core = (np.abs(v) < 3) & (f > 0) & (truth_on_v > 0)
    l2 = float(np.sqrt(np.mean((np.log10(f[core]) - np.log10(truth_on_v[core])) ** 2)))
    return bands | {"log10_L2_|v|<3": l2}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fits", nargs="+", required=True, help="label=root pairs, e.g. old=C:/tmp/oldfits new=C:/tmp/newfits")
    parser.add_argument("--truth-root", required=True, help="directory with <M_H|SG4_H>_learned_dist.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    fit_sets = dict(item.split("=", 1) for item in args.fits)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for key, (dataset, truth_name, truth_m) in DATASETS.items():
        vt_raw, ft_raw = load_truth(args.truth_root, truth_name)
        vt, ft, truth_var, truth_kurt = standardize(vt_raw, ft_raw)
        curves = {}
        for model in MODELS:
            for label, root in fit_sets.items():
                fit_dir = os.path.join(root, f"{key}_{model}")
                if not os.path.isdir(os.path.join(fit_dir, "csv")):
                    print(f"missing: {fit_dir}")
                    continue
                params, marginals, fe2, v2 = load_fit(fit_dir)
                row = {"dataset": dataset, "model": model, "fit": label}
                row |= {k: params.get(k, np.nan) for k in ("Te", "ne", "m")}
                row["Te_err_%"] = 100 * (row["Te"] / TRUTH["Te"] - 1)
                row["ne_err_%"] = 100 * (row["ne"] / TRUTH["ne"] - 1)
                v, f, var, kurt = standardize(*marginals["vx"])
                row["<v^2>"] = var
                row["Te_eff"] = row["Te"] * var
                row["Te_eff_err_%"] = 100 * (row["Te_eff"] / (TRUTH["Te"] * truth_var) - 1)
                row["kurtosis"] = kurt
                row |= shape_errors(v, f, vt, ft)
                curves[(model, label)] = (v, f)
                if "vy" in marginals:
                    vy, fy, var_y, _ = standardize(*marginals["vy"])
                    fx_on_y = np.interp(vy, v, f)
                    core = (np.abs(vy) < 3) & (fy > 0) & (fx_on_y > 0)
                    row["anisotropy_log10_L2"] = float(np.sqrt(np.mean((np.log10(fy[core]) - np.log10(fx_on_y[core])) ** 2)))
                    row["<vy^2>/<vx^2>"] = var_y / var
                    from tsadar.inverse.recovery_benchmark import edf_moments

                    mom = edf_moments(np.clip(fe2, 0, None), v2, v2)
                    row |= {f"2D_{k}": mom[k] for k in ("flow_vx", "flow_vy", "temperature_in_plane")}
                rows.append(row)

        # figures: vx marginal, standardized, every model in its own panel
        for scale in ("linear", "log"):
            for absv in (False, True):
                fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
                for ax, model in zip(axes.ravel(), MODELS):
                    x = np.abs(vt) if absv else vt
                    ax.plot(x, ft, "k-", lw=2.5, label=f"truth (DLM m={truth_m:g})")
                    for (mdl, label), (v, f) in curves.items():
                        if mdl == model:
                            ax.plot(np.abs(v) if absv else v, f, ".-" if absv else "-", ms=2, lw=1.2, label=label)
                    ax.set_title(f"{dataset} {model}")
                    ax.set_yscale(scale)
                    if scale == "log":
                        ax.set_ylim(1e-6, 1)
                    ax.set_xlim(0 if absv else -4.5, 4.5)
                    ax.legend(fontsize=8)
                for ax in axes[-1]:
                    ax.set_xlabel("|v| / sqrt(<v^2>)" if absv else "v / sqrt(<v^2>)")
                for ax in axes[:, 0]:
                    ax.set_ylabel("fe (unit area, <v^2> = 1)")
                fig.tight_layout()
                fig.savefig(os.path.join(args.out, f"fe_{key}_{scale}{'_absv' if absv else ''}.png"), dpi=130)
                plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(summary.to_markdown(index=False, floatfmt=".4g"))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
