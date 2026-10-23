"""Compare ARTS recovery fits of an anisotropic synthetic pair against its 2D truth EDF.

Each fit set is read from ``<root>/<prefix>_<model>/`` (``csv/learned_parameters.csv``, ``csv/learned_dist.csv`` or,
for DLM fits, the decks). A fit's velocity axis is in units of its own fitted vTe, so it is mapped to the truth's
physical scale with v * sqrt(Te_fit / Te_truth) before comparison; this way a fit that trades Te against EDF width
is judged on the physical distribution. Reported per fit:

  Te, ne against truth;
  in-plane moments in truth units: flow (ux, uy), temperature tensor (Txx, Tyy, Txy), anisotropy Txx/Tyy,
  principal-axis angle, against the truth's;
  2D shape error: rms log10(f / f_truth) where f_truth > 1e-4 max (2D fits only);
  marginal errors: rms log10 of the vx and vy marginals against the truth's, for |v| < 3 (1D fits are compared with
  both marginals, since a 1D fit models one isotropic projection).

Writes ``summary.csv``/``summary.md`` and figures (2D contours and marginal overlays) to --out.
"""

import argparse, os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_arts2d_recovery import dlm_fe

MODELS = ("dlm", "arb1D", "arb2D", "arbSH", "nnSH", "mySH")
TRUTH = {"Te": 1.0, "ne": 0.4}


def load_2d(path):
    d = pd.read_csv(path, index_col=0)
    v = d.index.to_numpy(float)
    return v, d.to_numpy(float)  # [vy, vx]


def moments(v, f):
    dv = v[1] - v[0]
    vx, vy = np.meshgrid(v, v)
    f = np.clip(f, 0, None)
    f = f / (f.sum() * dv * dv)
    ux, uy = (f * vx).sum() * dv * dv, (f * vy).sum() * dv * dv
    txx = ((vx - ux) ** 2 * f).sum() * dv * dv
    tyy = ((vy - uy) ** 2 * f).sum() * dv * dv
    txy = ((vx - ux) * (vy - uy) * f).sum() * dv * dv
    return {"ux": ux, "uy": uy, "Txx": txx, "Tyy": tyy, "Txy": txy, "Txx/Tyy": txx / tyy,
            "axis_deg": 0.5 * np.degrees(np.arctan2(2 * txy, txx - tyy))}


def regrid_2d(v_src, f_src, v_dst):
    """Bilinear interpolation of f[vy, vx] onto the v_dst grid (zero outside)."""
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator((v_src, v_src), f_src, bounds_error=False, fill_value=0.0)
    vx, vy = np.meshgrid(v_dst, v_dst)
    return interp(np.stack([vy.ravel(), vx.ravel()], axis=1)).reshape(vx.shape)


def marginals(v, f):
    dv = v[1] - v[0]
    f = np.clip(f, 0, None)
    f = f / (f.sum() * dv * dv)
    return f.sum(0) * dv, f.sum(1) * dv  # f(vx), f(vy)


def log_l2(v, a, b, vmax=3.0):
    mask = (np.abs(v) < vmax) & (a > 0) & (b > 0)
    return float(np.sqrt(np.mean((np.log10(a[mask]) - np.log10(b[mask])) ** 2)))


def load_fit(fit_dir, truth_v):
    """Returns params, and either a 2D fe on the truth grid (physical units) or a 1D fe on the truth axis."""
    p = pd.read_csv(os.path.join(fit_dir, "csv", "learned_parameters.csv"), index_col=0).iloc[0]
    params = {k: float(p[f"{k}_electron"]) for k in ("Te", "ne", "m") if f"{k}_electron" in p}
    scale = np.sqrt(params["Te"] / TRUTH["Te"])
    dist = os.path.join(fit_dir, "csv", "learned_dist.csv")
    if not os.path.exists(dist):
        v, f = dlm_fe(fit_dir, params["m"])
        return params, None, np.interp(truth_v, v * scale, f / scale, left=0, right=0)
    d = pd.read_csv(dist, index_col=0)
    if list(d.columns) == ["fe", "vx"]:
        v, f = d["vx"].to_numpy(), d["fe"].to_numpy()
        return params, None, np.interp(truth_v, v * scale, f / scale, left=0, right=0)
    v, f = d.index.to_numpy(float), d.to_numpy(float)
    return params, regrid_2d(v * scale, f / scale**2, truth_v), None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="label, e.g. biDLM")
    ap.add_argument("--fits", nargs="+", required=True, help="label=root:prefix:truth_csv, e.g. new=C:/tmp/anifits:B:C:/tmp/t.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, shown = [], {}
    for item in args.fits:
        label, rest = item.split("=", 1)
        root, prefix, truth_csv = rest.split("|")
        tv, tf = load_2d(truth_csv)
        tm = moments(tv, tf)
        tmx, tmy = marginals(tv, tf)
        rows.append({"dataset": args.dataset, "fit": label, "model": "TRUTH", **{k: TRUTH.get(k, np.nan) for k in ("Te", "ne")},
                     **{f"{k}": v for k, v in tm.items()}})
        for model in MODELS:
            fit_dir = os.path.join(root, f"{prefix}_{model}")
            if not os.path.isdir(os.path.join(fit_dir, "csv")):
                print("missing", fit_dir)
                continue
            params, f2, f1 = load_fit(fit_dir, tv)
            row = {"dataset": args.dataset, "fit": label, "model": model, "Te": params["Te"], "ne": params["ne"],
                   "Te_err_%": 100 * (params["Te"] / TRUTH["Te"] - 1), "ne_err_%": 100 * (params["ne"] / TRUTH["ne"] - 1)}
            if f2 is not None:
                m = moments(tv, f2)
                row |= m
                row["dT_aniso"] = m["Txx/Tyy"] - tm["Txx/Tyy"]
                row["d_axis_deg"] = (m["axis_deg"] - tm["axis_deg"] + 90) % 180 - 90
                row["d_flow"] = float(np.hypot(m["ux"] - tm["ux"], m["uy"] - tm["uy"]))
                big = tf > 1e-4 * tf.max()
                ok = big & (f2 > 0)
                row["log10_L2_2D"] = float(np.sqrt(np.mean((np.log10(f2[ok]) - np.log10(tf[ok])) ** 2)))
                row["frac_2D_nonpositive"] = float(np.mean(f2[big] <= 0))
                fx, fy = marginals(tv, f2)
                shown[(label, model)] = f2
            else:
                dv = tv[1] - tv[0]
                f1 = f1 / (f1.sum() * dv)
                fx = fy = f1
                row["Txx"] = row["Tyy"] = float((tv**2 * f1).sum() * dv)
            row["log10_L2_vx_marginal"] = log_l2(tv, fx, tmx)
            row["log10_L2_vy_marginal"] = log_l2(tv, fy, tmy)
            shown[(label, model, "marg")] = (fx, fy)
            rows.append(row)
        shown[(label, "truth")] = (tv, tf, tmx, tmy)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(df.to_markdown(index=False, floatfmt=".4g"))
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        cols = ["fit", "model", "Te_err_%", "ne_err_%", "ux", "uy", "Txx", "Tyy", "Txy", "Txx/Tyy", "axis_deg",
                "log10_L2_2D", "log10_L2_vx_marginal", "log10_L2_vy_marginal"]
        print(df[[c for c in cols if c in df]].round(4).to_string(index=False))

    labels = [item.split("=", 1)[0] for item in args.fits]
    # 2D contours: truth and each 2D model, per fit set
    for label in labels:
        tv, tf, _, _ = shown[(label, "truth")]
        models2d = [m for m in MODELS if (label, m) in shown]
        fig, axes = plt.subplots(1, len(models2d) + 1, figsize=(3.1 * (len(models2d) + 1), 3.3), layout="constrained", squeeze=False)
        axes = axes[0]
        levels = np.linspace(-5, 0, 11)
        for ax, (name, f) in zip(axes, [("truth", tf)] + [(m, shown[(label, m)]) for m in models2d]):
            ax.contourf(tv, tv, np.log10(np.clip(f / tf.max(), 1e-12, None)), levels=levels, cmap="viridis", extend="min")
            ax.contour(tv, tv, np.log10(np.clip(tf / tf.max(), 1e-12, None)), levels=levels[::2], colors="w", linewidths=0.5)
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ax.set_title(f"{name}", fontsize=10)
            ax.set_xlabel("vx / vTe")
        axes[0].set_ylabel("vy / vTe")
        fig.suptitle(f"{args.dataset} ({label}): log10 fe / max(truth); white: truth contours", fontsize=10)
        fig.savefig(os.path.join(args.out, f"fe2d_{label}.png"), dpi=130)
        plt.close(fig)
    # marginals, log scale, every model; all fit sets overlaid
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.2), sharex=True, sharey=True, layout="constrained")
    for j, model in enumerate(MODELS):
        for i, comp in enumerate(("vx", "vy")):
            ax = axes[i, j]
            for label in labels:
                tv, tf, tmx, tmy = shown[(label, "truth")]
                ax.semilogy(tv, tmx if i == 0 else tmy, "k-", lw=2 if label == labels[-1] else 1,
                            alpha=1 if label == labels[-1] else 0.4, label=f"truth ({label})")
                if (label, model, "marg") in shown:
                    ax.semilogy(tv, shown[(label, model, "marg")][i], "-", lw=1.2, label=label)
            ax.set_ylim(1e-6, 1); ax.set_xlim(-4.5, 4.5)
            ax.set_title(f"{model}: {comp} marginal", fontsize=10)
            if i == 1: ax.set_xlabel(f"{comp} / vTe (truth units)")
            if j == 0: ax.set_ylabel("marginal fe")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(os.path.join(args.out, "marginals.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
