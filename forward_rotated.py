"""Forward-mode run of an ARTS deck with the 2D EDF rotated, as a multiplexed fit models its second shot.

The anisotropic synthetic pairs are two images of one EDF, the second rotated by data.shot_rot. A fit evaluates the
second shot with ``ThomsonParams.electron.dist_rot = shot_rot``; this runs forward mode the same way, so the second
image and its saved truth EDF (``csv/learned_dist.csv``) are that exact rotation of the first.

Usage: python forward_rotated.py --cfg configs/arts-2d-recovery/truth_biDLM_rot90 [--rot 90]
"""

import argparse, os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
from jax import config

config.update("jax_enable_x64", True)

import equinox as eqx
from tsadar.forward import calc_series
from tsadar.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--rot", type=float, default=90.0, help="EDF rotation in degrees (counter-clockwise)")
    args = parser.parse_args()

    base = calc_series.ThomsonParams

    def rotated(*a, **k):
        return eqx.tree_at(lambda t: t.electron.dist_rot, base(*a, **k), args.rot)

    calc_series.ThomsonParams = rotated
    run(args.cfg, mode="forward")


if __name__ == "__main__":
    main()
