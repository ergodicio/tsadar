import tempfile
from pathlib import Path

import numpy as np
import pandas
import pytest
from jax import config

config.update("jax_enable_x64", True)
from jax import numpy as jnp

from tsadar.core.modules.distribution_functions.base import Numeric1V
from tsadar.core.modules.ts_params import ThomsonParams
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic

from tsverify.oracle import GaussianComponent as G
from tsverify.oracle import Ion, Plasma

from ..physics.harness import EPW_DETECTOR, IAW_DETECTOR, diagnostic_config

PLASMA = Plasma(
    ne=0.2,
    Te=0.6,
    ions=(Ion(A=1.0, Z=1.0, Ti=0.2, components=(G(0.5, -0.3, 1.0), G(0.5, 0.3, 1.0))),),
    electrons=(G(0.85, 0.0, 1.0), G(0.15, 0.0, 1.2)),
)


def _spectra(det, zero_tails=False):
    with tempfile.TemporaryDirectory() as folder:
        cfg = diagnostic_config(PLASMA, det, folder)
        p = cfg["parameters"]
        for block, cutoff in ((p["electron"]["fe"], 7.5), (p["ion-1"]["fi"], 6.0)):
            if zero_tails:
                df = pandas.read_csv(block["file"])
                df.loc[np.abs(df["v"]) > cutoff, "f"] = 0.0
                df.to_csv(block["file"], index=False)
        angles = {"sa": np.array([PLASMA.angle_deg]), "weights": np.ones((1, 1))}
        diag = ThomsonScatteringDiagnostic(cfg, angles)
        params = ThomsonParams(p, num_params=1, activate=False)
    n = det.pixels
    batch = {
        "e_data": jnp.ones((1, n)),
        "i_data": jnp.ones((1, n)),
        "noise_e": jnp.zeros((1, n)),
        "noise_i": jnp.zeros((1, n)),
        "e_amps": jnp.full(1, det.amplitude),
        "i_amps": jnp.full(1, det.amplitude),
    }
    channel = 0 if det.channel == "EPW" else 1
    return np.asarray(diag(params, batch)[channel]).ravel()


@pytest.mark.parametrize("det", [EPW_DETECTOR, IAW_DETECTOR], ids=["EPW", "IAW"])
def test_numeric_distributions_with_zero_tails_give_finite_spectra(det):
    with pytest.warns(UserWarning, match="raised to exp"):
        truncated = _spectra(det, zero_tails=True)
    full = _spectra(det)
    assert np.all(np.isfinite(truncated))
    assert np.max(np.abs(truncated - full)) < 1e-2 * np.max(np.abs(full))


def test_a_negative_numeric_distribution_is_rejected(tmp_path):
    v = np.linspace(-8.0, 8.0, 65)
    f = np.exp(-(v**2) / 2) - 0.01
    path = Path(tmp_path) / "fe.csv"
    pandas.DataFrame({"v": v, "f": f}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cannot be negative"):
        Numeric1V({"file": str(path), "nvx": v.size})
