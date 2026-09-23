import tempfile

import numpy as np

from tsadar.utils.plotting.plotters import save_sigmas_params


def test_save_sigmas_params_preserves_column_order_across_species():
    # Regression test: save_sigmas_params used to build each species' column index via a fresh
    # enumerate() per species, resetting to 0 every time -- so every species after the first silently
    # pulled an earlier species' sigma column instead of its own (e.g. "general"'s amp1 landing on
    # "electron"'s Te column) whenever more than one species had active parameters, the normal case.
    # sigmas' columns are ordered species-major across all_params combined (matching
    # postprocess.laplace.get_sigmas/postprocess.mcmc._active_param_keys), so the fix must use one
    # running index across that whole combined order, not one reset per species.
    num_lineouts = 4
    all_params = {
        "electron": {"Te": np.zeros(num_lineouts), "ne": np.zeros(num_lineouts)},
        "general": {"amp1": np.zeros(num_lineouts), "amp2": np.zeros(num_lineouts)},
        "ion-1": {"Z": np.zeros(num_lineouts)},
    }
    # columns match all_params' own combined order: Te, ne, amp1, amp2, Z
    sigmas = np.arange(num_lineouts * 5, dtype=float).reshape(num_lineouts, 5)

    config = {"data": {"lineouts": {"pixelE": slice(0, num_lineouts)}}}
    all_axes = {"x_label": "wavelength", "epw_x": np.arange(10)}

    with tempfile.TemporaryDirectory() as td:
        sigmas_ds = save_sigmas_params(config, all_params, sigmas, all_axes, td, filename="sigmas_test.nc")

    np.testing.assert_allclose(sigmas_ds["Te_electron"].values, sigmas[:, 0])
    np.testing.assert_allclose(sigmas_ds["ne_electron"].values, sigmas[:, 1])
    np.testing.assert_allclose(sigmas_ds["amp1_general"].values, sigmas[:, 2])
    np.testing.assert_allclose(sigmas_ds["amp2_general"].values, sigmas[:, 3])
    np.testing.assert_allclose(sigmas_ds["Z_ion-1"].values, sigmas[:, 4])
