import tempfile

import numpy as np

from tsadar.utils.plotting.plotters import save_sigmas_params


def test_save_sigmas_params_preserves_column_order_across_species():
    # regression test: the sigma column index must run across all species combined rather than
    # restarting at 0 for each species
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


def test_save_sigmas_params_skips_uncounted_distribution_function_entry():
    # regression test: the electron "fe" entry is always present in all_params but is not counted in
    # num_params, so it has no sigma column and must be skipped
    num_lineouts = 4
    all_params = {
        "electron": {"Te": np.zeros(num_lineouts), "ne": np.zeros(num_lineouts), "fe": np.zeros((num_lineouts, 64))},
        "general": {"amp1": np.zeros(num_lineouts)},
    }
    # only Te, ne, amp1 are active/counted -- "fe" has no corresponding column
    sigmas = np.arange(num_lineouts * 3, dtype=float).reshape(num_lineouts, 3)

    config = {"data": {"lineouts": {"pixelE": slice(0, num_lineouts)}}}
    all_axes = {"x_label": "wavelength", "epw_x": np.arange(10)}

    with tempfile.TemporaryDirectory() as td:
        sigmas_ds = save_sigmas_params(config, all_params, sigmas, all_axes, td, filename="sigmas_test.nc")

    assert "fe_electron" not in sigmas_ds
    np.testing.assert_allclose(sigmas_ds["Te_electron"].values, sigmas[:, 0])
    np.testing.assert_allclose(sigmas_ds["ne_electron"].values, sigmas[:, 1])
    np.testing.assert_allclose(sigmas_ds["amp1_general"].values, sigmas[:, 2])
