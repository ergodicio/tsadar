import copy
import warnings

import pytest
import yaml
from flatten_dict import flatten, unflatten

from tsadar.data.lineouts import spectral_smoothing_width
import mlflow

from tsadar.core.modules.ts_params import ThomsonParams
from tsadar.inverse import fitter
from tsadar.inverse.fitter import OMEGA_TS_GAIN, _check_ion_locks_, _validate_inputs_, reject_numeric_distributions
from tsadar.postprocess_runner import _reconstruct_fit_state


@pytest.fixture(scope="module")
def base_config():
    with open("tests/configs/time_test_defaults.yaml") as fi:
        flat = flatten(yaml.safe_load(fi))
    with open("tests/configs/time_test_inputs.yaml") as fi:
        flat.update(flatten(yaml.safe_load(fi)))
    return unflatten(flat)


def test_spectral_smoothing_defaults_to_the_square_kernel(base_config):
    cfg = _validate_inputs_(copy.deepcopy(base_config))
    assert cfg["data"]["spectral_smoothing"] == "default"
    assert spectral_smoothing_width(cfg) == 2 * cfg["data"]["dpixel"] + 1

    cfg["data"]["spectral_smoothing"] = 1
    assert spectral_smoothing_width(cfg) == 1


def test_a_config_without_spectral_smoothing_warns_about_the_0_4_0_lineout_change(base_config):
    cfg = copy.deepcopy(base_config)
    del cfg["data"]["spectral_smoothing"]
    with pytest.warns(UserWarning, match="predates tsadar 0.4.0"):
        cfg = _validate_inputs_(cfg)
    assert cfg["data"]["spectral_smoothing"] == "default"


@pytest.mark.parametrize("value", [0, 2, -3, "wide", 1.5, True])
def test_spectral_smoothing_must_be_default_or_a_positive_odd_integer(base_config, value):
    cfg = copy.deepcopy(base_config)
    cfg["data"]["spectral_smoothing"] = value
    with pytest.raises(ValueError, match="spectral_smoothing"):
        _validate_inputs_(cfg)


def test_covar_forces_unsmoothed_unsubtracted_data_and_checks_the_gain(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["optimizer"]["loss_method"] = "covar"
    cfg["data"]["background"]["bg_subtract"] = True
    cfg["other"]["detector_gain"] = 1
    with pytest.warns(UserWarning) as record:
        cfg = _validate_inputs_(cfg)
    messages = " ".join(str(w.message) for w in record)
    assert "requires unsmoothed data" in messages
    assert f"gain of {OMEGA_TS_GAIN}" in messages
    assert cfg["data"]["spectral_smoothing"] == 1
    assert cfg["data"]["background"]["bg_subtract"] is False


def test_deprecated_gain_keys_are_renamed_with_a_warning(base_config):
    cfg = copy.deepcopy(base_config)
    other = cfg["other"]
    del other["detector_gain"]
    other["gain"] = 150
    other["calibration_uncertainty"] = {"num_draws": 3, "gain_sigma": 0.05}
    # the old LPI keys sit beside the new-style defaults, as when an old inputs deck meets new defaults:
    # the new keys win, and only gain_cap, which has no new-style counterpart here, is converted
    other.update({"include_gains": True, "gain_cap": 50, "Ipump_14": 2.0, "beam_diam_um": 70})
    with pytest.warns(FutureWarning, match="0.5.0") as record:
        cfg = _validate_inputs_(cfg)
    message = " ".join(str(w.message) for w in record if w.category is FutureWarning)
    assert "other.include_gains=True (using other.LPI_gain.include_LPI_gains=False)" in message
    assert "other.gain -> other.detector_gain" in message

    other = cfg["other"]
    assert other["detector_gain"] == 150
    assert other["calibration_uncertainty"] == {"num_draws": 3, "detector_gain_sigma": 0.05}
    assert other["LPI_gain"] == {"include_LPI_gains": False, "LPI_gain_cap": 50, "Ipump_14": 0.0, "beam_diam_um": 100}
    assert not {"gain", "include_gains", "gain_cap", "Ipump_14", "beam_diam_um"} & set(other)


def test_current_gain_keys_do_not_warn(base_config):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _validate_inputs_(copy.deepcopy(base_config))


def test_calc_sigmas_with_smoothed_data_warns(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["other"]["calc_sigmas"] = True
    with pytest.warns(UserWarning, match="uncertainties are underestimated"):
        _validate_inputs_(cfg)


NUMERIC = {"dim": 1, "type": "numeric", "file": "fe.csv", "nvx": 513}


def _with_numeric(base_config, species):
    cfg = copy.deepcopy(base_config)
    if species == "electron":
        cfg["parameters"]["electron"]["fe"] = dict(NUMERIC)
    else:
        cfg["parameters"][species]["fi"] = dict(NUMERIC)
    return cfg


def test_a_numeric_electron_distribution_passes_validation_for_forward_use(base_config):
    _validate_inputs_(_with_numeric(base_config, "electron"))


@pytest.mark.parametrize("species", ["electron", "ion-1"])
def test_fits_and_replays_reject_numeric_distributions(base_config, species):
    cfg = _with_numeric(base_config, species)
    with pytest.raises(ValueError, match="forward mode only"):
        reject_numeric_distributions(cfg)
    with pytest.raises(ValueError, match="forward mode only"):
        _reconstruct_fit_state(copy.deepcopy(cfg), "unused.eqx")
    with mlflow.start_run(), pytest.raises(ValueError, match="forward mode only"):
        fitter.fit(copy.deepcopy(cfg))


@pytest.mark.parametrize("fe_type", ["dlm", "arbitrary", "arbitrary-nn", "mixture"])
def test_other_distribution_types_are_not_rejected(base_config, fe_type):
    cfg = copy.deepcopy(base_config)
    cfg["parameters"]["electron"]["fe"]["type"] = fe_type
    reject_numeric_distributions(cfg)


def _three_ions(base_config, ti_same=(False, False, False), va_same=(False, False, False)):
    params = copy.deepcopy(base_config["parameters"])
    for i, (Ti, Va) in enumerate([(0.2, -1.0), (0.5, 1.0), (1.5, 3.0)], start=1):
        params[f"ion-{i}"] = copy.deepcopy(base_config["parameters"]["ion-1"])
        params[f"ion-{i}"]["Ti"].update(val=Ti, same=ti_same[i - 1])
        params[f"ion-{i}"]["Va"].update(val=Va, same=va_same[i - 1])
    return params


def test_ion_locks_copy_ti_and_va_from_the_named_species(base_config):
    params = _three_ions(base_config, ti_same=(False, False, "ion-2"), va_same=(False, "ion-3", False))
    _check_ion_locks_(params)
    out = ThomsonParams(params, num_params=1).get_unnormed_params()
    assert float(out["ion-3"]["Ti"][0]) == pytest.approx(0.5)
    assert float(out["ion-2"]["Va"][0]) == pytest.approx(3.0)
    assert float(out["ion-2"]["Ti"][0]) == pytest.approx(0.5)
    assert float(out["ion-3"]["Va"][0]) == pytest.approx(3.0)
    assert float(out["ion-1"]["Ti"][0]) == pytest.approx(0.2)
    assert float(out["ion-1"]["Va"][0]) == pytest.approx(-1.0)


def test_legacy_same_true_locks_to_ion_1_with_a_deprecation_warning(base_config):
    params = _three_ions(base_config, ti_same=(False, False, True))
    with pytest.warns(FutureWarning, match="same: true is deprecated"):
        _check_ion_locks_(params)
    out = ThomsonParams(params, num_params=1).get_unnormed_params()
    assert float(out["ion-3"]["Ti"][0]) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "ti_same",
    [(False, "ion-2", "ion-1"), (False, "ion-2", False), (False, "ion-4", False), ("ion-2", "ion-1", False)],
)
def test_an_ion_lock_must_name_a_different_unlocked_species(base_config, ti_same):
    with pytest.raises(ValueError, match="Ti:same"):
        _check_ion_locks_(_three_ions(base_config, ti_same=ti_same))
