import copy
import warnings

import pytest
import yaml
from flatten_dict import flatten, unflatten

from tsadar.data.lineouts import spectral_smoothing_width
from tsadar.inverse.fitter import OMEGA_TS_GAIN, _validate_inputs_


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
