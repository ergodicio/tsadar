from jax import config
from tsadar.utils.process import prepare

config.update("jax_enable_x64", True)


import numpy as np
import matplotlib.pyplot as plt
import yaml
from flatten_dict import flatten, unflatten
from numpy.testing import assert_allclose

def test_feature_detector():
    """
    Test function for feature detection in Thomson scattering data.
    This function loads a configuration file, prepares the data, runs feature detection,
    and asserts that the detected features match expected values.
    """
    # Load configuration
    with open("tests/configs/detector_inputs.yaml", "r") as f:
        inputs = yaml.safe_load(f)
    
    # Flatten the configuration for easier access
    inputs = flatten(inputs)
    config = unflatten(inputs)

    prepare.prepare_data(config, config["data"]["shotnum"])

    known_values = {
        "lineout_start": -900,
        "lineout_end": 402,
        "iaw_min": 526.23,
        "iaw_max": 526.58,
        "iaw_cf_min": 526.33,
        "iaw_cf_max": 526.47, 
        "blue_min": 588,
        "blue_max": 598,    
        "red_min": 454,
        "red_max": 439,
    }

    assert_allclose(config["data"]["lineouts"]["start"], known_values["lineout_start"], rtol = 1e-1)
    assert_allclose(config["data"]["lineouts"]["end"], known_values["lineout_end"], rtol= 1e-1)
    assert_allclose(config["data"]["fit_rng"]["iaw_min"], known_values["iaw_min"], rtol=1e-1)
    assert_allclose(config["data"]["fit_rng"]["iaw_max"], known_values["iaw_max"], rtol=1e-1)
    assert_allclose(config["data"]["fit_rng"]["iaw_cf_min"], known_values["iaw_cf_min"], rtol=1e-1)
    assert_allclose(config["data"]["fit_rng"]["iaw_cf_max"], known_values["iaw_cf_max"], rtol=1e-1)
    assert_allclose(config["data"]["fit_rng"]["blue_min"], known_values["blue_min"], rtol=1e-1)
    assert_allclose(config["data"]["fit_rng"]["blue_max"], known_values["blue_max"], rtol=1e-1)
    assert_allclose(config["data"]["fit_rng"]["red_min"], known_values["red_min"], rtol=1e-1)
    assert_allclose(config["data"]["fit_rng"]["red_max"], known_values["red_max"], rtol=1e-1)


if __name__ == "__main__":
    test_feature_detector()