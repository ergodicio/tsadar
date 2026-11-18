"""The real-shot deck the inverse unit tests use."""

import yaml
from flatten_dict import flatten, unflatten


def base_config():
    """Shot 101675, two EPW lineouts, fe inactive (the MCMC sampler and get_sigmas require it), 10 epochs."""
    with open("tests/configs/time_test_defaults.yaml") as fi:
        defaults = yaml.safe_load(fi)
    with open("tests/configs/time_test_inputs.yaml") as fi:
        inputs = yaml.safe_load(fi)
    flat = flatten(defaults)
    flat.update(flatten(inputs))
    cfg = unflatten(flat)
    cfg["parameters"]["electron"]["fe"]["active"] = False
    cfg["data"]["launch_data_visualizer"] = False
    cfg["data"]["lineouts"]["val"] = list(
        range(cfg["data"]["lineouts"]["start"], cfg["data"]["lineouts"]["end"], cfg["data"]["lineouts"]["skip"])
    )
    cfg["optimizer"]["num_epochs"] = 10
    return cfg
