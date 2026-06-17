"""OMEGA Thomson scattering data provider.

Loads and preprocesses raw OMEGA data (file loading, background, throughput,
lineouts, calibration) into the ``(all_data, sa, all_axes)`` contract. This is
the reference implementation of the data-provider interface; see
``tsadar.data`` for how to register a provider for another diagnostic.
"""
from typing import Dict, Tuple

from . import prepare, register_data_provider


@register_data_provider("omega")
def load_omega_data(config: Dict) -> Tuple[Dict, Dict, Dict]:
    """Load and preprocess OMEGA TS data into ``(all_data, sa, all_axes)``.

    Handles the single-shot case as well as the multiplexed (rotated) angular
    case where ``config["data"]["shotnum"]`` is a two-element list.
    """
    if isinstance(config["data"]["shotnum"], list):
        startCCDsize = config["other"]["CCDsize"]
        all_data, sa, all_axes = prepare.prepare_data(config, config["data"]["shotnum"][0])
        config["other"]["CCDsize"] = startCCDsize
        all_data2, _, _ = prepare.prepare_data(config, config["data"]["shotnum"][1])
        all_data.update(
            {
                "e_data_rot": all_data2["e_data"],
                "e_amps_rot": all_data2["e_amps"],
                "rot_angle": config["data"]["shot_rot"],
                "noiseE_rot": all_data2["noiseE"],
            }
        )

        if config["other"]["extraoptions"]["spectype"] != "angular_full":
            raise NotImplementedError("Muliplexed data fitting is only availible for angular data")
    else:
        all_data, sa, all_axes = prepare.prepare_data(config, config["data"]["shotnum"])
    return all_data, sa, all_axes
