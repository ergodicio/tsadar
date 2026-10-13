"""Calibration-uncertainty draws for the MCMC postprocessor: draws independent realizations of the
instrument calibration values (gain, spectral IRF widths, dispersion, offset) and builds a
(config, all_data) pair per realization, so mcmc_postprocess.py can run one chain per realization and
pool the results. The perturbations are applied to the outputs of the data pipeline rather than by
re-running it.
"""
import copy
from typing import Dict, List, Tuple

import numpy as np

#: config["other"]["calibration_uncertainty"] field name -> nominal-value lookup, used to find each
#: quantity's current value before perturbing it. Each entry is (config_path, requires_axis_recompute).
_SIGMA_FIELDS = (
    "EPWDispersion_sigma",
    "IAWDispersion_sigma",
    "EPWoffset_sigma",
    "IAWoffset_sigma",
    "spect_stddev_ion_sigma",
    "spect_stddev_ele_sigma",
    "detector_gain_sigma",
)

#: Floor applied to sampled IRF widths so a large sigma draw can never make a convolution kernel
#: degenerate (zero or negative standard deviation).
_MIN_IRF_WIDTH = 1e-6


def _nominal_dispersion_offset(axis: np.ndarray) -> Tuple[float, float]:
    """Recovers (dispersion, offset) from a wavelength axis built as `axisy * dispersion + offset` with
    `axisy = np.arange(1, N+1)` (see calibration.get_calibrations)."""
    dispersion = float(axis[1] - axis[0])
    offset = float(axis[0] - dispersion)
    return dispersion, offset


def _calibration_cfg(config: Dict) -> Dict:
    return config.get("other", {}).get("calibration_uncertainty", {})


def _sigmas(config: Dict) -> Dict[str, float]:
    cal_cfg = _calibration_cfg(config)
    return {name: float(cal_cfg.get(name, 0.0)) for name in _SIGMA_FIELDS}


def draw_calibration_realizations(
    config: Dict, all_data: Dict, all_axes: Dict, rng: np.random.Generator
) -> List[Tuple[Dict, Dict]]:
    """
    Returns a list of K (config_k, all_data_k) pairs, K = config["other"]["calibration_uncertainty"]["num_draws"],
    one per MCMC chain. When num_draws <= 1, or every *_sigma is 0, the unperturbed (config, all_data) pair
    is returned (repeated K times) without copying.

    Otherwise each configured quantity is drawn from Normal(nominal, sigma):
      - EPW/IAW dispersion and offset: config_k["other"]["lamrangE"/"lamrangI"] are recomputed from the
        perturbed wavelength axis.
      - spect_stddev_ion/spect_stddev_ele: config_k["other"]["detector_specs"]["widIRF"], floored at
        _MIN_IRF_WIDTH when perturbed.
      - gain: config_k["other"]["detector_gain"], with all_data_k's e_data/i_data/noiseE/noiseI/e_amps/i_amps
        rescaled by old_gain / new_gain.

    Args:
        config: the merged input-deck config for the fit being post-processed.
        all_data: the data dict prepare_data produced for that fit.
        all_axes: the calibrated axes dict prepare_data produced.
        rng: a numpy random Generator.

    Returns:
        List[Tuple[Dict, Dict]] of length K.
    """
    sigmas = _sigmas(config)
    num_draws = int(_calibration_cfg(config).get("num_draws", 1))

    if num_draws <= 1:
        return [(config, all_data)]
    if not any(sigma > 0.0 for sigma in sigmas.values()):
        # nothing to perturb: reuse the same config/data for every chain
        return [(config, all_data)] * num_draws

    nominal_epw_disp, nominal_epw_off = _nominal_dispersion_offset(np.asarray(all_axes["epw_y"]))
    nominal_iaw_disp, nominal_iaw_off = _nominal_dispersion_offset(np.asarray(all_axes["iaw_y"]))
    widIRF = config["other"]["detector_specs"]["widIRF"]
    nominal_spect_stddev_ion = float(widIRF.get("spect_stddev_ion", 0.0))
    nominal_spect_stddev_ele = float(widIRF.get("spect_stddev_ele", 0.0))
    nominal_gain = float(config["other"]["detector_gain"])
    ccd_size = config["other"]["CCDsize"]

    draws: List[Tuple[Dict, Dict]] = []
    for _ in range(num_draws):
        epw_disp = rng.normal(nominal_epw_disp, sigmas["EPWDispersion_sigma"]) if sigmas["EPWDispersion_sigma"] > 0 else nominal_epw_disp
        epw_off = rng.normal(nominal_epw_off, sigmas["EPWoffset_sigma"]) if sigmas["EPWoffset_sigma"] > 0 else nominal_epw_off
        iaw_disp = rng.normal(nominal_iaw_disp, sigmas["IAWDispersion_sigma"]) if sigmas["IAWDispersion_sigma"] > 0 else nominal_iaw_disp
        iaw_off = rng.normal(nominal_iaw_off, sigmas["IAWoffset_sigma"]) if sigmas["IAWoffset_sigma"] > 0 else nominal_iaw_off
        # only a perturbed width is floored; a nominal width (including 0, which bypasses the IRF) is kept
        spect_stddev_ion = (
            max(rng.normal(nominal_spect_stddev_ion, sigmas["spect_stddev_ion_sigma"]), _MIN_IRF_WIDTH)
            if sigmas["spect_stddev_ion_sigma"] > 0
            else nominal_spect_stddev_ion
        )
        spect_stddev_ele = (
            max(rng.normal(nominal_spect_stddev_ele, sigmas["spect_stddev_ele_sigma"]), _MIN_IRF_WIDTH)
            if sigmas["spect_stddev_ele_sigma"] > 0
            else nominal_spect_stddev_ele
        )
        gain = rng.normal(nominal_gain, sigmas["detector_gain_sigma"]) if sigmas["detector_gain_sigma"] > 0 else nominal_gain
        if gain <= 0:
            gain = nominal_gain  # a non-positive gain draw is unphysical; keep this draw at the nominal value

        config_k = copy.deepcopy(config)
        axisy = np.arange(1, ccd_size[0] + 1)
        axisyE_k = axisy * epw_disp + epw_off
        axisyI_k = axisy * iaw_disp + iaw_off
        config_k["other"]["lamrangE"] = [float(axisyE_k[0]), float(axisyE_k[-1])]
        config_k["other"]["lamrangI"] = [float(axisyI_k[0]), float(axisyI_k[-1])]
        config_k["other"]["detector_specs"]["widIRF"]["spect_stddev_ion"] = float(spect_stddev_ion)
        config_k["other"]["detector_specs"]["widIRF"]["spect_stddev_ele"] = float(spect_stddev_ele)
        config_k["other"]["detector_gain"] = float(gain)

        gain_rescale = nominal_gain / gain
        all_data_k = dict(all_data)
        for key in ("e_data", "i_data", "noiseE", "noiseI", "e_amps", "i_amps"):
            if key in all_data_k:
                all_data_k[key] = all_data_k[key] * gain_rescale

        draws.append((config_k, all_data_k))

    return draws
