"""Calibration-uncertainty draws for the MCMC postprocessor: pre-draws a small number of independent
realizations of the instrument calibration values (gain, spectral IRF widths, dispersion, offset) and
builds a (config, all_data) pair per realization, so mcmc_postprocess.py can run one MCMC chain per
realization and pool the results -- see the module docstring in mcmc.py and the plan this implements for
why calibration values are perturbed this way rather than resampled at MCMC-step granularity.

Deliberately does not touch tsadar/data/calibration.py, tsadar/data/prepare.py, or
tsadar/data/lineouts.py: get_calibrations is a large, actively-maintained per-shot-number lookup table,
and every quantity perturbed here is a simple, branch-agnostic post-transform of its *output*
(axisyE/axisyI, detector_specs, gain) -- threading stochastic perturbation through the lookup table
itself would multiply bug surface for no benefit, and re-running the I/O-heavy prepare_data/get_lineouts
per draw would multiply real disk/compute cost for zero physical benefit, since none of that work
actually depends on the perturbed values.
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
    "gain_sigma",
)

#: Floor applied to sampled IRF widths so a large sigma draw can never make a convolution kernel
#: degenerate (zero or negative standard deviation).
_MIN_IRF_WIDTH = 1e-6


def _nominal_dispersion_offset(axis: np.ndarray) -> Tuple[float, float]:
    """Recovers (dispersion, offset) from a calibrated wavelength axis built as
    `axisy * dispersion + offset` with `axisy = np.arange(1, N+1)` (see
    calibration.get_calibrations, which builds axisyE/axisyI exactly this way). Exact and
    branch-agnostic: works regardless of which of get_calibrations' per-shot-number branches produced
    the axis, since it only looks at the axis values themselves.
    """
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
    Returns a list of K (config_k, all_data_k) pairs, K = config["other"]["calibration_uncertainty"]["num_draws"].

    Collapses to exactly [(config, all_data)] -- the identical objects, no copy, no RNG draw -- whenever
    num_draws <= 1 or every configured *_sigma is 0.0. This is the required backward-compatible,
    zero-overhead path: a deck that never configured calibration uncertainty behaves exactly as before.

    Otherwise draws K independent realizations from Normal(nominal, sigma) for each configured quantity
    and builds K (config_k, all_data_k) pairs:
      - EPWDispersion'/EPWoffset' -> config_k["other"]["lamrangE"], recomputed from
        axisyE' = np.arange(1, CCDsize[0]+1) * EPWDispersion' + EPWoffset' -- an exact reproduction of
        calibration.get_calibrations' own construction (calibration.py:587-589), not an approximation.
      - IAWDispersion'/IAWoffset' -> config_k["other"]["lamrangI"], analogously.
      - spect_stddev_ion'/spect_stddev_ele' -> config_k["other"]["detector_specs"]["widIRF"][...],
        floored at _MIN_IRF_WIDTH to keep the IRF convolution well-defined.
      - gain' -> config_k["other"]["gain"], with all_data_k's e_data/i_data/noiseE/noiseI/e_amps/i_amps
        rescaled by (old_gain / new_gain) -- the exact inverse of the division lineouts.get_lineouts
        already applies (tsadar/data/lineouts.py:147-175), reproducing what re-running prepare_data with
        that gain would have produced without repeating the (slow) data extraction.

    Args:
        config: the merged input-deck config for the fit being post-processed.
        all_data: the data dict prepare_data produced for that fit (only e_data/i_data/noiseE/noiseI/
            e_amps/i_amps are read/rescaled; every other key is passed through unchanged).
        all_axes: the calibrated axes dict prepare_data produced (epw_y/iaw_y are read to recover the
            nominal dispersion/offset).
        rng: a numpy random Generator, so callers control reproducibility via its seed.

    Returns:
        List[Tuple[Dict, Dict]]: length K (or 1 in the collapsed case).
    """
    sigmas = _sigmas(config)
    num_draws = int(_calibration_cfg(config).get("num_draws", 1))

    if num_draws <= 1 or not any(sigma > 0.0 for sigma in sigmas.values()):
        return [(config, all_data)]

    nominal_epw_disp, nominal_epw_off = _nominal_dispersion_offset(np.asarray(all_axes["epw_y"]))
    nominal_iaw_disp, nominal_iaw_off = _nominal_dispersion_offset(np.asarray(all_axes["iaw_y"]))
    widIRF = config["other"]["detector_specs"]["widIRF"]
    nominal_spect_stddev_ion = float(widIRF.get("spect_stddev_ion", 0.0))
    nominal_spect_stddev_ele = float(widIRF.get("spect_stddev_ele", 0.0))
    nominal_gain = float(config["other"]["gain"])
    ccd_size = config["other"]["CCDsize"]

    draws: List[Tuple[Dict, Dict]] = []
    for _ in range(num_draws):
        epw_disp = rng.normal(nominal_epw_disp, sigmas["EPWDispersion_sigma"]) if sigmas["EPWDispersion_sigma"] > 0 else nominal_epw_disp
        epw_off = rng.normal(nominal_epw_off, sigmas["EPWoffset_sigma"]) if sigmas["EPWoffset_sigma"] > 0 else nominal_epw_off
        iaw_disp = rng.normal(nominal_iaw_disp, sigmas["IAWDispersion_sigma"]) if sigmas["IAWDispersion_sigma"] > 0 else nominal_iaw_disp
        iaw_off = rng.normal(nominal_iaw_off, sigmas["IAWoffset_sigma"]) if sigmas["IAWoffset_sigma"] > 0 else nominal_iaw_off
        spect_stddev_ion = max(
            rng.normal(nominal_spect_stddev_ion, sigmas["spect_stddev_ion_sigma"]) if sigmas["spect_stddev_ion_sigma"] > 0 else nominal_spect_stddev_ion,
            _MIN_IRF_WIDTH,
        )
        spect_stddev_ele = max(
            rng.normal(nominal_spect_stddev_ele, sigmas["spect_stddev_ele_sigma"]) if sigmas["spect_stddev_ele_sigma"] > 0 else nominal_spect_stddev_ele,
            _MIN_IRF_WIDTH,
        )
        gain = rng.normal(nominal_gain, sigmas["gain_sigma"]) if sigmas["gain_sigma"] > 0 else nominal_gain
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
        config_k["other"]["gain"] = float(gain)

        gain_rescale = nominal_gain / gain
        all_data_k = dict(all_data)
        for key in ("e_data", "i_data", "noiseE", "noiseI", "e_amps", "i_amps"):
            if key in all_data_k:
                all_data_k[key] = all_data_k[key] * gain_rescale

        draws.append((config_k, all_data_k))

    return draws
