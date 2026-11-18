"""Scattering parameters of a result and whether they lie in the well-resolved regime."""

import logging

import mlflow
import numpy as np

ALPHA_LIMIT = 3.5
NUMERIC_BETA_LIMIT = 8.0

logger = logging.getLogger(__name__)


def scattering_parameters(ne, Te, ions, lam_nm, angles_deg):
    """
    Salpeter alpha = 1/(k lambda_De) and, per ion species, beta = sqrt(Z Te/Ti alpha^2/(1 + alpha^2)) at the
    smallest scattering angle.

    Args:
        ne: electron density in 1e20 cm^-3.
        Te: electron temperature in keV.
        ions: {species: (Z, Ti)} with Ti in keV.
        lam_nm: probe wavelength in nm.
        angles_deg: scattering angles in degrees.
    Returns:
        {"alpha": ..., "beta_<species>": ...}, each shaped like ne.
    """
    ne = np.asarray(ne, dtype=float)
    Te = np.asarray(Te, dtype=float)
    theta = np.deg2rad(np.min(angles_deg))
    nc = 1.115e21 / (np.asarray(lam_nm, dtype=float) * 1e-3) ** 2
    k_probe = 2 * np.pi / (np.asarray(lam_nm, dtype=float) * 1e-7) * np.sqrt(np.clip(1 - ne * 1e20 / nc, 0, None))
    debye = 743.4 * np.sqrt(Te * 1e3 / (ne * 1e20))
    alpha = 1.0 / (2 * k_probe * np.sin(theta / 2) * debye)
    out = {"alpha": alpha}
    for species, (Z, Ti) in ions.items():
        out[f"beta_{species}"] = np.sqrt(np.asarray(Z) * Te / np.asarray(Ti) * alpha**2 / (1 + alpha**2))
    return out


def plasma_conditions(fitted_weights):
    """
    ne, Te, lam and each ion species' Z and Ti for every fitted state, from the full physical parameters of a
    ThomsonParams (or a list of them, one per fit batch), fixed and fitted parameters alike.
    """
    weights = fitted_weights if isinstance(fitted_weights, (list, tuple)) else [fitted_weights]
    conditions = {}
    for weight in weights:
        physical = weight()
        for species, values in physical.items():
            keys = ("Z", "Ti") if "ion" in species else {"electron": ("ne", "Te"), "general": ("lam",)}.get(species, ())
            for key in keys:
                conditions.setdefault(species, {}).setdefault(key, []).append(np.ravel(np.asarray(values[key], float)))
    return {species: {key: np.concatenate(v) for key, v in c.items()} for species, c in conditions.items()}


def scattering_from_conditions(conditions, angles_deg):
    """scattering_parameters for the output of plasma_conditions."""
    ions = {species: (c["Z"], c["Ti"]) for species, c in conditions.items() if "ion" in species}
    electron = conditions["electron"]
    return scattering_parameters(electron["ne"], electron["Te"], ions, conditions["general"]["lam"], angles_deg)


def _limits(numeric_species):
    return {"alpha": ALPHA_LIMIT} | {f"beta_{species}": NUMERIC_BETA_LIMIT for species in numeric_species}


def validity_message(scattering, numeric_species=()):
    """
    None when every alpha <= 3.5 and every beta of a numeric ion species <= 8; otherwise the message naming the
    values past their limits. A non-finite value counts as past its limit; beta of a Maxwellian species is not
    limited.
    """
    limits = _limits(numeric_species)
    over = {}
    for name, limit in limits.items():
        value = np.asarray(scattering[name], dtype=float)
        if not np.all(value <= limit):
            over[name] = float(np.max(np.where(np.isfinite(value), value, np.inf)))
    if not over:
        return None
    values = ", ".join(f"{name}={value:.2f} (limit {limits[name]})" for name, value in over.items())
    return f"likely under-resolved: {values}"


def report_validity(scattering, numeric_species=()):
    """Log the validity of a result and tag the active MLflow run with it."""
    message = validity_message(scattering, numeric_species)
    if message is not None:
        logger.warning(message)
    if mlflow.active_run() is not None:
        limits = ", ".join(f"{name} <= {limit}" for name, limit in _limits(numeric_species).items())
        mlflow.set_tag("validity", message or f"within {limits}")
    return message


def numeric_ion_species(param_cfg):
    """Ion species whose distribution is ``type: numeric``."""
    return [
        species
        for species, cfg in param_cfg.items()
        if "ion" in species and cfg.get("fi", {}).get("type", "mx").casefold() == "numeric"
    ]
