"""Diagnostic data providers.

A *data provider* maps a config dict to the ``(all_data, sa, all_axes)`` tuple
that the forward/inverse pipeline consumes:

- ``all_data``: plain dict of arrays (``e_data``, ``i_data``, ``e_amps``,
  ``i_amps``, ``noiseE``, ``noiseI``, ...)
- ``sa``: scattering angles / weights
- ``all_axes``: spectral axes

Everything downstream (``LossFunction``, ``ThomsonScatteringDiagnostic``,
postprocessing) depends only on this contract -- not on how the data was
produced. All OMEGA-specific behaviour (raw file loading, lineouts, background,
throughput, calibration) lives *upstream* of this seam, inside the ``omega``
provider.

This is the extension point for adding a new diagnostic (e.g. NIF OTS) or for
supplying preprocessed data instead of raw OMEGA files -- without touching the
jax/AD model code. Implement a ``provider(config) -> (all_data, sa, all_axes)``
function and register it::

    from tsadar.data import register_data_provider

    @register_data_provider("nif_ots")
    def load_nif(config):
        ...
        return all_data, sa, all_axes

then select it with ``config["data"]["provider"] = "nif_ots"`` (default:
``"omega"``).
"""
from typing import Callable, Dict, Tuple

DataProvider = Callable[[Dict], Tuple[Dict, Dict, Dict]]

_DATA_PROVIDERS: Dict[str, DataProvider] = {}


def register_data_provider(name: str) -> Callable[[DataProvider], DataProvider]:
    """Register a ``config -> (all_data, sa, all_axes)`` provider under ``name``."""

    def _register(fn: DataProvider) -> DataProvider:
        _DATA_PROVIDERS[name.casefold()] = fn
        return fn

    return _register


def get_data_provider(config: Dict) -> DataProvider:
    """Return the provider selected by ``config["data"]["provider"]`` (default ``"omega"``)."""
    name = config.get("data", {}).get("provider", "omega").casefold()
    try:
        return _DATA_PROVIDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown data provider {name!r}. Registered providers: {sorted(_DATA_PROVIDERS)}. "
            f"Register one with tsadar.data.register_data_provider."
        )


# Import built-in providers for their registration side effects.
from . import omega  # noqa: E402,F401
