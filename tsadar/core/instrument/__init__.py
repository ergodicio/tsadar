"""Device-dependent instrument model: everything applied to a spectrum after the physics.

Modules here are organized by pipeline stage. Currently that is just the instrument
response function; aperture weighting, notch filters, and pixel reduction move here as
they are pulled out of ``core/physics/generate_spectra.py`` and
``core/thomson_diagnostic.py``.

Invariant for this package: no imports from the data or orchestration layers
(``tsadar.data``, ``tsadar.inverse``, ``tsadar.forward``, ``tsadar.runner``), and no
reading of a config dict. Callers translate their input deck into these value objects,
which is what makes a new device a matter of construction rather than of matching
OMEGA's deck layout.

This sits inside the light-dependency forward kernel described in #105, in the
"synthetic diagnostic" layer alongside ``core/thomson_diagnostic.py`` -- which is where
that issue's target layering already places ``irf``, rather than under ``physics``.
"""

from .irf import AngularIRF, SpectrometerIRF, add_ATS_IRF, add_electron_IRF, add_ion_IRF

__all__ = [
    "AngularIRF",
    "SpectrometerIRF",
    "add_ATS_IRF",
    "add_electron_IRF",
    "add_ion_IRF",
]
