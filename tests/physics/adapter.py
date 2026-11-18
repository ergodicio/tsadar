"""tsadar as a code under test for tsverify's conformance tests."""

import numpy as np

from tsverify.adapter import FEATURES, Adapter, NotSupported

from . import harness


class TsadarAdapter(Adapter):
    name = "tsadar"
    features = FEATURES

    def spectral_density(self, plasma, wavelengths_nm):
        """The 1-D ``FormFactor``, which evaluates on a uniform wavelength grid."""
        lam = np.asarray(wavelengths_nm, dtype=float)
        if not np.allclose(np.diff(lam), lam[1] - lam[0], rtol=1e-9, atol=0):
            raise NotSupported("FormFactor needs a uniform wavelength grid")
        lam_out, spectrum = harness.form_factor(plasma, (lam[0], lam[-1]), npts=lam.size)
        np.testing.assert_allclose(lam_out, lam, rtol=1e-12, atol=0, err_msg="FormFactor moved the wavelength grid")
        return spectrum

    def detector_spectrum(self, volume, detector):
        """The full diagnostic; tsadar's gradient model and aperture weights take the volume's values directly."""
        g = volume.gradient
        gradients = harness.Gradients(g.ne_percent, g.Te_percent, g.points) if g else harness.Gradients()
        a = volume.aperture
        aperture = (a.angles_deg, a.weights) if a else None
        return harness.detector(volume.plasma, detector, gradients=gradients, aperture=aperture)
