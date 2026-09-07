"""FitModel: wraps FormFactor to add finite-aperture/finite-volume instrument effects and produce the
electron (EPW) and ion (IAW) Thomson scattering spectra used for fitting and forward passes."""
from typing import Dict

from .form_factor import DEFAULT_N_BETA, FormFactor
from .resonance_quadrature import integrate_detector_bins

from jax import lax, numpy as jnp
from jax.tree_util import tree_map


class FitModel:
    """
    FitModel is a class that wraps the FormFactor class to add finite aperture and finite volume effects for generating Thomson scattering spectra. It manages configuration options, handles multiple scattering angles, and supports both electron and ion features, including gradient effects and angular spectra.

    Args: 

        config (Dict): Configuration dictionary built from the input deck, containing all static and runtime parameters for spectrum generation.
        scattering_angles (Dict): Dictionary containing the scattering angles at which the spectrum will be calculated and the relative weights for each angle.

    Methods:

        __call__(all_params: Dict):

            Calculates Thomson spectra corrected for finite aperture and optionally including plasma gradients, based on the current parameter dictionary. Does not compute or return the raw (pre-angle-averaged) theoretical spectra.

                all_params (Dict): Dictionary of current values for all active and static parameters.
                modlE: Electron plasma wave spectrum (array or int 0 if not loaded).
                modlI: Ion acoustic wave spectrum (array or int 0 if not loaded).
                lamAxisE: Wavelength axis for electron plasma wave (array or empty list if not loaded).
                lamAxisI: Wavelength axis for ion acoustic wave (array or empty list if not loaded).

        detailed_spectrum(all_params: Dict):

            Calculates both the total spectrum and all its components for postprocessing, including the raw (pre-angle-averaged) theoretical spectra ThryE/ThryI.

                all_params (Dict): Parameter dictionary.
                modlE: Electron plasma wave spectrum.
                modlI: Ion acoustic wave spectrum.
                ThryE: Detailed electron spectrum components.
                ThryI: Detailed ion spectrum components.
                lamAxisE: Wavelength axis for electron plasma wave.
                lamAxisI: Wavelength axis for ion acoustic wave.

        __call__ and detailed_spectrum share their implementation via the private _ion_spectrum_core/
        _electron_spectrum_core methods, which take a want_thry flag controlling whether the raw theoretical
        spectrum (and, for electrons, its extra filtering step) is computed and returned.

    """

    def __init__(self, config: Dict, scattering_angles: Dict):
        """
        Initializes the FitModel class, setting up static properties required for spectrum generation that remain unchanged across iterations.
        Args:
            config (Dict): Configuration dictionary constructed from the input deck, containing all necessary parameters for spectrum generation.
            scattering_angles (Dict): Dictionary containing the scattering angles at which the spectrum will be calculated, along with the relative weights for each angle in the final spectrum.
        Raises:
            AssertionError: If the number of gradient points for electron temperature (Te) and electron density (ne) are not the same.
        Attributes:
            config (Dict): Stores the provided configuration dictionary.
            scattering_angles (Dict): Stores the provided scattering angles and their weights.
            electron_form_factor (FormFactor): Form factor object for electrons, initialized with relevant parameters from the configuration.
            ion_form_factor (FormFactor): Form factor object for ions, initialized with relevant parameters from the configuration.
        """

        self.config = config
        self.scattering_angles = scattering_angles

        quadrature_config = config["other"].get("resonance_quadrature", {})
        self.electron_spectrum_is_detector_binned = bool(
            config["data"]["load_ele_spec"]
            and config["parameters"]["electron"]["fe"]["dim"] == 2
            and config["other"]["extraoptions"]["spectype"] == "angular_full"
            and quadrature_config.get("enabled", True)
        )
        if self.electron_spectrum_is_detector_binned:
            detector_specs = config["other"]["detector_specs"]
            self.electron_detector_edges_nm = jnp.asarray(
                detector_specs["electron_wavelength_edges"]
            )
            self.electron_detector_centers_nm = jnp.asarray(
                detector_specs.get(
                    "electron_wavelength_centers",
                    0.5
                    * (
                        self.electron_detector_edges_nm[:-1]
                        + self.electron_detector_edges_nm[1:]
                    ),
                )
            )
            self.electron_irf_sigma_nm = (
                detector_specs["widIRF"]["spect_FWHM_ele"] / 2.3548
            )
            tail_sigma = float(quadrature_config.get("tail_sigma", 6.0))
            self.resonance_quadrature_options = {
                "root_scan_panels": int(
                    quadrature_config.get("root_scan_panels", 4096)
                ),
                "integration_panels": int(
                    quadrature_config.get("integration_panels", 256)
                ),
                "regular_order": int(quadrature_config.get("regular_order", 8)),
                "root_order": int(quadrature_config.get("root_order", 32)),
                "max_roots": int(quadrature_config.get("max_roots", 16)),
                "neighbor_panels": int(quadrature_config.get("neighbor_panels", 1)),
                "bisection_iterations": int(
                    quadrature_config.get("bisection_iterations", 48)
                ),
                "tail_sigma": tail_sigma,
                "scan_phase": float(quadrature_config.get("scan_phase", 0.0)),
            }
            # ``lax.map`` keeps the expensive node-by-detector response matrix bounded
            # in memory. A small explicit batch can recover device parallelism without
            # ever materializing every scattering geometry at once.
            self.resonance_quadrature_map_batch_size = int(
                quadrature_config.get("map_batch_size", 1)
            )
            if self.resonance_quadrature_map_batch_size < 1:
                raise ValueError("resonance_quadrature.map_batch_size must be positive")

            self.electron_notch_filter = None
            if config["other"]["iawfilter"][0]:
                filter_center = float(config["other"]["iawfilter"][3])
                filter_width = float(config["other"]["iawfilter"][2])
                if filter_width <= 0:
                    raise ValueError("enabled iawfilter width must be positive")
                self.electron_notch_filter = (
                    filter_center - filter_width / 2,
                    filter_center + filter_width / 2,
                    10 ** (-float(config["other"]["iawfilter"][1])),
                )

                # The rectangular filter is applied to the continuous source spectrum,
                # before the Gaussian spectral response and detector-bin integration.
                # Preserve every discontinuity that actually lies inside the source
                # integration domain as an exact coarse-panel boundary. Boundaries on or
                # outside the domain do not split an integration panel and must not be
                # passed to the quadrature kernel, which requires strict interior points.
                source_lower = (
                    float(self.electron_detector_edges_nm[0])
                    - tail_sigma * self.electron_irf_sigma_nm
                )
                source_upper = (
                    float(self.electron_detector_edges_nm[-1])
                    + tail_sigma * self.electron_irf_sigma_nm
                )
                integration_breakpoints = tuple(
                    boundary
                    for boundary in self.electron_notch_filter[:2]
                    if source_lower < boundary < source_upper
                )
                if integration_breakpoints:
                    self.resonance_quadrature_options[
                        "integration_breakpoints_nm"
                    ] = jnp.asarray(integration_breakpoints)

        assert (
            config["parameters"]["general"]["Te_gradient"]["num_grad_points"]
            == config["parameters"]["general"]["ne_gradient"]["num_grad_points"]
        ), "Number of gradient points for Te and ne must be the same"
        num_grad_points = config["parameters"]["general"]["Te_gradient"]["num_grad_points"]

        ud_angle = (
            None
            if config["parameters"]["electron"]["fe"]["dim"] < 2
            else config["parameters"]["general"]["ud"]["angle"]
        )
        ion_species = [species for species in config["parameters"] if species.startswith("ion-")]
        va_angle = None
        if config["parameters"]["electron"]["fe"]["dim"] >= 2:
            va_angle = {
                species: config["parameters"][species]["Va"].get("angle", 0.0)
                for species in ion_species
            }

        # Angular resolution of the tabulated projection used by the 2D form factor. Set
        # to 0 to fall back to an exact rotation at every evaluation point, which is far
        # slower but is what the tabulation is validated against.
        n_beta = config["other"].get("n_beta", DEFAULT_N_BETA)

        if 'include_gains' in config["other"] and config["other"]["include_gains"]:
            calc_gain = {'calc': config["other"]["include_gains"], 'Ipump': config["other"]["Ipump_14"], 'beam_diam_um': config["other"]["beam_diam_um"]}
        else:
            calc_gain = {'calc': False, 'Ipump': 0, 'beam_diam_um': 0}

        self.electron_form_factor = FormFactor(
            config["other"]["lamrangE"],
            npts=config["other"]["npts"],
            lam_shift=config["data"]["ele_lam_shift"],
            scattering_angles=self.scattering_angles,
            num_grad_points=num_grad_points,
            va_ang=va_angle,
            ud_ang=ud_angle,
            calc_gain=calc_gain,
            n_beta=n_beta,
        )
        self.ion_form_factor = FormFactor(
            config["other"]["lamrangI"],
            npts=config["other"]["npts"],
            lam_shift=0,
            scattering_angles=scattering_angles,
            num_grad_points=num_grad_points,
            va_ang=va_angle,
            ud_ang=ud_angle,
            calc_gain=calc_gain,
            n_beta=n_beta,
        )

    def __call__(self, all_params: Dict):
        """
        Produces Thomson spectra corrected for finite aperture and optionally including gradients in the plasma
        conditions based off the current parameter dictionary. Calling this method will automatically choose the
        appropriate version of the formfactor class based off the dimension and distribute the conditions for
        multiple ion species to their respective inputs.


        Args:

            all_params: Parameter dictionary containing the current values for all active and static parameters. Only a few permanently static properties from the configuration dictionary will be used, everything else must be included in this input.

        Returns:
            
            modlE: calculated electron plasma wave spectrum as an array with length of npts. If an angular spectrum is calculated then it will be 2D. If the EPW is not loaded this is returned as the int 0.
            modlI: calculated ion acoustic wave spectrum as an array with length of npts. If the IAW is not loaded this is returned as the int 0.
            lamAxisE: electron plasma wave wavelength axis as an array with length of npts. If the EPW is not loaded this is returned as an empty list.
            lamAxisI: ion acoustic wave wavelength axis as an array with length of npts. If the IAW is not loaded this is returned as an empty list.
            all_params: The input all_params is returned

            
        """

        lamAxisI, modlI, _ = self._ion_spectrum_core(all_params, want_thry=False)
        lamAxisE, modlE, _ = self._electron_spectrum_core(all_params, want_thry=False)

        return modlE, modlI, lamAxisE, lamAxisI

    def detailed_spectrum(self, all_params: Dict):
        """
        Calculates detailed spectra for both electron plasma waves (EPW) and ion acoustic waves (IAW), including their wavelength axes and theoretical components, for postprocessing analysis. This method produces both the total spectrum and all its components for EPWs and IAWs, using the provided parameter dictionary. It is intended for postprocessing and requires all relevant parameters to be included in the input.
        Args:
            
            all_params (Dict): Dictionary containing current values for all active and static parameters. Most configuration properties must be included in this input, except for a few permanently static ones.
        
        Returns:
            
            modlE (np.ndarray or int): Calculated electron plasma wave spectrum as an array of length npts, or 0 if EPW is not loaded.
            modlI (np.ndarray or int): Calculated ion acoustic wave spectrum as an array of length npts, or 0 if IAW is not loaded.
            ThryE (np.ndarray): Theoretical components of the electron plasma wave spectrum.
            ThryI (np.ndarray): Theoretical components of the ion acoustic wave spectrum.
            lamAxisE (np.ndarray or list): Wavelength axis for the electron plasma wave spectrum, or empty list if EPW is not loaded.
            lamAxisI (np.ndarray or list): Wavelength axis for the ion acoustic wave spectrum, or empty list if IAW is not loaded.
        
        """

        lamAxisI, modlI, ThryI = self._ion_spectrum_core(all_params, want_thry=True)
        lamAxisE, modlE, ThryE = self._electron_spectrum_core(all_params, want_thry=True)

        return modlE, modlI, ThryE, ThryI, lamAxisE, lamAxisI

    def _ion_spectrum_core(self, all_params, want_thry):
        """
        Shared implementation behind ion_spectrum and ion_spectrum_detailed. want_thry controls whether the raw
        (pre-angle-averaged) ThryI is kept and returned; ion_spectrum passes False so callers that only need
        (lamAxisI, modlI) don't retain the larger intermediate array.
        """
        if self.config["data"]["load_ion_spec"]:
            if self.config["parameters"]["electron"]["fe"]["dim"] == 1:
                ThryI, lamAxisI = self.ion_form_factor(all_params)
            elif self.config["parameters"]["electron"]["fe"]["dim"] == 2:
                ThryI, lamAxisI = self.ion_form_factor.calc_in_2D(all_params)

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7  # TODO hardcoded
            modlI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(modlI * self.scattering_angles["weights"][0], axis=1)
            raw_thry = ThryI if want_thry else 0
        else:
            modlI = 0
            raw_thry = 0
            lamAxisI = jnp.zeros(1)
        return lamAxisI, modlI, raw_thry

    def _electron_spectrum_core(self, all_params, want_thry):
        """
        Shared implementation behind electron_spectrum and electron_spectrum_detailed. want_thry controls whether
        the raw (pre-angle-averaged) ThryE is filtered and returned. That filtering step is extra work needed only
        for the detailed/postprocessing output, so electron_spectrum passes False to skip it entirely rather than
        computing and discarding it on every forward pass.
        """
        if self.electron_spectrum_is_detector_binned:
            lamAxisE, modlE, ThryE, _ = self.detector_integrated_electron_spectrum(
                all_params
            )
            return lamAxisE, modlE, ThryE if want_thry else 0

        if self.config["data"]["load_ele_spec"]:
            if self.config["parameters"]["electron"]["fe"]["dim"] == 1:
                ThryE, lamAxisE_orig = self.electron_form_factor(all_params)
            elif self.config["parameters"]["electron"]["fe"]["dim"] == 2:
                ThryE, lamAxisE_orig = self.electron_form_factor.calc_in_2D(all_params)

            # remove extra dimensions and rescale to nm
            lamAxisE_orig *= 1e7
            lamAxisE = jnp.squeeze(lamAxisE_orig)  # TODO hardcoded

            modlE = jnp.mean(ThryE, axis=0)
            if self.config["other"]["extraoptions"]["spectype"] == "angular_full":
                modlE = jnp.matmul(self.scattering_angles["weights"], modlE.transpose())
            else:
                modlE = jnp.sum(modlE * self.scattering_angles["weights"][0], axis=1)

            lam = all_params["general"]["lam"]
            if self.config["other"]["iawoff"] and (
                self.config["other"]["lamrangE"][0] < lam < self.config["other"]["lamrangE"][1]
            ):
                # set the ion feature to 0 #should be switched to a range about lam
                lamlocb = jnp.argmin(jnp.abs(lamAxisE - lam - 3.0))
                lamlocr = jnp.argmin(jnp.abs(lamAxisE - lam + 3.0))
                modlE = jnp.concatenate(
                    [modlE[:lamlocb], jnp.zeros(lamlocr - lamlocb), modlE[lamlocr:]]
                )  # TODO hardcoded

            if self.config["other"]["iawfilter"][0]:
                filterb = self.config["other"]["iawfilter"][3] - self.config["other"]["iawfilter"][2] / 2
                filterr = self.config["other"]["iawfilter"][3] + self.config["other"]["iawfilter"][2] / 2

                if self.config["other"]["lamrangE"][0] < filterr and self.config["other"]["lamrangE"][1] > filterb:
                    indices = (filterb < lamAxisE) & (filterr > lamAxisE)
                    modlE = jnp.where(indices, modlE * 10 ** (-self.config["other"]["iawfilter"][1]), modlE)

                    if want_thry:
                        indices = (filterb < lamAxisE_orig) & (filterr > lamAxisE_orig)
                        ThryE = jnp.where(indices, ThryE * 10 ** (-9), ThryE)

            raw_thry = ThryE if want_thry else 0
        else:
            modlE = 0
            raw_thry = 0
            lamAxisE = []
        return lamAxisE, modlE, raw_thry

    def detector_integrated_electron_spectrum(self, all_params):
        """Integrate the ARTS2D electron spectrum directly into detector bins.

        Each gradient/scattering-angle spectrum gets an independent root search and
        tan-mapped quadrature because its dielectric roots occur at different physical
        wavelengths. The Gaussian spectral IRF is folded into the integral through exact
        CDF differences, so callers must not apply a second spectral convolution or
        wavelength reduction. An enabled rectangular ``iawfilter`` multiplies the
        continuous source numerator, with its in-domain edges inserted as exact
        integration breakpoints. ``iawoff`` remains a detector-bin mask.

        Returns:
            ``(wavelength_centers_nm, aperture_weighted_bin_means,
            per_geometry_bin_means, diagnostics)``. The first model array has shape
            ``[calibrated angle, detector bin]``; the per-geometry array has shape
            ``[gradient, detector bin, scattering angle]``. Every diagnostic field has
            leading ``[gradient, scattering angle]`` axes.
        """

        if not self.electron_spectrum_is_detector_binned:
            raise ValueError(
                "detector-integrated electron spectra require enabled ARTS2D "
                "resonance quadrature"
            )

        form_factor = self.electron_form_factor
        sinogram = form_factor.prepare_2D_sinogram(all_params)
        angles = jnp.asarray(self.scattering_angles["sa"])
        num_gradients = form_factor.num_grad_points
        num_angles = angles.size
        flat_indices = jnp.arange(num_gradients * num_angles, dtype=jnp.int32)

        def integrate_one(flat_index):
            gradient_index = flat_index // num_angles
            angle_index = flat_index % num_angles
            angle = angles[angle_index]

            def terms_at(wavelengths_nm):
                numerator, epsilon = form_factor.calc_2D_spectral_terms(
                    all_params,
                    wavelengths_nm,
                    sinogram=sinogram,
                    scattering_angles=angle,
                )
                if self.electron_notch_filter is not None:
                    filter_lower, filter_upper, attenuation = self.electron_notch_filter
                    transmission = jnp.where(
                        (wavelengths_nm > filter_lower)
                        & (wavelengths_nm < filter_upper),
                        attenuation,
                        1.0,
                    )
                    numerator = numerator * transmission[:, None, None]
                return numerator[:, gradient_index, 0], epsilon[:, gradient_index, 0]

            return integrate_detector_bins(
                terms_at,
                self.electron_detector_edges_nm,
                self.electron_irf_sigma_nm,
                **self.resonance_quadrature_options,
            )

        result = lax.map(
            integrate_one,
            flat_indices,
            batch_size=self.resonance_quadrature_map_batch_size,
        )
        per_geometry = result.bin_mean.reshape(
            num_gradients, num_angles, self.electron_detector_edges_nm.size - 1
        )
        # Restore FormFactor's historical raw ordering [gradient, wavelength, angle].
        ThryE = jnp.transpose(per_geometry, (0, 2, 1))
        gradient_average = jnp.mean(per_geometry, axis=0)
        modlE = jnp.matmul(self.scattering_angles["weights"], gradient_average)
        lamAxisE = self.electron_detector_centers_nm

        lam = all_params["general"]["lam"]
        if self.config["other"]["iawoff"]:
            ion_feature = (lamAxisE > lam - 3.0) & (lamAxisE < lam + 3.0)
            modlE = jnp.where(ion_feature[None, :], 0, modlE)
            ThryE = jnp.where(ion_feature[None, :, None], 0, ThryE)

        diagnostics = tree_map(
            lambda value: value.reshape(
                (num_gradients, num_angles) + value.shape[1:]
            ),
            result.diagnostics,
        )
        return lamAxisE, modlE, ThryE, diagnostics
