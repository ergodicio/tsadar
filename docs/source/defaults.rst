.. _configuring-the-default:

Default options
========================================

The code uses two input decks, which  are located in **configs/1d**. The primary input deck, ``inputs.yaml``, contains the parameters that are most likely to be changed on a shot-to-shot basis. The secondary input deck, ``defaults.yaml``, contains all parameters. This includes the parameters in the primary input deck, but also includes all other parameters that can be modified externally to the code. When running the code, the ``inputs.yaml`` deck will override any conflicting values in the ``defaults.yaml`` deck, allowing for quick changes to commonly modified parameters without needing to change the ``defaults.yaml`` deck.

This page contains a detailed description of the parameters in the ``defaults.yaml`` deck. For a more concise description of the parameters that are most commonly modified, see the :ref:`inputs page <inputs>`.

.. toctree::
   :maxdepth: 4
   :caption: Default Deck Fields
   

Parameters
^^^^^^^^^^^^

All fitting parameters are found in the ``parameters:`` section of the input decks. The fitting parameters are broken up based off their scope.
There must be at least 3 subfields referencing the different species in ``parameters`` called ``electron``, 
``ion-1``, and ``general``. If the problem has multiple ion species additional ion species fields can be added 
with increasing index numbers, i.e. ``ion-2``, ``ion-3``.

Each species has associated fitting parameters which generally have 4 attributes: ``val``, ``active``, 
``lb``, and ``ub``. ``val`` is the initial value for a fitted parameter or the constant value for a
static parameter. ``active`` is a boolean determining if the parameter is being fit, so parameters with ``active: False`` will be held at the value in ``val``. ``ub`` and ``lb`` are upper and lower bounds respectively for the parameter and are generally hard constraints on the values.

If fields require inputs other than the standard 4 attributes, the additional attributes will be described in the documentation for that field. 


Electron Parameters
~~~~~~~~~~~~~~~~~~~~
These parameters are found in the ``parameters:electron`` section of the input deck and describe various properties of the electrons.

- ``Te`` is the electron temperature in keV

- ``ne`` is the electron density in :math:`10^{20}` cm\ :sup:`-3`

- ``fe`` is the electron velocity distribution function. Like other parameters it has an active flag which controls whether it is being fit, but the other fields function differently. 
    
    - ``dim`` specifies the dimensionality of the distribution function. Options are 1 for a distribution function that is spherically symmetric in velocity space and 2 for a distribution function that has x and y dependence but is integrated along z where z is the probe's polarization direction. 2D options are very computationally expensive and most 2D effects are not visible in standard Thomson scattering, therefore 2D should only be used in forward modeling.
    
    - ``type`` specifies the class of distribution function. 1D Options are ``dlm`` for a Dum-Langdon-Matte distribution also known as a super-gaussian,  ``mx`` for a Maxwellian distribution, and ``arbitrary`` for an arbitrary numerically defined distribution. For most purposes the ``dlm`` option is sufficient and recommended, while the ``mx`` option only works if ``fe:active`` is set to False but ``dlm`` can also produce a Maxwellian distribution if the super-Gaussian order is set to 2. The ``arbitrary`` option allows the code to alter the distribution function point by point and is only tractable for special cases. 2D options are ``arbitrary`` for an arbitrary numerically defined distribution in 2D and ``spherical`` for a distribution based on spherical harmonics. Regardless of the dimensionality or model the distribution function is projected onto a Cartesian grid for use in the Thomson scattering calculations.

    - ``nvx`` is the number of velocity points along each direction of the final velocity grid

    - ``params`` is the container for various model specific parameters.

        - ``Nl`` is the maximum spherical-harmonic degree used to define the distribution function when the ``sphericalharmonic`` model is used. TSADAR uses the real cosine harmonics with JAX's ``(degree=l, order=m)`` convention. The physical ``(vx, vy)`` plane is embedded as ``(X, Y, Z) = (vy, 0, vx)``; therefore the ``(1, 0)`` and ``(1, 1)`` modes point along ``vx`` and ``vy``. This is a projected 3-D spherical-harmonic angular basis, not a 2-D polar Fourier basis.

        - ``dtx`` is the Knudsen number along the x direction for Spitzer-Harm type or the more general Mora-Yahi type distribution functions. This parameter set the amount of heat flux in this model. This parameter is only relevant if ``flm_type`` is set to ``mora-yahi`` and is ignored otherwise. See `Mora and Yahi (1982) <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.26.2259>`_.
        
        - ``dty`` is the Knudsen number along the y direction for the Mora-Yahi type distribution functions.

        - ``flm_type`` specifies the form of the spherical harmonic coefficients when the ``spherical`` model is used. These coefficients are the standard harmonic decomposition of the distribtuion function. The options are ``mora-yahi`` to generate the coefficients based on the Mora-Yahi distribution function (a generalization of the Spitzer-Harm distribution function that allows super-Gaussian distributions) and ``arbitrary`` to allow the coefficients to be fit with each velocity point being its own free parameter. 
        
        - ``m`` is the super-Gaussian order for the DLM distribution function. At m=2 this reduces to a Maxwellian distribution. When fitting the distribution function is activated with ``fe:active`` and ``type: dlm`` then the super-Gaussian order will be fit as the free parameter of the distribution. the ``val``, ``lb``, and ``ub`` fields for this parameter function the same as other parameters.

            - ``matte`` is a boolean which determines if the DLM distribution function follow the Matte formula which specifies the super-Gaussian order based on the electron temperature, ionization state, and overlapped laser intensity. This is an alternative to fitting the super-Gaussian order and has been found to be a good model for many experiments. If ``matte`` is set to True then the ``m`` parameter will be ignored and the super-Gaussian order will be determined by the Matte formula and therefore is not compatible with fitting the super-Gaussian order. When ``matte`` is used, the super-Gaussian will still evolve as the temperature and ionization state evolve to maintain consistency with the Matte formula. See `Matte et al. (1988) <https://iopscience.iop.org/article/10.1088/0741-3335/30/12/004>`_.
            - ``intens`` is the :math:`3\omega` laser intensity in units of :math:`10^{14}` W/cm\ :sup:`-2` and is used in the Matte formula to determine the super-Gaussian order. This is only relevant if ``matte`` is set to True.

        - ``nvr`` is the number of velocity points in the radial direction for the spherical harmonic decomposition. This is only relevant if ``type: spherical`` and is ignored otherwise.

        - ``smooth`` is a boolean which determines if the distribution function will be smoothed at every iteration of the fit. This is especially important if any of the distribution types are `arbitrary` as the large number of free parameters can lead to unphysical oscillations in the distribution function. If ``smooth`` is set to True then the distribution function will be convolved with a window specified by ``window`` at every iteration of the fit.

        - ``window`` specifies the type and size of the window used to smooth the distribution function if ``smooth`` is set to True. 
        
            - ``len`` defines the number of points used in the smoothing window, and is submitted as a parameter to the window function specified by ``type``.

            - ``type`` specifies the type of window used for smoothing. Current options are ``hanning`` and ``butterworth`` which are all standard smoothing functions.

.. :note:: Normalized velocities are used with these distributions so distributions at different temperatures converge to the same distribution. 



Ion Parameters
~~~~~~~~~~~~~~~~
These parameters are found in the ``parameters:ion-1`` section of the input deck and describe various properties of the ions. If multiple ion species are being fit additional sections such as ``parameters:ion-2`` which have the same parameters as ``parameters:ion-1`` can be appended.

- ``Ti`` is the ion temperature in keV
    
    - ``same`` is a special field for ion temperature; if multiple ions are used subsequent ions can have this boolean set to True in order to use a single ion temperature for all ion species

- ``Z`` is the average ionization state of the parent species

- ``A`` is the atomic mass number this should always be a constant and not fit

- ``fract`` is the element ratio for multispecies plasmas; the sum of ``fract`` for all species is held constant at 1. So if there are 2 ion species and the first has ``fract: 0.3`` then 30% of the ions are of that species. If the sum of the ``fract`` fields for all species is not 1 then the code will automatically rescale them to sum to 1. It is possible to fit the ``fract`` field for one or more species, but this parameter is self normalized between 0 and 1 so it does not have bounds.

- ``Va`` is the species lab-frame flow velocity in 10^6 cm/s. For 2-D distributions,
  ``angle`` gives its direction from the x-axis in degrees.
    
.. versionadded:: 0.2.0
    Va is now associated with indivdual ions allowing for counter-streaming plasmas or other complicated flow configurations. In previous versions the flow was a general parameter that was applied to all ions.

General parameters
~~~~~~~~~~~~~~~~~~~~
These parameters are found in the ``parameters:general`` section of the input deck and describe various properties of the plasma that are not specific to a single species.

- ``amp1`` is the blue-shifted EPW amplitude multiplier with 1 being the maximum of the data

- ``amp2`` is the red-shifted EPW amplitude multiplier with 1 being the maximum of the data

- ``amp3`` is the IAW amplitude multiplier with 1 being the maximum of the data

- ``lam`` is the probe wavelength in nanometers; a small shift (<5 nm) can be used to mimic wavelength calibration uncertainty

- ``Te_gradient`` is the electron temperature spatial gradient in % of ``Te``. ``Te`` will take the form ``linspace(Te-Te*Te_gradient.val/200, Te+Te*Te_gradient.val/200, Te_gradient.num_grad_points)``.

    - ``num_grad_points`` is the number of points in the spatial gradient. A higher number of points allows for a smoother gradient but also increases the computational cost of the fit as a spectrum is calculated for each gradient point. ``num_grad_points`` should be left at 1 if no gradient is desired.

    - ``val`` is the value of the gradient in % of ``Te``. For example if ``Te`` is 1 keV and ``val`` is 50 then the gradient will go from 0.75 keV to 1.25 keV.

- ``ne_gradient`` is the electron density spatial gradient in % of ``ne``. ``ne`` will take the form ``linspace(ne-ne*ne_gradient.val/200, ne+ne*ne_gradient.val/200, ne_gradient.num_grad_points)``. This works the same as ``Te_gradient`` but for the electron density instead of the temperature.

- ``ud`` is the electron drift velocity in 10^6 cm/s relative to the charge-weighted
  ion bulk velocity, ``sum(Z * fract * Va) / sum(Z * fract)``. For 2-D distributions,
  ``angle`` gives the drift direction from the x-axis in degrees.

MLFlow
^^^^^^^^

When running all code output is managed by MLFlow. This included the fitted parameters as well as the automated plots.
A copy of the inputs decks will also be saved by MLFlow for easier reference. The MLFlow options can be found in the ``mlflow:`` section.

- ``experiment`` is the name of the experiment folder that the run will be associated with. If running locally the experiment folder will be created in the ``mlruns`` folder in the current directory but will be named with a UUID. The MLFlow UI will display the experiment name and run name for easier navigation.

- ``run`` is the name of the analysis or forward model run. Run names do not need to be unique as they are also associated with a UUID. It is recommended that this is changed before each run.


.. _Data inputs:

Data
^^^^^

The ``data:`` section contains the specifics on which shot and what region of the shot should be analyzed. As well as all the extended options for how to process the data before fitting and how to apply background subtraction.

- ``shotnum`` is the OMEGA shot number. For non-OMEGA data please contact the developers.

- ``load_ion_spec`` is a boolean determining if IAW data will be loaded.

- ``load_ele_spec`` is a boolean determining if EPW data will be loaded.

- ``fit_IAW`` is a boolean determining if IAW data will be fit by including it in the loss metric.

- ``fit_EPWb`` is a boolean determining if the blue shifted EPW data will be fit by including it in the loss metric.

- ``fit_EPWr`` is a boolean determining if the red shifted EPW data will be fit by including it in the loss metric.

- ``absolute_timing`` is a boolean determining if the timing of the lineouts will be automatically determined based off the known relation fo the fiducials to t0 for the OMEGA data. This is recommended for time resolved data as long as the fiducials are visible in the raw data, as its the most accurate way of determining t0.

- ``lineouts`` specifies the region of the data to take lineouts from.

    - ``type`` specifies the units that the lineout locations are in. Options are ``um`` for microns in imaging data, ``ps`` for picoseconds in time resolved data, ``pixel`` is the general option to specify locations in pixel numbers.
  
    - ``start`` the first location where a lineout will be taken.

    - ``end`` the last location where a lineout will be taken.

    - ``skip`` the distance between lineouts in the same units specified by ``type``.

- ``background`` specifies the location and algorithm for background analysis.

    - ``type`` there are multiple background algorithms available. This field is used to select the appropriate one. The options are ``Fit`` in order to fit a model to the background, ``Shot`` in order to subtract a background shot, and ``pixel`` to specify a location with background data to be subtracted.

    - ``slice`` is the location for the background algorithm. If ``Fit`` or ``pixel`` are used this is the pixel location for the background lineout. If ``Shot`` is used this is the shot number for the background shot.

    - ``bg_alg`` specifies the simple function that is fit to the background when ``type`` is set to ``Fit``. The options are ``rat11``, ``rat21``, ``exp2`` and ``power2`` following the naming convention of the Matlab functions with the same names. These functions are all simple 3 or 4 parameter functions.

    -  ``bg_alg_domain`` sets the domain over which the background is fit. This field has 4 values, and the domain is constructed as a linear domain between the first two pixel values and the second two. For example if ``bg_alg_domain`` is [0, 100, 200, 300] then the background will be fit on a linear domain between pixels 0 and 100 and a linear domain between pixels 200 and 300. This allows for exclusion of regions with signal from the background fit. The algorithms other than ``Fit`` also rescale the background to match the data in the same domain as specified by ``bg_alg_domain``. 

    .. hint:: The most common reason fits look poor is due to a poor background subtraction which often results from a background domain that either includes signal or other features that are not representative of the true background such as fiducials of blocked regions. 

    - ``bg_alg_params`` are the starting values for the model selected by ``bg_alg``. ``power2`` and ``rat11`` take 3 arguments while ``exp2`` and ``rat21`` take 4.

    - ``bg_smothing_window`` is the size of the smoothing window applied to the background when ``type`` is set to ``pixel``. A larger window will result in a smoother background but can smear features such as the notch filter often used for EPWs affecting the fit quality and potentially fitted parameters.

    - ``bg_subtract`` is a boolean determining if the background will be subtracted from the data or if the background will be added to the fit model. Subtracting the background from the data is more common but can amplify noise and can affect statistical measures like chi-squared. 

    - ``report_background`` is a boolean determining if the background will plotted with the data in its own plots. This is useful for diagnosing background subtraction issues but is not necessary for the fitting process.

- ``bgshotmult`` multiplier on all background from a separate data file; this is used to rescale the background shot when ``type`` is set to ``Shot``. This can be used to account for differences in laser energy or other factors between the main shot and the background shot.

- ``bgscaleE`` multiplier on the background applied to EPW analysis.

- ``bgscaleI`` multiplier on the background applied to IAW analysis.

- ``dpixel`` determines the width of a lineout in pixels; the width of a lineout is 2*``dpixel`` + 1 centered about the values in ``lineouts``. 2 or 3 is a common value for this parameter to help improve the signal to noise ratio of the lineouts that are being fit, but a larger value can be used for very noisy data at the cost of potentially smearing features in the data due to changing conditions as a function of time or space.

- ``ele_lam_shift`` shifts the central frequency given by ``lam`` in the EPW spectrum, given in nm. This can be used to account for wavelength calibration uncertainty between different the electron and ion data.

- ``ele_t0`` shifts the time denoted as 0 for time resolved EPW data, given in the same units as the lineouts (ps or pixel). This can be used to set t0 to a specific location within the dataset.

- ``ion_t0_shift`` shifts the time denoted as 0 for time resolved IAW data relative to the EPW's zero, given in the same units as the lineouts (ps or pixel). This can be used to account for timing uncertainty between the electron and ion data.

- ``shotDay`` changes the default search path for analysis on a shot day *removal likely*

- ``launch_data_visualizer`` is a boolean determining if plots will be produced of the entire data set with the fitting regions highlighted. This is useful for verifying fitting regions and is generally recommended.

- ``fit_rng`` specifies the regions of the data to include in calculation of the fit metric (i.e. included in the fit)

    - ``blue_min`` starting wavelength for the analysis of the blue shifted EPW in nm

    - ``blue_max`` ending wavelength for the analysis of the blue shifted EPW in nm, must be larger than ``blue_min``
  
    - ``red_min`` starting wavelength for the analysis of the red shifted EPW in nm, must be larger than ``blue_max``

    - ``red_max`` ending wavelength for the analysis of the red shifted EPW in nm, must be larger than ``red_min``

    - ``iaw_min`` starting wavelength for the analysis of the IAW in nm

    - ``iaw_max`` ending wavelength for the analysis of the IAW in nm, must be larger than ``iaw_min`` and ``iaw_cf_max``

    - ``iaw_cf_min`` starting wavelength for a central feature in the IAW that is to be excluded from analysis in nm. This can be used to exclude s stray light feature or other zero-frequency features between the IAWs. While its not possible to turn off ``iaw_cf_min`` and ``iaw_cf_max`` can be arbitrarily close together to effectively exclude a single pixel.

    - ``iaw_cf_max`` ending wavelength for a central feature in the IAW that is to be excluded from analysis in nm, must be larger than ``iaw_cf_min``

    - ``forward_epw_start`` first wavelength center in nm for the EPW calculation in forward mode. For detector-integrated ARTS2D, the outer detector edge is inferred by half-spacing extrapolation.

    - ``forward_epw_end`` last wavelength center in nm for the EPW calculation in forward mode. For detector-integrated ARTS2D, the outer detector edge is inferred by half-spacing extrapolation.
    
    - ``forward_iaw_start`` starting wavelength in nm for the IAW calculation for forward model only
    
    - ``forward_iaw_end`` ending wavelength in nm for the IAW calculation for forward model only


- ``ion_loss_scale`` multiplier on the IAW component of the fit metric, allows for balancing of the electron and ion contribution to the total loss. This is useful for data with differing signal levels or to emphasize the fit quality of one feature over another. 

- ``probe_beam`` identifies the beam on OMEGA used as the probe, automatically adjusts the scattering angle and finite aperture calculations. Currently available options are P9 (standard for most experiments), B12, B15, B23, B26, B35, B42, B46, B58, and B62.

Optimizer
^^^^^^^^^^^

The ``optimizer:`` section includes options specifying the behavior of the optimizer used to fit the data.

- ``batch_size`` number of lineouts to be fit simultaneously; the code is designed to fit multiple lineouts at once in order to speed up the fitting process by taking advantage of parallelization on a GPU. Selecting an appropriate batch size requires balancing speed with potential plasma condition changes from one lineout to another. For example, if the plasma conditions are changing rapidly as a function of time then it may be best to use a smaller batch size to avoid fitting lineouts with significantly different plasma conditions at the same time. On the other hand, if the plasma conditions are relatively constant across the lineouts being fit then a larger batch size can be used to speed up the fitting process. Generally, a batch size of 4 or 6 is best form the majority of cases.

- ``method`` gradient descent algorithm employed by the minimizer; this can be ``l-bfgs-b`` in which case scipy's implementation of the L-BFGS-B algorithm will be used, or almost any of the optimizers from the optax library which include ``adam``, ``sgd``, and ``rmsprop``. For a full list see the `optax documentation <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_. L-BFGS-B is recommended for simpler problems as its more efficient but the optax algorithms provide more control when needed and ``adam`` is usualy a good choice for IAW data and combined fits.

- ``moment_loss`` boolean, adds a penalty to maintain the moments of the EDF when fitting EDFs numerically *needs more testing*

- ``loss_method`` metric minimized in order to match data; ``l2`` is recommended but ``l1``, ``log-cosh``, and ``poisson`` are also available

- ``hessian`` boolean, determines if the hessian will be supplied to the minimizer (*Not recommended; Likely to be removed*)

- ``jit`` boolean, determines if just-in-time compilation will be used for the optimization; TSADAR is designed to be JITed and therefore it is always recommended to use JIT compilation.

- ``learning_rate_init`` initial learning rate for the optimizers that require a learning rate, i.e. any of the optax optimizers. The learning rate is the scale factor for the step taken by the minimizer at each iteration, so a larger learning rate will result in larger steps and a faster but potentially more unstable fit while a smaller learning rate will result in smaller steps and a slower but potentially more stable fit. Each optimizer has a different interpretation of the learning rate and therefore the optimal learning rate can differ between optimizers. This parameter is used in conjunction with the ``learning_rate_final`` parameter to set the learning rate schedule for the fit. This schedule is a cosine decay from the initial learning rate to the final learning rate over the first 75% of the total number of epochs, after which the learning rate is held constant at the final learning rate for the remaining 25% of the epochs.

- ``learning_rate_final`` final learning rate for the optimizers that require a learning rate, i.e. any of the optax optimizers. This is used in conjunction with the ``learning_rate_init`` parameter to set the learning rate schedule for the fit.

- ``param_method`` determines how the plasma parameters are updated during the fit. When an optax minimizer is used in ``method`` then the plasma condition parameters are update based off this method while the distribution function is updated via the ``method`` parameter. Options for ``param_method`` are the same as for ``method``. This allows for different learning rates or stepping algorithms between the distribtuion function and the plasma parameters which can be very useful due to the scale separation between the plasma parameters and the distribution function values. When ``method`` is set to ``l-bfgs-b`` then the plasma parameters will always be updated with L-BFGS-B regardless of the value of ``param_method``.

- ``param_learning_rate`` independent learning rate for the plasma parameters when an optax minimizer is used in ``method`` and ``param_method`` is set to an optax method. This allows for a different learning rate between the plasma parameters.

- ``y_norm`` boolean, normalizes data to a maximum value of 1 to improve minimizer behavior; true values are still used for error analysis

- ``x_norm`` boolean, normalizes data to a maximum value of 1 as an input to the neural network *deprecated*

- ``grad_method`` AD or FD determining if gradients are computed with automatic difference or finite difference, AD is always recommended but FD can be used for performance testing

- ``num_epochs`` max number of iterations of the minimizer for each batch

- ``patience`` number of consecutive steps without an improvement of at least ``min_delta`` before an Optax fit stops early. Smaller improvements are still checkpointed, so the returned loss is always the true best evaluated loss. Set to 0 to disable early stopping.

- ``min_delta`` minimum loss improvement that resets ``patience``.

- ``seed`` seed recorded with optimizer telemetry for reproducibility. The current Optax loops and continuation stages are deterministic and do not randomize their initial parameters.

- ``validate_active_leaves`` boolean, validates before optimization that every parameter marked active in the deck exists as a finite differentiable leaf and has detector-space forward sensitivity in the fitted geometry.

- ``sensitivity_tol`` lower bound for the detector-space JVP norm used by active-leaf validation. The default of zero rejects exactly insensitive parameters.

- ``learning_rate`` scale factor for step sizes taken by the minimizer

- ``parameter_norm`` boolean, determines if the fitted parameters will be rescaled to 0 to 1, this is always recommended as it improves the behavior of the minimizer by keeping all parameters on the same scale, but the true values of the parameters are still used for error analysis and plotting.

- ``refine_factor`` factor used to rescale the EDF domain during multiple minimizations of ARTS data

- ``num_mins`` number of sequential continuation/refinement stages for ARTS data; it does not affect non-ARTS data. These are not independent random restarts: each stage starts from the preceding stage's best checkpoint, and the EDF grid is refined by ``refine_factor`` between stages. The globally best evaluated checkpoint across all stages is returned.

- ``save_state`` boolean, determines if the state of the minimizer will be saved at some regular period during the minimization, primarily used for understanding the behavior of the minimizer

- ``save_state_freq`` how often the state of the minimizer will be saved during the minimization in terms of number of epochs between saves, primarily used for understanding the behavior of the minimizer

- ``sequential`` boolean, if true the final parameters from each batch will be used as initial conditions for the following batch. If false, input deck initial conditions will be used for all lineouts.


Other options
^^^^^^^^^^^^^^^
 
The ``other:`` section includes options specifying the types of data that are being fit and other options on how to perform the fit.

- ``include_gains`` is a boolean determining whether to compute the SRS and SBS amplification of the Thomson scattered light. Having ``include_gains`` set to True will require the pump intensity and beam diameter to be specified in the ``other:`` section of the input deck.

- ``Ipump_14`` is the intensity relevant to computing the SRS and SBS amplification in units of :math:`10^{14}` W/cm\ :sup:`-2`. This is likely the probe beam intensity for most experiments at that is the beam that is overlapped with the scattering volume, but for some experiments it may differ.

- ``beam_diam_um`` is the beam diameter in microns of the beam that pumps SRS and SBS, again this is likely the probe beam diameter. This is used in conjunction with the scattering angle to compute the gain length for the SRS and SBS amplification.

- ``refit`` is a boolean determining if poor fits will attempt to be refit; it is recommended this is only turned on once most fits look good.

- ``refit_thresh`` is the value of the loss metric above which refits will be performed; this threshold must be determined empirically from the other fits.

- ``calc_sigmas`` is a boolean determining if a Hessian will be computed to determine the uncertainty in fitted parameters.

.. deprecated:: 0.2.0
    ``calc_sigmas`` is currently deprecated. Implementation of uncertainty quantification is planned for a future release but the current method of computing sigmas from the Hessian is not robust and therefore has been deprecated until a more robust method can be implemented.

- ``extraoptions`` is a container for some additional options

    - ``spectype`` specifies the type of spectrum such as temporal or spatial, it is populated by the code based on the data file and does not need to be set.

- ``expandedions`` is a boolean determining if a non-linear wavelength grid will be used allowing IAW and EPW spectra to be resolved simultaneously *currently deprecated*.

- ``detector_specs`` is a dictionary that is assigned within the code and stores detector information. It also serves as the fallback calibration source in ``tsadar.data.calibration`` whenever a shot number isn't in the hardcoded calibration table there -- in that case a warning is raised and the values below are used instead of a shot-specific calibration:

    - ``EPWDispersion``, ``IAWDispersion``, ``EPWoffset``, ``IAWoffset`` are the fallback spectral dispersion and offset values (see :ref:`Getting Started <getting started>` for how these are normally determined per-shot).

    - ``EPWtcc`` and ``IAWtcc`` are the fallback target-chamber-center pixel offsets, only required for imaging-type data; if a shot with unknown calibration and ``spectype: imaging`` is analyzed and these are not set, a clear error is raised rather than guessing a value, since a wrong TCC offset would silently miscalibrate the spatial axis.

- ``iawoff`` is a boolean determining if the iaw will be suppressed in plotting of the EPW feature

- ``iawfilter`` is an alternative to ``iawoff`` that suppresses the IAW with a notch filter. The list has 4 elements: boolean for on/off, OD of the filter, spectral width of the filter in nm, and central wavelength of the filter in nm.

- ``CCDsize`` size of the CCD in pixels

- ``flatbg`` flat (applied to all pixes) value added to the background

- ``gain`` CCD counts per photo-electron; the standard OMEGA ROSS has a gain of 144. Gain must be accurate for appropriate use of Poisson statistics but the gain is generaly not important for the fitting process as the data is normalized by default.

- ``points_per_pixel`` number of wavelength points computed per detector pixel by the legacy sampled-spectrum path. ARTS2D does not use this setting to resolve a narrow resonance: it integrates the continuous spectrum directly into the calibrated detector-bin edges.

- ``resonance_quadrature`` controls the ARTS2D detector-bin integration and is enabled by default. ``root_scan_panels`` (4096 by default) is the fine grid used only to bracket sign-changing zeros of the real dielectric, while ``integration_panels`` (256 by default) controls the coarser regular quadrature grid. Separating them resolves closely spaced roots without paying the fine-grid cost in the detector response matrix. ``regular_order`` and the even ``root_order`` set the Gauss--Legendre rules away from and near a root; ``neighbor_panels`` expands the root-mapped neighborhood; ``max_roots`` is the fixed root capacity (16 by default); ``bisection_iterations`` controls the differentiable root solve; and ``tail_sigma`` extends the source integration domain beyond the detector by that many spectral-IRF standard deviations. When ``iawfilter`` is enabled, every filter edge strictly inside that source domain is inserted as an exact integration breakpoint: attenuation is applied to the continuous source spectrum before Gaussian spectral blur and detector integration, including when a filter cuts through a detector bin. ``iawoff`` remains a detector-space mask. ``scan_phase`` is intended for convergence tests and must lie strictly between -1 and 1. ``map_batch_size`` trades device parallelism for peak memory across scattering geometries. A detected root overflow, zero-width resonance, invalid breakpoint, invalid detector geometry, or non-finite evaluation produces a non-finite model rather than silently returning a partial integral.

- ``ang_res_unit`` is the number of pixels in an angular resolution unit for ARTS

- ``lam_res_unit`` is the number of pixels in an specular resolution unit for ARTS


Plotting
^^^^^^^^^^^
These options only alter the plotting of that data and fits; they do not influence the fits. The plotting and fitting ranges are intentionally decoupled in order to allow visualization of data that is not fit. 

- ``n_sigmas`` is the number of standard deviations to plot the uncertainty region over

- ``rolling_std_width`` number of lineouts used to calculate the standard deviation for the moving window error region

- ``data_cbar_u`` upper limit for the colorbar in plotting the data and fit; also limits the lineout plots. Can be given as a number of counts or as ``data`` to automatically use the maximum of the data

- ``data_cbar_l`` lower limit for the colorbar in plotting the data and fit; also limits the lineout plots. Can be given as a number of counts or as ``data`` to automatically use the minimum of the data

- ``ion_window_start`` determines the spectral range of the IAW fit plots; this gives the lower bound in nm

- ``ion_window_end`` determines the spectral range of the IAW fit plots; this gives the upper bound in nm

- ``ele_window_start`` determines the spectral range of the EPW fit plots; this gives the lower bound in nm

- ``ele_window_end`` determines the spectral range of the EPW fit plots; this gives the upper bound in nm

- ``detailed_breakdown`` when active all lineout plots will show the constituent curves within the fit model. This includes the angle extremes, gradient extremes, IRF, and background.
