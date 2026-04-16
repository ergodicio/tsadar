.. _inputs_deck:

Inputs Deck
-------------------------------------


The code uses two input decks, which are located in ``configs/1d``. 
The inputs deck contains parameters that are frequently altered. Since frequently altered is subjective, the fields in the inputs deck may change based on user's needs. Any parameter can be removed from the inputs deck or added from the defaults deck. Here we will describe some of the more commonly used parameters in the inputs deck, but for a full list of parameters and their descriptions see the :ref:`Defaults page <configuring-the-default>`.

.. Tip:: The inputs deck will override the defaults deck when values conflict. 

.. contents:: Input Deck Fields
   :local:
   :depth: 3


Parameters
^^^^^^^^^^^^

All fitting parameters are found in the ``parameters:`` section of the input decks. The fitting parameters are broken up based off their scope.
There must be at least 3 subfields referencing the different species in ``parameters`` called ``electron``, 
``ion-1``, and ``general``. If the problem has multiple ion species additional ion species fields can be added 
with increasing index numbers, i.e. ``ion-2``, ``ion-3``.

Each species has associated fitting parameters which generally have 4 attributes: ``val``, ``active``, 
``lb``, and ``ub``. ``val`` is the initial value for a fitted parameter or the constant value for a
static parameter. ``active`` is a boolean determining if the parameter is being fit, so parameters with ``active: False``
will be held at the value in ``val``. ``ub`` and ``lb`` are upper and lower bounds respectively 
for the parameter and are generally hard constraints on the values.

If fields require inputs other than the standard 4 attributes, the additional attributes will be described in the documentation for that field. 


Electron Parameters
~~~~~~~~~~~~~~~~~~~

These parameters are found in the ``parameters:electron`` section of the input deck and describe various properties of the electrons.

- ``Te`` is the electron temperature in keV

- ``ne`` is the electron density in :math:`10^{20}` cm\ :sup:`-3`

- ``fe`` is the electron velocity distribution function. Like other parameters it has an active flag which controls whether it is being fit, but the other fields function differently. 
    
    - ``type`` specifies the class of distribution function. Options are ``dlm`` for a Dum-Langdon-Matte distribution also known as a super-gaussian,  ``mx`` for a Maxwellian distribution, and ``arbitrary`` for an arbitrary numericaly defined distribution. For most purposes the ``dlm`` option is sufficent and recomended, while the ``mx`` option only works if ``fe:active`` is set to False but ``dlm`` can also produce a Maxwellian distribution if the super-Gaussian order is set to 2. The ``arbitrary`` option allow the code to alter the distribution function point by point and is only tractable for special cases. 

    - ``params`` is the container for various model specific parameters.

        - ``m`` is the super-Gaussian order for the DLM distribution function. At m=2 this reduces to a Maxwellian distribution. When fitting the distribution function is activated with ``fe:active`` and ``type: dlm`` then the super-gausian order will be fit as the free parameter of the distribution. the ``val``, ``lb``, and ``ub`` fields for this parameter function the same as other parameters.

            - ``matte`` is a boolean which determines if the DLM distribution function follow the Matte formula which specifies the super-gaussian order based on the electron temperature, ionization state, and overlapped laser intensity. This is an alternative to fitting the super-Gaussian order and has been found to be a good model for many experiements. If ``matte`` is set to True then the ``m`` parameter will be ignored and the super-Gaussian order will be determined by the Matte formula and therefore is not compatable with fitting the super-Gaussian order. When ``matte`` is used, the super-Gaussian will still evolve as the temperature and ionization state evolve to maintain consistency with the Matte formula. See `Matte et al. (1988) <https://iopscience.iop.org/article/10.1088/0741-3335/30/12/004>`_.
            - ``intens`` is the :math:`3\omega` laser intensity in units of :math:`10^{14}` W/cm\ :sup:`-2` and is used in the Matte formula to determine the super-Gaussian order. This is only relevant if ``matte`` is set to True.

.. :note:: Normalized velocities are used with these distribtuions so distributions at different temperatures converge to the same distribution. 



Ion Parameters
~~~~~~~~~~~~~

These parameters are found in the ``parameters:ion-1`` section of the input deck and describe various properties of the ions. If multiple ion species are being fit additional sections such as ``parameters:ion-2`` which have the same parameters as ``parameters:ion-1`` can be appended.

- ``Ti`` is the ion temperature in keV
    
    - ``same`` is a special field for ion temperature; if multiple ions are used subsequent ions can have this boolean set to True in order to use a single ion temperature for all ion species

- ``Z`` is the average ionization state of the parent species

- ``A`` is the atomic mass number this should always be a constant and not fit

- ``fract`` is the element ratio for multispecies plasmas; the sum of ``fract`` for all species is held constant at 1. So if there are 2 ion species and the first has ``fract: 0.3`` then 30% of the ions are of that species. If the sum of the ``fract`` fields for all species is not 1 then the code will automatically rescale them to sum to 1. It is possible to fit the ``fract`` field for one or more species, but this parameter is self normalized between 0 and 1 so it does not have bounds.

- ``Va`` is the species flow velocity in 10^6 cm/s 
    
.. versionadded:: 0.2.0
    Va is now associated with indivdual ions allowing for counter-streaming plasmas or other complicated flow configurations. In previous versions the flow was a general parameter that was applied to all ions.

General parameters
~~~~~~~~~~~~~~~~~~

These parameters are found in the ``parameters:general`` section of the input deck and describe various properties of the plasma that are not specific to a single species.

- ``amp1`` is the blue-shifted EPW amplitude multiplier with 1 being the maximum of the data

- ``amp2`` is the red-shifted EPW amplitude multiplier with 1 being the maximum of the data

- ``amp3`` is the IAW amplitude multiplier with 1 being the maximum of the data

- ``lam`` is the probe wavelength in nanometers; a small shift (<5 nm) can be used to mimic wavelength calibration uncertainty

- ``Te_gradient`` is the electron temperature spatial gradient in % of ``Te``. ``Te`` will take the form ``linspace(Te-Te*Te_gradient.val/200, Te+Te*Te_gradient.val/200, Te_gradient.num_grad_points)``.

    - ``num_grad_points`` is the number of points in the spatial gradient. A higher number of points allows for a smoother gradient but also increases the computational cost of the fit as a spectrum is calculated for each gradient point. ``num_grad_points`` should be left at 1 if no gradient is desired.

    - ``val`` is the value of the gradient in % of ``Te``. For example if ``Te`` is 1 keV and ``val`` is 50 then the gradient will go from 0.75 keV to 1.25 keV.

- ``ne_gradient`` is the electron density spatial gradient in % of ``ne``. ``ne`` will take the form ``linspace(ne-ne*ne_gradient.val/200, ne+ne*ne_gradient.val/200, ne_gradient.num_grad_points)``. This works the same as ``Te_gradient`` but for the electron density instead of the temperature.

- ``ud`` is the electron drift velocity (relative to the ions) in 10^6 cm/s

MLFlow
^^^^^^^^

When running all code output is managed by MLFlow. This included the fitted parameters as well as the automated plots.
A copy of the inputs decks will also be saved by MLFlow for easier reference. The MLFlow options can be found in the ``mlflow:`` section.

- ``experiment`` is the name of the experiment folder that the run will be associated with. If running locally the experiment folder will be created in the ``mlruns`` folder in the current directory but will be named with a UUID. The MLFlow UI will display the experiment name and run name for easier navigation.

- ``run`` is the name of the analysis or forward model run. Run names do not need to be unique as they are also associated with a UUID. It is recommended that this is changed before each run.


Data
^^^^^
The ``data:`` section contains the specifics on which shot and what region of the shot should be analyzed.

- ``shotnum`` is the OMEGA shot number. For non-OMEGA data please contact the developers.

- ``load_ion_spec`` is a boolean determining if IAW data will be loaded.

- ``load_ele_spec`` is a boolean determining if EPW data will be loaded.

- ``fit_IAW`` is a boolean determining if IAW data will be fit by including it in the loss metric.

- ``fit_EPWb`` is a boolean determining if the blue shifted EPW data will be fit by including it in the loss metric.

- ``fit_EPWr`` is a boolean determining if the red shifted EPW data will be fit by including it in the loss metric.

- ``lineouts`` specifies the region of the data to take lineouts from.

    - ``type`` specifies the units that the lineout locations are in. Options are ``um`` for microns in imaging data, ``ps`` for picoseconds in time resolved data, ``pixel`` is the general option to specify locations in pixel numbers.
  
    - ``start`` the first location where a lineout will be taken.

    - ``end`` the last location where a lineout will be taken.

    - ``skip`` the distance between lineouts in the same units specified by ``type``.

- ``background`` specifies the location where the background will be analyzed.

    - ``type`` there are multiple background algorithms available. This field is used to select the appropriate one. The options are ``Fit`` in order to fit a model to the background, ``Shot`` in order to subtract a background shot, and ``pixel`` to specify a location with background data to be subtracted.

    - ``slice`` is the location for the background algorithm. If ``Fit`` or ``pixel`` are used this is the pixel location for the background lineout. If ``Shot`` is used this is the shot number for the background shot.


Other options
^^^^^^^^^^^^^^^
 
The ``other:`` section includes options specifying the types of data that are being fit and other options
on how to perform the fit.

- ``include_gains`` is a boolean determining whether to compute the SRS and SBS amplification of the Thomson scattered light. Having ``include_gains`` set to True will require the pump intensity and beam diameter to be specified in the ``other:`` section of the input deck.

- ``Ipump_14`` is the intensity relevant to computing the SRS and SBS amplification in units of :math:`10^{14}` W/cm\ :sup:`-2`. This is likely the probe beam intensity for most experiments at that is the beam that is overlapped with the scattering volume, but for some experiments it may differ.

- ``beam_diam_um`` is the beam diameter in microns of the beam that pumps SRS and SBS, again this is likely the probe beam diameter. This is used in conjunction with the scattering angle to compute the gain length for the SRS and SBS amplification.

- ``refit`` is a boolean determining if poor fits will attempt to be refit; it is recommended this is only turned on once most fits look good.

- ``refit_thresh`` is the value of the loss metric above which refits will be performed; this threshold must be determined empirically from the other fits.

- ``calc_sigmas`` is a boolean determining if a Hessian will be computed to determine the uncertainty in fitted parameters.

.. deprecated:: 0.2.0
    ``calc_sigmas`` is currently deprecated. Implementation of uncertainty quantification is planned for a future release but the current method of computing sigmas from the Hessian is not robust and therefore has been deprecated until a more robust method can be implemented.
