
Inputs Deck
-------------------------------------

The code uses two input decks, which  are located in **inverse-thomson-scattering/configs/1d**. 
The inputs deck contains all the commonly altered options which will be discussed in detail in the following section.
More information on the specifics of each deck can be found by clicking on the cards bellow. 
The defaults deck contains additional options and default values for all options.

.. grid:: 2

    .. grid-item-card::  Inputs.yaml
        :link: inputs
        :link-type: ref

        Primary input deck 

    .. grid-item-card::  Defaults.yaml
        :link: configuring-the-default
        :link-type: ref

        Secondary input deck 





.. Tip:: The primary input deck will override the secondary input deck when values conflict. 

Parameters
^^^^^^^^^^^^

All fitting parameters are found in the :bdg-success-line:`parameters:` section of the input decks. These parameters are separated into
species fields. These species can be called anything but best practice is to name them :bdg-success-line:`species1` through
:bdg-success-line:`speciesn`.

Each species must have a :bdg-success-line:`type` field which specifies weather the species is an electron, ion, or the unique general
type. These three key word should be entered as fields of the :bdg-success-line:`type` field. Any number of ion species can be included,
and while the code currently only supports one electron species this will be expanded in the future. The :bdg-success-line:`general`
species is used to specify properties that apply to the system as a whole and are not unique to a species, therefore only one can be included.

Within each species live the parameters that are relevent to fitting that species, each parameter has at least 4
attributes. :bdg-success-line:`val` is the initial value used as a starting condition for the minimizer. :bdg-success-line:`active` is a boolean
determining if a parameter is to be fit, i.e :bdg-success-line:`active: True` means a parameter with be fit and :bdg-success-line:`active: False` means
a parameter with be held constant at :bdg-success-line:`val`. :bdg-success-line:`ub` and :bdg-success-line:`lb` are upper and lower bounds respectively 
for the parameters.

Electron parameters
^^^^^^^^^^^^^^^^^^^
- :bdg-success-line:`Te` is the electron temperature in keV

- :bdg-success-line:`ne` is the electron density in 10^20 cm^-3

- :bdg-success-line:`m` is the electron distribution function super-Gaussian parameter

- :bdg-success-line:`fe` contains additional options for controlling the distribution function *more info to come*


Ion parameters
^^^^^^^^^^^^^^^^^^^
- :bdg-success:`Ti` is the ion temperature in keV
    - :bdg-success-line:`same` is a special field for ion temperature, if multiple ions are used subsequent ions can have this booleanset to True in order to use a single ion temperature for all ion species

- :bdg-success-line:`Z` is the average ionization state

- :bdg-success-line:`A` is the atomic mass

- :bdg-success-line:`fract` is the element ratio for multispecies plasmas, the sum of fract for all species should be 1

General parameters
^^^^^^^^^^^^^^^^^^^

- :bdg-success-line:`amp1` is the blue-shifted EPW amplitude multiplier with 1 being the maxmimum of the data

- :bdg-success-line:`amp2` is the red-shifted EPW amplitude multiplier with 1 being the maxmimum of the data

- :bdg-success-line:`amp3` is the IAW amplitude multiplier with 1 being the maxmimum of the data

- :bdg-success-line:`lam` is the probe wavelength in nanometers, small shift (<5nm) can be used to mimic wavelength calibration uncertainty

- :bdg-success:`Te_gradient` is the electron temperature spatial gradient in % of :bdg-success-line:`Te`. 
    - :bdg-success-line:`Te` will take the form 
    - :bdg-success-line:`linspace(Te-Te*Te_gradient.val/200, Te+Te*Te_gradient.val/200, Te_gradient.num_grad_points)` 
    - :bdg-success-line:`val!=0` will calculate the spectrum with a gradient.

- :bdg-success:`ne_gradient` is the electron density spatial gradient in % of :bdg-success-line:`ne`. 
    - :bdg-success-line:`ne` will take the form :bdg-success-line:`linspace(ne-ne*ne_gradient.val/200, ne+ne*ne_gradient.val/200, ne_gradient.num_grad_points)`
    - :bdg-success-line:`val!=0` will calculate the spectrum with a gradient.

- :bdg-success-line:`ud` is the electron drift velocity (relative to the ions) in 10^6 cm/s

- :bdg-success-line:`Va` is the plasma fluid velocity or flow velocity in 10^6 cm/s

MLFlow
^^^^^^^^

When running all code output is managed by MLFlow. This included the fitted parameters as well as the automated plots.
A copy of the inputs decks will also be saved by MLFlow for easier reference. The MLFlow options can be found at the
end of :bdg-success-line:`inputs.yaml` in the :bdg-success:`mlflow:` section.

- :bdg-success-line:`experiment` is the name of the experiment folder that the run will be associated with.

- :bdg-success-line:`run` is the name of the analysis or forward model run. Run names do not need to be unique as many runs can be created with the same name. It is recomended that this is changed before each run.


.. _Data inputs:
Data
^^^^^
The :bdg-success-line:`data:` section contains the specifics on which shot and what region of the shot should be analyzed.

- :bdg-success-line:`shotnum` is the OMEGA shot number. For non-OMEGA data please contact the developers.

- :bdg-success:`lineouts` specifies the region of the data to take lineouts from.
    - :bdg-success-line:`type` specifies the units that the linout locations are in. 
        - :bdg-success-line:`um` for microns in imaging data.
        - :bdg-success-line:`ps` for picoseconds in time resolved data.
        - :bdg-success-line:`pixel` is the general option to specify locations in pixel numbers.
    - :bdg-success-line:`start` the first location where a lineout will be taken.

    - :bdg-success-line:`end` the last location where a lineout will be take

    - :bdg-success-line:`skip` the distance between lineouts in the same units specified by :bdg-success-line:`type`

- :bdg-success:`background` specifies the location where the background will be analyzed.
    - :bdg-success-line:`type` there are multiple background algorithms availible. This field is used to select the approprate one. The options are :bdg-success-line:`Fit` in order to fit a model to the background, :bdg-success-line:`Shot` in order to subtract a background shot, and :bdg-success-line`pixel` to specify a location with background data to be subtracted.

    - :bdg-success-line:`slice` is the location for the background algorithm. 
        - If :bdg-success-line:`Fit` or :bdg-success-line:`pixel` are used this is the pixel location'
        - If :bdg-success-line:`Shot` is used this is the shot number.


Other options
^^^^^^^^^^^^^^^
 
The :bdg-success:`other:` section includes options specifying the types of data that are being fit and other options
on how to perform the fit.

- :bdg-success-line:`load_ion_spec` is a boolean determining if IAW data will be loaded.

- :bdg-success-line:`load_ele_spec` is a boolean determining if EPW data will be loaded.

- :bdg-success-line:`fit_IAW` is a boolean determining if IAW data will be fit by including it in the loss metric.

- :bdg-success-line:`fit_EPWb` is a boolean determining if the blue shifted EPW data will be fit by including it in the loss metric.

- :bdg-success-line:`fit_EPWr` is a boolean determining if the red shifted EPW data will be fit by including it in the loss metric.

- :bdg-success-line:`refit` is a boolean determinging if poor fits will attempt to be refit.

- :bdg-success-line:`refit_thresh` is the value of the loss metric below above which refits will be performed.

- :bdg-success-line:`calc_sigmas` is a boolean determining if a Hessian will be computed to determine the uncertainty in fitted parameters.
