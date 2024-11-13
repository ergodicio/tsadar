.. _configuring-the-default:

Default options
========================================

The code uses two input decks, which  are located in **configs/1d**. 
The defaults deck contains default values for all options in the input deck and other options which will be discussed in detail in the following section.
The inputs deck contains all the commonly altered options.
More information on the specifics of each deck can be found by clicking on the cards bellow. 


.. grid:: 2

    .. grid-item-card::  Inputs.yaml
        :link: inputs_deck
        :link-type: ref

        Primary input deck 

    .. grid-item-card::  Defaults.yaml
        :link: configuring-the-default
        :link-type: ref

        Secondary input deck 


.. Tip:: The primary input deck will override the secondary input deck when values conflict. 

Parameters
^^^^^^^^^^^

- :bdg-success-line:`blur` is an additional smoothing for ARTS *currently depreciated*

- :bdg-success-line:`specCurvature` corrects angle dependence of the central wavleength in ARTS *currently depreciated*

- :bdg-success-line:`fitprops` *currently depreciated, removal likely*


..  _Data default:

Data
^^^^^^^
The :bdg-success:`data:` section contains the specifics on which shot and what region of the shot should be analyzed.

- :bdg-success-line:`shotDay` changes the default search path for analysis on a shot day *removal likely*

- :bdg-success-line:`launch_data_visualizer` is a boolean determining if plots will be produced of the entire data set with the fitting regions highlighted

- :bdg-success:`fit_rng` specifies the regions of the data to include in calculation of the fit metric (i.e. included in the fit)

    - :bdg-success-line:`blue_min` starting wavelength for the analysis of the blue shifted EPW in nm

    - :bdg-success-line:`blue_max` ending wavelength for the analysis of the blue shifted EPW in nm
  
    - :bdg-success-line:`red_min` starting wavelength for the analysis of the red shifted EPW in nm

    - :bdg-success-line:`red_max` ending wavelength for the analysis of the red shifted EPW in nm

    - :bdg-success-line:`iaw_min` starting wavelength for the analysis of the IAW in nm

    - :bdg-success-line:`iaw_max` ending wavelength for the analysis of the IAW in nm

    - :bdg-success-line:`iaw_cf_min` starting wavelength for a central feature in the IAW that is to be excluded from analysis in nm

    - :bdg-success-line:`iaw_cf_max` ending wavelength for a central feature in the IAW that is to be excluded from analysis in nm

- :bdg-success-line:`bgscaleE` multiplier on the background applied to EPW analysis

- :bdg-success-line:`bgscaleI` multiplier on the background applied to IAW analysis

- :bdg-success-line:`bgshotmult` multiplier on all background from a separate data file

- :bdg-success-line:`ion_loss_scale` multiplier on the IAW component of the fit metric, allows for balancing of data with differing signal levels

- :bdg-success-line:`ele_t0` shifts the time denoted as 0 for time resolved EPW data, given in the same units as the lineouts (ps or pixel)

- :bdg-success-line:`ion_t0_shift` shifts the time denoted as 0 for time resolved IAW data relative to the EPWs zero, given in the same units as the lineouts (ps or pixel)

- :bdg-success-line:`ele_lam_shift` shifts the central frequency given by `lam` in the EPW spectrum, given in nm

- :bdg-success-line:`probe_beam` identifies the beam on OMEGA used as the probe, automatically adjusts the scattering angle and finite aperture calculations. Currently availible options are P9, B15, B23, B26, B35, B42, B46, and B58.

- :bdg-success-line:`dpixel` determined the width of a lineout in pixels, the width of a lineout is 2*`dpixel` + 1 centered about the values in `lineouts`


Other options
^^^^^^^^^^^^^^^
 
The :bdg-success:`other:` section includes options specifying the types of data that are being fit and other options
on how to perform the fit.

- :bdg-success-line:`expandedions` is a boolean determining if a non-linear wavelength grid will be used allowing IAW and EPW spectra to be resolved simultaneously *currently depreciated*.

- :bdg-success-line:`PhysParams` is a dictionary that is assigned within the code and stores detector information.

- :bdg-success-line:`iawoff` is a boolean determining if the iaw will be suppressed in plotting of the EPW feature

- :bdg-success-line:`iawfilter` is an alternative to iawoff that suppresses the IAW with a notch filter. The list has 4 elements, boolean for on/off, OD of the filter, spectral width of the filter in nm, and central wavelength of the filter in nm.

- :bdg-success-line:`CCDsize` size of the CCD in pixels

- :bdg-success-line:`flatbg` flat (applied to all pixes) value added to the background

- :bdg-success-line:`gain` CCD counts per photo-electron, the standard OMEGA ROSS has a gain of 144. Gain must be accurate for appropriate use of Poisson statistics

- :bdg-success-line:`points_per_pixel` number of wavelength points computed in the spectrum per pixel in the data being analyzed

- :bdg-success-line:`ang_res_unit` is the number of pixels in an angular resolution unit for ARTS

- :bdg-success-line:`lam_res_unit` is the number of pixels in an specular resolution unit for ARTS


Plotting
^^^^^^^^^^^

- :bdg-success-line:`n_sigmas` is the number of standard deviations to plot the uncertainty region over

- :bdg-success-line:`rolling_std_width` number of lineouts used to calculate the standard deviation for the moving window error region

- :bdg-success-line:`data_cbar_u` upper limit for the colorbar in plotting the data and fit, can be given as a number of counts or as `data` to automatically use the maximum of the data

- :bdg-success-line:`data_cbar_l` lower limit for the colorbar in plotting the data and fit, can be given as a number of counts or as `data` to automatically use the minimum of the data

- :bdg-success-line:`ion_window_start` determines the spectral range of the IAW fit plots, this gives the lower bound in nm

- :bdg-success-line:`ion_window_end` determines the spectral range of the IAW fit plots, this gives the upper bound in nm

- :bdg-success-line:`ele_window_start` determines the spectral range of the EPW fit plots, this gives the lower bound in nm

- :bdg-success-line:`ele_window_end` determines the spectral range of the EPW fit plots, this gives the upper bound in nm


Optimizer
^^^^^^^^^^^

- :bdg-success-line:`method` gradient descent algorithm employed by the minimizer, current options are `adam` and `l-bfgs-b`

- :bdg-success-line:`hessian` boolean, determines if the hessian will be supplied to the minimizer

- :bdg-success-line:`y_norm` boolean, normalizes data to a maximum value of 1 to improve minimizer behavior, true values are still used for error analysis

- :bdg-success-line:`x_norm` boolean, normalizes data to a maximum value of 1 as an input to the neural network *depreciated*

NN
^^^^
Options for the NN version of the code which is currently depreciated.

Dist_fit
^^^^^^^^^^

- :bdg-success:`window` options the smoothing function applied to the distribution function while fitting ARTS

    - :bdg-success-line:`len` length of the smoothing kernel relative the the length of the velocity vector

    - :bdg-success-line:`type` type of smoothing function used, can be `hamming`, `hann`, or `bartlett`

