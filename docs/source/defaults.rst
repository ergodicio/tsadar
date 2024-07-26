.. _configuring-the-default:

Default options
========================================

This code utilizes 2 input deck called `inputs.yaml` and `defaults.yaml`. The primary input deck `inputs.yaml` contains
all the commonly altered options, things like which parameters are going to be fit, the initial values, the shotnumber
and the lineouts you want to fit. The secondary input deck `defaults.yaml` contains additional options that rarely need
to be altered and default values for all options. This page details the additional options in `defaults.yaml` that are
not found in `inputs.yaml`.

Parameters
---------------------

- ``blur`` is an additional smoothing for ARTS *currently depreciated*

- ``specCurvature`` corrects angle dependence of the central wavleength in ARTS *currently depreciated*

- ``fitprops`` *currently depreciated, removal likely*


..  _Data default:
Data
--------------
The ``data:`` section contains the specifics on which shot and what region of the shot should be analyzed.

- ``shotDay`` changes the default search path for analysis on a shot day *removal likely*

- ``launch_data_visualizer`` is a boolean determining if plots will be produced of the entire data set with the fitting regions highlighted

- ``fit_rng`` specifies the regions of the data to include in calculation of the fit metric (i.e. included in the fit)

    - ``blue_min`` starting wavelength for the analysis of the blue shifted EPW in nm

    - ``blue_max`` ending wavelength for the analysis of the blue shifted EPW in nm
  
    - ``red_min`` starting wavelength for the analysis of the red shifted EPW in nm

    - ``red_max`` ending wavelength for the analysis of the red shifted EPW in nm

    - ``iaw_min`` starting wavelength for the analysis of the IAW in nm

    - ``iaw_max`` ending wavelength for the analysis of the IAW in nm

    - ``iaw_cf_min`` starting wavelength for a central feature in the IAW that is to be excluded from analysis in nm

    - ``iaw_cf_max`` ending wavelength for a central feature in the IAW that is to be excluded from analysis in nm

- ``bgscaleE`` multiplier on the background applied to EPW analysis

- ``bgscaleI`` multiplier on the background applied to IAW analysis

- ``bgshotmult`` multiplier on all background from a separate data file

- ``ion_loss_scale`` multiplier on the IAW component of the fit metric, allows for balancing of data with differing signal levels

- ``ele_t0`` shifts the time denoted as 0 for time resolved EPW data, given in the same units as the lineouts (ps or pixel)

- ``ion_t0_shift`` shifts the time denoted as 0 for time resolved IAW data relative to the EPWs zero, given in the same units as the lineouts (ps or pixel)

- ``ele_lam_shift`` shifts the central frequency given by `lam` in the EPW spectrum, given in nm

- ``probe_beam`` identifies the beam on OMEGA used as the probe, automatically adjusts the scattering angle and finite aperture calculations. Currently availible options are P9, B15, B23, B26, B35, B42, B46, and B58.

- ``dpixel`` determined the width of a lineout in pixels, the width of a lineout is 2*`dpixel` + 1 centered about the values in `lineouts`


Other options
--------------------
 
The ``other:`` section includes options specifying the types of data that are being fit and other options
on how to perform the fit.

- ``expandedions`` is a boolean determining if a non-linear wavelength grid will be used allowing IAW and EPW spectra to be resolved simultaneously *currently depreciated*.

- ``PhysParams`` is a dictionary that is assigned within the code and stores detector information.

- ``iawoff`` is a boolean determining if the iaw will be suppressed in plotting of the EPW feature

- ``iawfilter`` is an alternative to iawoff that suppresses the IAW with a notch filter. The list has 4 elements, boolean for on/off, OD of the filter, spectral width of the filter in nm, and central wavelength of the filter in nm.

- ``CCDsize`` size of the CCD in pixels

- ``flatbg`` flat (applied to all pixes) value added to the background

- ``gain`` CCD counts per photo-electron, the standard OMEGA ROSS has a gain of 144. Gain must be accurate for appropriate use of Poisson statistics

- ``points_per_pixel`` number of wavelength points computed in the spectrum per pixel in the data being analyzed

- ``ang_res_unit`` is the number of pixels in an angular resolution unit for ARTS

- ``lam_res_unit`` is the number of pixels in an specular resolution unit for ARTS


Plotting
--------------------

- ``n_sigmas`` is the number of standard deviations to plot the uncertainty region over

- ``rolling_std_width`` number of lineouts used to calculate the standard deviation for the moving window error region

- ``data_cbar_u`` upper limit for the colorbar in plotting the data and fit, can be given as a number of counts or as `data` to automatically use the maximum of the data

- ``data_cbar_l`` lower limit for the colorbar in plotting the data and fit, can be given as a number of counts or as `data` to automatically use the minimum of the data

- ``ion_window_start`` determines the spectral range of the IAW fit plots, this gives the lower bound in nm

- ``ion_window_end`` determines the spectral range of the IAW fit plots, this gives the upper bound in nm

- ``ele_window_start`` determines the spectral range of the EPW fit plots, this gives the lower bound in nm

- ``ele_window_end`` determines the spectral range of the EPW fit plots, this gives the upper bound in nm


Optimizer
--------------------

- ``method`` gradient descent algorithm employed by the minimizer, current options are `adam` and `l-bfgs-b`

- ``hessian`` boolean, determines if the hessian will be supplied to the minimizer

- ``y_norm`` boolean, normalizes data to a maximum value of 1 to improve minimizer behavior, true values are still used for error analysis

- ``x_norm`` boolean, normalizes data to a maximum value of 1 as an input to the neural network *depreciated*

NN
--------------------
Options for the NN version of the code which is currently depreciated.

Dist_fit
--------------------

- ``window`` options the smoothing function applied to the distribution function while fitting ARTS

    - ``len`` length of the smoothing kernel relative the the length of the velocity vector

    - ``type`` type of smoothing function used, can be `hamming`, `hann`, or `bartlett`

