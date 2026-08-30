.. _forward:

Configuring options for forward pass
========================================

In addition to the options covered in :ref:`Configuring the inputs <inputs_deck>` and :ref:`Default options <configuring-the-default>` there are some options which are unique to running the code in ``forward`` or ``series`` mode.

To run a series of forward passes, any of the fit parameters can be turned into a list of values. This list will then be iterated over to produce a series of spectra relflecting the series of parameters. For example, if the electron temperature is turned into a list of values, then a series of spectra will be produced for each value of the electron temperature. Multiple parameters can be turned into lists at the same time, but they must all be the same length.


Other
-----------------------------

- ``extraoptions``

    - ``spectype`` the type of spectrum to be computed. This field is determined from the data when fitting, but must be supplied for a forward pass. Options include ``temporal``, ``imaging``, and ``angular``.

- ``detector_specs`` defines instrumental properties and is a sibling of ``extraoptions`` under ``other``.

    - ``widIRF`` defines the instrumental response widths. They are read from calibration files when fitting data and must be supplied in forward mode. Its subfields are:

        - ``spect_stddev_ion`` the standard deviation of the Gaussian ion spectral response in nanometers.

        - ``spect_stddev_ele`` the standard deviation of the Gaussian electron spectral response in nanometers for non-ARTS spectra.

        - ``spect_FWHM_ele`` and ``ang_FWHM_ele`` the spectral width in nanometers and angular width in degrees, respectively, for ARTS. These two values are full widths at half maximum.
