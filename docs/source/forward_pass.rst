.. _forward:

Configuring options for forward pass
========================================

In addition to the options covered in :ref:`Configuring the inputs <inputs_deck>` and :ref:`Default options <configuring-the-default>` there are some options which are unique to running the code in ``forward`` or ``series`` mode.

To run a series of forward passes, any of the fit parameters can be turned into a list of values. This list will then be iterated over to produce a series of spectra relflecting the series of parameters. For example, if the electron temperature is turned into a list of values, then a series of spectra will be produced for each value of the electron temperature. Multiple parameters can be turned into lists at the same time, but they must all be the same length.


Other
-----------------------------

- ``extraoptions``

    - ``spectype`` the type of spectrum to be computed. This field is self determined from the data when fitting. For a forward pass somthing has to be specified but it deos not effect the spectrum. Options are ``temporal`` or  ``imaging`` In this context they produce the same spectrum.

    - ``PhysParams`` the subfields define instrumental properties

        - ``widIRF`` the subfields define the instrumental response functions, when fitting data it is determined from the calibration files but when running in forward mode these must be supplied. The subfields are:

            - ``spect_std_ion`` the standard deviation of the gaussian ion instrumental response function in nanometers

            - ``spect_std_ele`` the standard deviation of the gaussian electron instrumental response function in nanometers

