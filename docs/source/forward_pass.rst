.. _forward:

Configuring options for forward pass
========================================

In addition to the options covered in :ref:`Configuring the inputs <inputs_deck>` and :ref:`Default options <configuring-the-default>` there are some options which are
unique to running the code in ``forward`` or ``series`` mode. This includes a new field of the input deck ``series`` and additions to standard fields.

Series
--------------------------------

This section of the input deck specifies which variables are to be looped over in order to produce a series of spectra.
Up to 4 parameters can be specified. For a single spectrum this field can be omitted and for a series of less then 4
parameters additional ``param`` fields should be omitted.

- ``param1`` the parameter field to be looped over. Must be an subfield of the ``parameters`` field. This parameter will be used to name plots.

- ``vals1`` a list of values to use for ``param1``. The elements of the list must be the same type and shape as the corresponding field, i.e if running Te each element should be a float and if running Z it shoudl be a list the length of the number of species.

- ``param2`` the second parameter to be looped over. Omit to loop over 1 variable.

- ``vals2`` a list of values to use form ``param2``. Must be the same length as ``vals1``. Omit to loop over 1 variable.

- ``param3`` the third parameter to be looped over. Omit to loop over 2 variables.

- ``vals3`` a list of values to use form ``param3``. Must be the same length as ``vals1``.

- ``param4`` the fourth parameter to be looped over. Omit to loop over 3 variables.

- ``vals4`` a list of values to use form ``param4``. Must be the same length as ``vals1``.


Other
-----------------------------

- ``extraoptions``

    - ``spectype`` the type of spectrum to be computed. This field is self determined from the data when fitting. Options are "temporal", "imaging", or "angular_full". In this context "temporal" and "imaging produce the same spectrum".

    - ``PhysParams`` the subfields define instrumental properties

        - ``widIRF`` the subfields define the instrumental response functions

            - ``spect_std_ion`` the standard deviation of the gaussian ion instrumental response function in nanometers

            - ``spect_std_ele`` the standard deviation of the gaussian electron instrumental response function in nanometers

