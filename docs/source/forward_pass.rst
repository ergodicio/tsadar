.. _forward:

Configuring options for forward pass
========================================

In addition to the options covered in :ref:`Configuring the inputs <inputs_deck>` and :ref:`Default options <configuring-the-default>` there are some options which are
unique to running the code in **forward** or **series** mode. This includes a new field of the input deck **series** and additions to standard fields.

Series
--------------------------------

This section of the input deck specifies which variables are to be looped over in order to produce a series of spectra.
Up to 4 parameters can be specified. For a single spectrum this field can be omitted and for a series of less then 4
parameters additional :bdg-success:`param` fields should be omitted.

- :bdg-success-line:`param1` the parameter field to be looped over. Must be an subfield of the :bdg-success-line:`parameters` field. This parameter will be used to name plots.

- :bdg-success-line:`vals1` a list of values to use for :bdg-success-line:`param1`. The elements of the list must be the same type and shape as the corresponding field, i.e if running Te each element should be a float and if running Z it shoudl be a list the length of the number of species.

- :bdg-success-line:`param2` the second parameter to be looped over. Omit to loop over 1 variable.

- :bdg-success-line:`vals2` a list of values to use form :bdg-success-line:`param2`. Must be the same length as :bdg-success-line:`vals1`. Omit to loop over 1 variable.

- :bdg-success-line:`param3` the third parameter to be looped over. Omit to loop over 2 variables.

- :bdg-success-line:`vals3` a list of values to use form :bdg-success-line:`param3`. Must be the same length as :bdg-success-line:`vals1`.

- :bdg-success-line:`param4` the fourth parameter to be looped over. Omit to loop over 3 variables.

- :bdg-success-line:`vals4` a list of values to use form :bdg-success-line:`param4`. Must be the same length as :bdg-success-line:`vals1`.


Other
-----------------------------

-:bdg-success:`extraoptions`

    - :bdg-success-line:`spectype` the type of spectrum to be computed. This field is self determined from the data when fitting. Options are "temporal", "imaging", or "angular_full". In this context "temporal" and "imaging produce the same spectrum".

    - :bdg-success:`PhysParams` the subfields define instrumental properties

        - :bdg-success:`widIRF` the subfields define the instrumental response functions

            - :bdg-success-line:`spect_std_ion` the standard deviation of the gaussian ion instrumental response function in nanometers

            - :bdg-success-line:`spect_std_ele` the standard deviation of the gaussian electron instrumental response function in nanometers

