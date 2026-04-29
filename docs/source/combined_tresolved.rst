Combined Time Resolved 
=========================

This example illustrates how to fit time-resolved data for both EPW and IAW.

.. Tip:: To fix co-timing issues adjust the ``ion_t0_shift`` and ``ele_t0`` variables, which are found in the default deck.

    .. code-block:: yaml
        :caption: Inputs.yaml
        :emphasize-lines: 3,5

        data:
            ...
            ele_t0: 
            ...
            ion_t0_shift: 

Load the provided data, update the input decks to mimc those used here, and use **fit** mode to run the code. 

.. image:: _elfolder/fit_and_data_ele.png
    :scale: 35%

.. image:: _elfolder/fit_and_data_ion.png
    :scale: 35%

::download:`electron data <examples/time_resolved/EPW-s116773.hdf>`
::download:`ion data <examples/time_resolved/IAW-s116773.hdf>` 
::download:`input deck <examples/time_resolved/Combined_tresolved_inputs.yaml>`
::download:`input deck <examples/time_resolved/Combined_tresolved_defaults.yaml>` 
::download:`output plots <examples/time_resolved/Time_resolved_combined.zip>`
