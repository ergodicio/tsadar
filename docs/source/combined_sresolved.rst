
Combined Spatial Resolved 
================================

This example illustrates how to fit Spatially-resolved data for both EPW and IAW.

.. Tip:: To fix co-timing issues, adjust the tcc locations which are defined in the :bdg-success-line:`calibration.py` file.
   
    .. code-block:: python
        :caption: calibration.py
        :emphasize-lines: 3,4

        def get_calibrations(shotNum, tstype, CCDsize):
            ...
            EPWtcc = 1024 - 503 
            IAWtcc = 1024 - 578  

    If you need to shift the EPW are IAW position beyond that, you should use :bdg-success-line:`ion_t0_shift` and :bdg-success-line:`ele_t0`` variables, which are found in the default deck.
    
    
    .. code-block:: yaml
        :caption: Inputs.yaml
        :emphasize-lines: 3,5

        data:
            ...
            ele_t0:
            ...
            ion_t0_shift: 

Load the provided data, update the input decks to mimc those used here, and use **fit** mode to run the code. 

.. image:: examples/combined_s_ele.png
    :scale: 35%

.. image:: examples/combined_s_ion.png
    :scale: 35%

::download:`data <examples/combined_s_data.zip>` 
::download:`input decks <examples/combined_s_inputs.zip>` 
::download:`output plots <examples/combined_s_outputs.zip>`
