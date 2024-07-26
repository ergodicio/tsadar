Example: Fitting mode
-------------------------
This is an example for running the fit mode

Firstly, you will want to download this TSADAR, and install all the necesary requirements. 
Instructions for that can be found in the :ref:`getting started<getting started the remix>` page.

Once you have created a virtual environment, you should add your raw data, which should be a .hdf file 
into the folder where you created the virtual envirnment 

 [picture of the data in the folder]

The code uses two input decks. Which can be found on inverse-thomson-scattering/configd/1d

.. card:: Inputs.yaml
    :link: inputs
    :link-type: ref

    Primary input deck, which contains the shotnumber, lineouts, and other fun things. 

.. card:: Defaults.yalm
    :link: configuring-the-default
    :link-type: ref

    Secondary imput deck, contains the blue and red shift minimum and maximum values

In the docs folder, you will fing a source folder, and in that folder you will want to look at two main 

Fitting a new data set 
^^^^^^^^^^^^^^^^^^^^^^^
 For fitting a new data set, it is recomended to fit a small region of the data using a small number of
 linouts. This can be acomplished by setting the lineout:start and lineout:end to be close, or by increasing 
 lineouts:skip

 Once you have adjusted the inputs and outputs 