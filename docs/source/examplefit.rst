Example: Fitting time-resolved data
---------------------------------------
This is an example for fitting time-resolved data. 

Firstly, you will want to download this TSADAR, install all the necesary requirements, and create a virtual environment. 
Instructions for that can be found in the :ref:`getting started<getting started>` page.

Importing raw data
^^^^^^^^^^^^^^^^^^^

.. image:: _elfolder/data_to_env.png
    :width: 413
    :height: 163
    :align: right 

------------------

Once you have created a virtual environment, you should add your raw data, which should be a :bdg-primary-line:`.hdf`` file 
into the folder where you created the virtual envirnment 

------------------

Adjusting parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

Set up the inputs and defaults to best fit your data. This can be acomplished by configuring the input decks.The code uses two input decks. 
Which can be found on inverse-thomson-scattering/configd/1d.

For fitting a new data set, it is recomended that start by fitting a small region of the data using a small number of lineouts. 
Set the :bdg-light:`lineout:start` and :bdg-info:`lineout:end` close together, to select a small region. decrease the :bdg:`lineouts:skip` to accelerate the runnig time.
Setting the  :bdg-success:`lineout:start` and :bdg-success-line:`lineout:end` close together will  a small region
if this is the first time you are fitting a data set, it is recommended that you fit a small region of the data, 
using a small number of lineouts. To fit a small region of the data, edit the lineout:start and lineout:end parameeters to be close, and increase lineouts:skip. 
These parameters can be found in the :bdg:`Inputs.yalm.` deck. Make sure to save your changes, and get ready to run the code.

.. card:: Inputs.yaml
    :link: inputs_deck
    :link-type: ref

    Primary input deck will override defaults deck.  

.. code-block:: yaml
    :emphasize-lines: 3,6,7,8

    data:
        shotnum: 1234567
        lienouts:
            type:
                pixel
            start: 100
            end: 900
            skip: 10
        background:
            type:
                pixel
            slice: 900

.. card:: Defaults.yaml
    :link: configuring-the-default
    :link-type: ref

    Secondary imput deck, contains the blue and red shift minimum and maximum values

.. code-block:: yaml
    :emphasize-lines: 6,7,8,9

    data:
    shotnum: 1234567
    shotDay: False
    launch_data_visualizer: True
    fit_rng:
        blue_min: 460
        blue_max: 510
        red_min: 545
        red_max: 600
        iaw_min: 525.5
        iaw_max: 527.5
        iaw_cf_min: 526.49
        iaw_cf_max: 526.51
        forward_epw_start: 400
        forward_epw_end: 700
        forward_iaw_start: 525.75
        forward_iaw_end: 527.25

Running the code
^^^^^^^^^^^^^^^^^
Once you have adjusted the parameters, and saved the changes made. You will want to implement the run command.

.. code-block:: python

    python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode fit

This command will yeild the following output, indicating the the fit is completed:

The following command will allow you to visualize the results of the fitting. The output link will redirect you to a local site where the outputs can be viewed. 
 
.. code-block:: shell

    mlflow ui

.. image:: _elfolder\mlflow_home.png

Click the follow the link to vizialize the data. The resulting plots can be founs in the :bdg:`Artifacts` unedr the folder :bdg:`plots`. 
Best and worst folders contain the best and worst fits respectively. `



Fitting a new data set 
^^^^^^^^^^^^^^^^^^^^^^^
For fitting a new data set, it is recomended to fit a small region of the data using a small number of
linouts. This can be acomplished by setting the lineout:start and lineout:end to be close, or by increasing 
lineouts:skip
Once you have adjusted the inputs and outputs 


.. grid:: 2

    .. grid-item-card::  Inputs.yaml
        :link: inputs_deck
        :link-type: ref

        Primary input deck 

    .. grid-item-card::  Defaults.yaml
        :link: configuring-the-default
        :link-type: ref

        Secondary input deck 

Extra info in the inputs
^^^^^^^^^^^^^^^^^^^^^^^^^

The inputs for the code are stored in an input deck. The default location for this input deck and therefore
the starting path for running jobs is :code:`inverse_thomson_scattering/configs/1d`. These inputs should be
modified to change the specifics to fit your analysis needs. More information on the Input deck can be found 
on the :ref:`Configuring the inputs <inputs_deck>` page.

Extra info on fitting the data set for a small region 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For fitting a new data set, it is recomended to start by fitting a small region of the data using a small number of lineouts. 
Set the :bdg-light:`lineout:start` and :bdg-info:`lineout:end` close together, to select a small region. 
Increase the :bdg:`lineouts:skip` to decrease the resolution.
