.. _getting started the remix:

Getting Started
================

This page should provide you with the basics to set up the repo and get going.


Installation 
^^^^^^^^^^^^^^^
1. Clone the `repo <https://github.com/ergodicio/inverse-thomson-scattering>`_ to the local or remote machine where you will be running analysis.
2. Install using the commands bellow, or following your prefered method.


.. tab-set::

    .. tab-item:: Python

        .. code-block:: shell
            
            python --version                # hopefully this says >= 3.9
            python -m venv venv             # make an environment in this folder here
            source venv/bin/activate        # activate the new environment
            pip install -r requirements.txt # install dependencies


    .. tab-item:: Conda CPU

        .. code-block:: shell

            conda env create -f env.yml
            conda activate tsadar-cpu

    .. tab-item:: Conda GPU

        .. code-block:: shell

            conda env create -f env_gpu.yml
            conda activate tsadar-gpu

Importing raw data
^^^^^^^^^^^^^^^^^^^

.. image:: _elfolder/data_to_env.png
    :width: 413
    :height: 163
    :align: right 

------------------

Once you have created a virtual environment, you should add your raw data, which should be a :bdg-primary-line:`.hdf`` file 
into the folder where you created the virtual envirnment 

Adjusting parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

Set up the :ref:`inputs <inputs>` and :ref:` defaults <defaults>` to best fit your data. 
This can be acomplished by configuring the input decks.The code uses two input decks. 
Which can be found on inverse-thomson-scattering/configs/1d.

For fitting a new data set, it is recomended to start by fitting a small region of the data using a small number of lineouts. 
Set the :bdg-light:`lineout:start` and :bdg-info:`lineout:end` close together, to select a small region. 
Increase the :bdg:`lineouts:skip` to decrease the resolution.
These parameters can be found in the :bdg:`Inputs.yalm.` deck. Make sure to save your changes, and get ready to run the code.

.. card:: Inputs.yaml
    :link: inputs
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

.. card:: Defaults.yalm
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


Run command
^^^^^^^^^^^^^^^
Run the code using a run command.

There are :bdg-info:`2` run "modes".

**fit** performs a time resolved fitting procedure.

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode fit

**fordward** performs a forward pass and gives you the spectra given some input parameters.
 Additionally, it can get spectra for a series of plasma conditions. 
 For more information on specifying the inputs see :ref:`Configuring the inputs<inputs>` . 

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode forward

The inputs for the code are stored in an input deck. The default location for this input deck and therefore
the starting path for running jobs is :code:`inverse_thomson_scattering/configs/1d`. These inputs should be
modified to change the specifics to fit your analysis needs. More information on the Input deck can be found 
on the :ref:`Configuring the inputs<inputs>` page.

Output visualization
^^^^^^^^^^^^^^^^^^^^^
To visualize the outputs run the following commnand, and follow the resultant link. 

.. code-block:: bash

   mlflow ui 