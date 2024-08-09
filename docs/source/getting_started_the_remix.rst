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

Input decks
^^^^^^^^^^^^

The code uses two input decks, which  are located in **inverse-thomson-scattering/configs/1d**. The primary input deck will override the secondary input deck when values conflict. 
More information on the specifics of each deck can be found by clicking on the cards bellow. 

.. grid:: 2

    .. grid-item-card::  Inputs.yaml
        :link: inputs
        :link-type: ref

        Primary input deck 

    .. grid-item-card::  Defaults.yaml
        :link: configuring-the-default
        :link-type: ref

        Secondary input deck 

Experiment information
^^^^^^^^^^^^^^^^^^^^^^^
Indicate the shotnumber of the experimnet in the :bdg-link-primary:`Input.yaml <inputs>` deck.
The code will identify the data as time resolved for OMEGA experients, based of the data file. 
For fitting data files from other sources, please contact the authors.

.. code-block:: yaml
    :caption: Inputs.yaml
    :emphasize-lines: 2

    data:
        shotnum: 101675
        lineouts:
            type:
                pixel

Load the spectra you are interested in visualizing by activating its corresponding boolean.

.. code-block:: yaml
    :caption: Inputs.yaml
    :emphasize-lines: 3,4

    other:
        extraoptions:
            load_ion_spec: True
            load_ele_spec: True
            fit_IAW: True
            fit_EPWb: True
            fit_EPWr: True
        PhysParams:

Fitting a new data set
^^^^^^^^^^^^^^^^^^^^^^^^
Select the data you are interested in fitting by activating its corresponding boolean. 

.. code-block:: yaml
    :caption: Inputs.yaml
    :emphasize-lines: 5,6,7

    other:
        extraoptions:
            load_ion_spec: True
            load_ele_spec: True
            fit_IAW: True
            fit_EPWb: True
            fit_EPWr: True
        PhysParams:

The fit will start at :bdg-light:`lineout:start` and will end at :bdg-light:`lineout:end`. Lineouts will be fit every :bdg:`lineout:skip`of the unit type defined. 
For fitting a new data set, it is recomended to start by fitting a small region of the data using a small number of lineouts. 

.. code-block:: yaml
    :caption: Inputs.yaml
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

Adjusting parameters
^^^^^^^^^^^^^^^^^^^^^

Set up the input deckst best fit your data. **value** sets the initial value for the first itteration, or the static value of unfit parameters.
These values are bounded by **lb** and **ub** indicating the lower and upper bound respectively.

.. code-block:: yaml
    :caption: Inputs.yaml
    :emphasize-lines: 7,,9,10

    parameters:
        species1:
            type:
                electron:
                active: False
            Te:
                val: .6
                active: True
                lb: 0.01
                ub: 1.25

The secondary imput deck, contains the minimum and maximum values for the blue and red shifts.

.. code-block:: yaml
    :caption: Defaults.yaml
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

Run command
^^^^^^^^^^^^^^^
Name the run in the input deck. 

.. code-block:: yaml
    :caption: Input.yaml 
    :emphasize-lines: 3

    mlflow:
    experiment: inverse-thomson-scattering
    run: name of the run

Run the code using a run command.

There are :bdg:`2` run "modes".

**Fit mode** fitperfoms fitting procedure.

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode fit

**Forward mode** performs a forward pass and gives you the spectra given some input parameters. Additionally, it can get spectra for a series of plasma conditions. 
 
.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode forward

Output visualization
^^^^^^^^^^^^^^^^^^^^^
To visualize the outputs run the following commnand, and follow the resultant link. 

.. code-block:: bash

   mlflow ui 