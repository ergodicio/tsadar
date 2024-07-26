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




1. Run using a run command.

Run command
^^^^^^^^^^^^^^^

.. note:: 
    The run command will vary for each "mode".

There are :bdg-success:`3` run "modes".

One performs a fitting procedure

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode fit

The second just performs a forward pass and gives you the spectra given some input parameters

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode forward

The last can be used to perform forward passes and get spectra for a series of plasma conditions. For more information on specifying the inputs see :ref:`Configuring the inputs<inputs>` .

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode series


The inputs for the code are stored in an input deck. The default location for this input deck and therefore
the starting path for running jobs is :code:`inverse_thomson_scattering/configs/1d`. These inputs should be
modified to change the specific to fit your analysis needs. More information on the Input deck can be found 
on the :ref:`Configuring the inputs<inputs>` page.

**Output visualization**:

Visualizing the outputs requires 
After 
To visualize the outputs run the following commnand, and follow the link provided 

.. code-block:: bash

   mlflow ui 