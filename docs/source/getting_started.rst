Getting Started
================

*Only MacOS and Linux supported at this time. It may or may not work on Windows but is not yet tested so YMMV*

1. Clone the repo to the local or remote machine where you will be running analysis

2. Install using below instructions or your own way

3. Run using the run command below

**Python virtual environment**

.. code-block:: bash

   python --version                # hopefully this says >= 3.9
   python -m venv venv             # make an environment in this folder here
   source venv/bin/activate        # activate the new environment
   pip install -r requirements.txt # install dependencies

**Conda CPU**

.. code-block:: bash

   conda env create -f env.yml
   conda activate tsadar-cpu

**Conda GPU**

.. code-block:: bash

   conda env create -f env_gpu.yml
   conda activate tsadar-gpu

**Run command**

There are three run "modes".

One performs a fitting procedure

.. card:: Fitting 

    Performs a fitting procedure
   .. code block:: Python

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode fit

The second just performs a forward pass and gives you the spectra given some input parameters

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode forward

The last can be used to perform forward passes and get spectra for a series of plasma conditions. For more information on specifying the inputs see `Configuring the inputs`_

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode series


The inputs for the code are stored in an input deck. The default location for this input deck and therefore
the starting path for running jobs is :code:`inverse_thomson_scattering/configs/1d`. These inputs should be
modified to change the specific to fit your analysis needs. More information on the Input deck can be found 
on the `Configuring the inputs`_ page.

**Output visualization**:

.. code-block:: bash

   mlflow ui 