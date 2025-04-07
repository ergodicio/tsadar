.. _top:

TSADAR Documentation
==================================

Welcome to the TSADAR documentation site! 

TSADAR stands for Thomson Scattering with Automatic Differentiation for Analysis and Regression; it is a Thomson scattering 
data analysis software written in Python. It helps determine plasma parameters given
Thomson scattering measurements by fitting the collisionless spectral density function to the observed spectra.
More detail on the theory is provided elsewhere [1] and the specifics of implementation can be found in :doc:`math` [2]. 
If you are new to Thomson scattering, go to :ref:`Fundamentals of Thomson Scattering <ts_fundamentals>` for a quick introduction.
The fitting is performed via gradient descent. The gradients are acquired using automatic differentiation and JAX [3].
More details on the numerics will be added soon.

Table of Contents
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   Getting Started Tutorial<getting_started>
   Example Gallery<examples>
   Inputs<inputs>
   Defaults<defaults>
   Forward<forward_pass>
   Fundamentals of Thomson Scattering<TS_fundamentals>
   FAQ
   contributing
   math
   api_main
   


References
------------

[1] - Sheffield, J., Froula, D., Glenzer, S.H. and Luhmann Jr, N.C., 2010. Plasma scattering of electromagnetic radiation: theory and measurement techniques. Academic press.

[2] - Milder, A & Joglekar, Archis & Rozmus, Wojciech & Froula, Dustin. (2024). Qualitative and quantitative enhancement of parameter estimation for model-based diagnostics using automatic differentiation with an application to inertial fusion. Machine Learning: Science and Technology. 5. 10.1088/2632-2153/ad2493. 

[3] - https://github.com/google/jax




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
