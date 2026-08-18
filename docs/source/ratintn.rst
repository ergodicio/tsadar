Dispersion Relation Integral
=============================

The dispersion relation integral is the most computationally expensive component. Because there is a pole in the
integral, we perform a rational integral that is described by the following routines


.. autofunction::   tsadar.core.physics.ratintn.ratintn
.. autofunction::   tsadar.core.physics.ratintn.ratcen

The integral is linear in the numerator, so wherever the denominator and the integration variable
are fixed across calls the whole quadrature can be precomputed as a single matrix.

.. autofunction::   tsadar.core.physics.ratintn.ratintn_operator
