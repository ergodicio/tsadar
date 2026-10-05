Contributing
============

Issues and pull requests
------------------------

Please open an issue for a proposed feature or a reproducible problem on the
`TSADAR issue tracker <https://github.com/ergodicio/tsadar/issues>`_. Pull requests
should explain the physical or user-facing behavior being changed, include focused
tests, and avoid unrelated formatting or generated-file changes.

Running tests
-------------

Install the test dependencies and run the fast suite before opening a pull request:

.. code-block:: console

   pip install -e ".[test]"
   pytest tests/ -m "not slow"

The physics-only commands are:

.. code-block:: console

   pytest tests/ -m "physics and not slow"
   pytest tests/ -m "physics and slow"

The second command is the scheduled reference/NERSC lane. See
:ref:`physics_validation` for the case inventory, tolerance evidence, and detailed
requirements for adding a physics-validation case.

Physics-test review checklist
-----------------------------

Before adding or changing a physics case, verify that:

* its docstring cites an equation, exact invariant, or trusted reference;
* the assertion measures the physical feature rather than a much larger carrier or
  normalization term;
* its tolerance is justified by convergence, approximation error, and backend/dtype
  variation;
* failure messages identify the case and the physical quantity that failed;
* random inputs use an explicit seed;
* pull-request cases do not write plots or contact MLflow; and
* expensive sweeps are marked ``slow`` and pass in the reference lane.

When a stored reference artifact is necessary, include its generator, input deck,
units, source or derivation, TSADAR commit, backend/dtype, and checksum. Do not replace
a reference merely to make a changed implementation pass; first establish why the new
result is more trustworthy.
