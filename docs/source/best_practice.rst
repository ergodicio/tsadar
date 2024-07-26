Best Practices for Fittings
================================

This page gives input deck snippets which produce the highest quality fits using TSADAR. It is recomended that 
when fitting new data a small region of the data is fit with a small number of lineouts. This can be accomplished
by setting the :code:`lineouts: start` and :code:`lineouts: end` to be close or increasing :code:`lineouts: skip`.
This will allow fast fits that can be used to dial in the starting conditions and the free parameters.
This can also be used to check and adjust the fitting ranges. Once the best inital conditions have been identified 
the entire dataset can be fit.

Background and lineout selection
---------------------------------

There are multiple options for background algorithms and types of fitting. These tend to be the best options for 
various data types. All of these options are editable in the input deck.

**Best operation for time resolved data:**

.. code-block:: yaml

    background:
        type: pixel
        slice: 900

**Best operation for spatially resolved data:**

.. code-block:: yaml

    background:
        type: fit
        slice: 900 <or background slice for IAW>

**Best operation for lineouts of angular:**

.. code-block:: yaml

    background:
	    type: fit
	    slice: <background shot number>

**Best operation for full angular:**

.. code-block:: yaml

    background:
	    type: fit
	    val: <background shot number>
    lineouts:
	    type: range
	    start: 90
    end: 950
