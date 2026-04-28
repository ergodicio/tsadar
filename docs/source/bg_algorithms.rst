.. _bg_algs:

Background Algorithms
===============================

When analyzing Thomson scattering data it is common for the desired signal to be sitting on a background signal which must be
accounted for. This background tends to come in two flavors, bremsstrahlung background and other Thomson signals. Occasionaly
other sources may be observed such as line emission or stray light. 

TSADAR only attempts to account for bremsstrahlung background. Other sources of background should be windowed out using the
spectral fit range options. 

Three algorithms are included in TSADAR and accesible through the ``data:background:type`` field. These are ``shot``, ``pixel``, and ``fit``.

.. toctree::
   :maxdepth: 4


Shot background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``shot`` option for the ``data:background:type`` field can be selected if a dedicate background shot was taken with no probe beam.
In this case the background shot number should be supplied to the ``data:background:slice`` field. This algorithm is quite simple,
A small smoothing kernel is applied to the background data and corresponding lineouts are taken in order to provide backgounds for
each lineout.

Pixel background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the workhorse algorithm for time resolved data which tends to have lower background levels. This algorithm can be called 
by supplying ``pixel`` to the ``data:background:type`` field and the pixel location where the background is to be analyzed should be supplied to to the ``data:background:slice`` field.

The pixel algorithm takes a group of lineouts centered at the specified location with a width 2* ``data:dpixel`` +1 and averages them. 
This background lineout is then smoothed. The smoothing is done with a moving average whose window size is set with the ``data:background:bg_smoothing_window``. At this point the algorithm divereges for the EPW and IAW. For IAW the spectral range is very
small and very little spectral dependence is usualy seen in the background, so the background is averaged again to produce a scalar background value. For the EPW there is far more spectral dependence so the smothed background used as is.
Finaly this idealized background is scaled in magnitude to match the data at that lineout using the endges of the data.


Fit background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fitted background only applies to the EPW data, if applied to IAW data or combined fits the pixel algorithm will be applied to the IAW.
When applied to the EPW this algorithm attempts to fit a theoretical model to the edges of the data outside the EPW allowing interpolation of the background 
into regions with data. This is adventageous for imaging data where the bremstruhlung can change singificantly as a function of space and there may be no location 
with background and no data. The domain for this fit must not include any Thomson scattering data or contamination of Thomson scattering of other beams.

Multiple fit models are availible for this and specified with the field ``data:background:bg_alg``. The options are ``rat11``, ``rat21``, ``exp2`` and ``power2``. Each of these has 3 or 4
fitting parameters whose inital guesses must be supplied with the ``data:background:bg_alg_params`` field. If there are stray sources of light or the fit is seeing the Thomson
scattering signal, the domain of the fit can be altered with ``data:background:bg_alg_domain``. This field has 4 values, and the domain is constructed as a linear domain 
between the first two pixel values and the second two.