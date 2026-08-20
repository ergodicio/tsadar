.. _artifacts:

Fit Artifacts
=============

Every fit logs its results to MLflow as artifacts. This page documents what a run
produces, so that downstream readers -- the Thomson analysis browser, analysis
notebooks, anything reading a run after the fact -- can rely on the contract
rather than on the internals of :mod:`tsadar.utils.plotting.plotters`.

manifest.json
-------------

Each fit writes a ``manifest.json`` at the root of its artifacts, describing the
tree that was actually produced:

.. code-block:: json

   {
     "schema_version": 1,
     "tsadar_version": "0.1.1",
     "mode": "fit",
     "kind": "one_d",
     "files": [
       {
         "path": "binary/ele_fit_and_data.nc",
         "role": "spectrogram",
         "species": "ele",
         "bytes": 65536,
         "dims": {"Time (ps)": 40, "Wavelength": 1024},
         "coords": ["Time (ps)", "Wavelength"],
         "data_vars": ["fit", "data"]
       }
     ]
   }

The manifest is built by walking the artifact directory after plotting has
finished, so it describes what a run really wrote rather than what it was
supposed to write. A fit that loads only the ion spectrum simply has no electron
entry.

``schema_version`` is bumped whenever a file or a variable is renamed or removed
-- that is, whenever a reader written against the previous manifest could break.
Adding a new file or a new role is additive and does not bump it.

Readers should switch on ``role``, not on filename, so that a future rename costs
them nothing beyond the version bump.

Roles
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Role
     - Meaning
   * - ``spectrogram``
     - Data and fit over the lineout axis and wavelength; what an interactive
       visualizer renders
   * - ``parameter_profiles``
     - Fitted plasma parameters along the lineout axis
   * - ``losses``
     - Per-lineout loss values
   * - ``distribution``
     - Learned distribution functions
   * - ``uncertainty``
     - Parameter uncertainties from the Hessian
   * - ``plot``
     - Rendered PNG figures
   * - ``config``
     - The input deck the run was launched with
   * - ``other``
     - Anything else the run happened to log

``kind``: 1D versus angular
---------------------------

``kind`` is ``one_d``, ``angular``, or ``unknown``, and it is decided from the
artifacts themselves.

This matters more than it looks. The logged
``other.extraoptions.spectype`` param is **not** reliable:
:func:`tsadar.utils.misc.log_mlflow` runs before the fit, and ``loadData``
overwrites ``spectype`` from the data file during ``prepare``, so a deck that
says ``temporal`` against angular data logs ``temporal``. The artifacts are
written afterwards, so they are the ground truth.

The two cases are also not distinguishable by shape: an angular
``fit_and_data.nc`` holds the same two variables with the same dimensionality as
a 1D ``ele_fit_and_data.nc``. Only the x coordinate differs. A reader switching
on array shape would silently render angular data through a 1D code path, which
means something different.

Datasets
--------

``binary/ele_fit_and_data.nc``, ``binary/ion_fit_and_data.nc`` (1D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Written by :func:`tsadar.utils.plotting.plotters.plot_ts_data`, one per loaded
spectrum.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - dims
     - ``(<lineout axis>, Wavelength)``
   * - lineout axis
     - ``Time (ps)`` for temporal data, ``Radius (\mum)`` for imaging data --
       the calibrated axis the lineouts were taken along
   * - ``Wavelength``
     - Wavelength axis in nm, over the fit region only
   * - ``data``
     - The measured spectrum, background-subtracted when
       ``data.background.bg_subtract`` is set
   * - ``fit``
     - The fitted spectrum over the same grid

``binary/fit_and_data.nc`` (angular)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Written by :func:`tsadar.utils.plotting.plotters.plot_data_angular`. Same
``data`` and ``fit`` variables, but the first dimension is
``Scattering angle (degrees)``.

``sigmas.nc``, ``binary/sigma-params.nc``, ``binary/sigma-fe.nc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameter uncertainties derived from the Hessian. Note that ``sigmas.nc`` sits at
the artifact **root**, not under ``binary/``. Negative values are meaningful:
they mark parameters whose inverse Hessian diagonal was negative, which normally
indicates a non-optimal point -- see :func:`tsadar.inverse.postprocess.get_sigmas`.

CSVs
----

``csv/learned_parameters.csv``
   The fitted parameters per lineout. 1D fits include the lineout axis column;
   angular fits do not, since they have no such axis, so they cannot be plotted
   as profiles.

``csv/losses.csv``
   Loss values per lineout.

``csv/learned_dist.csv``, ``csv/learned_flm.csv``
   The learned distribution function, and its spherical harmonic components when
   the run used them.

Plots
-----

PNG figures live under ``plots/``, ``lineouts/``, ``best/`` and ``worst/``. They
remain the fallback view for runs whose datasets a reader cannot interpret,
including runs that predate this contract.

Config
------

The input deck travels with the run: ``config.yaml`` for app-queued runs
(:func:`tsadar.runner.run_for_app`), or ``defaults.yaml`` plus ``inputs.yaml``
for NERSC-queued runs (:func:`tsadar.runner.load_and_make_folders`).

Runs that predate the manifest
------------------------------

Historical runs have no ``manifest.json``. Readers need a heuristic fallback for
those: infer ``kind`` from which ``binary/*fit_and_data.nc`` files exist, and
fall back to the PNG gallery when the datasets cannot be interpreted.
