"""
The fit artifact contract: a versioned ``manifest.json`` describing what a run
produced (ergodicio/tsadar#116).

A fit's artifacts are a read API for the Thomson analysis browser
(ergodicio/tsadar-app), but the set of files and their schemas is implicit in
:mod:`tsadar.utils.plotting.plotters` and can drift silently with any plotting
change. Downstream readers were left sniffing filenames and guessing at
dimensionality. The manifest states it instead: what was written, what role each
file plays, and -- for the datasets an interactive visualizer renders from --
the dims, coords and variables actually stored.

The manifest is *descriptive*: it is built by walking the artifact directory
after the plotting code has filled it, not from a hardcoded list of what should
be there. A fit that skips the ion spectrum simply has no ion entry, and a new
plot appears in the manifest without anyone remembering to declare it.

``schema_version`` is bumped when a file or variable is renamed or removed --
that is, whenever a reader written against the old manifest could break.
Additive changes (a new file, a new role) do not need a bump.
"""

import json
import os

#: Bump when a file or variable is renamed or removed. See the module docstring.
SCHEMA_VERSION = 1

MANIFEST_NAME = "manifest.json"

#: What a file is *for*, as opposed to what it is called. Readers should switch
#: on role, so that a future rename costs them nothing but a schema_version bump.
ROLE_SPECTROGRAM = "spectrogram"
ROLE_PARAMETER_PROFILES = "parameter_profiles"
ROLE_LOSSES = "losses"
ROLE_DISTRIBUTION = "distribution"
ROLE_UNCERTAINTY = "uncertainty"
ROLE_DIAGNOSTICS = "diagnostics"
ROLE_PLOT = "plot"
ROLE_CONFIG = "config"
ROLE_OTHER = "other"

#: Exact artifact paths whose role is known. Everything else falls back to the
#: directory-based rules in :func:`_role_of`.
KNOWN_ROLES = {
    "binary/ele_fit_and_data.nc": ROLE_SPECTROGRAM,
    "binary/ion_fit_and_data.nc": ROLE_SPECTROGRAM,
    "binary/fit_and_data.nc": ROLE_SPECTROGRAM,
    "csv/learned_parameters.csv": ROLE_PARAMETER_PROFILES,
    "csv/losses.csv": ROLE_LOSSES,
    "csv/learned_dist.csv": ROLE_DISTRIBUTION,
    "csv/learned_flm.csv": ROLE_DISTRIBUTION,
    "sigmas.nc": ROLE_UNCERTAINTY,
    "binary/sigma-fe.nc": ROLE_UNCERTAINTY,
    "binary/sigma-params.nc": ROLE_UNCERTAINTY,
    "angular_objective_diagnostics.npz": ROLE_DIAGNOSTICS,
    "angular_objective_terms.json": ROLE_DIAGNOSTICS,
    "config.yaml": ROLE_CONFIG,
    "defaults.yaml": ROLE_CONFIG,
    "inputs.yaml": ROLE_CONFIG,
}

#: Directories whose contents are all plots.
PLOT_DIRS = ("plots", "lineouts", "best", "worst")

#: The 1D spectrogram datasets, keyed by the species they hold. Their presence is
#: what makes a run 1D rather than angular.
ONE_D_SPECTROGRAMS = {"binary/ele_fit_and_data.nc": "ele", "binary/ion_fit_and_data.nc": "ion"}

#: The angular spectrogram dataset. Same two variables with the same
#: dimensionality as the 1D files, but its x axis is scattering angle -- which is
#: exactly why a reader must not tell them apart by shape alone.
ANGULAR_SPECTROGRAM = "binary/fit_and_data.nc"

KIND_ONE_D = "one_d"
KIND_ANGULAR = "angular"
KIND_UNKNOWN = "unknown"


def _role_of(relpath: str) -> str:
    """Classifies an artifact by its path."""

    if relpath in KNOWN_ROLES:
        return KNOWN_ROLES[relpath]

    head = relpath.split("/")[0]
    if head in PLOT_DIRS:
        return ROLE_PLOT
    if relpath.endswith(".png"):
        return ROLE_PLOT

    return ROLE_OTHER


def _describe_dataset(path: str) -> dict:
    """
    Records the schema of a netCDF dataset: its dims with sizes, coord names and
    data variables. This is what lets a visualizer know it can render a file
    without opening it first.

    Returns an ``unreadable`` note instead of raising if the file cannot be
    opened -- most often because no netCDF engine is installed in the reading
    environment, which is not a reason to fail a finished fit.
    """

    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            return {
                "dims": {str(k): int(v) for k, v in ds.sizes.items()},
                "coords": [str(c) for c in ds.coords],
                "data_vars": [str(v) for v in ds.data_vars],
            }
    except Exception as e:
        return {"unreadable": f"{type(e).__name__}: {e}"}


def _describe_csv(path: str) -> dict:
    """Records a csv's columns and row count."""

    try:
        import pandas as pd

        frame = pd.read_csv(path)

        return {"columns": [str(c) for c in frame.columns], "rows": int(len(frame))}
    except Exception as e:
        return {"unreadable": f"{type(e).__name__}: {e}"}


def _walk(artifact_dir: str):
    """Yields every file under the artifact directory as a posix-style relative path."""

    for root, _dirs, files in os.walk(artifact_dir):
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, artifact_dir).replace(os.sep, "/")
            yield rel, full


def _kind_of(paths) -> str:
    """
    Decides whether the run is 1D or angular from the artifacts themselves.

    The logged ``other.extraoptions.spectype`` param is not usable for this:
    ``misc.log_mlflow`` runs before the fit, and ``loadData`` overwrites spectype
    from the data file during prepare, so a deck saying "temporal" against
    angular data logs "temporal". The artifacts are written after that, so they
    are the ground truth -- see the scope note on ergodicio/tsadar-app#37.
    """

    paths = set(paths)
    if paths & set(ONE_D_SPECTROGRAMS):
        return KIND_ONE_D
    if ANGULAR_SPECTROGRAM in paths:
        return KIND_ANGULAR

    return KIND_UNKNOWN


def build_manifest(artifact_dir: str, mode: str = "fit") -> dict:
    """
    Builds the manifest by describing what is actually in the artifact directory.

    Args:
        artifact_dir: the directory that is about to be logged to MLflow
        mode: the run mode that produced these artifacts

    Returns:
        manifest: a JSON-serializable dict; see the module docstring
    """

    from . import misc

    entries = []
    for rel, full in _walk(artifact_dir):
        if rel == MANIFEST_NAME:
            continue

        entry = {"path": rel, "role": _role_of(rel), "bytes": os.path.getsize(full)}

        if rel in ONE_D_SPECTROGRAMS:
            entry["species"] = ONE_D_SPECTROGRAMS[rel]

        if rel.endswith(".nc"):
            entry.update(_describe_dataset(full))
        elif rel.endswith(".csv"):
            entry.update(_describe_csv(full))

        entries.append(entry)

    entries.sort(key=lambda e: e["path"])

    return {
        "schema_version": SCHEMA_VERSION,
        "tsadar_version": misc.get_version(),
        "mode": str(mode).casefold(),
        "kind": _kind_of(e["path"] for e in entries),
        "files": entries,
    }


def write_manifest(artifact_dir: str, mode: str = "fit") -> dict:
    """
    Writes ``manifest.json`` into the artifact directory, next to the files it
    describes, so that it travels with them wherever they are copied.

    Never raises: a run that has finished fitting and plotting must not fail
    because its manifest could not be written. A failure is reported and the
    artifacts are logged without it, which is exactly the situation readers
    already handle for historical runs.

    Args:
        artifact_dir: the directory that is about to be logged to MLflow
        mode: the run mode that produced these artifacts

    Returns:
        manifest: the manifest that was written, or an empty dict on failure
    """

    try:
        manifest = build_manifest(artifact_dir, mode=mode)
        with open(os.path.join(artifact_dir, MANIFEST_NAME), "w") as fi:
            json.dump(manifest, fi, indent=2, sort_keys=False)

        return manifest
    except Exception as e:
        print(f"WARNING: could not write {MANIFEST_NAME}: {type(e).__name__}: {e}")

        return {}
