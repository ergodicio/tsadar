"""
Tests for the fit artifact contract (ergodicio/tsadar#116).

These build the artifact tree the way the plotting code does -- real xarray
datasets written with ``to_netcdf``, real csvs -- and assert the manifest
describes exactly what ended up on disk. Building the tree directly rather than
running a fit keeps the test fast enough to be worth running, and it is the
manifest, not the fit, that is under test; the coupling that actually matters is
manifest-vs-disk, and that is what is asserted.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsadar.utils import manifest


def spectrogram(x_label="Time (ps)", nx=4, nlam=8):
    """A fit_and_data dataset shaped like the ones plotters.py writes."""

    coords = ((x_label, np.arange(nx, dtype=float)), ("Wavelength", np.linspace(520.0, 530.0, nlam)))

    return xr.Dataset({k: xr.DataArray(np.zeros((nx, nlam)), coords=coords) for k in ("fit", "data")})


@pytest.fixture
def one_d_tree(tmp_path):
    """An artifact tree for a 1D electron+ion fit."""

    for sub in ("binary", "csv", "plots", "lineouts"):
        os.makedirs(tmp_path / sub)

    spectrogram().to_netcdf(tmp_path / "binary" / "ele_fit_and_data.nc")
    spectrogram().to_netcdf(tmp_path / "binary" / "ion_fit_and_data.nc")

    pd.DataFrame({"lineout pixel": [1, 2], "Te": [0.6, 0.7], "ne": [0.22, 0.23]}).to_csv(
        tmp_path / "csv" / "learned_parameters.csv", index=False
    )
    pd.DataFrame({"loss": [1.0, 0.5]}).to_csv(tmp_path / "csv" / "losses.csv", index=False)

    (tmp_path / "plots" / "fit_and_data.png").write_bytes(b"png")
    (tmp_path / "lineouts" / "lineout_1.png").write_bytes(b"png")

    return tmp_path


def entry_for(built, path):
    return next(e for e in built["files"] if e["path"] == path)


# -- what the manifest says ---------------------------------------------------


def test_manifest_lists_every_file_on_disk(one_d_tree):
    built = manifest.build_manifest(str(one_d_tree))

    on_disk = {rel for rel, _ in manifest._walk(str(one_d_tree))}

    assert {e["path"] for e in built["files"]} == on_disk


def test_manifest_is_versioned_and_stamped(one_d_tree):
    built = manifest.build_manifest(str(one_d_tree))

    assert built["schema_version"] == manifest.SCHEMA_VERSION
    assert built["tsadar_version"]
    assert built["mode"] == "fit"


def test_spectrogram_schema_is_recorded(one_d_tree):
    entry = entry_for(manifest.build_manifest(str(one_d_tree)), "binary/ele_fit_and_data.nc")

    assert entry["role"] == manifest.ROLE_SPECTROGRAM
    assert entry["species"] == "ele"
    assert entry["dims"] == {"Time (ps)": 4, "Wavelength": 8}
    assert sorted(entry["data_vars"]) == ["data", "fit"]
    assert "Time (ps)" in entry["coords"]


def test_dataset_schema_matches_the_file_it_describes(one_d_tree):
    # The point of recording dims is that a reader can trust them without opening
    # the file, so they had better agree with the file.
    entry = entry_for(manifest.build_manifest(str(one_d_tree)), "binary/ion_fit_and_data.nc")

    with xr.open_dataset(one_d_tree / "binary" / "ion_fit_and_data.nc") as ds:
        assert entry["dims"] == {str(k): int(v) for k, v in ds.sizes.items()}
        assert sorted(entry["data_vars"]) == sorted(str(v) for v in ds.data_vars)


def test_csv_columns_are_recorded(one_d_tree):
    entry = entry_for(manifest.build_manifest(str(one_d_tree)), "csv/learned_parameters.csv")

    assert entry["role"] == manifest.ROLE_PARAMETER_PROFILES
    assert "Te" in entry["columns"]
    assert entry["rows"] == 2


def test_plots_are_rolled_up_by_directory(one_d_tree):
    built = manifest.build_manifest(str(one_d_tree))

    assert entry_for(built, "plots/fit_and_data.png")["role"] == manifest.ROLE_PLOT
    assert entry_for(built, "lineouts/lineout_1.png")["role"] == manifest.ROLE_PLOT


def test_unknown_files_get_a_role_rather_than_being_dropped(one_d_tree):
    (one_d_tree / "state_weights.txt").write_text("weights")

    entry = entry_for(manifest.build_manifest(str(one_d_tree)), "state_weights.txt")

    assert entry["role"] == manifest.ROLE_OTHER


# -- 1D vs angular ------------------------------------------------------------


def test_a_1d_run_is_identified_from_its_artifacts(one_d_tree):
    assert manifest.build_manifest(str(one_d_tree))["kind"] == manifest.KIND_ONE_D


def test_an_angular_run_is_identified_from_its_artifacts(tmp_path):
    # Angular writes binary/fit_and_data.nc, with the same two variables and the
    # same dimensionality as the 1D files -- only the x coord differs. A reader
    # that switched on shape would get this wrong, which is why the manifest
    # states the kind outright.
    os.makedirs(tmp_path / "binary")
    spectrogram(x_label="Scattering angle (degrees)").to_netcdf(tmp_path / "binary" / "fit_and_data.nc")

    built = manifest.build_manifest(str(tmp_path))

    assert built["kind"] == manifest.KIND_ANGULAR
    assert entry_for(built, "binary/fit_and_data.nc")["role"] == manifest.ROLE_SPECTROGRAM


def test_a_run_with_no_spectrogram_is_unknown_not_guessed(tmp_path):
    os.makedirs(tmp_path / "plots")
    (tmp_path / "plots" / "only.png").write_bytes(b"png")

    assert manifest.build_manifest(str(tmp_path))["kind"] == manifest.KIND_UNKNOWN


def test_an_ion_only_fit_has_no_electron_entry(tmp_path):
    os.makedirs(tmp_path / "binary")
    spectrogram().to_netcdf(tmp_path / "binary" / "ion_fit_and_data.nc")

    built = manifest.build_manifest(str(tmp_path))

    assert [e["path"] for e in built["files"]] == ["binary/ion_fit_and_data.nc"]
    assert built["kind"] == manifest.KIND_ONE_D


# -- writing ------------------------------------------------------------------


def test_write_manifest_lands_next_to_the_artifacts(one_d_tree):
    written = manifest.write_manifest(str(one_d_tree))

    with open(one_d_tree / manifest.MANIFEST_NAME) as fi:
        assert json.load(fi) == written


def test_the_manifest_does_not_describe_itself(one_d_tree):
    manifest.write_manifest(str(one_d_tree))
    built = manifest.build_manifest(str(one_d_tree))

    assert manifest.MANIFEST_NAME not in {e["path"] for e in built["files"]}


def test_rewriting_is_stable(one_d_tree):
    # postprocess writes once, but a manifest that changed on every rebuild would
    # make schema_version meaningless.
    first = manifest.write_manifest(str(one_d_tree))
    second = manifest.write_manifest(str(one_d_tree))

    assert first == second


def test_a_broken_manifest_never_fails_the_fit(tmp_path, monkeypatch):
    # The fit is done and the artifacts are good by this point. Losing the
    # manifest is a degraded read experience, not a lost run.
    monkeypatch.setattr(manifest, "build_manifest", lambda *a, **kw: 1 / 0)

    assert manifest.write_manifest(str(tmp_path)) == {}


def test_an_unreadable_dataset_is_noted_not_fatal(tmp_path):
    os.makedirs(tmp_path / "binary")
    (tmp_path / "binary" / "ele_fit_and_data.nc").write_bytes(b"not actually netcdf")

    entry = entry_for(manifest.build_manifest(str(tmp_path)), "binary/ele_fit_and_data.nc")

    assert entry["role"] == manifest.ROLE_SPECTROGRAM
    assert "unreadable" in entry


# -- the wiring into postprocess ---------------------------------------------


def test_postprocess_logs_a_manifest_with_the_artifacts(tmp_path, monkeypatch):
    """
    Guards the wiring, not the manifest: a refactor that moved or dropped the
    write_manifest call would leave every subsequent run without a contract, and
    nothing else in this file would notice.
    """

    import mlflow

    from tsadar.inverse import postprocess

    def fake_process_data(
        config, sample_indices, all_data, all_axes, loss_fn, fitted_weights, sa, init_losses, t1, td, all_params, num_params
    ):
        spectrogram().to_netcdf(os.path.join(td, "binary", "ele_fit_and_data.nc"))

        return t1, {}

    monkeypatch.setattr(postprocess, "process_data", fake_process_data)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    client = mlflow.tracking.MlflowClient()
    client.create_experiment("manifest-tests", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("manifest-tests")

    config = {"other": {"extraoptions": {"spectype": "temporal"}, "refit": False}}
    try:
        with mlflow.start_run() as run:
            postprocess.postprocess(config, [], {}, {}, None, None, [])

        logged = {f.path for f in client.list_artifacts(run.info.run_id)}
    finally:
        mlflow.set_tracking_uri(None)

    assert manifest.MANIFEST_NAME in logged
