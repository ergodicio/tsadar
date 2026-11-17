"""The fit artifact contract (ergodicio/tsadar#116): manifests compared with golden files."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsadar.utils import manifest

GOLDEN = Path(__file__).parent / "golden"


def spectrogram(x_label="Time (ps)", nx=4, nlam=8):
    """A fit_and_data dataset shaped like the ones plotters.py writes."""
    coords = ((x_label, np.arange(nx, dtype=float)), ("Wavelength", np.linspace(520.0, 530.0, nlam)))
    return xr.Dataset({k: xr.DataArray(np.zeros((nx, nlam)), coords=coords) for k in ("fit", "data")})


def _one_d(root):
    for sub in ("binary", "csv", "plots", "lineouts"):
        os.makedirs(root / sub)
    spectrogram().to_netcdf(root / "binary" / "ele_fit_and_data.nc")
    spectrogram().to_netcdf(root / "binary" / "ion_fit_and_data.nc")
    pd.DataFrame({"lineout pixel": [1, 2], "Te": [0.6, 0.7], "ne": [0.22, 0.23]}).to_csv(
        root / "csv" / "learned_parameters.csv", index=False
    )
    pd.DataFrame({"loss": [1.0, 0.5]}).to_csv(root / "csv" / "losses.csv", index=False)
    (root / "plots" / "fit_and_data.png").write_bytes(b"png")
    (root / "lineouts" / "lineout_1.png").write_bytes(b"png")
    (root / "state_weights.txt").write_text("weights")


def _angular(root):
    os.makedirs(root / "binary")
    spectrogram(x_label="Scattering angle (degrees)").to_netcdf(root / "binary" / "fit_and_data.nc")
    (root / "angular_objective_diagnostics.npz").write_bytes(b"npz")
    (root / "angular_objective_terms.json").write_text("{}")


def _ion_only(root):
    os.makedirs(root / "binary")
    spectrogram().to_netcdf(root / "binary" / "ion_fit_and_data.nc")


def _no_spectrogram(root):
    os.makedirs(root / "plots")
    (root / "plots" / "only.png").write_bytes(b"png")


TREES = {"1d": _one_d, "angular": _angular, "ion_only": _ion_only, "no_spectrogram": _no_spectrogram}


def _masked(built):
    built = json.loads(json.dumps(built))
    built["tsadar_version"] = "<masked>"
    for entry in built["files"]:
        assert isinstance(entry["bytes"], int) and entry["bytes"] > 0, entry
        entry["bytes"] = "<masked>"
        for key in ("coords", "data_vars"):
            if key in entry:
                entry[key] = sorted(entry[key])
    return built


@pytest.mark.parametrize("case", TREES)
def test_manifest_matches_its_golden_file(case, tmp_path):
    TREES[case](tmp_path)
    written = manifest.write_manifest(str(tmp_path))

    with open(tmp_path / manifest.MANIFEST_NAME) as fi:
        assert json.load(fi) == written
    assert written == manifest.write_manifest(str(tmp_path))
    on_disk = {rel for rel, _ in manifest._walk(str(tmp_path))} - {manifest.MANIFEST_NAME}
    assert {e["path"] for e in written["files"]} == on_disk

    actual = _masked(written)
    golden = GOLDEN / f"manifest_{case}.json"
    if os.environ.get("TSADAR_REGENERATE_GOLDEN"):
        golden.parent.mkdir(exist_ok=True)
        golden.write_text(json.dumps(actual, indent=2) + "\n")
    assert actual == json.loads(golden.read_text()), (
        f"manifest for the {case} tree differs from {golden.name}; if the change is intended, regenerate "
        "(see the module docstring) and bump SCHEMA_VERSION for a rename or removal"
    )


def test_a_broken_manifest_never_fails_the_fit(tmp_path, monkeypatch):
    # The fit is done and the artifacts are good by this point. Losing the
    # manifest is a degraded read experience, not a lost run.
    monkeypatch.setattr(manifest, "build_manifest", lambda *a, **kw: 1 / 0)

    assert manifest.write_manifest(str(tmp_path)) == {}


def test_an_unreadable_dataset_is_noted_not_fatal(tmp_path):
    os.makedirs(tmp_path / "binary")
    (tmp_path / "binary" / "ele_fit_and_data.nc").write_bytes(b"not actually netcdf")

    entry = next(e for e in manifest.build_manifest(str(tmp_path))["files"] if e["path"] == "binary/ele_fit_and_data.nc")

    assert entry["role"] == manifest.ROLE_SPECTROGRAM
    assert "unreadable" in entry


def test_postprocess_logs_a_manifest_with_the_artifacts(tmp_path, monkeypatch):
    """
    Guards the wiring, not the manifest: a refactor that moved or dropped the
    write_manifest call would leave every subsequent run without a contract, and
    nothing else in this file would notice.
    """

    import mlflow

    from tsadar.inverse import postprocess
    from tsadar.inverse.postprocess import laplace

    def fake_process_data(
        config, sample_indices, all_data, all_axes, loss_fn, fitted_weights, sa, init_losses, t1, td, all_params, num_params
    ):
        spectrogram().to_netcdf(os.path.join(td, "binary", "ele_fit_and_data.nc"))

        return t1, {}

    # postprocess.postprocess (re-exported from .laplace) resolves `process_data` by bare-name lookup
    # in laplace's own module globals, not the postprocess package's -- so the patch target has to be
    # the laplace submodule, not the package attribute of the same name.
    monkeypatch.setattr(laplace, "process_data", fake_process_data)

    previous_uri = mlflow.get_tracking_uri()
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
        mlflow.set_tracking_uri(previous_uri)

    assert manifest.MANIFEST_NAME in logged
