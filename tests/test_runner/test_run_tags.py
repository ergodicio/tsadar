"""
Tests for the canonical run tags and the status lifecycle (ergodicio/tsadar#115).

The fit itself is stubbed out here: what is under test is the tagging contract
that ``_run_`` wraps around the fit, not the fit. Runs land in a file-backed
MLflow store in a temporary directory, so these exercise the real
``mlflow.set_tag`` path without needing a tracking server.
"""

import mlflow
import pytest

from tsadar import runner
from tsadar.utils import misc


@pytest.fixture
def tracking(tmp_path):
    """
    Points MLflow at a throwaway sqlite store and yields a client for reading back.

    sqlite rather than the ``file://`` store because MLflow put the filesystem
    backend into maintenance mode and now refuses it without an opt-out env var;
    a database backend is also what production runs against.
    """

    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    client = mlflow.tracking.MlflowClient()
    client.create_experiment("tag-tests", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("tag-tests")
    yield client
    mlflow.set_tracking_uri(previous_uri)


def tags_of(client, run_id):
    return client.get_run(run_id).data.tags


CONFIG = {
    "data": {"shotnum": 101675, "load_ele_spec": True, "load_ion_spec": False},
    "other": {"username": "avi"},
}


# -- canonical_tags -----------------------------------------------------------

CANONICAL_TAGS = [
    pytest.param(
        CONFIG, "fit", {"tsadar.mode": "fit", "tsadar.shotnum": "101675", "tsadar.data": "epw", "tsadar.user": "avi"},
        [], id="filterable-fields",
    ),
    *[
        pytest.param({"data": {"load_ele_spec": ele, "load_ion_spec": ion}}, "fit", {"tsadar.data": kind}, [], id=f"data-{kind}")
        for ele, ion, kind in [(True, False, "epw"), (False, True, "iaw"), (True, True, "both"), (False, False, "none")]
    ],
    pytest.param(CONFIG, "Fit", {"tsadar.mode": "fit"}, [], id="mode-normalized"),
    pytest.param({"data": {"shotnum": [101675, 101676]}}, "fit", {"tsadar.shotnum": "101675,101676"}, [], id="multi-shot"),
    pytest.param({"data": {"load_ele_spec": True}}, "fit", {}, ["tsadar.shotnum", "tsadar.user"], id="unknown-omitted"),
    pytest.param({"data": None, "other": None}, "fit", {"tsadar.data": "none"}, [], id="null-sections"),
    pytest.param({"data": {"shotnum": None}}, "fit", {"tsadar.data": "none"}, ["tsadar.shotnum"], id="null-shot"),
    pytest.param({}, "forward", {"tsadar.mode": "forward", "tsadar.data": "none"}, [], id="forward-empty"),
]


@pytest.mark.parametrize("config, mode, present, absent", CANONICAL_TAGS)
def test_canonical_tags(config, mode, present, absent):
    tags = misc.canonical_tags(config, mode=mode)

    assert tags["tsadar.version"]
    assert {k: tags.get(k) for k in present} == present
    assert not set(absent) & set(tags)


def test_tag_values_are_clipped_to_the_mlflow_limit():
    value = misc.canonical_tags({"data": {"shotnum": list(range(5000))}})["tsadar.shotnum"]

    assert len(value) == misc.MAX_TAG_LEN
    assert value.endswith("...")


# -- format_error -------------------------------------------------------------


@pytest.mark.parametrize(
    "error, expected",
    [
        (ValueError("something\n  broke   badly"), "ValueError: something broke badly"),
        (KeyboardInterrupt(), "KeyboardInterrupt"),
    ],
)
def test_format_error_is_one_readable_line(error, expected):
    assert misc.format_error(error) == expected


def test_format_error_is_clipped():
    assert len(misc.format_error(ValueError("x" * 10_000))) == misc.MAX_TAG_LEN


# -- the lifecycle around _run_ ----------------------------------------------


def test_canonical_tags_are_set_before_the_fit_starts(tracking, monkeypatch):
    # The point of tagging up front is that an in-flight run is findable. If the
    # tags only landed at the end, a queue of running fits would be unfilterable.
    seen = {}

    def capture(config):
        # Re-read through the client: active_run() holds a snapshot taken when
        # the run started, so it would not show tags written since.
        seen.update(tags_of(tracking, mlflow.active_run().info.run_id))
        return {}, 0.0

    monkeypatch.setattr(runner.fitter, "fit", capture)

    with mlflow.start_run():
        runner._run_(dict(CONFIG), mode="fit")

    assert seen["tsadar.shotnum"] == "101675"
    assert seen["status"] == misc.STATUS_RUNNING


def _fit_returns(config):
    return {}, 0.0


def _fit_raises(config):
    raise RuntimeError("minimizer diverged")


def _fit_is_cancelled(config):
    raise KeyboardInterrupt


def _fit_dies_mid_stage(config):
    mlflow.set_tag("status", "minimizing")
    raise RuntimeError("boom")


LIFECYCLE = [
    pytest.param(_fit_returns, "fit", None, misc.STATUS_COMPLETED, None, id="completed"),
    pytest.param(_fit_raises, "fit", RuntimeError, misc.STATUS_FAILED, "RuntimeError: minimizer diverged", id="raises"),
    pytest.param(_fit_is_cancelled, "fit", KeyboardInterrupt, misc.STATUS_FAILED, "KeyboardInterrupt", id="cancelled"),
    pytest.param(_fit_returns, "nonsense", NotImplementedError, misc.STATUS_FAILED, "NotImplementedError", id="unknown-mode"),
    pytest.param(_fit_dies_mid_stage, "fit", RuntimeError, misc.STATUS_FAILED, "RuntimeError: boom", id="dies-mid-stage"),
]


@pytest.mark.parametrize("fit, mode, raises, status, error", LIFECYCLE)
def test_runs_end_in_a_terminal_status(tracking, monkeypatch, fit, mode, raises, status, error):
    monkeypatch.setattr(runner.fitter, "fit", fit)

    if raises is None:
        with mlflow.start_run() as run:
            runner._run_(dict(CONFIG), mode=mode)
    else:
        with pytest.raises(raises):
            with mlflow.start_run() as run:
                runner._run_(dict(CONFIG), mode=mode)

    tags = tags_of(tracking, run.info.run_id)
    assert tags["status"] == status
    assert tags["status"] in misc.TERMINAL_STATUSES
    if error is None:
        assert "error" not in tags
        assert tags["tsadar.shotnum"] == "101675" and tags["tsadar.mode"] == "fit"
    else:
        assert error in tags["error"]


def test_a_tagging_failure_does_not_bury_the_real_error(tracking, monkeypatch):
    # A tracking server that drops out mid-run makes the failure-tagging call
    # raise too. The fit's own error is the one worth propagating; the tagging
    # error would bury why the run actually died.
    real_set_tag = mlflow.set_tag

    def flaky_set_tag(key, value, *args, **kwargs):
        if value in (misc.STATUS_FAILED,) or key == "error":
            raise ConnectionError("tracking server unreachable")

        return real_set_tag(key, value, *args, **kwargs)

    def explode(config):
        raise RuntimeError("the real problem")

    monkeypatch.setattr(runner.fitter, "fit", explode)
    monkeypatch.setattr(runner.mlflow, "set_tag", flaky_set_tag)

    with pytest.raises(RuntimeError, match="the real problem"):
        with mlflow.start_run():
            runner._run_(dict(CONFIG), mode="fit")


def test_runs_are_searchable_by_their_tags(tracking, monkeypatch):
    # The reason the tags exist: MLflow filter strings, which is what the
    # browser's run table is built on.
    monkeypatch.setattr(runner.fitter, "fit", lambda config: ({}, 0.0))

    with mlflow.start_run():
        runner._run_(dict(CONFIG), mode="fit")

    found = mlflow.search_runs(
        experiment_names=["tag-tests"],
        filter_string="tags.`tsadar.shotnum` = '101675' and tags.`tsadar.data` = 'epw'",
        output_format="list",
    )

    assert len(found) == 1
