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

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    client = mlflow.tracking.MlflowClient()
    client.create_experiment("tag-tests", artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment("tag-tests")
    yield client
    mlflow.set_tracking_uri(None)


def tags_of(client, run_id):
    return client.get_run(run_id).data.tags


CONFIG = {
    "data": {"shotnum": 101675, "load_ele_spec": True, "load_ion_spec": False},
    "other": {"username": "avi"},
}


# -- canonical_tags -----------------------------------------------------------


def test_canonical_tags_covers_the_filterable_fields():
    tags = misc.canonical_tags(CONFIG, mode="fit")

    assert tags["tsadar.mode"] == "fit"
    assert tags["tsadar.shotnum"] == "101675"
    assert tags["tsadar.data"] == "epw"
    assert tags["tsadar.user"] == "avi"
    assert tags["tsadar.version"]


@pytest.mark.parametrize(
    "ele, ion, expected",
    [(True, False, "epw"), (False, True, "iaw"), (True, True, "both"), (False, False, "none")],
)
def test_data_kind_follows_the_load_switches(ele, ion, expected):
    config = {"data": {"load_ele_spec": ele, "load_ion_spec": ion}}

    assert misc.canonical_tags(config)["tsadar.data"] == expected


def test_mode_is_normalized():
    assert misc.canonical_tags(CONFIG, mode="Fit")["tsadar.mode"] == "fit"


def test_multi_shot_fits_join_their_shot_numbers():
    # fitter.load_data_for_fitting accepts a list of shots; a LIKE filter on any
    # one of them still has to match.
    config = {"data": {"shotnum": [101675, 101676]}}

    assert misc.canonical_tags(config)["tsadar.shotnum"] == "101675,101676"


def test_unknown_fields_are_omitted_not_blanked():
    # An absent tag means "unknown". A tag set to "" would claim the run genuinely
    # had no shot number, which is a different and wrong statement.
    tags = misc.canonical_tags({"data": {"load_ele_spec": True}})

    assert "tsadar.shotnum" not in tags
    assert "tsadar.user" not in tags


def test_tag_values_are_clipped_to_the_mlflow_limit():
    config = {"data": {"shotnum": list(range(5000))}}

    value = misc.canonical_tags(config)["tsadar.shotnum"]

    assert len(value) == misc.MAX_TAG_LEN
    assert value.endswith("...")


@pytest.mark.parametrize("config", [{"data": None, "other": None}, {"data": {"shotnum": None}}, {}])
def test_canonical_tags_survives_null_sections(config):
    # Decks in the wild carry keys set to null. This runs at the top of every
    # run, so it must not be the thing that breaks a run that would otherwise
    # work.
    assert misc.canonical_tags(config)["tsadar.data"] == "none"


def test_canonical_tags_survives_a_config_missing_every_key():
    # Forward-only decks do not carry the data section the fit path assumes.
    tags = misc.canonical_tags({}, mode="forward")

    assert tags["tsadar.mode"] == "forward"
    assert tags["tsadar.data"] == "none"


# -- format_error -------------------------------------------------------------


def test_format_error_is_one_readable_line():
    error = ValueError("something\n  broke   badly")

    assert misc.format_error(error) == "ValueError: something broke badly"


def test_format_error_handles_an_empty_message():
    assert misc.format_error(KeyboardInterrupt()) == "KeyboardInterrupt"


def test_format_error_is_clipped():
    assert len(misc.format_error(ValueError("x" * 10_000))) == misc.MAX_TAG_LEN


# -- the lifecycle around _run_ ----------------------------------------------


def test_successful_run_is_tagged_completed(tracking, monkeypatch):
    monkeypatch.setattr(runner.fitter, "fit", lambda config: ({}, 0.0))

    with mlflow.start_run() as run:
        runner._run_(dict(CONFIG), mode="fit")

    tags = tags_of(tracking, run.info.run_id)
    assert tags["status"] == misc.STATUS_COMPLETED
    assert tags["tsadar.shotnum"] == "101675"
    assert tags["tsadar.mode"] == "fit"
    assert "error" not in tags


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


def test_a_raising_fit_is_tagged_failed_and_reraises(tracking, monkeypatch):
    def explode(config):
        raise RuntimeError("minimizer diverged")

    monkeypatch.setattr(runner.fitter, "fit", explode)

    with pytest.raises(RuntimeError, match="minimizer diverged"):
        with mlflow.start_run() as run:
            runner._run_(dict(CONFIG), mode="fit")

    tags = tags_of(tracking, run.info.run_id)
    assert tags["status"] == misc.STATUS_FAILED
    assert tags["error"] == "RuntimeError: minimizer diverged"


def test_a_cancelled_run_is_tagged_failed(tracking, monkeypatch):
    # A Batch timeout arrives as a KeyboardInterrupt, which is not an Exception.
    def cancel(config):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.fitter, "fit", cancel)

    with pytest.raises(KeyboardInterrupt):
        with mlflow.start_run() as run:
            runner._run_(dict(CONFIG), mode="fit")

    assert tags_of(tracking, run.info.run_id)["status"] == misc.STATUS_FAILED


def test_an_unknown_mode_is_tagged_failed(tracking):
    with pytest.raises(NotImplementedError):
        with mlflow.start_run() as run:
            runner._run_(dict(CONFIG), mode="nonsense")

    tags = tags_of(tracking, run.info.run_id)
    assert tags["status"] == misc.STATUS_FAILED
    assert "NotImplementedError" in tags["error"]


def test_status_ends_terminal_whatever_the_stage_tag_said(tracking, monkeypatch):
    # fitter/postprocess overwrite `status` with their progress stages. The
    # failure branch has to win over whichever stage the run died in, since a
    # run stuck at "minimizing" is what #115 exists to make distinguishable.
    def die_mid_stage(config):
        mlflow.set_tag("status", "minimizing")
        raise RuntimeError("boom")

    monkeypatch.setattr(runner.fitter, "fit", die_mid_stage)

    with pytest.raises(RuntimeError):
        with mlflow.start_run() as run:
            runner._run_(dict(CONFIG), mode="fit")

    assert tags_of(tracking, run.info.run_id)["status"] in misc.TERMINAL_STATUSES


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
