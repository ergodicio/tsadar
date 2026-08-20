"""
Tests for the historical-run tag backfill (scripts/backfill_run_tags.py).

The script is not an importable package, so it is loaded by path. Only the pure
derivation is exercised here -- the MLflow walk around it is thin, and what can
actually go wrong is misreading the string-typed params of an old run.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "backfill_run_tags.py"


@pytest.fixture(scope="module")
def backfill():
    spec = importlib.util.spec_from_file_location("backfill_run_tags", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


HISTORICAL_PARAMS = {
    "data.shotnum": "101675",
    "data.load_ele_spec": "True",
    "data.load_ion_spec": "False",
    "other.username": "avi",
    "optimizer.batch_size": "6",
}


def test_tags_are_recovered_from_flattened_params(backfill):
    tags = backfill.tags_from_params(HISTORICAL_PARAMS)

    assert tags["tsadar.shotnum"] == "101675"
    assert tags["tsadar.data"] == "epw"
    assert tags["tsadar.user"] == "avi"


def test_params_are_parsed_as_strings_not_truthiness(backfill):
    # MLflow hands params back as strings, and the string "False" is truthy --
    # reading it naively would tag every historical run as loading both spectra.
    tags = backfill.tags_from_params({"data.load_ele_spec": "False", "data.load_ion_spec": "True"})

    assert tags["tsadar.data"] == "iaw"


def test_unknowable_fields_are_not_guessed(backfill):
    # mode and version were never logged as params. Inventing them would put a
    # confident wrong value on the tags whose whole job is provenance.
    tags = backfill.tags_from_params(HISTORICAL_PARAMS)

    assert "tsadar.mode" not in tags
    assert "tsadar.version" not in tags


def test_absent_load_switches_mean_unknown_not_none(backfill):
    tags = backfill.tags_from_params({"data.shotnum": "101675"})

    assert "tsadar.data" not in tags


def test_a_run_with_nothing_useful_yields_nothing(backfill):
    assert backfill.tags_from_params({}) == {}
