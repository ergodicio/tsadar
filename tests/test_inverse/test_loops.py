import pytest

from tsadar.inverse.loops import advance_refinement_shape, apply_ang_res_unit


def test_apply_ang_res_unit():
    config = {"data": {"lineouts": {"start": 100, "end": 205}}, "other": {"ang_res_unit": 10}}

    apply_ang_res_unit(config)

    assert config["data"]["lineouts"]["start"] == 10
    assert config["data"]["lineouts"]["end"] == 20


def test_advance_refinement_shape():
    # matches multirun_angular_optax's nvx *= refine_factor, window.len = window.len * refine_factor + 1
    config = {
        "parameters": {"electron": {"fe": {"dim": 1, "nvx": 64, "params": {"window": {"len": 5}}}}},
        "optimizer": {"refine_factor": 2},
    }

    advance_refinement_shape(config)

    fe = config["parameters"]["electron"]["fe"]
    assert fe["nvx"] == 128
    assert fe["params"]["window"]["len"] == 11


def test_advance_refinement_shape_multiple_steps_matches_manual_iteration():
    # regression guard for postprocess_runner, which calls this num_mins - 1 times to replay the shape
    # multirun_angular_optax's refinement loop would have reached, without redoing the fit
    config = {
        "parameters": {"electron": {"fe": {"dim": 1, "nvx": 32, "params": {"window": {"len": 3}}}}},
        "optimizer": {"refine_factor": 2},
    }

    for _ in range(3):
        advance_refinement_shape(config)

    fe = config["parameters"]["electron"]["fe"]
    assert fe["nvx"] == 32 * 2**3
    assert fe["params"]["window"]["len"] == 31  # 3 -> 7 -> 15 -> 31


def test_advance_refinement_shape_rejects_non_1d():
    config = {"parameters": {"electron": {"fe": {"dim": 2}}}, "optimizer": {"refine_factor": 2}}

    with pytest.raises(ValueError):
        advance_refinement_shape(config)
