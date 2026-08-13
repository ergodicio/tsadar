import time
import multiprocessing as mp
import yaml
import mlflow
from flatten_dict import flatten, unflatten
from numpy.testing import assert_allclose
from jax import config

config.update("jax_enable_x64", True)

from tsadar.inverse import fitter
from tsadar.utils import misc


def test_refit():
    # Regression test for the refit_bad_fits path (postprocess.py), which is otherwise never exercised by
    # the other inverse tests since they all run with other.refit = false. Forces every lineout to be flagged
    # as "bad" so refit_bad_fits actually attempts a refit, guarding against the pytree-structure mismatch
    # that used to crash this path (fitted_weights[idx] wasn't dict-subscriptable, and the "m" override was
    # nested one level too shallow relative to the fe.params.m config schema).
    #
    # This config has 2 lineouts (lineouts.start=500, end=510, skip=5) with batch_size=2, which matters because
    # refit_bad_fits hardcodes `if i == 0: continue`, i.e. the first lineout in the batch can never be refit -
    # at least 2 lineouts are needed for the refit loop body to actually run.
    with open("tests/configs/time_test_defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("tests/configs/time_test_inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)

    config["other"]["refit"] = True
    config["other"]["refit_thresh"] = -1.0  # losses are non-negative, so this flags every lineout as "bad"

    mlflow.set_experiment("tsadar-tests")

    with mlflow.start_run() as run:
        misc.log_mlflow(config)
        config["num_cores"] = int(mp.cpu_count())

        t0 = time.time()
        fit_results, loss = fitter.fit(config=config)
        metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
        mlflow.log_metrics(metrics=metrics_dict)
        mlflow.set_tag("status", "completed")
        print(fit_results)

        # loose sanity bounds (not tight regression values like test_1d_data.py) - the point of this test is
        # that fitter.fit() completes without raising when refit is forced on, and still returns physically
        # reasonable parameters for both lineouts, not that refit reproduces a specific known-good fit.
        assert len(fit_results["Te_electron"]) == 2
        for i in range(2):
            assert 0.0 < fit_results["Te_electron"][i] < 5.0
            assert 0.0 < fit_results["ne_electron"][i] < 1.0
            assert 0.0 < fit_results["m_electron"][i] < 10.0


if __name__ == "__main__":
    test_refit()
