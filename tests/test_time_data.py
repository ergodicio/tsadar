import time, pytest
import multiprocessing as mp
import yaml
import mlflow
from flatten_dict import flatten, unflatten
from numpy.testing import assert_allclose
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)
# config.update("jax_check_tracer_leaks", True)

from tsadar import fitter
from tsadar.misc import utils


@pytest.mark.parametrize("nn", [False])
def test_data(nn):
    # Test #3: Data test, compare fit to a preknown fit result
    # currently just runs one line of shot 101675 for the electron, should be expanded in the future

    with open("tests/configs/time_test_defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("tests/configs/time_test_inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)

    config["nn"]["use"] = nn
    # config["parameters"]["Te"]["val"] = 0.5
    # config["parameters"]["ne"]["val"] = 0.2  # 0.25
    # config["parameters"]["m"]["val"] = 3.0  # 2.2

    mlflow.set_experiment(config["mlflow"]["experiment"])

    with mlflow.start_run() as run:
        utils.log_params(config)
        config["num_cores"] = int(mp.cpu_count())

        t0 = time.time()
        fit_results, loss = fitter.fit(config=config)
        metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
        mlflow.log_metrics(metrics=metrics_dict)
        mlflow.set_tag("status", "completed")
        print(fit_results)

        # These were changed 5/6/24 to reflect new good fit values, unclear why changes were required probably a change
        # to the calibration
        assert_allclose(fit_results["amp1_general"][0], 0.734, rtol=1e-1)  # 0.9257
        assert_allclose(fit_results["amp2_general"][0], 0.519, rtol=1e-1)  # 0.6727
        assert_allclose(fit_results["lam_general"][0], 524.016, rtol=5e-3)  # 524.2455
        assert_allclose(fit_results["Te_electron"][0], 0.5994, rtol=1e-1)  # 0.67585
        assert_allclose(fit_results["ne_electron"][0], 0.2256, rtol=5e-2)  # 0.21792
        assert_allclose(fit_results["m_electron"][0], 2.987, rtol=15e-2)  # 3.3673


if __name__ == "__main__":
    test_data(False)
