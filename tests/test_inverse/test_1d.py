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

from tsadar.inverse import fitter
from tsadar.utils import misc


def test_data():
    # Test #3: Data test, compare fit to a preknown fit result
    # currently just runs one line of shot 101675 for the electron, should be expanded in the future

    with open("tests/configs/time_test_defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("tests/configs/time_test_inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)

    # config["parameters"]["Te"]["val"] = 0.5
    # config["parameters"]["ne"]["val"] = 0.2  # 0.25
    # config["parameters"]["m"]["val"] = 3.0  # 2.2

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

        # These are the best values as of 3/22/25 and represent a good fit (although potentially not as good as the matlab fit)
        # and have been rigourously investigated for any potential source of inaccuracy
        real_values = {
            "amp1_general": 0.733,
            "amp2_general": 0.520,
            "lam_general": 523.855,
            "Te_electron": 0.641,
            "ne_electron": 0.228,
            "m_electron": 3.20,
        } 
        
        # These were changed 5/6/24 to reflect new good fit values, unclear why changes were required probably a change
        # to the calibration (note 3/22/25: These numbers also probably changed due to some resolution and interpolation changes)
        # real_values = {
        #     "amp1_general": 0.734,
        #     "amp2_general": 0.519,
        #     "lam_general": 524.016,
        #     "Te_electron": 0.5994,
        #     "ne_electron": 0.2256,
        #     "m_electron": 2.987,
        # }
        #These values are what the origenal Matlab code found
        # real_values = {
        #     "amp1_general": 0.9257,
        #     "amp2_general": 0.6727,
        #     "lam_general": 524.2455,
        #     "Te_electron": 0.67585,
        #     "ne_electron": 0.21792,
        #     "m_electron": 3.3673,
        # }

        assert_allclose(fit_results["amp1_general"][0], real_values["amp1_general"], rtol=1e-1)
        assert_allclose(fit_results["amp2_general"][0], real_values["amp2_general"], rtol=1e-1)
        assert_allclose(fit_results["lam_general"][0], real_values["lam_general"], rtol=5e-3)
        assert_allclose(fit_results["Te_electron"][0], real_values["Te_electron"], rtol=1e-1)
        assert_allclose(fit_results["ne_electron"][0], real_values["ne_electron"], rtol=5e-2)
        assert_allclose(fit_results["m_electron"][0], real_values["m_electron"], rtol=15e-2)

        mlflow.log_metrics({"gt_" + k: real_values[k] for k in real_values.keys()})
        mlflow.log_metrics({"fit_" + k: fit_results[k][0] for k in real_values.keys()})


if __name__ == "__main__":
    test_data()
