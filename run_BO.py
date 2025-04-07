import tempfile, time, os, yaml
import multiprocessing as mp
import mlflow
from flatten_dict import flatten, unflatten
from jax import config
from bayes_opt import BayesianOptimization

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

from tsadar import fitter
from tsadar.misc import utils


def bbf(window_factor, grad_scalar):
    all_configs["defaults"]["dist_fit"]["window"]["len"] = window_factor
    all_configs["defaults"]["optimizer"]["grad_scalar"] = grad_scalar

    with mlflow.start_run(run_name=run_name) as run:
        with tempfile.TemporaryDirectory() as td:
            for k in ["defaults", "inputs"]:
                with open(os.path.join(td, f"{k}.yaml"), "w") as fi:
                    yaml.dump(all_configs[k], fi)

            mlflow.log_artifacts(td)

    defaults = flatten(all_configs["defaults"])
    defaults.update(flatten(all_configs["inputs"]))
    config = unflatten(defaults)

    return one_run(config)


def one_run(config):
    utils.log_params(config)

    t0 = time.time()
    fit_results, loss = fitter.fit(config=config)
    metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
    mlflow.log_metrics(metrics=metrics_dict)
    mlflow.set_tag("status", "completed")

    return -loss


if __name__ == "__main__":
    all_configs = {}
    for k in ["defaults", "inputs"]:
        with open(f"{k}.yaml", "r") as fi:
            all_configs[k] = yaml.safe_load(fi)

    if "mlflow" in all_configs["inputs"].keys():
        experiment = all_configs["inputs"]["mlflow"]["experiment"]
        run_name = all_configs["inputs"]["mlflow"]["run"]

    else:
        experiment = all_configs["defaults"]["mlflow"]["experiment"]
        run_name = all_configs["defaults"]["mlflow"]["run"]

    mlflow.set_experiment(experiment)

    # Bounded region of parameter space
    pbounds = {"window_factor": (0.1, 0.95), "grad_scalar": (0.1, 0.95)}

    optimizer = BayesianOptimization(
        f=bbf,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(init_points=2, n_iter=3)

    # divs = np.arange(2, 13)
    # np.random.shuffle(divs)
    #
    # grads = np.linspace(0.05, 0.5, 10)
    # np.random.shuffle(grads)
    #
    # for window_divisor in divs:
    #
    #     for grad_scalar in grads:
    #
    #         bbf()
