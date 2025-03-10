from jax import config, jit, numpy as jnp
from copy import deepcopy
import yaml, mlflow, numpy as np, tempfile
from flatten_dict import flatten, unflatten
import matplotlib.pyplot as plt


config.update("jax_enable_x64", True)
from numpy.testing import assert_allclose
from scipy.signal import find_peaks
from tsadar.core.physics.form_factor import FormFactor
from tsadar.core.modules import ThomsonParams


def test_iaw():
    """
    Test #2: IAW test, calculate a spectrum and compare the resonance to the IAW dispersion relation

    Returns:

    """
    mlflow.set_experiment(experiment_name="tsadar-tests")
    with mlflow.start_run(run_name="iaw_form_factor_test") as run:
        with open("tests/configs/epw_defaults.yaml", "r") as fi:
            defaults = yaml.safe_load(fi)

        with open("tests/configs/epw_inputs.yaml", "r") as fi:
            inputs = yaml.safe_load(fi)

        defaults = flatten(defaults)
        defaults.update(flatten(inputs))
        config = unflatten(defaults)

        C = 2.99792458e10
        Me = 510.9896 / C**2  # electron mass keV/C^2
        Mp = Me * 1836.1  # proton mass keV/C^2
        re = 2.8179e-13  # classical electron radius cm
        Esq = Me * C**2 * re  # sq of the electron charge keV cm

        ion_form_factor = FormFactor(
            [525, 528],
            npts=8192,
            lam_shift=0.0,
            scattering_angles={"sa": np.array([60])},
            num_grad_points=config["parameters"]["general"]["ne_gradient"]["num_grad_points"],
            ud_ang=None,
            va_ang=None,
        )
        constants = jnp.sqrt(4 * jnp.pi * Esq / Me)

        ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False)
        physical_params = ts_params()
        ThryI, lamAxisI = jit(ion_form_factor)(physical_params)
        ThryI = jnp.mean(ThryI, axis=0)

        ThryI = np.squeeze(ThryI)
        test = deepcopy(np.asarray(ThryI))
        peaks, peak_props = find_peaks(test, height=0.1, prominence=0.2)
        highest_peak_index = peaks[np.argmax(peak_props["peak_heights"])]
        second_highest_peak_index = peaks[np.argpartition(peak_props["peak_heights"], -2)[-2]]

        lams = lamAxisI[0, [highest_peak_index, second_highest_peak_index], 0]
        omgpe = constants * jnp.sqrt(0.2 * 1e20)
        omgL = 2 * np.pi * 1e7 * C / config["parameters"]["general"]["lam"]["val"]  # laser frequency Rad / s
        kL = jnp.sqrt(omgL**2 - omgpe**2) / C

        model_omegas = 2 * jnp.pi * C / lams  # peak frequencies
        omg = 2 * kL * jnp.sqrt((0.5 + 3 * 0.2) / Mp)
        theory_omegas = [omgL + omg, omgL - omg]

        assert_allclose(theory_omegas, model_omegas, rtol=1e-2)

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.log_metrics(
                {
                    "model_omega_1": float(model_omegas[0]),
                    "model_omega_2": float(model_omegas[1]),
                    "theory_omega_1": float(theory_omegas[0]),
                    "theory_omega_2": float(theory_omegas[1]),
                }
            )

            fig, ax = plt.subplots(1, 1, figsize=(5, 3), tight_layout=True)
            ax.plot(lamAxisI[0, :, 0], ThryI)
            ax.set_xlabel("Wavelength", fontsize=14)
            ax.set_ylabel("Amplitude", fontsize=14)
            ax.grid()
            fig.savefig(f"{tmpdir}/iaw_form_factor_test.png", bbox_inches="tight")

            mlflow.log_artifacts(tmpdir)


if __name__ == "__main__":
    test_iaw()
