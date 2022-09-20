from jax import numpy as jnp
from jax import jit
import jax
from inverse_thomson_scattering.v0.fitmodl import get_fit_model


def get_loss_function(TSinputs, xie, sas, data):

    fit_model = get_fit_model(TSinputs, xie, sas)
    lam = TSinputs["lam"]["val"]
    amp1 = TSinputs["amp1"]["val"]
    amp2 = TSinputs["amp2"]["val"]
    amp3 = TSinputs["amp3"]["val"]

    stddev = TSinputs["D"]["PhysParams"]["widIRF"]

    def load_ion_spec(lamAxisI, modlI, lamAxisE):
        originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
        inst_funcI = jnp.squeeze(
            (1.0 / (stddev[1] * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddev[1]) ** 2.0))
        )  # Gaussian
        ThryI = jnp.convolve(modlI, inst_funcI, "same")
        ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
        ThryI = jnp.average(ThryI.reshape(1024, -1), axis=1)

        if TSinputs["D"]["PhysParams"]["norm"] == 0:
            lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = amp3 * TSinputs["D"]["PhysParams"]["amps"][1] * ThryI / jnp.amax(ThryI)
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)

        return lamAxisI, lamAxisE, ThryI

    def load_electron_spec(lamAxisE, modlE):
        # Conceptual_origin so the convolution donsn't shift the signal
        originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
        inst_funcE = jnp.squeeze(
            (1.0 / (stddev[0] * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddev[0]) ** 2.0))
        )  # Gaussian
        ThryE = jnp.convolve(modlE, inst_funcE, "same")
        ThryE = (jnp.amax(modlE) / jnp.amax(ThryE)) * ThryE

        if TSinputs["D"]["PhysParams"]["norm"] > 0:
            ThryE = jnp.where(
                lamAxisE < lam,
                amp1 * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
                amp2 * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
            )

        ThryE = jnp.average(ThryE.reshape(1024, -1), axis=1)
        if TSinputs["D"]["PhysParams"]["norm"] == 0:
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)
            ThryE = TSinputs["D"]["PhysParams"]["amps"][0] * ThryE / jnp.amax(ThryE)
            ThryE = jnp.where(lamAxisE < lam, amp1 * ThryE, amp2 * ThryE)

        return lamAxisE, ThryE

    def get_spectra(modlE, modlI, lamAxisE, lamAxisI):

        if TSinputs["D"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, lamAxisE, ThryI = load_ion_spec(lamAxisI, modlI, lamAxisE)

        if TSinputs["D"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = load_electron_spec(lamAxisE, modlE)

        return ThryE, ThryI, lamAxisE, lamAxisI

    def loss_fn(x):

        modlE, modlI, lamAxisE, lamAxisI = fit_model(x)
        ThryE, ThryI, lamAxisE, lamAxisI = get_spectra(modlE, modlI, lamAxisE, lamAxisI)
        chisq = 0
        if TSinputs["D"]["extraoptions"]["fit_IAW"]:
            #    chisq=chisq+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
            chisq = chisq + jnp.sum((data[1, :] - ThryI) ** 2)

        if TSinputs["D"]["extraoptions"]["fit_EPWb"]:
            chisq = chisq + jnp.sum(
                (data[0, (lamAxisE > 410) & (lamAxisE < 510)] - ThryE[(lamAxisE > 410) & (lamAxisE < 510)]) ** 2
            )

        if TSinputs["D"]["extraoptions"]["fit_EPWr"]:
            chisq = chisq + jnp.sum(
                (data[0, (lamAxisE > 540) & (lamAxisE < 680)] - ThryE[(lamAxisE > 540) & (lamAxisE < 680)]) ** 2
            )

        return chisq

    vg_func = jax.value_and_grad(loss_fn)

    def val_and_grad_loss(x):
        x = jnp.array(x)
        value, grad = vg_func(x)

        return value, jnp.array(grad)

    return loss_fn, val_and_grad_loss