import copy

from inverse_thomson_scattering.model.physics.form_factor import get_form_factor_fn
from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func

from jax import numpy as jnp


def get_fit_model(config, sa, backend: str = "haiku"):
    nonMaxwThomsonE_jax = get_form_factor_fn(config["other"]["lamrangE"], npts=config["other"]["npts"], backend=backend)
    nonMaxwThomsonI_jax = get_form_factor_fn(config["other"]["lamrangI"], npts=config["other"]["npts"], backend=backend)
    num_dist_func = get_num_dist_func(config["parameters"]["fe"]["type"], config["velocity"])

    def fit_model(fitted_params):
        parameters = copy.deepcopy(config["parameters"])
        for key in parameters.keys():
            # if parameters[key]["active"]:
            parameters[key]["val"] = jnp.squeeze(fitted_params[key])

        # if parameters["fe"]["active"]:
        #     parameters["fe"]["val"] = fitted_params[-parameters["fe"]["length"] : :]
        if parameters["m"]["active"]:
            parameters["fe"]["val"] = jnp.log(num_dist_func(parameters["m"]["val"]))

        # Add gradients to electron temperature and density just being applied to EPW
        cur_Te = jnp.linspace(
            (1 - parameters["Te_gradient"]["val"] / 200) * parameters["Te"]["val"],
            (1 + parameters["Te_gradient"]["val"] / 200) * parameters["Te"]["val"],
            parameters["Te_gradient"]["num_grad_points"]
        )
        cur_ne = jnp.linspace(
            (1 - parameters["ne_gradient"]["val"] / 200) * parameters["ne"]["val"],
            (1 + parameters["ne_gradient"]["val"] / 200) * parameters["ne"]["val"],
            parameters["Te_gradient"]["num_grad_points"]
        )

        fecur = jnp.exp(parameters["fe"]["val"])
        vcur = config["velocity"]
        if config["parameters"]["fe"]["symmetric"]:
            fecur = jnp.concatenate((jnp.flip(fecur[1:]),fecur))
            vcur = jnp.concatenate((-jnp.flip(vcur[1:]),vcur))
        
        #fecur = parameters["fe"]["val"]
        lam = parameters["lam"]["val"]

        if config["other"]["extraoptions"]["load_ion_spec"]:
            ThryI, lamAxisI = nonMaxwThomsonI_jax(
                cur_Te,
                parameters["Ti"]["val"],
                parameters["Z"]["val"],
                parameters["A"]["val"],
                parameters["fract"]["val"],
                cur_ne * jnp.array([1e20]),  # TODO hardcoded
                parameters["Va"]["val"],
                parameters["ud"]["val"],
                sa["sa"],
                (fecur, vcur),
                lam,
            )

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7  # TODO hardcoded

            ThryI = jnp.real(ThryI)
            ThryI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(ThryI * sa["weights"][0], axis=1)
        else:
            modlI = 0
            lamAxisI = []

        if config["other"]["extraoptions"]["load_ele_spec"]:
            ThryE, lamAxisE = nonMaxwThomsonE_jax(
                cur_Te,
                parameters["Ti"]["val"],
                parameters["Z"]["val"],
                parameters["A"]["val"],
                parameters["fract"]["val"],
                cur_ne * 1e20,  # TODO hardcoded
                parameters["Va"]["val"],
                parameters["ud"]["val"],
                sa["sa"],
                (fecur, vcur),
                lam,
            )

            # if parameters.fe['Type']=='MYDLM':
            #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,parameters.fe['thetaphi'])
            # elif parameters.fe['Type']=='Numeric':
            #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,[2*np.pi/3,0])
            # else:
            #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,expion=D['expandedions'])
            # nonMaxwThomson,_ =get_form_factor_fn(D['lamrangE'],lam)
            # [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,sa['sa'], [fecur,xie])

            # remove extra dimensions and rescale to nm
            lamAxisE = jnp.squeeze(lamAxisE) * 1e7  # TODO hardcoded
        
            ThryE = jnp.real(ThryE)
            ThryE = jnp.mean(ThryE, axis=0)
            if config["other"]["extraoptions"]["spectype"] == "angular_full":
                modlE = jnp.matmul(sa["weights"], ThryE.transpose())
            else:
                modlE = jnp.sum(ThryE * sa["weights"][0], axis=1)

            if config["other"]["iawoff"] and (config["other"]["lamrangE"][0] < lam < config["other"]["lamrangE"][1]):
                # set the ion feature to 0 #should be switched to a range about lam
                lamloc = jnp.argmin(jnp.abs(lamAxisE - lam))
                modlE = jnp.concatenate(
                    [modlE[: lamloc - 2000], jnp.zeros(4000), modlE[lamloc + 2000 :]]
                )  # TODO hardcoded

            if config["other"]["iawfilter"][0]:
                filterb = config["other"]["iawfilter"][3] - config["other"]["iawfilter"][2] / 2
                filterr = config["other"]["iawfilter"][3] + config["other"]["iawfilter"][2] / 2

                if config["other"]["lamrangE"][0] < filterr and config["other"]["lamrangE"][1] > filterb:
                    indices = (filterb < lamAxisE) & (filterr > lamAxisE)
                    modlE = jnp.where(indices, modlE * 10 ** (-config["other"]["iawfilter"][1]), modlE)
        else:
            modlE = 0
            lamAxisE = []

        return modlE, modlI, lamAxisE, lamAxisI, parameters

    return fit_model