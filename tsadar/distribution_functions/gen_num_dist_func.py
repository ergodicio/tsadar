from jax import numpy as jnp
import scipy.io as sio
import jax

from tsadar.distribution_functions import dist_functional_forms


# needs the ability to enforce symetry
class DistFunc:
    """
    Distribution function class used to generate numerical distribution functions based off some known functional forms.
    Eventually this class will be expanded to handle loading of numerical distribution function from text files.

    """

    def __init__(self, cfg):
        """
        Distribution function class constructor, reads the inout deck and used the relevant fields to set static
        parameters for the distribution function creation. These include properties like the dimension and velocity grid
        spacing that are static.


        Args:
            cfg: Dictionary for the electron species, a subfield of the input deck dictionary

        Returns:
            DistFunc: An instance of the DistFunc class

        """
        self.velocity_res = cfg["fe"]["v_res"]
        self.fe_name = list(cfg["fe"]["type"].keys())[-1]

        if "dim" in cfg["fe"].keys():
            self.dim = cfg["fe"]["dim"]
        else:
            self.dim = 1

        if "dt" in cfg["fe"].keys():
            self.dt = cfg["fe"]["dt"]

        # normalized here so it only is done once
        if "f1_direction" in cfg["fe"].keys():
            self.f1_direction = jnp.array(cfg["fe"]["f1_direction"]) / jnp.sqrt(
                jnp.sum(jnp.array([ele**2 for ele in cfg["fe"]["f1_direction"]]))
            )
        # temperature asymetry for biDLM with Tex = Te and Tey = Te*temp_asym
        if "temp_asym" in cfg["fe"].keys():
            self.temp_asym = cfg["fe"]["temp_asym"]
        else:
            self.temp_asym = 1.0

        # m asymetry for biDLM with mx = m and my = m*m_asym (with a min of 2)
        if "m_asym" in cfg["fe"].keys():
            self.m_asym = cfg["fe"]["m_asym"]
        else:
            self.m_asym = 1.0

        # rotion angle for the biDLM defined counter clockwise from the x-axis in degrees
        if "m_theta" in cfg["fe"].keys():
            self.m_theta = cfg["fe"]["m_theta"] / 180.0 * jnp.pi
        else:
            self.m_theta = 0.0

    def __call__(self, mval):
        """
        Distribution function class call, produces a numerical distribution function based of the object and the current
        m-value.


        Args:
            mval: super-gaussian order to be used in calculation must be a float or shape (1,)

        Returns:
            v: Velocity grid, for 1D distribution this is a single array, for 2D this is a tuple of arrays
            fe: Numerical distribution function

        """
        if self.fe_name == "DLM":
            if self.dim == 1:
                #v, fe = dist_functional_forms.DLM_1D(mval, self.velocity_res)
                tabl="numDistFuncs/DLM_x_-3_-10_10_m_-1_2_5.mat"
                tablevar = sio.loadmat(tabl, variable_names="IT")
                IT = tablevar["IT"]
                vx = jnp.arange(-8, 8, self.velocity_res)
                xs = jnp.arange(-10,10,.001)
                ms = jnp.arange(2,5,.1)
                x_float_inds = jnp.interp(vx, xs, jnp.linspace(0,xs.shape[0]-1,xs.shape[0]))
                m_float_inds = jnp.interp(mval, ms, jnp.linspace(0,ms.shape[0]-1,ms.shape[0]))

                                          #np.linspace(0, params["x"].size - 1, params["x"].size))
                #m_float_inds = jnp.array(jnp.interp(m, params["m"], np.linspace(0, params["m"].size - 1, params["m"].size)))
                m_float_inds = m_float_inds.reshape((1,))
                ind_x_mesh, ind_m_mesh = jnp.meshgrid(x_float_inds, m_float_inds)
                indices = jnp.concatenate([ind_x_mesh.flatten()[:, None], ind_m_mesh.flatten()[:, None]], axis=1)

                fe = jax.scipy.ndimage.map_coordinates(IT, indices.T, order=1, mode="constant", cval=0.0)
                v=vx
            elif self.dim == 2:
                # v, fe = dist_functional_forms.DLM_2D(mdict["val"], self.velocity_res)
                # v, fe = dist_functional_forms.BiDLM(
                #    mval,
                #    jnp.max(jnp.array([mval * self.m_asym, 2.0])),
                #    jnp.max(jnp.array([jnp.array(mval * self.m_asym).squeeze(), 2.0])),
                #    self.temp_asym,
                #    self.m_theta,
                #    self.velocity_res,
                # )
                # this will cause issues if my is less then 2
                v, fe = dist_functional_forms.BiDLM(
                    mval, mval * self.m_asym, self.temp_asym, self.m_theta, self.velocity_res
                )

        elif self.fe_name == "Spitzer":
            if self.dim == 2:
                if len(self.f1_direction) == 2:
                    v, fe = dist_functional_forms.Spitzer_2V(self.dt, self.f1_direction, self.velocity_res)
                elif len(self.f1_direction) == 3:
                    v, fe = dist_functional_forms.Spitzer_3V(self.dt, self.f1_direction, self.velocity_res)
            else:
                raise ValueError("Spitzer distribution can only be computed in 2D")

        elif self.fe_name == "MYDLM":
            if self.dim == 2:
                if len(self.f1_direction) == 2:
                    v, fe = dist_functional_forms.MoraYahi_2V(self.dt, self.f1_direction, self.velocity_res)
                elif len(self.f1_direction) == 3:
                    v, fe = dist_functional_forms.MoriYahi_3V(self.dt, self.f1_direction, self.velocity_res)
            else:
                raise ValueError("Mora and Yahi distribution can only be computed in 2D")

        return v, fe
