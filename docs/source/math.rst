Implementation of Thomson Scattering
-------------------------------------

Electron susceptibility convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TSADAR uses perturbations proportional to
:math:`\exp(i\mathbf{k}\mathbin{\cdot}\mathbf{x}-i\omega t)` and the corresponding
Landau contour.  If :math:`f_k(v)` is the normalized electron distribution projected
along :math:`\widehat{\mathbf{k}}`, the collisionless longitudinal susceptibility is

.. math::

   \chi_e = -\frac{1}{(k\lambda_{De})^2}
   \left[\mathop{\mathrm{PV}}\int
   \frac{f_k'(v)}{v-\xi}\,dv + i\pi f_k'(\xi)\right].

For a Maxwellian, with :math:`\zeta=\xi/\sqrt{2}`, this is
:math:`\chi_e=[1+\zeta Z(\zeta)]/(k\lambda_{De})^2`, where
:math:`Z(\zeta)=i\sqrt{\pi}\,w(\zeta)` and :math:`w` is the Faddeeva function.
This convention fixes the sign of both the 1-D and 2-D implementations.

Flow and projection frames
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ion species retains its own lab-frame flow :math:`\mathbf{V}_s`.  The ion-fluid
reference velocity is the charge-weighted mean

.. math::

   \mathbf{V}_{i} =
   \frac{\sum_s Z_s f_s\mathbf{V}_s}{\sum_s Z_s f_s},

and ``ud`` is the electron drift relative to that reference.  Thus the electron
lab-frame flow is :math:`\mathbf{u}_e=\mathbf{V}_i+\mathbf{u}_d`.  The 2-D electron
distribution is projected along
:math:`\beta=\operatorname{atan2}(\widehat{k}_y,\widehat{k}_x)` and sampled at the
signed coordinate

.. math::

   \xi = \frac{\omega/k-\mathbf{u}_e\mathbin{\cdot}\widehat{\mathbf{k}}}{v_{Te}}.

A flow perpendicular to :math:`\mathbf{k}` therefore cannot rotate or translate the
longitudinal projection.
