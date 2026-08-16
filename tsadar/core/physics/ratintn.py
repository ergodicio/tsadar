from jax import custom_jvp, numpy as jnp


def ratintn(f: jnp.ndarray, g: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    Integrate f / g dz taking each to be piecwise linear.This is more accurate when f / g has a near - pole in an
    interval f, g and z are 1D complex arrays.

    Based on newlip routine by Ed Williams.
    Args:
        f (jnp.ndarray): 1D complex array representing the numerator of the rational function to be integrated.
        g (jnp.ndarray): 1D complex array representing the denominator of the rational function.
        z (jnp.ndarray): 1D complex array representing the variable of integration.
    Returns:
        jnp.ndarray: The integrated values of f / g over z.
    """

    if len(jnp.shape(f)) == 1:
        f = jnp.transpose(f[..., jnp.newaxis])

    zdif = z[1:-1] - z[0:-2]
    out = jnp.sum(ratcen(f, g) * zdif, 1)
    return out


def ratcen(f: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """
    Return "rationally centered" f / g such that int_s(1) ^ s(0) ds f(s) / g(s) = sum(ratcen(f, g) * s(dif)) when
    f and g are linear functions of s.
    This allows accurate integration through near poles of f / g

    Based on newlip routine by Ed Williams.

    ``ratcen`` is exactly linear in ``f``: it is the weighted sum ``P * f_av + Q * f_dif`` of the two-point
    stencils ``f_av`` and ``f_dif``, with coefficients ``P`` and ``Q`` that depend only on ``g``. Written that
    way, ``f`` stays out of every branch, so reverse mode differentiates the linear map alone and its VJP is
    the adjoint of the same stencil -- no complex ``log`` and no branch selection anywhere on the ``f`` path.
    The whole ``g`` dependence is isolated in :func:`_ratcen_coeffs`, which carries its own ``custom_jvp``.

    Args:
        f (jnp.ndarray): 2D complex array (shape: [batch, N]) representing the numerator values of the rational function.
        g (jnp.ndarray): 1D complex array (shape: [N]) representing the denominator values of the rational function.
    Returns:
        jnp.ndarray: 2D real array (shape: [batch, N-2]) containing the rationally centered values for integration.

    """

    fdif = f[:, 1:-1] - f[:, 0:-2]
    fav = 0.5 * (f[:, 1:-1] + f[:, 0:-2])

    p_coeff, q_coeff = _ratcen_coeffs(g)

    return jnp.real(p_coeff[None, :] * fav + q_coeff[None, :] * fdif)


def _ratcen_intervals(g: jnp.ndarray):
    """
    Per-interval stencils of ``g``, the near-pole branch mask, and the branch logarithm.

    ``gav`` and ``gdif`` are guarded before use: near a pole ``gav -> 0``, so the (unselected) Taylor branch
    would evaluate to inf, and where the Taylor branch *is* selected ``gdif`` may be zero, so the (unselected)
    exact branch would too. ``jnp.where`` discards those values either way, but they are cheaper never to
    create. Unlike the double-``where`` this replaced, the guard is now purely defensive -- the coefficients
    carry a hand-written tangent rule, so no derivative depends on it.

    Args:
        g (jnp.ndarray): 1D complex array (shape: [N]) representing the denominator values.
    Returns:
        tuple: ``(gav, gdif, gav_safe, gdif_safe, g_lo, g_hi, use_rf, log_term)``, each of shape ``[N-2]``.
        ``g_lo`` and ``g_hi`` are the guarded interval endpoints ``gav -/+ gdif / 2``, i.e. ``g[0:-2]`` and
        ``g[1:-1]`` wherever the exact branch is selected. ``use_rf`` selects the Taylor branch, valid where
        ``gdif`` is negligible against ``gav`` -- i.e. away from a pole.
    """

    gdif = g[1:-1] - g[0:-2]
    gav = 0.5 * (g[1:-1] + g[0:-2])

    use_rf = jnp.abs(gdif) < 1.0e-4 * jnp.abs(gav)
    gav_safe = jnp.where(use_rf, gav, 1.0)
    gdif_safe = jnp.where(use_rf, 1.0, gdif)

    g_lo = gav - 0.5 * gdif_safe
    g_hi = gav + (0.5 + 0j) * gdif_safe

    # g_hi is promoted to complex so the log picks up the i * pi when the pole sits inside the interval and
    # the ratio goes negative; only the real part survives in ratcen.
    log_term = jnp.log(g_hi / g_lo)

    return gav, gdif, gav_safe, gdif_safe, g_lo, g_hi, use_rf, log_term


def _ratcen_coeffs_impl(g: jnp.ndarray):
    """
    Shared implementation behind :func:`_ratcen_coeffs` and its tangent rule.

    Args:
        g (jnp.ndarray): 1D complex array (shape: [N]) representing the denominator values.
    Returns:
        tuple: ``((P, Q), intervals)``, where ``intervals`` is the tuple from :func:`_ratcen_intervals`.
    """

    intervals = _ratcen_intervals(g)
    gav, gdif, gav_safe, gdif_safe, _, _, use_rf, log_term = intervals

    p_rf = 1.0 / gav_safe + gdif**2 / (12.0 * gav_safe**3)
    q_rf = -gdif / (12.0 * gav_safe**2)

    p_rfn = log_term / gdif_safe
    q_rfn = 1.0 / gdif_safe - gav * log_term / gdif_safe**2

    return (jnp.where(use_rf, p_rf, p_rfn), jnp.where(use_rf, q_rf, q_rfn)), intervals


@custom_jvp
def _ratcen_coeffs(g: jnp.ndarray):
    """
    Coefficients ``(P, Q)`` of the linear map ``ratcen(f, g) = P * f_av + Q * f_dif``.

    Away from a pole (``use_rf``) the interval integral is Taylor expanded in ``gdif / gav``; near one it is
    evaluated exactly through the complex logarithm ``L``. The two branches are

    ``P = 1 / gav + gdif**2 / (12 * gav**3)``,  ``Q = -gdif / (12 * gav**2)``        (Taylor)
    ``P = L / gdif``,                           ``Q = 1 / gdif - gav * L / gdif**2``  (exact)

    The custom rule in :func:`_ratcen_coeffs_jvp` differentiates only the branch ``use_rf`` actually selects,
    so the unselected branch can no longer contribute a ``0 * inf`` -- the failure mode that made the
    double-``where`` guard load-bearing on GPU, where FMA rounding can land ``gav`` at exactly zero.

    Args:
        g (jnp.ndarray): 1D complex array (shape: [N]) representing the denominator values.
    Returns:
        tuple: ``(P, Q)``, each a 1D complex array of shape ``[N-2]``.
    """

    return _ratcen_coeffs_impl(g)[0]


@_ratcen_coeffs.defjvp
def _ratcen_coeffs_jvp(primals, tangents):
    """
    Tangent rule for :func:`_ratcen_coeffs`: the analytic derivative of the selected branch only.

    The rule is stated in forward mode so that both directions are covered by one derivation. It is linear in
    the tangent and built from constants that depend only on ``g``, so JAX transposes it to obtain the VJP --
    which is then exactly the adjoint of this stencil, evaluated against coefficients the primal already
    computed. Writing it as a ``custom_vjp`` instead would cover reverse mode only and make ``jacfwd``
    (used by the Levenberg-Marquardt least-squares path) raise outright.

    The partials are taken against the interval *endpoints* ``lo = g[0:-2]`` and ``hi = g[1:-1]`` rather than
    against ``gav`` and ``gdif``. That choice is load-bearing for accuracy, not cosmetic: when the pole sits
    close to a grid point one endpoint approaches zero and ``dP/dlo``, ``dQ/dlo`` genuinely diverge like
    ``1 / lo``, while the ``hi`` partials stay O(1). Expressed in ``(gav, gdif)`` both tangents carry that
    same divergent term and the stencil has to cancel it to recover the small ``hi`` partial, which costs
    most of the significant digits exactly where the integrand needs them.

    With ``a = gav``, ``d = gdif`` and ``L = log(hi / lo)`` the branch logarithm, the exact branch gives

    ``dP/dhi = 1 / (hi * d) - L / d**2``,  ``dP/dlo = -1 / (lo * d) + L / d**2``
    ``dQ/dhi = -1 / d**2 - L / (2 * d**2) - a / (hi * d**2) + 2 * a * L / d**3``
    ``dQ/dlo = 1 / d**2 - L / (2 * d**2) + a / (lo * d**2) - 2 * a * L / d**3``

    while the Taylor branch is naturally written in ``(a, d)`` -- where no denominator is near zero, so the
    conditioning argument above does not apply -- and rotated onto the endpoints via
    ``d/dhi = d/da / 2 + d/dd`` and ``d/dlo = d/da / 2 - d/dd``:

    ``dP/da = -1 / a**2 - d**2 / (4 * a**4)``,  ``dP/dd = d / (6 * a**3)``
    ``dQ/da = d / (6 * a**3)``,                 ``dQ/dd = -1 / (12 * a**2)``

    (``dQ/da`` coincides with ``dP/dd``, the coefficients being second derivatives of a common potential.)

    Args:
        primals (tuple): A 1-tuple holding ``g``.
        tangents (tuple): A 1-tuple holding the tangent of ``g``.
    Returns:
        tuple: ``((P, Q), (P_dot, Q_dot))``.
    """

    (g,) = primals
    (g_dot,) = tangents

    coeffs, intervals = _ratcen_coeffs_impl(g)
    gav, gdif, gav_safe, gdif_safe, g_lo, g_hi, use_rf, log_term = intervals

    # Taylor branch, valid where gdif is negligible against gav. Derived in (gav, gdif), then rotated.
    dp_dgav_rf = -1.0 / gav_safe**2 - gdif**2 / (4.0 * gav_safe**4)
    dp_dgdif_rf = gdif / (6.0 * gav_safe**3)
    dq_dgav_rf = dp_dgdif_rf
    dq_dgdif_rf = -1.0 / (12.0 * gav_safe**2)

    dp_dhi_rf = 0.5 * dp_dgav_rf + dp_dgdif_rf
    dp_dlo_rf = 0.5 * dp_dgav_rf - dp_dgdif_rf
    dq_dhi_rf = 0.5 * dq_dgav_rf + dq_dgdif_rf
    dq_dlo_rf = 0.5 * dq_dgav_rf - dq_dgdif_rf

    # Exact branch, valid through a near-pole. Derived directly against the endpoints.
    inv_d2 = 1.0 / gdif_safe**2
    log_over_d2 = log_term * inv_d2
    two_a_log_over_d3 = 2.0 * gav * log_term * inv_d2 / gdif_safe

    dp_dhi_rfn = 1.0 / (g_hi * gdif_safe) - log_over_d2
    dp_dlo_rfn = -1.0 / (g_lo * gdif_safe) + log_over_d2
    dq_dhi_rfn = -inv_d2 - 0.5 * log_over_d2 - gav * inv_d2 / g_hi + two_a_log_over_d3
    dq_dlo_rfn = inv_d2 - 0.5 * log_over_d2 + gav * inv_d2 / g_lo - two_a_log_over_d3

    dp_dhi = jnp.where(use_rf, dp_dhi_rf, dp_dhi_rfn)
    dp_dlo = jnp.where(use_rf, dp_dlo_rf, dp_dlo_rfn)
    dq_dhi = jnp.where(use_rf, dq_dhi_rf, dq_dhi_rfn)
    dq_dlo = jnp.where(use_rf, dq_dlo_rf, dq_dlo_rfn)

    hi_dot, lo_dot = g_dot[1:-1], g_dot[0:-2]
    tangents_out = (dp_dhi * hi_dot + dp_dlo * lo_dot, dq_dhi * hi_dot + dq_dlo * lo_dot)

    return coeffs, tangents_out
