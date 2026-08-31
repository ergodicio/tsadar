"""Complete, node-safe quadrature of ``f / g`` over a one-dimensional grid.

The susceptibility call sites use the affine denominator ``g = z - xi``. For that case
the Cauchy principal value is evaluated by singularity subtraction against a C2 spline of
``f``. General, nonsingular denominators retain the original piecewise-rational rule.
"""

from interpax import approx_df
from jax import custom_jvp, lax, numpy as jnp


def ratintn(f: jnp.ndarray, g: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Integrate ``f / g`` over ``z``, including all ``N - 1`` intervals.

    When ``g`` is affine in ``z`` and its zero lies inside the integration domain,
    singularity subtraction turns the Cauchy principal value into an ordinary integral
    plus the analytic endpoint logarithm. ``f`` at the pole comes from a C2 cubic spline.
    On the interval containing the pole, removable divided differences are evaluated
    from the cubic coefficients rather than by subtracting nearly equal values. The
    result is finite and differentiable when the pole is exactly on a grid node, without
    moving the pole by an epsilon.

    Other denominators use the piecewise-rational rule implemented by :func:`ratcen`.

    Based on newlip routine by Ed Williams.

    Args:
        f: Numerator samples, shape ``[N]`` or ``[batch, N]``.
        g: Denominator samples, shape ``[N]``.
        z: Strictly increasing real integration grid, shape ``[N]``.

    Returns:
        The real integral for every row of ``f``, shape ``[batch]``.
    """

    if len(jnp.shape(f)) == 1:
        f = jnp.transpose(f[..., jnp.newaxis])

    # Complex grids are outside the principal-value call sites. Preserve the general
    # piecewise-rational implementation rather than asking searchsorted to order them.
    if jnp.iscomplexobj(g) or jnp.iscomplexobj(z):
        return _rational_integral(f, g, z)

    slope, pole, use_principal_value = _affine_interior_pole(g, z)
    return lax.cond(
        use_principal_value,
        lambda _: _principal_value_integral(f, slope, pole, z),
        lambda _: _rational_integral(f, g, z),
        operand=None,
    )


def _affine_interior_pole(g: jnp.ndarray, z: jnp.ndarray):
    """Return the affine scale, zero, and whether ``g = scale * (z - zero)``.

    The tolerance only recognizes roundoff in an affine sampled denominator; it never
    shifts or floors the pole. A zero slope and roots outside the open integration
    interval are handled by the ordinary rational quadrature.
    """

    slope = (g[..., -1] - g[..., 0]) / (z[-1] - z[0])
    nonzero_slope = slope != 0
    slope_safe = jnp.where(nonzero_slope, slope, 1.0)
    pole = z[0] - g[..., 0] / slope_safe

    affine_values = slope_safe[..., None] * (z - pole[..., None])
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(g), axis=-1))
    tolerance = 64.0 * jnp.finfo(z.dtype).eps * scale
    is_affine = jnp.max(jnp.abs(g - affine_values), axis=-1) <= tolerance
    is_interior = (pole > z[0]) & (pole < z[-1])

    return slope_safe, pole, nonzero_slope & is_affine & is_interior


def _spline_pole_data(f: jnp.ndarray, z: jnp.ndarray, pole: jnp.ndarray):
    """Evaluate a C2 spline and its stable endpoint divided differences.

    If ``P(t) = c0 + c1*t + c2*t**2 + c3*t**3`` is the Hermite cubic on the
    interval containing the pole, then both endpoint divided differences factor
    exactly. These factored expressions supply the derivative limit at a node and
    avoid a hidden ``0 / 0`` branch.
    """

    slopes = approx_df(z, f.T, method="cubic2", axis=0).T
    interval = jnp.clip(jnp.searchsorted(z, pole, side="right") - 1, 0, z.size - 2)
    width = z[interval + 1] - z[interval]
    t = (pole - z[interval]) / width

    f0, f1 = f[:, interval], f[:, interval + 1]
    slope0, slope1 = slopes[:, interval], slopes[:, interval + 1]
    c0 = f0
    c1 = width * slope0
    c2 = 3.0 * (f1 - f0) - width * (2.0 * slope0 + slope1)
    c3 = 2.0 * (f0 - f1) + width * (slope0 + slope1)

    value = c0 + c1 * t + c2 * t**2 + c3 * t**3
    divided_left = (c1 + c2 * t + c3 * t**2) / width
    divided_right = (c1 + c2 * (1.0 + t) + c3 * (1.0 + t + t**2)) / width

    return interval, value, divided_left, divided_right


def _principal_value_integral(f: jnp.ndarray, slope: jnp.ndarray, pole: jnp.ndarray, z: jnp.ndarray):
    """Singularity-subtracted principal value for one affine denominator."""

    interval, value_at_pole, divided_left, divided_right = _spline_pole_data(f, z, pole)

    indices = jnp.arange(z.size)
    is_left = indices == interval
    is_right = indices == interval + 1
    is_bracketing = is_left | is_right

    # The direct quotient is never formed at either bracketing endpoint. This is
    # important near a node as well as exactly on it: both endpoints use the factored
    # cubic divided difference, so no cancellation loses the pole displacement.
    denominator = jnp.where(is_bracketing, 1.0, z - pole)
    regularized = (f - value_at_pole[:, None]) / denominator
    regularized = jnp.where(is_left[None, :], divided_left[:, None], regularized)
    regularized = jnp.where(is_right[None, :], divided_right[:, None], regularized)

    dz = z[1:] - z[:-1]
    regular_integral = jnp.sum(0.5 * (regularized[:, :-1] + regularized[:, 1:]) * dz, axis=1)
    log_term = jnp.log(jnp.abs(z[-1] - pole)) - jnp.log(jnp.abs(z[0] - pole))

    return jnp.real((regular_integral + value_at_pole * log_term) / slope)


def _rational_integral(f: jnp.ndarray, g: jnp.ndarray, z: jnp.ndarray):
    """Original piecewise-rational rule, now over all adjacent grid pairs."""

    return jnp.sum(ratcen(f, g) * (z[1:] - z[:-1]), axis=1)


def ratintn_operator(g: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    Assemble the constant matrix M for which `M @ f == ratintn(f, g, z)` for any real 1D `f`.

    ``ratintn`` is exactly linear in ``f`` in both its singularity-subtracted and
    piecewise-rational forms. Whenever ``g`` and ``z`` are fixed across calls, the
    quadrature therefore collapses to a single matrix multiply.

    Args:
        g (jnp.ndarray): Denominator samples, shape [..., N]. Leading axes are carried through as
            batch axes, giving one row of M per batch element.
        z (jnp.ndarray): 1D array of the variable of integration, shape [N].

    Returns:
        jnp.ndarray: M, of shape `g.shape[:-1] + (N,)`.
    """

    if jnp.iscomplexobj(g) or jnp.iscomplexobj(z):
        return _rational_operator(g, z)

    slope, pole, use_principal_value = _affine_interior_pole(g, z)

    # Keep both sides finite before selecting batched rows. The rational formula is
    # undefined at an exact node, while the principal-value formula requires nonzero
    # affine scale; the replacements affect only the unselected side.
    rational_g = jnp.where(use_principal_value[..., None], jnp.ones_like(g), g)
    principal_slope = jnp.where(use_principal_value, slope, 1.0)
    principal_pole = jnp.where(use_principal_value, pole, 0.5 * (z[0] + z[-1]))

    rational = _rational_operator(rational_g, z)
    principal = _principal_value_operator(principal_slope, principal_pole, z)
    return jnp.where(use_principal_value[..., None], principal, rational)


def _rational_operator(g: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Matrix form of the complete piecewise-rational quadrature."""

    gdif = g[..., 1:] - g[..., :-1]
    gav = 0.5 * (g[..., 1:] + g[..., :-1])
    zdif = z[1:] - z[:-1]

    # same branch selection and guarded denominator as `ratcen`
    use_rf = jnp.abs(gdif) < 1.0e-4 * jnp.abs(gav)
    gav_safe = jnp.where(use_rf, gav, 1.0)
    gdif_safe = jnp.where(use_rf, 1.0, gdif)
    log_ratio = jnp.real(jnp.log((gav + (0.5 + 0j) * gdif_safe) / (gav - 0.5 * gdif_safe)))

    # `ratcen(f, g) == p * fav + q * fdif`, obtained by collecting the fav/fdif terms of rf and rfn
    p = jnp.where(use_rf, 1.0 / gav_safe + gdif**2 / (12.0 * gav_safe**3), log_ratio / gdif_safe)
    q = jnp.where(
        use_rf,
        -gdif / (12.0 * gav_safe**2),
        1.0 / gdif_safe - gav * log_ratio / gdif_safe**2,
    )

    # fav and fdif are two-point stencils, so interval j contributes to grid points j and j+1
    lower = (0.5 * p - q) * zdif
    upper = (0.5 * p + q) * zdif

    M = jnp.zeros(jnp.shape(g), dtype=lower.dtype)
    M = M.at[..., :-1].add(lower)
    M = M.at[..., 1:].add(upper)

    return M


def _principal_value_operator(slope: jnp.ndarray, pole: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Matrix form of the singularity-subtracted C2-spline quadrature."""

    size = z.size
    identity = jnp.eye(size, dtype=z.dtype)
    derivative = approx_df(z, identity, method="cubic2", axis=0)

    interval = jnp.clip(jnp.searchsorted(z, pole, side="right") - 1, 0, size - 2)
    width = z[interval + 1] - z[interval]
    t = (pole - z[interval]) / width

    endpoint0, endpoint1 = identity[interval], identity[interval + 1]
    slope0, slope1 = derivative[interval], derivative[interval + 1]
    width = width[..., None]
    t = t[..., None]

    c0 = endpoint0
    c1 = width * slope0
    c2 = 3.0 * (endpoint1 - endpoint0) - width * (2.0 * slope0 + slope1)
    c3 = 2.0 * (endpoint0 - endpoint1) + width * (slope0 + slope1)

    interpolation_weights = c0 + c1 * t + c2 * t**2 + c3 * t**3
    divided_left = (c1 + c2 * t + c3 * t**2) / width
    divided_right = (c1 + c2 * (1.0 + t) + c3 * (1.0 + t + t**2)) / width

    dz = z[1:] - z[:-1]
    trapezoid_weights = jnp.concatenate(
        (0.5 * dz[:1], 0.5 * (dz[:-1] + dz[1:]), 0.5 * dz[-1:])
    )
    indices = jnp.arange(size)
    is_bracketing = (indices == interval[..., None]) | (indices == interval[..., None] + 1)
    denominator = jnp.where(is_bracketing, 1.0, z - pole[..., None])
    quotient_weights = jnp.where(is_bracketing, 0.0, trapezoid_weights / denominator)

    log_term = jnp.log(jnp.abs(z[-1] - pole)) - jnp.log(jnp.abs(z[0] - pole))
    operator = quotient_weights
    operator += (log_term - jnp.sum(quotient_weights, axis=-1))[..., None] * interpolation_weights
    operator += trapezoid_weights[interval][..., None] * divided_left
    operator += trapezoid_weights[interval + 1][..., None] * divided_right

    return operator / slope[..., None]


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
        jnp.ndarray: 2D real array (shape: [batch, N-1]) containing the rationally centered values for integration.

    """

    fdif = f[:, 1:] - f[:, :-1]
    fav = 0.5 * (f[:, 1:] + f[:, :-1])

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
        tuple: ``(gav, gdif, gav_safe, gdif_safe, g_lo, g_hi, use_rf, log_term)``, each of shape ``[N-1]``.
        ``g_lo`` and ``g_hi`` are the guarded interval endpoints ``gav -/+ gdif / 2``, i.e. ``g[:-1]`` and
        ``g[1:]`` wherever the exact branch is selected. ``use_rf`` selects the Taylor branch, valid where
        ``gdif`` is negligible against ``gav`` -- i.e. away from a pole.
    """

    gdif = g[1:] - g[:-1]
    gav = 0.5 * (g[1:] + g[:-1])

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
        tuple: ``(P, Q)``, each a 1D complex array of shape ``[N-1]``.
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

    The partials are taken against the interval *endpoints* ``lo = g[:-1]`` and ``hi = g[1:]`` rather than
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

    hi_dot, lo_dot = g_dot[1:], g_dot[:-1]
    tangents_out = (dp_dhi * hi_dot + dp_dlo * lo_dot, dq_dhi * hi_dot + dq_dlo * lo_dot)

    return coeffs, tangents_out
