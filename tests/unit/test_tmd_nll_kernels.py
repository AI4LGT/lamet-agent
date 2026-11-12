"""Independent reference expressions and infrared-policy checks for TMD NLL."""

import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad

from lamet_agent.kernels.tmd import cg_nll, cg_nlo, coefficients


def reference_alpha(q, mu=2.0, alpha=.293):
    # LaMETLat constants.py order=2, expressed independently of production code.
    a = alpha/(4*np.pi)
    b0 = 11 - 2*3/3
    b1 = 102 - 38*3/3
    b2 = 2857/2 - 5033*3/18 + 325*9/54
    t = 1 + a*b0*np.log((q/mu)**2)
    return alpha/(t + a*b1/b0*np.log(t) + a*a*(
        b2/b0*(1 - 1/t) + (b1/b0)**2*(np.log(t)/t + 1/t - 1)))


def reference_single_leg(x, p, mu=2.0):
    # Original tmd_kernel_RGR, with tree matching and its published anomalous dimensions.
    cf, ca, tf, nf = 4/3, 3, .5, 3
    c1, c2 = 2*cf, 2*cf*((67/9 - np.pi*np.pi/3)*ca - 20/9*tf*nf)
    b0, b1, gc = 9, 64, -6*cf
    a0, amu = reference_alpha(2*x*p), reference_alpha(mu)
    r = amu/a0
    kc = -c1/(4*b0*b0)*(4*np.pi/a0*(1 - 1/r - np.log(r))
                       + (c2/c1 - b1/b0)*(1 - r + np.log(r))
                       + b1/(2*b0)*np.log(r)**2)
    kg = -gc/(2*b0)*np.log(r)
    return np.exp(-2*kc + kg)


def reference_sudakov(q, mu=2.0):
    def integrand(u):
        return -(4/3)*reference_alpha(u)/(4*np.pi)*(4*np.log(u*u/(q*q)) + 6)/u
    return quad(integrand, q, mu, epsrel=1e-3)


def test_nlo_dispatch_is_bitwise_unchanged():
    x = np.linspace(.1, .9, 13)
    for obs in ("tmdwf", "tmdpdf"):
        np.testing.assert_array_equal(
            coefficients.hard_factor(x, observable=obs, momentum_gev=4),
            cg_nlo.hard_factor(x, observable=obs, momentum_gev=4),
        )
    np.testing.assert_array_equal(
        coefficients.form_factor_kernel(x[:, None], x, momentum_gev=2.58),
        cg_nlo.form_factor_kernel(x[:, None], x, momentum_gev=2.58),
    )
    assert coefficients.coefficient_metadata() == {
        "order": "nlo", "resummation": "", "kernel_id": "tmd_cg_nlo",
    }


def test_reference_running_and_hard_coefficients_at_safe_scales():
    scales = np.array([.5, .7, 1., 2., 3., 5., 10.])
    np.testing.assert_allclose(cg_nll.running_coupling(scales), reference_alpha(scales), rtol=2e-14)
    x = np.array([.1, .2, .34, .5, .66, .8, .9])
    single = reference_single_leg(x, 4.3)
    np.testing.assert_allclose(cg_nll.tmd_coefficient(x, momentum_gev=4.3), single, rtol=2e-14)
    np.testing.assert_allclose(
        coefficients.hard_factor(x, observable="tmdwf", momentum_gev=4.3, order="nll"),
        single*reference_single_leg(1-x, 4.3), rtol=2e-14,
    )
    np.testing.assert_allclose(cg_nll.hard_factor(x, observable="tmdpdf", momentum_gev=4.3), single**2)
    assert cg_nll.coupling_pole_gev() == pytest.approx(.391910396244829, abs=2e-13)
    assert cg_nll.running_coupling(2.) == pytest.approx(.293, rel=1e-15)


def test_sudakov_and_double_kernel_agree_with_reference_above_pole():
    scales = np.array([.6, 1., 1.7, 2., 3., 5.])
    expected = [np.exp(reference_sudakov(q)[0]) for q in scales]
    np.testing.assert_allclose(cg_nll.sudakov_factor(scales), expected, rtol=3e-10)
    x = np.array([.2, .4, .6, .8])
    result = cg_nll.form_factor_kernel(x[:, None], x, momentum_gev=2.58)
    expected = np.empty_like(result)
    for i, x1 in enumerate(x):
        for j, x2 in enumerate(x):
            q, qb = 2*2.58*np.sqrt(x1*x2), 2*2.58*np.sqrt((1-x1)*(1-x2))
            expected[i, j] = np.exp(reference_sudakov(q)[0] + reference_sudakov(qb)[0])
    np.testing.assert_allclose(result, expected, rtol=3e-10)
    np.testing.assert_allclose(result, result.T, rtol=2e-14)
    np.testing.assert_allclose(result, result[::-1, ::-1], rtol=2e-14)


def test_boundary_zero_coupling_and_weak_coupling_limit():
    assert cg_nll.tmd_coefficient(.25, momentum_gev=4) == 1
    assert cg_nll.sudakov_factor(2.) == 1
    np.testing.assert_array_equal(cg_nll.hard_factor(
        [.02, .3, .8], observable="tmdwf", momentum_gev=3, alpha_s=0), np.ones(3))
    np.testing.assert_array_equal(cg_nll.sudakov_factor([.02, 2, 5], alpha_s=0), np.ones(3))
    alpha = 1e-6
    x, p, mu = .3, 4., 2.
    logarithm = np.log(mu**2/(2*x*p)**2)
    expected = (4/3)/(4*np.pi)*(-.5*logarithm**2 - 3*logarithm)
    actual = np.log(cg_nll.tmd_coefficient(x, momentum_gev=p, alpha_s=alpha))/alpha
    assert actual == pytest.approx(expected, rel=2e-5)


def test_default_policy_rejects_pole_and_freeze_is_explicit_scale_floor():
    with pytest.raises(ValueError, match="pole"):
        cg_nll.form_factor_kernel(.05, .05, momentum_gev=2.58)
    with pytest.raises(ValueError, match="pole"):
        cg_nll.tmd_coefficient(.05, momentum_gev=2.58, infrared_policy="reference_quad")
    frozen = dict(infrared_policy="freeze", freeze_scale_gev=1.)
    values = cg_nll.sudakov_factor([.258, .6, 1., 2.], **frozen)
    np.testing.assert_array_equal(values[:3], np.repeat(cg_nll.sudakov_factor(1.), 3))
    hard = cg_nll.tmd_coefficient([.05, .1, .25], momentum_gev=2, **frozen)
    np.testing.assert_array_equal(hard, np.repeat(cg_nll.tmd_coefficient(.25, momentum_gev=2), 3))
    assert np.isfinite(cg_nll.form_factor_kernel(.05, .05, momentum_gev=2.58, **frozen))
    diagnostics = cg_nll.scale_diagnostics([.258, .6, 1., 2.], **frozen)
    assert diagnostics["minimum_scale_gev"] == .258
    assert diagnostics["minimum_effective_scale_gev"] == 1.
    assert diagnostics["floored_node_count"] == 2
    assert diagnostics["below_pole_node_count"] == 1


@pytest.mark.parametrize("kwargs", [
    {"infrared_policy": "clip"}, {"infrared_policy": "freeze"},
    {"infrared_policy": "freeze", "freeze_scale_gev": .3},
    {"infrared_policy": "freeze", "freeze_scale_gev": 3.},
    {"infrared_policy": "freeze", "freeze_scale_gev": True},
    {"infrared_policy": "error", "freeze_scale_gev": 1.},
])
def test_invalid_infrared_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        coefficients.coefficient_metadata(order="nll", **kwargs)


def test_reference_quad_pole_crossing_is_visible_and_cached():
    cg_nll.reset_reference_quad_diagnostics()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        reference_value, reference_error = reference_sudakov(.258)
    assert caught
    value = cg_nll.sudakov_factor(.258, infrared_policy="reference_quad")
    # A pole-crossing numerical return is deliberately unstable under roundoff;
    # its warning and pole flags, rather than high precision, are the requirement.
    assert np.isfinite(value) and value > 0
    assert np.isfinite(reference_value) and reference_error > .001
    diagnostics = cg_nll.reference_quad_diagnostics()
    assert diagnostics["evaluation_count"] == diagnostics["pole_crossing_count"] == 1
    assert diagnostics["warning_count"] == 1
    assert diagnostics["controlled_integral"] is False
    row = diagnostics["evaluations"][0]
    assert row["quad_estimated_absolute_error"] > .001
    assert row["q_gev"] == .258
    assert cg_nll.sudakov_factor(.258, infrared_policy="reference_quad") == value
    assert cg_nll.reference_quad_diagnostics()["evaluation_count"] == 1
    with pytest.raises(ValueError, match="logarithm branch"):
        cg_nll.sudakov_factor(.1, infrared_policy="reference_quad")


def test_frozen_double_integral_converges_and_is_positive():
    # Segment at the diagonal's floor crossing and use smooth symmetric shape.
    def integral(order):
        x0, w0 = np.polynomial.legendre.leggauss(order)
        x, w = .45*x0 + .5, .45*w0
        kernel = cg_nll.form_factor_kernel(x[:, None], x, momentum_gev=2.58,
                                          infrared_policy="freeze", freeze_scale_gev=1.)
        shape = x*(1-x)
        return (w*shape) @ kernel @ (w*shape)
    low, high = integral(32), integral(64)
    assert high > 0
    assert abs(high-low)/high < 2e-4
