"""Spacelike form-factor references and fixed-order expansion of NNLL H_F."""

import numpy as np
import pytest
from scipy.special import zeta

from lamet_agent.kernels.tmd import cg_nll, cg_nlo, cg_nnll, coefficients


@pytest.mark.parametrize("q,mu,alpha,expected", [
    (0.8, 2.0, 0.293, 0.394704164528020089995163),
    (1.3, 2.0, 0.293, 0.653329445987503825824858),
    (2.0, 2.0, 0.293, 0.802432022678498745242355),
    (3.2, 2.0, 0.293, 0.885315243105545994978631),
    (6.0, 2.0, 0.293, 0.886067494735613039100676),
    (0.8, 3.0, 0.22, 0.413451787873276371034629),
    (1.3, 3.0, 0.22, 0.617074850896766082884120),
    (2.0, 3.0, 0.22, 0.758377313330848592533420),
    (3.2, 3.0, 0.22, 0.862794631526357867872646),
    (6.0, 3.0, 0.22, 0.919455111951412421339067),
])
def test_nnll_matches_independent_high_precision_integral(q, mu, alpha, expected):
    # mpmath, 70 decimal digits, integration in u (production uses log u).
    # Transcribe gamma_j and gamma_K directly from arXiv:1705.07167 (58),(59),
    # integrate [-gamma_j + gamma_K/2 * log(Q^2/u^2)]/u from Q to mu,
    # multiply by 1 + CF*a(Q)*(-8+pi^2/6). Use 1002.2213 (155) for a(u),
    # with nf=3, beta=(9,64,3863/6), a=alpha_s/(4*pi).
    actual = cg_nnll.sudakov_factor(q, mu=mu, alpha_s=alpha)
    assert actual == pytest.approx(expected, rel=3e-13, abs=1e-14)


def test_one_loop_boundary_and_zero_coupling():
    assert cg_nnll.sudakov_factor(2.0) == pytest.approx(0.8024320226784987, abs=2e-15)
    assert coefficients.form_factor_kernel(.5, .5, momentum_gev=2, order="nnll") == pytest.approx(
        0.8024320226784987**2, abs=2e-15,
    )
    np.testing.assert_array_equal(cg_nnll.sudakov_factor([.01, 2, 5], alpha_s=0), np.ones(3))
    np.testing.assert_array_equal(
        cg_nnll.form_factor_kernel([.01, .5, .99], .5, momentum_gev=2, alpha_s=0), np.ones(3),
    )


def test_weak_coupling_recovers_complete_nlo_kernel():
    x, y = np.array([.2, .4, .8]), np.array([.3, .7])
    errors = []
    for alpha in (1e-3, 5e-4, 2.5e-4):
        kwargs = dict(momentum_gev=2.58, alpha_s=alpha)
        fixed = cg_nlo.form_factor_kernel(x[:, None], y, **kwargs)
        resummed = cg_nnll.form_factor_kernel(x[:, None], y, **kwargs)
        errors.append(np.max(np.abs(resummed - fixed)))
    # The remainder is O(alpha_s^2), including the product of the two legs.
    np.testing.assert_allclose(np.array(errors[1:]) / errors[:-1], .25, rtol=.01)
    assert errors[-1] < 1e-7


def _two_loop_log_terms(q, mu):
    # Direct spacelike fixed-order expression, arXiv:1705.07167 (60).
    # Subtract only its two-loop constant: NNLL has a one-loop boundary.
    t = np.log(np.asarray(q)**2 / mu**2)
    cf, ca, nf, pi, z3 = 4/3, 3, 3, np.pi, zeta(3, 1)
    return (
        cf**2 * (t*(-45/2 - 3*pi**2/2 + 24*z3) + t**2*(25/2 - pi**2/6) - 3*t**3 + t**4/2)
        + cf*ca * (t*(2545/54 + 11*pi**2/9 - 26*z3) + t**2*(-233/18 + pi**2/3) + 11*t**3/9)
        + cf*nf * (t*(-209/27 - 2*pi**2/9) + 19*t**2/9 - 2*t**3/9)
    )


def test_two_loop_logarithms_match_published_fixed_order_form_factor():
    q, mu = np.array([.8, 1.3, 3.2, 6.0]), 2.0
    expected = _two_loop_log_terms(q, mu)
    remainders = []
    for alpha in (2e-3, 1e-3, 5e-4):
        a = alpha / (4*np.pi)
        first = cg_nlo.sudakov_correction(q, mu=mu, alpha_s=alpha)
        result = cg_nnll.sudakov_factor(q, mu=mu, alpha_s=alpha)
        extracted = ((result - 1) - first) / a**2
        remainders.append(extracted - expected)
    # After extracting a_s^2, the omitted a_s^3 terms decrease linearly.
    np.testing.assert_allclose(np.array(remainders[1:]) / remainders[:-1], .5, rtol=.03)
    # Extrapolate the a_s^3 contamination away, independently of RGE coefficients.
    np.testing.assert_allclose(2*remainders[-1] - remainders[-2], 0, atol=2e-4)


def test_nnll_scale_floor_and_pole_checks_follow_nll():
    with pytest.raises(ValueError, match="pole"):
        cg_nnll.form_factor_kernel(.05, .05, momentum_gev=2.58)
    frozen = dict(infrared_policy="freeze", freeze_scale_gev=1.)
    values = cg_nnll.sudakov_factor([.1, .5, 1., 2.], **frozen)
    np.testing.assert_array_equal(values[:3], np.repeat(cg_nnll.sudakov_factor(1.), 3))
    assert np.isfinite(cg_nnll.form_factor_kernel(.05, .05, momentum_gev=2.58, **frozen))
    for invalid in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            cg_nnll.sudakov_factor(invalid)


@pytest.mark.parametrize("kwargs", [
    {"infrared_policy": "reference_quad"}, {"infrared_policy": "freeze"},
    {"infrared_policy": "freeze", "freeze_scale_gev": .3},
    {"infrared_policy": "freeze", "freeze_scale_gev": 3.},
    {"infrared_policy": "freeze", "freeze_scale_gev": True},
    {"freeze_scale_gev": 1.}, {"mu": 0}, {"alpha_s": -1},
])
def test_invalid_nnll_controls(kwargs):
    with pytest.raises(ValueError):
        coefficients.form_factor_kernel(.4, .6, momentum_gev=2.58, order="nnll", **kwargs)


def test_nnll_scope_is_explicit_and_existing_dispatch_is_unchanged():
    metadata = coefficients.coefficient_metadata(order="nnll")
    assert metadata["resummation"] == "nnll"
    assert metadata["kernel_scope"] == "form_factor"
    assert metadata["sudakov_boundary"] == "one_loop"
    assert metadata["running_flavors"] == 3
    assert (metadata["cusp_loops"], metadata["noncusp_loops"], metadata["running_loops"]) == (3, 2, 3)
    assert "hard_boundary" not in metadata
    for observable in ("tmdwf", "tmdpdf"):
        with pytest.raises(ValueError, match="only for form_factor_kernel"):
            coefficients.hard_factor(.4, observable=observable, momentum_gev=3, order="nnll")
    for order, module in (("nlo", cg_nlo), ("nll", cg_nll), ("nnll", cg_nnll)):
        kwargs = dict(momentum_gev=2.58)
        np.testing.assert_array_equal(
            coefficients.form_factor_kernel([.2, .4, .8], .5, order=order, **kwargs),
            module.form_factor_kernel([.2, .4, .8], .5, **kwargs),
        )
