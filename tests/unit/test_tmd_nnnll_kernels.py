"""N3LL H_F: published fixed-order coefficients and independent RG checks."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from lamet_agent.kernels.tmd import cg_nlo, cg_nnnll as kernel, coefficients


def _reference_alpha(q, mu, alpha):
    # Independent implicit four-loop beta solution in inverse a; production
    # integrates an ODE in log(mu). beta3 from hep-ph/9701390 (11), nf=3.
    z0 = 4*np.pi/alpha

    def inverse_beta(z):
        return z**3/(2*(9*z**3 + 64*z**2 + (3863/6)*z + 12090.37813080371))

    def residual(z):
        return quad(inverse_beta, z0, z, epsabs=1e-12, epsrel=1e-12)[0] - np.log(q/mu)

    return 4*np.pi/brentq(residual, .001, 10000, xtol=1e-12)


@pytest.mark.parametrize("q", [.8, 1.3, 2., 3.2, 6.])
def test_against_independent_implicit_running(q):
    # Implicit inverse-beta quadrature is independent of the production ODE.
    expected = _reference_alpha(q, 2., .293)
    assert kernel.running_coupling(q) == pytest.approx(expected, rel=2e-11)



def _fixed_two_loop(q, mu, alpha):
    # hep-ph/0607228 (50),(51), including the finite two-loop constant.
    logarithm = np.log(np.asarray(q)**2/mu**2)
    cf, ca, tf, nf, p, z = 4/3, 3, .5, 3, np.pi, 1.2020569031595943
    c2 = (
        cf**2*(logarithm**4/2 - 3*logarithm**3 + (25/2-p*p/6)*logarithm*logarithm + (-45/2-3*p*p/2+24*z)*logarithm
               + 255/8 + 7*p*p/2 - 83*p**4/360 - 30*z)
        + cf*ca*(11*logarithm**3/9 + (-233/18+p*p/3)*logarithm*logarithm + (2545/54+11*p*p/9-26*z)*logarithm
                 - 51157/648 - 337*p*p/108 + 11*p**4/45 + 313*z/9)
        + cf*tf*nf*(-4*logarithm**3/9 + 38*logarithm*logarithm/9 + (-418/27-4*p*p/9)*logarithm
                    + 4085/162 + 23*p*p/27 + 4*z/9)
    )
    return 1 + cg_nlo.sudakov_correction(q, mu=mu, alpha_s=alpha) + (alpha/(4*p))**2*c2


def test_complete_two_loop_expansion_and_product_cross_term():
    errors = []
    for alpha in (.008, .004, .002):
        q = np.array([.8, 1.3, 2., 3.2, 6.])
        errors.append(np.max(np.abs(kernel.sudakov_factor(q, alpha_s=alpha) - _fixed_two_loop(q, 2., alpha))))
    np.testing.assert_allclose(np.array(errors[1:])/errors[:-1], .125, rtol=.04)
    # At natural scales each factor is exactly its polynomial boundary.
    alpha = .293
    expected = _fixed_two_loop(2., 2., alpha)
    assert kernel.sudakov_factor(2.) == pytest.approx(expected, abs=1e-15)
    a = alpha/(4*np.pi)
    hf = 1+2*kernel.BOUNDARY1*a+(kernel.BOUNDARY1**2+2*kernel.BOUNDARY2)*a*a
    assert coefficients.form_factor_kernel(.5, .5, momentum_gev=2., order="nnnll") == pytest.approx(hf)
    assert abs(hf-expected**2) > 1e-3
    errors = []
    for alpha in (.008, .004, .002):
        q, qb = 2*3.*np.sqrt(.3*.4), 2*3.*np.sqrt(.7*.6)
        exact = _fixed_two_loop(q, 2., alpha)*_fixed_two_loop(qb, 2., alpha)
        actual = kernel.form_factor_kernel(.3, .4, momentum_gev=3., alpha_s=alpha)
        errors.append(abs(actual-exact))
    np.testing.assert_allclose(np.array(errors[1:])/errors[:-1], .125, rtol=.08)


def test_rge_residual_is_higher_order():
    # A consistently expanded coefficient obeys the RGE to its perturbative
    # accuracy, rather than the exact RGE of the unexpanded truncated solution.
    q, mu, h = 3.2, 2., 2e-4
    errors = []
    for alpha in (.02, .01, .005):
        values = []
        for sign in (-1, 1):
            shifted = mu*np.exp(sign*h)
            values.append(kernel.sudakov_factor(q, mu=shifted, alpha_s=_reference_alpha(shifted, mu, alpha)))
        derivative = np.log(values[1]/values[0])/(2*h)
        a = alpha/(4*np.pi)
        cusp = a*(kernel.GAMMA0+a*(kernel.GAMMA1+a*(kernel.GAMMA2+a*kernel.GAMMA3)))
        gamma = a*(kernel.NONCUSP0+a*(kernel.NONCUSP1+a*kernel.NONCUSP2))
        errors.append(abs(derivative-(cusp*np.log(q*q/(mu*mu))+gamma)))
    assert errors[1] < .2*errors[0]
    assert errors[2] < .2*errors[1]


def test_broadcast_symmetries_zero_coupling_and_scope():
    x = np.array([.25, .4, .7])[:, None]
    y = np.array([.3, .6])[None, :]
    args = dict(momentum_gev=3., order="nnnll")
    actual = coefficients.form_factor_kernel(x, y, **args)
    assert actual.shape == (3, 2)
    np.testing.assert_allclose(actual, coefficients.form_factor_kernel(y, x, **args))
    np.testing.assert_allclose(actual, coefficients.form_factor_kernel(1-x, 1-y, **args))
    np.testing.assert_array_equal(coefficients.form_factor_kernel(x, y, alpha_s=0., **args), np.ones((3, 2)))
    assert kernel.sudakov_factor(np.empty((0, 2))).shape == (0, 2)
    metadata = coefficients.coefficient_metadata(order="nnnll")
    assert (metadata["cusp_loops"], metadata["noncusp_loops"], metadata["running_loops"]) == (4, 3, 4)
    assert metadata["sudakov_boundary"] == "two_loop"
    assert metadata["running_scheme"] == "msbar_four_loop_numeric"
    assert metadata["coupling_pole_gev"] == pytest.approx(.5861456678664407)
    for observable in ("tmdwf", "tmdpdf"):
        with pytest.raises(ValueError, match="only for form_factor_kernel"):
            coefficients.hard_factor(.4, observable=observable, **args)


def test_pole_and_explicit_scale_floor():
    pole = kernel.coupling_pole_gev()
    for q in (.1, pole, .99*pole):
        with pytest.raises(ValueError, match="pole"):
            kernel.sudakov_factor(q)
    frozen = dict(infrared_policy="freeze", freeze_scale_gev=1.)
    np.testing.assert_array_equal(kernel.sudakov_factor([.1, .5, 1.], **frozen),
                                  np.repeat(kernel.sudakov_factor(1.), 3))
    assert np.isfinite(kernel.form_factor_kernel(.001, .001, momentum_gev=2., **frozen))
    assert kernel.coupling_pole_gev(alpha_s=0) == 0
    assert kernel.sudakov_factor(.01, alpha_s=0) == 1


@pytest.mark.parametrize("kwargs", [
    {"infrared_policy": "reference_quad"}, {"infrared_policy": "freeze"},
    {"infrared_policy": "freeze", "freeze_scale_gev": .55},
    {"infrared_policy": "freeze", "freeze_scale_gev": 3.},
    {"infrared_policy": "freeze", "freeze_scale_gev": True},
    {"freeze_scale_gev": 1.}, {"mu": 0}, {"alpha_s": -1},
])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        coefficients.form_factor_kernel(.4, .6, momentum_gev=3., order="nnnll", **kwargs)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_scale(value):
    with pytest.raises(ValueError, match="finite and positive"):
        kernel.sudakov_factor(value)


@pytest.mark.parametrize("value", [0, 1, -1, np.nan, np.inf])
def test_invalid_fraction(value):
    with pytest.raises(ValueError, match="0 < x1,x2 < 1"):
        kernel.form_factor_kernel(value, .5, momentum_gev=3.)


def test_historical_dis_figures_use_production_hard_evolution():
    import runpy
    from pathlib import Path
    benchmark = runpy.run_path(str(Path(__file__).resolve().parents[2]/"examples"/"validate_dis_nnnll.py"))
    errors = benchmark["benchmark_errors"]()
    assert len(errors) == 24
    assert max(abs(r[-1]) for r in errors) < .005
