import mpmath as mp
import numpy as np
import pytest

from lamet_agent.kernels import matching_kernel_id, load_kernel
from lamet_agent.kernels import quark_pdf_gi_gt_hybrid_nlo_rgr_tr as tr
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nlo_rgr import kernel as rgr


def _star_reference(y, p, mu, eta):
    with mp.workdps(max(mp.mp.dps, 45)):
        e = mp.mpf(eta)
        dy = mp.mpf(str(y[1] - y[0]))
        lo, hi = mp.mpf(str(y[0])) - dy/2, mp.mpf(str(y[-1])) + dy/2
        f = mp.sin(mp.pi*e/2)*mp.gamma(1-e)*mp.exp(-mp.euler*e)/mp.pi
        matrix = mp.matrix(len(y))
        for j, yy in enumerate(y):
            yy = mp.mpf(str(yy))
            total = mp.mpf(0)
            for i, xx in enumerate(y):
                if i != j:
                    d = abs(mp.mpf(str(xx))-yy)
                    matrix[i,j] = dy*f*(2*p*d/mu)**e/d
                    total += matrix[i,j]
            mass = f/e*((2*p*(yy-lo)/mu)**e + (2*p*(hi-yy)/mu)**e) if e else 1
            matrix[j,j] = mass-total
        return matrix


@pytest.mark.parametrize("eta", [-.65, -.1, 0., 1e-8, .1, .65])
def test_star_against_high_precision_distribution(eta):
    y = np.linspace(-1.9, 1.9, 20)
    ref = np.array(_star_reference(y, 2, 1.2, eta).tolist(), dtype=float)
    np.testing.assert_allclose(tr._star_matrix(y, 2, 1.2, eta), ref, atol=2e-14, rtol=2e-14)


def test_soft_against_distribution_derivatives():
    y = np.array([-1.5, -.5, .5, 1.5])
    with mp.workdps(max(mp.mp.dps, 45)):
        ref = np.empty((4,4))
        for i in range(4):
            for j in range(4):
                def f(e):
                    return _star_reference(y, 2, 1.2, e)[i,j]
                ref[i,j] = f(0) - mp.mpf('.27')*mp.mpf(4)/3/(2*mp.pi)*(
                    (2+mp.pi**2/12)*f(0)-2*mp.diff(f,0)+2*mp.diff(f,0,2))
    np.testing.assert_allclose(tr._soft_nlo(y,2,1.2,.27),ref,atol=2e-13,rtol=2e-13)


@pytest.mark.parametrize("x", [[-.5,-.3,.3,.5],[-.2,.2]])
def test_tr_off_bitwise_rgr(x):
    y=np.linspace(-2,2,40)
    args=dict(momentum_gev=2.,scale_gev=2.,zs_fm=.18,evolution_points=10,evolution_steps=4)
    np.testing.assert_array_equal(tr.kernel(x,y,**args),rgr(x,y,**args))


def test_rectangular_output_is_projection_of_complete_matching():
    from lamet_agent.kernels.implementation import _lo_interp_matrix
    y=np.linspace(-2,2,40)
    radius=.7
    mu=4*radius
    constant,slope=tr._gi_gt_hybrid_nlo_tables(y,.18*2/tr.GEV_FM,1e-12)
    fixed=np.eye(len(y))-tr.alpha_s(2,mu,"running")/(2*np.pi)*(constant+2*np.log(mu/2)*slope)
    matched=fixed+tr._threshold_correction(y,2,mu,4*(1-radius),.18,.4)
    actual=tr.kernel([radius],y,momentum_gev=2,scale_gev=mu,zs_fm=.18,evolution_points=10)
    np.testing.assert_allclose(actual,_lo_interp_matrix(np.array([radius]),y)@matched,atol=3e-14)


def test_switch_continuity_and_nonzero_effect():
    y=np.linspace(-2,2,40)
    assert np.array_equal(tr._threshold_correction(y,2,2,2,.18,.4),np.zeros((40,40)))
    near=tr._threshold_correction(y,2,2+1e-7,2-1e-7,.18,.4)
    assert np.max(np.abs(near))<1e-6
    assert np.max(np.abs(tr._threshold_correction(y,2,2.8,1.2,.18,.4)))>1e-3


def test_signed_reflection():
    y=np.linspace(-2,2,40)
    m=tr.kernel([-.7,.7],y,momentum_gev=2,scale_gev=2,zs_fm=.18,evolution_points=10,evolution_steps=4)
    np.testing.assert_allclose(m[0],m[1,::-1],atol=2e-13)


@pytest.mark.parametrize("extra", [dict(threshold_cut=0),dict(threshold_cut=1),dict(threshold_cut=np.nan),
                                   dict(spectator_kappa=0),dict(spectator_kappa=True),dict(zs_fm=-1)])
def test_invalid_controls(extra):
    args=dict(momentum_gev=2,scale_gev=2,zs_fm=.18)
    args.update(extra)
    with pytest.raises(ValueError):
        tr.kernel([.7],np.linspace(-2,2,20),**args)


@pytest.mark.parametrize("x", [.9,1.,-1.])
def test_spectator_cutoff_rejects(x):
    with pytest.raises(ValueError,match="spectator scale"):
        tr.kernel([x],np.linspace(-2,2,20),momentum_gev=2,scale_gev=2,zs_fm=.18)


@pytest.mark.parametrize("eta", [-1,1,np.nan])
def test_star_domain(eta):
    with pytest.raises(ValueError,match="eta"):
        tr._star_matrix(np.linspace(-2,2,20),2,1,eta)


def test_discovery_and_scope():
    attrs=dict(parton="quark",observable="pdf",gfix="gi",operator="gt",renormalization_scheme="hybrid",
               source_component="both")
    name=matching_kernel_id(attrs,order="nlo",resummation="rgr_tr")
    assert load_kernel(name).__name__=="kernel"
    for changes,order in [({},"n3lo"),({"gfix":"cg"},"nlo"),({"observable":"gpd"},"nlo"),
                          ({"source_component":"invalid"},"nlo")]:
        with pytest.raises(ValueError):
            matching_kernel_id(attrs|changes,order=order,resummation="rgr_tr")


def test_threshold_correction_starts_beyond_nlo(monkeypatch):
    y = np.linspace(-2, 2, 40)
    errors = []
    for a in (.01, .005, .0025):
        monkeypatch.setattr(tr, "alpha_s", lambda loops, mu, mode: a / (1 + 9/(2*np.pi)*a*np.log(mu/2.8)))
        errors.append(np.max(np.abs(tr._threshold_correction(y, 2, 2.8, 1.2, .18, .4))))
    assert all(3.8 < a/b < 4.2 for a,b in zip(errors, errors[1:]))


def test_reporting_labels_tr():
    from lamet_agent.stages.perturbative_matching.reporting import _kernel_structure
    assert _kernel_structure("quark_pdf_gi_gt_hybrid_nlo_rgr_tr")["resummation"] == "RGR+TR"


@pytest.mark.parametrize("complex_samples", [False, True])
def test_tr_stage_publishes_samples_and_provenance(tmp_path, complex_samples):
    order = "nlo"
    from lamet_agent.agent import ToolContext
    from lamet_agent.data import EnsembleData
    from lamet_agent.stages.perturbative_matching._apply import run
    from lamet_agent.stages.perturbative_matching._inspection import run as inspect_kernel

    grid = np.linspace(-2, 2, 20)
    output_grid = np.array([-0.7, 0.3, 0.7])
    samples = [np.exp(-(grid**2)), 1.01 * np.exp(-(grid**2))]
    if complex_samples:
        samples = [sample + 0.2j * grid for sample in samples]
    quasi = EnsembleData(
        None,
        "bootstrap",
        samples,
        ["x"],
        {"x": grid.tolist()},
        attrs={
            "parton": "quark",
            "target_observable": "pdf",
            "gfix": "gi",
            "kernel_operator": "gt",
            "renormalization_scheme": "hybrid",
            "momentum_gev": 2.0,
            "zs_fm": 0.2,
            "source_component": "both",
            "output_component": "both" if complex_samples else "re",
        },
        name="quasi_distribution",
    )
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "perturbative_matching",
        "match",
        {
            "order": order,
            "resummation": "rgr_tr",
            "mu": 2.0,
            "lc_x_ls": output_grid.tolist(),
            "kernel_parameters": {"evolution_points": 10, "evolution_steps": 4, "y_max": 4.,
                                  **({"quadrature_order": 16} if order == "nnlo" else {})},
        },
        {"quasi": quasi},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    inspect_kernel(context)
    assert context.params["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr_tr"
    assert "antiquark" in context.state["kernel_inspection"]["document"]
    run(context)
    assert context.output is not None
    assert context.output.attrs["order"] == order
    assert context.output.attrs["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr_tr"
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_tr")
    matrix = kernel(output_grid, grid, momentum_gev=2., scale_gev=2., zs_fm=.2,
                    evolution_points=10, evolution_steps=4, y_max=4.)
    np.testing.assert_allclose(context.output.array.values, np.asarray(samples) @ matrix.T)
    restored = EnsembleData.from_netcdf(tmp_path / "output.nc")
    np.testing.assert_allclose(restored.array.values, context.output.array.values)
    assert restored.attrs["order"] == order
    assert restored.attrs["output_component"] == ("both" if complex_samples else "re")
