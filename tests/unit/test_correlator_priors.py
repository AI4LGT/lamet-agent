"""Shared Gaussian prior semantics and manifest validation."""

from types import SimpleNamespace

import gvar as gv
import numpy as np
import pytest

from lamet_agent.stages.correlator_analysis._priors import (
    build_priors,
    declared_prior_names,
    sample_prior_scale,
)
from lamet_agent.stages.correlator_analysis.contract import check_prior_controls
from lamet_agent.stages._priors import prior_diagnostics, validate_prior_overrides


def test_authored_priors_scale_widths_once_and_condition_physical_parameters():
    defaults = {"z0": gv.gvar(1, 3), "log(E0)": gv.gvar(-1, 0.5), "O00_re": gv.gvar(0, 2)}
    prior = build_priors(
        defaults,
        {"z0": "2.0(0.3)e-8", "log(E0)": "-0.7(0.2)"},
        prior_width=2,
        conditioning={"z0": 1e8, "log(E0)": 4},
    )
    assert gv.mean(prior["z0"]) == pytest.approx(2)
    assert gv.sdev(prior["z0"]) == pytest.approx(0.6)
    assert gv.mean(prior["log(E0)"]) == pytest.approx(-0.7 + np.log(4))
    assert gv.sdev(prior["log(E0)"]) == pytest.approx(0.4)
    assert gv.mean(prior["O00_re"]) == 0
    assert gv.sdev(prior["O00_re"]) == 4


def test_extra_overrides_are_inactive_without_requiring_a_declared_state_union():
    overrides = {"z1": "0.1(0.3)", "unused": "2(1)"}
    prior = build_priors({"z0": gv.gvar(1, 2)}, overrides)
    assert set(prior) == {"z0"}
    assert prior_diagnostics(prior, overrides)["inactive_prior_overrides"] == ["unused", "z1"]
    assert gv.mean(prior["z0"]) == 1
    assert gv.sdev(prior["z0"]) == 2


@pytest.mark.parametrize(
    "value",
    [
        {"mean": 1, "sdev": 0.3},
        ["1.0(0.3)"],
        True,
        1.0,
        "",
        "1.0",
        "invalid",
        "inf +- 1",
        "1 +- inf",
        "1(0)",
        "1(-1)",
    ],
)
def test_rejects_invalid_scalar_prior_definitions(value):
    with pytest.raises(ValueError, match="prior"):
        validate_prior_overrides({"z0": value})


@pytest.mark.parametrize("value", ["1.0(0.3)", "1.00(30)", "2.00(30)e-08", "1.0 +- 0.3", "0 ± 3.0"])
def test_prior_strings_follow_gvar_parsing(value):
    expected = gv.gvar(value)
    prior = build_priors({"z0": gv.gvar(0, 1)}, {"z0": value})
    assert gv.mean(prior["z0"]) == gv.mean(expected)
    assert gv.sdev(prior["z0"]) == gv.sdev(expected)


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf"), True])
def test_rejects_invalid_global_prior_width(width):
    with pytest.raises(ValueError, match="prior_width"):
        build_priors({"z0": gv.gvar(1, 2)}, prior_width=width)


def test_replica_prior_scale_is_independent_of_base_width():
    assert sample_prior_scale("original", 10) is None
    assert sample_prior_scale("center_posterior", 7) == 7
    with pytest.raises(ValueError, match="sample_prior_mode"):
        sample_prior_scale("invalid", 10)
    with pytest.raises(ValueError, match="posterior_prior_error_scale"):
        sample_prior_scale("center_posterior", 0)


def test_manifest_prior_validation_accepts_names_unused_by_the_declared_models():
    params = {
        "analysis_method": "lsqfit",
        "fit_scope": ["2pt", "3pt_over_2pt"],
        "nstate": {"2pt": [1, 2], "3pt_over_2pt": [1, 2]},
        "fitting_form": "Breit",
        "priors": {"O11_re": "0.4(0.8)"},
    }
    assert "O11_re" in declared_prior_names(params)
    assert check_prior_controls(SimpleNamespace(params=params)) is None
    params["priors"] = {"O12_re": "0.4(0.8)"}
    assert check_prior_controls(SimpleNamespace(params=params)) is None
    params["fit_scope"] = ["2pt"]
    params["nstate"] = {"2pt": [1, 2]}
    params["priors"] = {"E1": "0.8(0.2)"}
    assert check_prior_controls(SimpleNamespace(params=params)) is None
    assert declared_prior_names(params) == {"E0", "E1", "z0", "z1"}
    params["priors"] = {"log(E0)": "-0.7(0.3)"}
    assert check_prior_controls(SimpleNamespace(params=params)) is None


def test_qda_declared_priors_use_the_implemented_breit_model_without_optional_form():
    names = declared_prior_names({"fit_scope": ["qda"], "nstate": {"qda": [1, 2]}})
    assert {"log(E0)", "log(dE1)", "z0", "z1", "O00_re", "O01_im", "zprime0"} <= names
