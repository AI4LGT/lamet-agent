"""Shared Gaussian prior semantics and manifest validation."""

from types import SimpleNamespace

import gvar as gv
import numpy as np
import pytest

from lamet_agent.stages.correlator_analysis._priors import (
    build_priors,
    declared_prior_names,
    prior_diagnostics,
    sample_prior_scale,
    validate_prior_overrides,
)
from lamet_agent.stages.correlator_analysis.contract import check_prior_controls


def test_authored_priors_scale_widths_once_and_condition_physical_parameters():
    defaults = {"z0": gv.gvar(1, 3), "log(E0)": gv.gvar(-1, 0.5), "O00_re": gv.gvar(0, 2)}
    prior = build_priors(
        defaults,
        {"z0": {"mean": 2e-8, "sdev": 3e-9}, "log(E0)": {"mean": -0.7, "sdev": 0.2}},
        prior_width=2,
        conditioning={"z0": 1e8, "log(E0)": 4},
    )
    assert gv.mean(prior["z0"]) == pytest.approx(2)
    assert gv.sdev(prior["z0"]) == pytest.approx(0.6)
    assert gv.mean(prior["log(E0)"]) == pytest.approx(-0.7 + np.log(4))
    assert gv.sdev(prior["log(E0)"]) == pytest.approx(0.4)
    assert gv.mean(prior["O00_re"]) == 0
    assert gv.sdev(prior["O00_re"]) == 4


def test_candidate_union_reports_inactive_overrides_without_accepting_typos():
    overrides = {"z1": {"mean": 0.1, "sdev": 0.3}}
    prior = build_priors({"z0": gv.gvar(1, 2)}, overrides, allowed_names={"z0", "z1"})
    assert prior_diagnostics(prior, overrides)["inactive_prior_overrides"] == ["z1"]
    with pytest.raises(ValueError, match="unknown prior parameter 'zI'"):
        build_priors({"z0": gv.gvar(1, 2)}, {"zI": overrides["z1"]}, allowed_names={"z0", "z1"})
    with pytest.raises(ValueError, match="unknown prior parameter 'z1'"):
        build_priors({"z0": gv.gvar(1, 2)}, overrides)


@pytest.mark.parametrize(
    "value",
    [
        {"mean": [1], "sdev": 1},
        {"mean": 1, "sdev": [1]},
        {"mean": float("nan"), "sdev": 1},
        {"mean": 1, "sdev": float("inf")},
        {"mean": True, "sdev": 1},
        {"mean": 1, "sdev": False},
        {"mean": 1, "sdev": 0},
        {"mean": 1, "sdev": -1},
        {"mean": 1},
        {"mean": 1, "sdev": 1, "width": 2},
    ],
)
def test_rejects_invalid_scalar_prior_definitions(value):
    with pytest.raises(ValueError, match="prior"):
        validate_prior_overrides({"z0": value})


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


def test_manifest_prior_validation_uses_declared_state_union_and_model_names():
    params = {
        "analysis_method": "lsqfit",
        "fit_scope": ["2pt", "3pt_over_2pt"],
        "nstate": {"2pt": [1, 2], "3pt_over_2pt": [1, 2]},
        "fitting_form": "Breit",
        "priors": {"O11_re": {"mean": 0.4, "sdev": 0.8}},
    }
    assert "O11_re" in declared_prior_names(params)
    assert check_prior_controls(SimpleNamespace(params=params)) is None
    params["priors"] = {"O12_re": {"mean": 0.4, "sdev": 0.8}}
    assert "unknown prior parameter" in check_prior_controls(SimpleNamespace(params=params)).message
    params["fit_scope"] = ["2pt"]
    params["nstate"] = {"2pt": [1, 2]}
    params["priors"] = {"E1": {"mean": 0.8, "sdev": 0.2}}
    assert check_prior_controls(SimpleNamespace(params=params)) is None
    assert declared_prior_names(params) == {"E0", "E1", "z0", "z1"}
    params["priors"] = {"log(E0)": {"mean": -0.7, "sdev": 0.3}}
    assert "unknown prior parameter" in check_prior_controls(SimpleNamespace(params=params)).message


def test_qda_declared_priors_use_the_implemented_breit_model_without_optional_form():
    names = declared_prior_names({"fit_scope": ["qda"], "nstate": {"qda": [1, 2]}})
    assert {"log(E0)", "log(dE1)", "z0", "z1", "O00_re", "O01_im", "zprime0"} <= names
