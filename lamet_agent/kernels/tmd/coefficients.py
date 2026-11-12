"""Shared, validated choice of CG TMD perturbative coefficients."""

from __future__ import annotations

from . import cg_nll, cg_nlo, cg_nnll


def validate_settings(*, order="nlo", mu=2.0, alpha_s=0.293,
                      infrared_policy="error", freeze_scale_gev=None):
    cg_nlo._controls(1.0, mu, alpha_s)
    if order not in {"nlo", "nll", "nnll"}:
        raise ValueError("TMD coefficient order must be 'nlo', 'nll', or 'nnll' (Sudakov only)")
    if order == "nlo":
        if infrared_policy != "error" or freeze_scale_gev is not None:
            raise ValueError(
                "infrared controls apply only to resummed coefficients; NLO has no running-scale integration"
            )
    else:
        module = cg_nnll if order == "nnll" else cg_nll
        module.validate_controls(mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                                 freeze_scale_gev=freeze_scale_gev)


def coefficient_metadata(*, order="nlo", mu=2.0, alpha_s=0.293,
                         infrared_policy="error", freeze_scale_gev=None):
    validate_settings(order=order, mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                      freeze_scale_gev=freeze_scale_gev)
    metadata = {"order": order, "resummation": "" if order == "nlo" else order,
                "kernel_id": f"tmd_cg_{order}"}
    if order in {"nll", "nnll"}:
        metadata.update({
            "running_scheme": "reference_nnlo_analytic", "running_flavors": 3,
            "coupling_reference_scale_gev": float(mu), "coupling_reference_alpha_s": float(alpha_s),
            "coupling_pole_gev": cg_nll.coupling_pole_gev(mu=mu, alpha_s=alpha_s),
            "infrared_policy": infrared_policy,
        })
        if order == "nll":
            metadata.update({"hard_boundary": "tree", "sudakov_boundary": "tree"})
        else:
            metadata.update({"kernel_scope": "form_factor", "sudakov_boundary": "one_loop",
                             "cusp_loops": 3, "noncusp_loops": 2, "running_loops": 3,
                             "reference": "arXiv:1705.07167, Eqs. (58)-(60)"})
        if infrared_policy == "freeze":
            metadata.update({"freeze_scale_gev": float(freeze_scale_gev),
                             "freeze_prescription": "natural_and_evolution_scale_floor"})
        if infrared_policy == "reference_quad":
            metadata["infrared_integral_controlled"] = False
    return metadata


def _kwargs(order, mu, alpha_s, infrared_policy, freeze_scale_gev):
    validate_settings(order=order, mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                      freeze_scale_gev=freeze_scale_gev)
    if order == "nlo":
        return cg_nlo, {"mu": mu, "alpha_s": alpha_s}
    module = cg_nnll if order == "nnll" else cg_nll
    return module, {"mu": mu, "alpha_s": alpha_s, "infrared_policy": infrared_policy,
                    "freeze_scale_gev": freeze_scale_gev}


def hard_factor(x, *, observable, momentum_gev, order="nlo", mu=2.0, alpha_s=0.293,
                infrared_policy="error", freeze_scale_gev=None):
    module, kwargs = _kwargs(order, mu, alpha_s, infrared_policy, freeze_scale_gev)
    if order == "nnll":
        raise ValueError("order='nnll' is available only for form_factor_kernel, not TMDWF/TMDPDF hard_factor")
    return module.hard_factor(x, observable=observable, momentum_gev=momentum_gev, **kwargs)


def form_factor_kernel(x1, x2, *, momentum_gev, order="nlo", mu=2.0, alpha_s=0.293,
                       infrared_policy="error", freeze_scale_gev=None):
    module, kwargs = _kwargs(order, mu, alpha_s, infrared_policy, freeze_scale_gev)
    return module.form_factor_kernel(x1, x2, momentum_gev=momentum_gev, **kwargs)
