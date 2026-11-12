"""Shared, validated choice of CG TMD perturbative coefficients."""

from __future__ import annotations

from . import cg_nll, cg_nlo


def validate_settings(*, order="nlo", mu=2.0, alpha_s=0.293,
                      infrared_policy="error", freeze_scale_gev=None):
    cg_nlo._controls(1.0, mu, alpha_s)
    if order not in {"nlo", "nll"}:
        raise ValueError("TMD coefficient order must be 'nlo' or 'nll'")
    if order == "nlo":
        if infrared_policy != "error" or freeze_scale_gev is not None:
            raise ValueError("infrared controls apply only to NLL; fixed-order NLO has no running-scale integration")
    else:
        cg_nll.validate_controls(mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                                 freeze_scale_gev=freeze_scale_gev)


def coefficient_metadata(*, order="nlo", mu=2.0, alpha_s=0.293,
                         infrared_policy="error", freeze_scale_gev=None):
    validate_settings(order=order, mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                      freeze_scale_gev=freeze_scale_gev)
    metadata = {"order": order, "resummation": "" if order == "nlo" else "nll",
                "kernel_id": f"tmd_cg_{order}"}
    if order == "nll":
        metadata.update({
            "running_scheme": "reference_nnlo_analytic", "running_flavors": 3,
            "coupling_reference_scale_gev": float(mu), "coupling_reference_alpha_s": float(alpha_s),
            "coupling_pole_gev": cg_nll.coupling_pole_gev(mu=mu, alpha_s=alpha_s),
            "infrared_policy": infrared_policy, "hard_boundary": "tree",
            "sudakov_boundary": "tree",
        })
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
    return cg_nll, {"mu": mu, "alpha_s": alpha_s, "infrared_policy": infrared_policy,
                    "freeze_scale_gev": freeze_scale_gev}


def hard_factor(x, *, observable, momentum_gev, order="nlo", mu=2.0, alpha_s=0.293,
                infrared_policy="error", freeze_scale_gev=None):
    module, kwargs = _kwargs(order, mu, alpha_s, infrared_policy, freeze_scale_gev)
    return module.hard_factor(x, observable=observable, momentum_gev=momentum_gev, **kwargs)


def form_factor_kernel(x1, x2, *, momentum_gev, order="nlo", mu=2.0, alpha_s=0.293,
                       infrared_policy="error", freeze_scale_gev=None):
    module, kwargs = _kwargs(order, mu, alpha_s, infrared_policy, freeze_scale_gev)
    return module.form_factor_kernel(x1, x2, momentum_gev=momentum_gev, **kwargs)
