"""Shared, validated choice of CG TMD perturbative coefficients."""

from __future__ import annotations

from . import cg_nll, cg_nlo, cg_nnll, cg_nnnll

_RESUMMED = {"nll": cg_nll, "nnll": cg_nnll, "nnnll": cg_nnnll}


def validate_settings(*, order="nlo", mu=2.0, alpha_s=0.293,
                      infrared_policy="error", freeze_scale_gev=None):
    cg_nlo._controls(1.0, mu, alpha_s)
    if order not in {"nlo", *_RESUMMED}:
        raise ValueError("TMD coefficient order must be 'nlo', 'nll', 'nnll', or 'nnnll' (Sudakov only)")
    if order == "nlo":
        if infrared_policy != "error" or freeze_scale_gev is not None:
            raise ValueError(
                "infrared controls apply only to resummed coefficients; NLO has no running-scale integration"
            )
    else:
        module = _RESUMMED[order]
        module.validate_controls(mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                                 freeze_scale_gev=freeze_scale_gev)


def coefficient_metadata(*, order="nlo", mu=2.0, alpha_s=0.293,
                         infrared_policy="error", freeze_scale_gev=None):
    validate_settings(order=order, mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                      freeze_scale_gev=freeze_scale_gev)
    metadata = {"order": order, "resummation": "" if order == "nlo" else order,
                "kernel_id": f"tmd_cg_{order}"}
    if order in _RESUMMED:
        running = cg_nnnll if order == "nnnll" else cg_nll
        metadata.update({
            "running_scheme": "msbar_four_loop_numeric" if order == "nnnll" else "reference_nnlo_analytic",
            "running_flavors": 3,
            "coupling_reference_scale_gev": float(mu), "coupling_reference_alpha_s": float(alpha_s),
            "coupling_pole_gev": running.coupling_pole_gev(mu=mu, alpha_s=alpha_s),
            "infrared_policy": infrared_policy,
        })
        if order == "nll":
            metadata.update({"hard_boundary": "tree", "sudakov_boundary": "tree"})
        elif order == "nnll":
            metadata.update({"kernel_scope": "form_factor", "sudakov_boundary": "one_loop",
                             "cusp_loops": 3, "noncusp_loops": 2, "running_loops": 3,
                             "reference": "arXiv:1705.07167, Eqs. (58)-(60)"})
        else:
            metadata.update({"kernel_scope": "form_factor", "sudakov_boundary": "two_loop",
                             "cusp_loops": 4, "noncusp_loops": 3, "running_loops": 4,
                             "rg_expansion": "consistent_nnlo",
                             "reference": "hep-ph/0607228 (50),(51),(96); 2002.04617 (6); hep-ph/9701390 (11)"})
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
    module = _RESUMMED[order]
    return module, {"mu": mu, "alpha_s": alpha_s, "infrared_policy": infrared_policy,
                    "freeze_scale_gev": freeze_scale_gev}


def hard_factor(x, *, observable, momentum_gev, order="nlo", mu=2.0, alpha_s=0.293,
                infrared_policy="error", freeze_scale_gev=None):
    module, kwargs = _kwargs(order, mu, alpha_s, infrared_policy, freeze_scale_gev)
    if order in {"nnll", "nnnll"}:
        raise ValueError(f"order='{order}' is available only for form_factor_kernel, not TMDWF/TMDPDF hard_factor")
    return module.hard_factor(x, observable=observable, momentum_gev=momentum_gev, **kwargs)


def form_factor_kernel(x1, x2, *, momentum_gev, order="nlo", mu=2.0, alpha_s=0.293,
                       infrared_policy="error", freeze_scale_gev=None):
    module, kwargs = _kwargs(order, mu, alpha_s, infrared_policy, freeze_scale_gev)
    return module.form_factor_kernel(x1, x2, momentum_gev=momentum_gev, **kwargs)
