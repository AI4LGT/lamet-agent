"""Explicit contract for notebook operations and the native TMDWF data bridge."""

import math
from typing import Literal

from lamet_agent.contract import CheckContext, Depends, Issue, Source, Value, stage_job_rules


PHYSICS = "Preserve paired bootstrap replicas and the explicitly authored notebook conventions."
PARAM_RULES = (
    Depends("", "operation", physics=PHYSICS),
    Value("operation", Literal["bare_fit", "wilson_loop_fit", "renormalize", "fourier", "native_bridge"],
          physics=PHYSICS),
    Depends("", "settings", physics=PHYSICS),
    Value("settings", dict, physics=PHYSICS),
)
INPUT_RULES = tuple(Source(name, physics=PHYSICS) for name in ("raw", "target", "targets", "denominator", "input"))
JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)


def check(context: CheckContext):
    operation = context.params["operation"]
    cfg = context.params["settings"]
    meta = context.manifest["metadata"]
    issues = []

    def require(ok, path, message):
        if not ok:
            issues.append(Issue(path, message, PHYSICS))

    def number(value, positive=False):
        return (type(value) in (int, float) and math.isfinite(value)
                and (not positive or value > 0))

    roles = {"bare_fit": {"raw"}, "wilson_loop_fit": {"raw"},
             "renormalize": {"target", "denominator"}, "fourier": {"input"},
             "native_bridge": {"targets", "denominator"}}[operation]
    require(set(context.inputs) == roles, "inputs", f"requires exactly {sorted(roles)}")
    require(meta["target_observable"] == "tmdwf", "operation", "requires target_observable=tmdwf")
    require(meta["resample_mode"] == "bootstrap" and meta["bin_size"] == 1,
            "operation", "requires existing bootstrap samples and bin_size=1")
    fields = {
        "bare_fit": {"pz", "p1", "p2", "b", "a_fm", "L_s", "windows", "svdcut"},
        "wilson_loop_fit": {"b", "a_fm", "L", "ZO", "fit_start", "fit_stop", "priors", "svdcut"},
        "renormalize": set(),
        "native_bridge": {"projection", "negative_z"},
        "fourier": {"tail_start", "tail_stop", "stitch_index", "extent_factor", "priors", "svdcut",
                    "x_start", "x_stop", "x_points", "negative_z", "phase_sign", "x_shift"},
    }[operation]
    require(set(cfg) == fields, "settings", f"requires exactly {sorted(fields)}")
    if set(cfg) != fields:
        return issues
    if operation == "native_bridge":
        require(cfg["projection"] in {"real", "imag"}, "settings.projection", "requires real or imag")
        require(cfg["negative_z"] == "even_both_notebook", "settings.negative_z", "requires explicit even components")
        require(isinstance(context.inputs.get("targets"), list) and len(context.inputs["targets"]) == 4,
                "inputs.targets", "requires four source/direction fits")
    for field in ("a_fm", "ZO", "svdcut", "extent_factor"):
        if field in cfg:
            require(number(cfg[field], True), f"settings.{field}", "must be finite and positive")
    if operation in {"bare_fit", "wilson_loop_fit"}:
        require(type(cfg["b"]) is int and 1 <= cfg["b"] <= 10, "settings.b", "supported range is 1..10")
    if operation == "bare_fit":
        require((cfg["pz"], cfg["p1"], cfg["p2"]) in ((4, 2, -2), (5, 2, -3), (5, 3, -2), (6, 3, -3)),
                "settings.pz", "requires an available (pz,p1,p2) combination")
        require(type(cfg["L_s"]) is int and cfg["L_s"] > 0, "settings.L_s", "must be a positive integer")
        windows = cfg["windows"]
        valid = isinstance(windows, list) and len(windows) == 13
        if valid:
            valid = all(isinstance(w, list) and len(w) == 5 and all(type(v) is int for v in w)
                        and w[0] in (1, 2) and 0 <= w[1] < w[2] <= 15 and 0 <= w[3] < w[4] <= 15
                        for w in windows)
        require(valid, "settings.windows", "requires 13 [nstate,re_start,re_stop,im_start,im_stop] index windows")
    if "priors" in cfg:
        names = {"c", "E", "c1", "delta_E"} if operation == "wilson_loop_fit" else {
            "d", "lambda0", "c1", "c2", "f1", "f2"}
        priors = cfg["priors"]
        valid = isinstance(priors, dict) and set(priors) == names
        if valid:
            valid = all(isinstance(v, list) and len(v) == 2 and number(v[0]) and number(v[1], True)
                        for v in priors.values())
        require(valid, "settings.priors", f"requires mean/positive-width pairs for {sorted(names)}")
    if operation == "wilson_loop_fit":
        valid = all(type(cfg[k]) is int for k in ("fit_start", "fit_stop", "L"))
        require(valid and 0 <= cfg["fit_start"] < cfg["fit_stop"] <= 26 and cfg["L"] >= 0,
                "settings.fit_start", "requires an ordered window in 0..26 and nonnegative L")
    if operation == "fourier":
        valid = all(type(cfg[k]) is int for k in ("tail_start", "tail_stop", "stitch_index", "x_points"))
        require(valid and 1 <= cfg["tail_start"] < cfg["tail_stop"] <= 13
                and cfg["tail_start"] <= cfg["stitch_index"] < cfg["tail_stop"] and cfg["x_points"] >= 2,
                "settings.tail_start", "invalid positive-z tail, stitching, or x grid")
        require(number(cfg["x_start"]) and number(cfg["x_stop"]) and cfg["x_start"] < cfg["x_stop"],
                "settings.x_start", "requires a finite increasing x interval")
        require(cfg["negative_z"] == "even_both_notebook", "settings.negative_z", "requires explicit legacy convention")
        require(cfg["phase_sign"] == 1 and cfg["x_shift"] == 0.5,
                "settings.phase_sign", "this reproduction uses exp(+i*(x-0.5)*Pz*z)")
    return issues


CHECKS = (check,)
