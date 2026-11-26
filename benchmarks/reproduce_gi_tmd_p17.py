"""Independent P17 reproduction with production analytic finite remainder.

Scale factors .75/1/1.5 and symmetric raw half-difference bands are
figure-specific. Production Z_O uses .8/1/1.2 and fixed-flavor running.
"""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import solve_ivp, quad
from lamet_agent.kernels.tmd.zo import anomalous_dimension
from lamet_agent.kernels.tmd.zo_nnlo import color_coefficients, analytic_same_side_sum as sigma
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Reproduce TMD_V3.nb P17; historical threshold coupling, not production settings."
)
parser.add_argument("output", type=Path)
out = parser.parse_args().output
out.mkdir(parents=True, exist_ok=True)
HC = 0.1973269631


def beta(a, nf, loops):
    b0 = 11 - 2 * nf / 3
    b1 = 102 - 38 * nf / 3
    b2 = 2857 / 2 - 5033 * nf / 18 + 325 * nf * nf / 54
    x = a / (4 * np.pi)
    return -2 * a * (b0 * x + b1 * x * x + (b2 * x**3 if loops == 3 else 0))


segments = {}
for loops in [2, 3]:
    ai = 0.1179
    segments[loops] = []
    for hi, lo, nf in [(91.1876, 4.18, 5), (4.18, 1.27, 4), (1.27, 0.6, 3)]:
        sol = solve_ivp(
            lambda t, y: beta(y, nf, loops), (np.log(hi), np.log(lo)), [ai], rtol=1e-11, atol=1e-13, dense_output=True
        )
        assert sol.success
        ai = sol.y[0, -1]
        segments[loops].append((lo, hi, sol.sol))


def alpha(mu, loops):
    for lo, hi, f in segments[loops]:
        if lo <= mu <= hi:
            return float(f(np.log(mu))[0])
    raise ValueError(mu)


def h(z, k, loops):
    r = z / 0.1
    mu0 = k * 2 * np.exp(-np.euler_gamma) * HC / np.hypot(z, 0.1)
    ai = alpha(mu0, loops)
    af = alpha(2, loops)
    ell = 2 * np.log(k)
    as_ = ai / (4 * np.pi)
    fixed = 1 + as_ * 4 / 3 * (3 * ell + 1 - 4 * r * np.arctan(r))
    if loops == 3:
        ff, fa, ferm = color_coefficients(r, ell, sigma(r))
        fixed += as_**2 * ((4 / 3) ** 2 * ff + 4 * fa + 2 * ferm)
    return fixed * np.exp(quad(lambda a: anomalous_dimension(a, 3, loops) / beta(a, 3, loops), ai, af, epsabs=1e-11)[0])


z = np.arange(26) * 0.01
curves = {}
for loops in [2, 3]:
    v = np.array([[h(zi, k, loops) for zi in z] for k in [0.75, 1, 1.5]])
    curves[loops] = {"central": v[1], "error": abs(v[2] - v[0]) / 2}
reference = json.loads(Path(__file__).with_name("gi_tmd_p17_reference.json").read_text())
report = {}
for loops in [2, 3]:
    pred = curves[loops]
    cached = reference[str(loops)]
    report[loops] = {
        "central_max_abs_diff": float(max(abs(pred["central"] - cached["central"]))),
        "band_max_abs_diff": float(max(abs(pred["error"] - cached["half_width"]))),
    }
    assert report[loops]["central_max_abs_diff"] < 2e-5, report
    assert report[loops]["band_max_abs_diff"] < 5e-5, report
    print(loops, report[loops])
print("samples", [(float(z[i]), *[float(curves[n]["central"][i]) for n in [2, 3]]) for i in [0, 10, 15, 20, 25]])
(out / "notebook_validation.json").write_text(json.dumps(report, indent=2))
fig, ax = plt.subplots(figsize=(9, 5))
for loops, c, label in [(2, "blue", "NLO+RGR"), (3, "red", "NNLO+RGR")]:
    v = curves[loops]["central"]
    e = curves[loops]["error"].copy()
    # P17 matches the cached 'Plot - each point' output, without neighbor smoothing.
    ax.plot(z, v, c=c, label=label)
    ax.fill_between(z, v - e, v + e, color=c, alpha=0.2)
ax.set(
    xlim=(0, 0.25), ylim=(0.46, 1.04), xlabel="z (fm)", ylabel=r"$\tilde h(z,b=0.1\,\mathrm{fm},\mu=2\,\mathrm{GeV})$"
)
ax.legend()
fig.tight_layout()
fig.savefig(out / "p17_reproduced.png", dpi=180)
np.savetxt(
    out / "p17_values.csv",
    np.column_stack([z, curves[2]["central"], curves[2]["error"], curves[3]["central"], curves[3]["error"]]),
    delimiter=",",
    header="z_fm,NLO_RGR,NLO_raw_half_difference,NNLO_RGR,NNLO_raw_half_difference",
)
