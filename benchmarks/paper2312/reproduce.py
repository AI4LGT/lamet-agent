"""Figure-1 digitization benchmark; no synthetic lattice samples are generated.

Source: Holligan and Lin, arXiv:2312.10829v2, Fig. 1, CC BY 4.0.
Run from the repository root: python benchmarks/paper2312/reproduce.py
"""

from pathlib import Path
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from lamet_agent.kernels.implementation import ZMSbar, alpha_s  # noqa: E402
from lamet_agent.kernels.z_msbar_pdf_nnlo import kernel as nnlo  # noqa: E402

HC = 0.1973269804
# SVG axes: x=0,2 fm at 87.641,782.523; y=0,1 at 427.988,25.367.
X0, XS, Y0, YS = 87.641, (782.523 - 87.641) / 2, 427.988, 427.988 - 25.367
tree = ET.parse(ROOT / "LinearDivergence.svg")
bars = {}
for elem in tree.iter():
    if elem.get("stroke") == "#0000ff":
        match = re.fullmatch(r"M([\d.]+) ([\d.]+)V([\d.]+)", elem.get("d", ""))
        if match:
            x, y1, y2 = map(float, match.groups())
            bars.setdefault(x, []).extend([y1, y2])
data = np.array(
    [[(x - X0) / XS, (Y0 - (min(v) + max(v)) / 2) / YS, (max(v) - min(v)) / 2 / YS] for x, v in sorted(bars.items())]
)
assert len(data) == 21
assert np.max(abs(data[:, 0] - np.arange(21) * 0.09)) < 1e-5
with (ROOT / "pion_digitized.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["z_fm", "h_bare", "plotted_marginal_error"])
    writer.writerows(data)
z, h, err = data.T
select = (z >= 0.54 - 1e-5) & (z <= 1.53 + 1e-5)


def exponential(z, b, mass):
    return b * np.exp(-mass * z / HC)


fit, covariance = curve_fit(exponential, z[select], h[select], sigma=err[select], absolute_sigma=True, p0=[2.4, 0.668])
unweighted, _ = curve_fit(exponential, z[select], h[select], p0=fit)
chi2 = float(np.sum(((h[select] - exponential(z[select], *fit)) / err[select]) ** 2))

# Independent extraction of the red published fit curve (the first SVG red path).
path = next(e.get("d") for e in tree.iter() if e.get("stroke") == "#ff0000")
# Before the first H command all vertices are explicit x,y pairs.
vertices = np.array(list(map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", path.split("H")[0])))).reshape(-1, 2)
rz = (vertices[:, 0] - X0) / XS
rh = (Y0 - vertices[:, 1]) / YS
keep = (rz >= 0.54) & (rz <= 1.53)
red_fit, _ = curve_fit(exponential, rz[keep], rh[keep], p0=fit)

# Central orange curves manually identified in the downloaded 960x600 panels.
# Calibration from visible ticks: y=0 at 388, y=1 at25.5; x=.08 at119.5,
# x=.20 at931.5. Entries refer to central curves, not band midpoints.
# Use only five unambiguously traceable points; overlapping curves omitted.
centers = np.array([0.08, 0.10, 0.12, 0.14, 0.16])
pixel_y = {"NLO": np.array([411.5, 437.5, 450.5, 456.5, 459.5]), "NNLO": np.array([334.0, 366.3, 384.5, 397.0, 404.5])}
interp = CubicSpline(z, h)  # explicit reconstruction assumption, not specified by paper
scan = np.linspace(0.08, 0.20, 121)


def local_mass(c, coefficient):
    q = c + np.array([-0.02, 0, 0.02])
    return np.polyfit(q / HC, np.log(coefficient(q) / interp(q)), 1)[0] - fit[1]


coefficients = {"NLO": lambda q: ZMSbar(q, mu=2.0, offset=2.5), "NNLO": nnlo}
comparison = []
for order, coefficient in coefficients.items():
    for c, py in zip(centers, pixel_y[order]):
        digitized = (388 - py) / 362.5
        calculated = local_mass(c, coefficient)
        comparison.append([order, c, py, digitized, calculated, calculated - digitized])
with (ROOT / "m0_fixed_order_comparison.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["order", "z_fm", "source_y_pixel", "digitized_m0_gev", "computed_m0_gev", "difference_gev"])
    writer.writerows(comparison)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
ax = axes[0]
ax.errorbar(z, h, yerr=err, fmt="o", ms=3, color="#2369bd", label="Digitized pion points")
fine = np.linspace(0.40, 1.8, 400)
ax.plot(fine, exponential(fine, *fit), color="#c23b22", label=rf"Refit: $\delta m={fit[1]:.5f}$ GeV")
ax.axvspan(0.54, 1.53, color="green", alpha=0.08)
ax.set(xlabel="$z$ (fm)", ylabel=r"$h^B_\pi(z,0)$", ylim=(-0.025, 1.05), title="Fig. 1: pion reference")
ax.legend(fontsize=8)
for ax, (order, coefficient) in zip(axes[1:], coefficients.items()):
    ax.plot(scan, [local_mass(c, coefficient) for c in scan], label="Recomputed, cubic interpolation")
    values = (388 - pixel_y[order]) / 362.5
    ax.scatter(centers, values, color="#e68016", marker="x", s=40, label="Paper central curve, digitized")
    ax.set(xlabel="$z$ (fm)", ylabel=r"$m_0$ (GeV)", title=order + ", fixed order")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
fig.suptitle("2312.10829 Fig. 1: numerical check from digitized inputs", fontsize=13)
fig.savefig(ROOT / "figure1_reproduction.png", dpi=180)
result = {
    "source": "https://arxiv.org/html/2312.10829v2",
    "figure": 1,
    "delta_m_gev": float(fit[1]),
    "B": float(fit[0]),
    "diagonal_only_mass_error_gev": float(np.sqrt(covariance[1, 1])),
    "residual_scaled_diagonal_mass_error_gev": float(np.sqrt(covariance[1, 1] * chi2 / (select.sum() - 2))),
    "chi2_diagonal": chi2,
    "dof": int(select.sum() - 2),
    "unweighted_delta_m_gev": float(unweighted[1]),
    "published_red_curve_delta_m_gev": float(red_fit[1]),
    "paper_delta_m_gev": 0.668,
    "paper_mass_error_gev": 0.010,
    "alpha_s_at_2_gev": alpha_s(3, 2.0, "running"),
    "m0_max_abs_difference_gev": max(abs(row[-1]) for row in comparison),
    "limitations": [
        "No covariance recovered; diagonal error is not the paper uncertainty.",
        "Cubic interpolation is a reconstruction choice; paper does not specify interpolation details.",
        "m0 digitization precision about 1-2 pixels (0.003-0.006 GeV).",
        "RGR/LRR curves and final PDFs/GPDs are not reproduced by this benchmark.",
    ],
}
(ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
