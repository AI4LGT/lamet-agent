"""Reproduce the supplied notebook's Z_O prescription from original input tables."""

from pathlib import Path
import argparse
import re
import json
import csv
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Reproduce the preliminary GI TMD Z_O notebook; no raw data are bundled.")
parser.add_argument("source_dir", type=Path, help="Directory containing the notebook and original ensemble folders")
parser.add_argument("output_dir", type=Path, help="Directory for generated comparison artifacts")
args = parser.parse_args()
ROOT = args.source_dir.resolve()
OUT = args.output_dir.resolve()
if ROOT == OUT or ROOT in OUT.parents:
    parser.error("Choose an output directory outside the original source tree")
OUT.mkdir(parents=True, exist_ok=True)
HC = 0.1973269631
CF = 4 / 3


def betas(nf):
    return (33 - 2 * nf) / (12 * np.pi), (153 - 19 * nf) / (24 * np.pi**2)


segments = []
initial = 0.1179
for high, low, nf in [(91.1876, 4.18, 5), (4.18, 1.27, 4), (1.27, 0.6, 3)]:
    b0, b1 = betas(nf)
    sol = solve_ivp(
        lambda t, y: -2 * (b0 * y * y + b1 * y * y * y),
        (np.log(high), np.log(low)),
        [initial],
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
    )
    assert sol.success
    segments.append((low, high, sol.sol))
    initial = sol.y[0, -1]
b0, b1 = betas(3)


def asymptotic(mu, lam):
    L = np.log(mu**2 / lam**2)
    return 1 / (b0 * L) - b1 * np.log(L) / (b0**3 * L**2)


lam = brentq(lambda scale: asymptotic(0.6, scale) - initial, 0.05, 0.59)


def alpha(mu):
    if mu <= lam:
        return np.nan
    if mu < 0.6:
        return asymptotic(mu, lam)
    for low, high, sol in segments:
        if low <= mu <= high:
            return float(sol(np.log(mu))[0])
    raise ValueError(mu)


def evolution(ai, af, nf=3):
    b0, b1 = betas(nf)
    g0 = 3 * CF / (2 * np.pi)
    g1 = (
        2
        / (4 * np.pi) ** 2
        * ((49 / 6 - 2 * np.pi**2 / 3) * CF * 3 - 10 / 3 * CF * nf * 0.5 - (2.5 - 8 * np.pi**2 / 3) * CF**2)
    )
    return np.exp(
        0.5
        * (
            g0 / b0 * np.log((b0 + b1 * af) * ai / (af * (b0 + b1 * ai)))
            + g1 / b1 * np.log((b0 + b1 * ai) / (b0 + b1 * af))
        )
    )


def h(z, b, mu=2):
    return 1 + alpha(mu) * CF / (2 * np.pi) * (
        0.5
        + 1.5 * np.log(mu**2 * (z * z + b * b) / HC**2 * np.exp(2 * np.euler_gamma) / 4)
        - 2 * z / b * np.arctan(z / b)
    )


def hrgr(z, b, c=1):
    mu0 = c * 2 * np.exp(-np.euler_gamma) * HC / np.hypot(z, b)
    return h(z, b, mu0) * evolution(alpha(mu0), alpha(2))


specs = [
    ("N203", 0.064, "N203_pion", "N203r001_p0_b{}_subtr_quasi_TMDPDF.txt", range(1, 5), 5),
    ("H102", 0.085, "H102_pion", "H102r002_p0_b{}_subtr_quasi_TMDPDF.txt", range(1, 4), 4),
    ("X650 pion", 0.098, "X650_pion", "X650r000r001r002_p0_b{}_subtr_quasi_TMDPDF.txt", range(1, 4), 4),
    ("X650 nucleon", 0.098, "X650_nucleon", "X650_p0_b{}_substr.txt", range(2, 4), 4),
]
rows = []
summaries = {}
fig, axs = plt.subplots(4, 2, figsize=(12, 15))
for row, (name, a, folder, pattern, bs, nz) in enumerate(specs):
    selected = []
    for bi in bs:
        matches = list((ROOT / (folder + "_Boer-Mulders_data_preliminary")).rglob(pattern.format(bi)))
        assert len(matches) == 1, matches
        dat = np.loadtxt(matches[0])
        if name == "X650 nucleon":
            dat = np.vstack([np.arange(len(dat)), dat.T])
        z = dat[0, :nz] * a
        v = dat[1, :nz]
        err = dat[2, :nz]
        fo = np.array([h(zi, bi * a) for zi in z])
        co = np.array([[hrgr(zi, bi * a, c) for zi in z] for c in [0.8, 1, 1.2]])
        zo = v / co[1]
        stat = err / np.abs(co[1])
        syst = (v / co[2] - v / co[0]) / 2
        total = np.hypot(stat, syst)
        for i, zi in enumerate(z):
            rows.append([name, bi, zi / a, zi, v[i], err[i], v[i] / fo[i], zo[i], stat[i], syst[i], total[i]])
            if bi in (2, 3) and i < 2:
                selected.append((zo[i], stat[i], syst[i]))
        for col, values, errors in [(0, v / fo, err / np.abs(fo)), (1, zo, total)]:
            axs[row, col].errorbar(z + 0.0012 * bi, values, yerr=errors, fmt="o", ms=4, label=f"b/a={bi}")
    s = np.array(selected)
    summaries[name] = {
        "mean_4_points": s[:, 0].mean(),
        "notebook_independent_combined_error": np.sqrt(np.sum(s[:, 1:] ** 2)) / len(s),
        "stat_independent_error": np.linalg.norm(s[:, 1]) / len(s),
        "coherent_scale_half_difference": abs(s[:, 2].mean()),
    }
    for col, order in enumerate(["NLO", "NLO+RGR"]):
        ax = axs[row, col]
        ax.set(
            title=f"{name}: {order}", xlabel="z (fm)", ylabel=r"$Z_O$", ylim=(0, 0.5 if name == "X650 nucleon" else 2)
        )
        ax.legend()
        ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "zo_panels.png", dpi=160)
fig.savefig(OUT / "zo_panels.pdf")
plt.close(fig)
with (OUT / "zo_points.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "ensemble",
            "b_over_a",
            "z_over_a",
            "z_fm",
            "subtracted_matrix_element",
            "input_error",
            "ZO_NLO",
            "ZO_RGR",
            "stat_error",
            "signed_scale_half_difference",
            "combined_error",
        ]
    )
    w.writerows(rows)
print("alpha(2)", alpha(2), "lambda", lam)
print(json.dumps(summaries, indent=2))
(OUT / "summary.json").write_text(
    json.dumps({"alpha_s_2": alpha(2), "lambda_low_scale": lam, "ensembles": summaries}, indent=2)
)
src = (ROOT / "TMD_Boer-Mulders_Resummation.nb").read_text(encoding="utf-8")
# Compare the final four-point averages to the original cached outputs.
checks = {}
for name, label in [("N203", "213"), ("H102", "236"), ("X650 pion", "142")]:
    pos = src.index('"Out[' + label + "]=")
    pair = re.findall(r"Around\[([\d.`]+), ([\d.`]+)\]", src[pos - 600 : pos])[-1]
    cached_mean, cached_error = [float(x.replace("`", "")) for x in pair]
    result = summaries[name]
    delta = abs(result["mean_4_points"] - cached_mean)
    error_delta = abs(result["notebook_independent_combined_error"] - cached_error)
    checks[name] = dict(
        cached_mean=cached_mean,
        cached_error=cached_error,
        mean_absolute_difference=delta,
        error_absolute_difference=error_delta,
    )
    if delta > 1e-5 or error_delta > 2e-6:
        raise RuntimeError(f"Notebook reproduction failed for {name}: {checks[name]}")
(OUT / "cached_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
print("All three GI TMD Z_O averages match the cached notebook within tolerance.")
