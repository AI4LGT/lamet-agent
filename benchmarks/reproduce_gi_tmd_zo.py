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

