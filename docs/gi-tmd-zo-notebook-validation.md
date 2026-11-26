# Reproduction of the Boer–Mulders residual Z_O calculation

The supplied Mathematica notebook and original tables in the user-supplied data directory were used. The original files were read only. Python independently implements the notebook's coefficient, running coupling, analytic evolution and table processing. Mathematica 12 could not be launched because Windows reported a DLL relocation failure.

## Reproduced averages

The arithmetic average uses b/a = 2,3 and z/a = 0,1, exactly as the notebook's averaging cells. These are the preliminary slide/notebook results, not the later paper's different windows/results.

| Ensemble | Cached notebook | Reproduced |
|---|---:|---:|
| N203 | 1.53572354898 ± 0.03068591740 | 1.53572577169 ± 0.03068583631 |
| H102 | 1.40495203574 ± 0.02352225434 | 1.40495640595 ± 0.02352176503 |
| X650 pion | 1.23271805882 ± 0.02668391293 | 1.23272351183 ± 0.02668317969 |

The central values agree within 5.5e-6; the quoted errors agree within 7.4e-7. The X650 nucleon table gives approximately 0.30067 for the same four-point average. The input tables themselves yield this smaller normalization; the reproduction does not establish its physical cause.

`coefficient.png` reproduces the geometry and curves of slide 6. `zo_panels.pdf` and `zo_panels.png` reproduce the NLO and NLO+RGR Z_O panels of slides 9–12 from the original data. Plot limits intentionally clip unphysical large-distance excursions as in the slides; all calculated points remain in `zo_points.csv`.

## Conventions needed for agreement

- Wilson-loop division is retained: input tables already contain the subtracted matrix element, and Z_O = h_sub/C. No second Wilson-loop division is performed.
- alpha_s(M_Z)=0.1179, M_Z=91.1876 GeV, m_b=4.18 GeV, m_c=1.27 GeV; two-loop running with continuous thresholds. This gives alpha_s(2 GeV)=0.29587965525.
- Below 0.6 GeV the notebook uses a two-loop asymptotic expression whose Lambda is fitted for continuity. The translated value is 0.29208942934 GeV.
- hRGR uses these threshold-evolved couplings but n_f=3 in the analytic evolution factor, matching the supplied notebook. This is different from consistently fixed-n_f running. The agent comparison uses the same alpha_s(2 GeV) to isolate that difference.
- mu=2 GeV, hbar*c=0.1973269631 GeV fm, mu0=kappa*2*exp(-EulerGamma)*hbar*c/sqrt(z^2+b^2), kappa=0.8,1,1.2.
- The notebook defines each scale uncertainty as half the difference between kappa=1.2 and 0.8. It combines that with the point's statistical error in quadrature, then averages Mathematica Around objects assuming independent uncertainties.

## Uncertainty caveat for the production extraction

The last step reproduces the notebook, but it does not preserve correlated scale variations across points. For a common scale variation applied to all four points, the half-difference of the averaged Z_O is 0.02351 (N203), 0.03784 (H102), and 0.04627 (X650). These should not be reduced as independent pointwise statistical errors. The supplied tables contain central values and point errors, not bootstrap covariance; a correlated statistical uncertainty cannot be reconstructed from these tables alone. The production extraction should retain its sample correlations and coherent scale variants.

## Validation and scope

`cached_validation.json` compares the averages directly with saved Mathematica Around outputs. It also compares packed numerical coordinates in the notebook's b=0.255 fm perturbative plot with the translated functions. The NLO curve differs by at most 1.5e-6, central RGR by 1.7e-5, and upper-scale curve by 4.9e-5. The lower-scale filled-polygon coordinates have a larger discrepancy up to 0.00143, which is recorded rather than claimed as exact agreement. The polygon comparison assigns each coordinate to the closest of the four predicted curves and is not a substitute for a fresh Mathematica evaluation.

This reproduction covers the renormalization/Z_O calculation. The later hard-matching and final momentum-space TMD panels have not been validated in this run. No published-paper reproduction or full covariance reproduction is claimed.

## Run the committed benchmark

No original lattice tables or full notebook are redistributed in this PR.
The source directory must contain TMD_Boer-Mulders_Resummation.nb and its original
N203, H102, X650 pion and X650 nucleon data folders.

```
python benchmarks/reproduce_gi_tmd_zo.py /path/to/original/data /path/to/output
```

This creates the Z_O panels, per-point CSV, summary and cached-average validation.
The output directory must be outside the original source tree. The coefficient
curve/packed-polygon comparison described above was performed during the local
audit; the committed benchmark verifies the three cached Z_O averages.
