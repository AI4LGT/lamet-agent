# Single-spacing residual Z_O for gauge-invariant TMDPDFs

`strategy="tmd_zo"` implements the short-distance extraction of
[arXiv:2412.19988](https://arxiv.org/abs/2412.19988), Eqs. (5)-(8), Sec. III.5.
It runs deterministically through the renormalization stage, without an LLM.

## Inputs and normalization

Prepare both reference and target by dividing the bare staple matrix element
by sqrt(Z_E(2L+z,b;a)), using the correct geometry, common replicas and an
adequate L plateau. These jobs accept **already subtracted** matrix elements;
they do not fit/extrapolate Wilson loops or take their square roots. Preserve
the uncertainty of the loop in the supplied samples.

Reference: one sample-bearing EnsembleData NetCDF with axes z and bT (in either
order), z coordinate in fm, bT_unit="fm", and at least three bootstrap or
jackknife replicas. Declare:

- observable="tmdpdf", gfix="GI", operator="gt", momentum_gev=0
- coord_unit="fm", wilson_loop_subtracted="true"
- reference_normalization="local_vector_charge"
- renormalization_family="<your action, link smearing, staple/endpoints and operator equivalence class>"
- the lattice EnsembleInfo, and the standard sampling lineage metadata

The reference must already be normalized by its local vector-charge matrix
element with kinematic factors removed, so its short-distance leading local
matrix element is one. The target must use the same local-vector-charge
normalization and declare reference_normalization="local_vector_charge" too.
In particular, dividing the reference by a bare local current changes the
extracted factor by that current's renormalization: do not apply it to an
unnormalized target. Local-charge division must preserve correlations.
Project onto the real part explicitly before extraction. Imaginary reference
values are rejected rather than silently discarded. Positive reference replicas
are required throughout the selected window. The target's complex part is retained.

Target: same ensemble and renormalization_family, GI tmdpdf with z and optional
bT axes. A fixed-b target uses bT_fm. Declare the actual target operator; the
family identifier asserts that the scalar endpoint factor applies to it,
including any needed control of lattice operator mixing. Sharing a lattice
spacing alone does not establish this equivalence. Already-MSbar inputs are rejected.

## Extraction and uncertainty

Select exact [z,b] pairs in `points_fm`. No interpolation is performed. The
window must contain at least three distinct points, span both z and b, and have
b>a. There is no z>a requirement: z=0 and z approximately a are allowed.
The b>a check is only a minimum guard; the data must show a credible stable
window between discretization and long-distance contamination.

For each replica and each scale variant, compute Z_eff=h_sub/C and take its
equal-weight average over the authored points. This follows the paper's
averaging prescription rather than a covariance-weighted constant fit. No
point selection, plateau certification or extrapolation is automatic.

NLO+RGR uses kappa=0.8,1,1.2. The factor output has one `scale_factor` axis;
these are deterministic systematic variants, not extra statistical replicas.
The application job chooses a stored variant and divides every target
coordinate by that sample's factor. Run all three application jobs to propagate
scale uncertainty coherently into later analysis. The central-value maximum
absolute shift is reported separately from the statistical standard deviation.
Use the same coupling and flavor prescription in all variants.

`diagnostics/zo.json` retains every selected effective factor with statistical
errors, the reference-point covariance, averaged factors, scale envelope,
settings and an exploratory residual chi-square/rank/Q diagnostic. The latter
uses covariance projected perpendicular to the arithmetic mean, with a relative
1e-10 pseudoinverse cutoff. A high Q is not proof of negligible cutoff or power
corrections. Compare different authored windows and fixed-order/RGR runs.
Independent jackknife groups are aligned through the package's existing
superjackknife mechanism. Shared replicas and central rows remain correlated.

## Manifest stage fragment

Place this under the manifest's `stages` object, retaining the normal run-wide
metadata. Paths refer to prepared NetCDF artifacts, not raw correlation files.
The sample points illustrate a=0.08 fm, not a universal window. alpha_s is the
fixed-nf coupling at mu, explicitly supplied by the analysis.

```json
{
  "renormalization": {
    "defaults": {"strategy": "tmd_zo", "normalization": false, "mu": 2.0},
    "jobs": [
      {
        "id": "zo", "type": "fit",
        "inputs": {"reference": {"file": "reference_subtracted.nc"}},
        "points_fm": [[0.0, 0.16], [0.08, 0.16], [0.0, 0.24], [0.08, 0.24]],
        "order": "nlo_rgr", "alpha_s": 0.293, "n_f": 3,
        "min_scale_gev": 0.7
      },
      {
        "id": "renormalized", "type": "apply",
        "inputs": {"target": {"file": "target_subtracted.nc"}, "zO": "zo"},
        "scale_factor": 1.0
      }
    ]
  }
}
```

Repeat the apply job with different IDs and scale_factor=0.8 or 1.2 for
systematics. For a fixed-order comparison select order="nlo" in a separate
extraction; its only factor coordinate is 1.0. Application mu must match the
stored extraction scale. Fit output is an intermediate factor, not a matrix
element for Fourier transformation. Apply output records scheme="msbar".

## Scope and validation

This adds residual-factor extraction and application, not an end-to-end
gauge-invariant Boer-Mulders analysis. Existing CG ratio workflows and their
matching kernels retain their conventions; do not feed GI MSbar results to a
CG ratio matching kernel. Soft factors, CS evolution and hard matching remain
separate. Candidate NNLO finite terms and LRR are outside this implementation.

Synthetic tests exercise coefficient normalization, independent RG quadrature,
scale variation, complex application, axis order, statistical correlations,
invalid metadata/windows, manifest execution, NetCDF and stage reporting.
They do not reproduce the paper's ensemble results without its lattice samples.

The supplied preliminary GI TMD notebook has separately been reproduced from
its original central-value/error tables; see
[the notebook validation](gi-tmd-zo-notebook-validation.md). Its threshold-running
couplings combined with a fixed-n_f evolution factor differ from this production
kernel's consistent fixed-flavor prescription by up to 2.8% in the compared
four-point windows. The benchmark is an explicit historical prescription,
not a selectable production coupling mode. Its independent Around-error
averaging is not used in sample-based production extraction.
