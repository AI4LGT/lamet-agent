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

Both RGR orders use kappa=0.8,1,1.2. The factor output has one `scale_factor` axis;
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
separate. The notebook NNLO extension is described below. LRR remains outside this implementation.

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


### NNLO boundary coefficient with RGR

Select `order: "nnlo_rgr"` in the extraction job (`"nnlo"` selects fixed order).
The Wilson-loop division remains an input requirement and is not applied twice.
This option ports the **NNLO Fixed order** module of the supplied `TMD_V3.nb`,
using its closed analytic remainder (`qTMDNNLOClosed-2`). It is an additional
notebook result, not an NNLO zero-momentum formula attributed to arXiv:2412.19988.

With `a_s = alpha_s/(4*pi)`, `r=z/b`, and
`ell=log(mu^2*(z^2+b^2)*exp(2*EulerGamma)/4)` (distances in GeV^-1),

```text
C_NNLO = 1 + a_s*CF*(3*ell+1-4*r*atan(r))
         + a_s^2*(CF^2*HFF + CF*CA*HFA + CF*TF*n_f*HFf).
C_NNLO+RGR(mu) = U(mu,mu0) * C_NNLO(mu0),
mu0 = kappa * 2*exp(-EulerGamma)/sqrt(z^2+b^2).
```

`nnlo_rgr` uses the **three-loop beta function and three-loop endpoint
anomalous dimension (NNLL evolution)** with the NNLO boundary.
`nlo_rgr` retains two-loop running and two-loop anomalous dimension (NLL). Supply `alpha_s(mu)` in the same explicit
fixed-flavor convention as the other extraction modes. The notebook's
threshold-running `alpha_sNNLO` function is not silently substituted.
Boundary logarithms are retained at each of the 0.8/1.0/1.2 scale variants.
The sample-wise average and application of the resulting factor are unchanged.

The finite remainder uses the closed principal-polylogarithm expression from
`TMD_V3.nb`, cached by absolute `z/b`, with at least 40 decimal working digits.
The original quadrature is retained as an independent diagnostic only. Extraction
no longer depends on its refinement guard, which rejected some points at
`z/b >= 1.5` with the old default grid. Scale and coupling guards still apply.
The analytic prescription and boundary source are recorded in `zo_settings`.
Production extraction retains coherent scale factors **0.8, 1.0, 1.2**.
The P17 benchmark uses 0.75, 1.0, 1.5 solely to reproduce that figure.

The notebook result assumes massless, zero external momentum, pure dimensional
regularization, the nonsinglet gamma-t projection, and MSbar renormalization.
Its equivalence to a separately taken finite-momentum limit remains unestablished.
The reference tests check the notebook's color coefficients and ordered kernel,
independent integration refinements, two-loop RG logarithms, sign symmetry,
and manifest-driven extraction/application with correlated replicas.
