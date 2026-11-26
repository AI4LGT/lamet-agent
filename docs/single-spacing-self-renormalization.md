# Single-spacing self-renormalization for collinear PDFs and GPDs

The renormalization strategy `single_spacing_self_renormalization` implements
the two-step mass determination and hybrid-ratio application in Sec. II,
Eq. (3) and the discussion following Eq. (10), of
[Holligan and Lin, arXiv:2312.10829](https://arxiv.org/abs/2312.10829).
It is separate from the existing multi-spacing `self_renormalization` strategy.

## Physics and conventions

1. Fit the real zero-momentum forward PDF reference to
   \(h^B(z)=B\exp[-\delta m z/(\hbar c)]\) in an explicit large-distance window.
2. At explicitly selected short-distance points fit
   \(\log[C_0(z,\mu)/h^B(z)]-\delta m z/(\hbar c)=m_0z/(\hbar c)+c\).
   The intercept is free. The short-distance data are linearly interpolated
   **matrix elements**, before taking logarithms, separately in every replica.
3. Apply

\[
h^R(z)=\begin{cases}
h^B_{\rm target}(z)/h^B_{\rm ref}(|z|),& |z|\le z_s,\\
e^{(\delta m+m_0)(|z|-z_s)/(\hbar c)}
h^B_{\rm target}(z)/h^B_{\rm ref}(z_s),& |z|>z_s.
\end{cases}
\]

Masses are in GeV, lengths in fm, and \(\hbar c=0.1973269804\) GeV fm.
The large-distance mass is a subtraction convention: physical decay in that
window can contribute. The short-distance fit compensates it. Only their sum
enters the renormalized matrix element. Tests explicitly verify this cancellation.
This operation is hybrid-ratio renormalization, not a direct all-distance MSbar
conversion; use the corresponding hybrid light-cone matching downstream.

The paper's large-distance window is [0.54,1.53] fm and its local short-distance
fits use {z-0.02,z,z+0.02} fm. These are not universal defaults. Supply a window
covered by your data, inspect the scan, and author the selected points. No
automatic plateau claim is made. Interpolation adds no independent observations;
the covariance rank and SVD regularization are reported. Cutoff effects and
OPE power corrections are not fitted away at a single lattice spacing.

## Inputs and scope

Reference: one NetCDF `EnsembleData` with `dims=["z"]`, physical `coord_unit="fm"`
(or declared lattice units), a nonnegative increasing z grid, and at least
three bootstrap or jackknife replicas. Declare `observable="pdf"`,
`gfix="gi"`, `operator="gt"`, `momentum_gev=0`, and the lattice ensemble.
The real part is used; negative or zero reference values in the fits or ratio
are rejected rather than replaced by absolute values. Both fit windows need
at least three points and at least two independent covariance directions.

Target: the same lattice ensemble/regularization, gauge-invariant gamma-t quark
PDF or GPD. GPDs may have nonzero momentum transfer or skewness; the forward
zero-momentum **PDF** reference still supplies the subtraction. Targets support
`z` and optional `t`, `xi`, or `Q2` axes in any order, and signed z grids. The
reference grid must contain each target |z| <= z_s and z_s exactly; the
long-distance target tail does not need reference points beyond z_s.
The fitted masses and denominator must come from identical reference samples.
Users must keep the link smearing and lattice operator identical; ensemble
identity and spacing alone cannot establish operator equivalence.

The GPD application requires `normalization=false` so H/E local form factors
are retained. For PDFs, `normalization=true` optionally normalizes the final
hybrid result to one at the origin. The fit itself uses `normalization=false`.

The fit exports `output.nc` with one fitted parameter vector per input replica
and a `parameter` axis:
`delta_m_gev`, `m0_gev`, `total_mass_gev`, `B`, `c`. This is a mass artifact,
not a multi-spacing zR artifact. The apply job consumes it as `inputs.masses`.
Shared sample lineage is retained, including independent jackknife-group
alignment. Fits use full covariance regardless of the plotting error mode.

## Perturbative coefficient

`kernel_id="z_msbar_pdf_nlo"` reuses the existing NLO coefficient;
`z_msbar_pdf_nnlo` adds Eq. (5) with three-loop threshold-matched running and
explicit `kernel_parameters.n_f` (default 3). The kernel must be an unpolarized
PDF coordinate-space coefficient. Its scale and parameters are recorded in the
mass artifact and inherited by application jobs; they cannot be silently changed
at application. Scale/order variations require separate fit-and-apply jobs.

These built-in coordinate coefficients are fixed order. The mass-fitting method
does not itself perform RGR/LRR. A resummed C0 must be implemented as an explicit
`z_msbar_pdf_*` coordinate kernel and used consistently with the downstream
matching prescription; selecting a resummed momentum-space matching kernel alone
does not resum the upstream mass subtraction.

## Manifest example

Insert the `renormalization` entry below under the manifest's top-level `stages`
object, after the jobs `ca_p0` and `ca_target` that provide the sample-bearing
inputs described above. This is a stage fragment, not a standalone manifest;
retain the package's required run-wide `metadata`. The distances below illustrate the paper's
choice; adjust to your grid and perturbative window.

```json
{
  "renormalization": {
    "defaults": {
      "strategy": "single_spacing_self_renormalization",
      "scheme": "hybrid",
      "normalization": false
    },
    "jobs": [
      {
        "id": "single_a_masses",
        "type": "fit",
        "inputs": {"reference": "ca_p0"},
        "kernel_id": "z_msbar_pdf_nnlo",
        "kernel_parameters": {"n_f": 3},
        "mu": 2.0,
        "long_distance_window_fm": [0.54, 1.53],
        "short_distance_points_fm": [0.16, 0.18, 0.20],
        "scan_centers_fm": [0.14, 0.16, 0.18],
        "scan_half_width_fm": 0.02,
        "svdcut": 1e-10
      },
      {
        "id": "single_a_target",
        "type": "apply",
        "inputs": {
          "target": "ca_target",
          "denominator": "ca_p0",
          "masses": "single_a_masses"
        },
        "zs_fm": 0.27
      }
    ]
  }
}
```

Diagnostics include both fit qualities, covariance rank, all masses with
statistical errors, selected windows, and correlated local m0/total-mass scans.
They do not include a continuum extrapolation or a systematic error estimate
from fit-window, scale, perturbative-order, or interpolation variations.

## Package integration

Run this workflow through the normal manifest validation and execution commands
in the [package README](../README.md). The fit and apply jobs run deterministically;
no LLM selects a plateau. Unlike the multi-spacing `self_renormalization` fit,
this strategy performs covariance-weighted fits without Gaussian priors and
accepts no `priors`, `LambdaQCD_gev`, `d`, or `zR` fields.

Only the apply job's `output.nc` is a coordinate-space matrix element suitable
for the downstream Fourier stage. The fit's parameter-axis `output.nc` is an
intermediate mass artifact. Use the usual Fourier and matching jobs with a
compatible hybrid prescription; this stage does not select or implement new
momentum-space matching kernels. Its fixed-order mass subtraction alone does
not provide a consistent upstream RGR/LRR prescription.

This guide is available in the repository and source distribution. Installed
wheels contain the implementation and paired NNLO coefficient document; the
repository-level guide and digitization benchmark are not wheel resources.
The [paper benchmark](../benchmarks/paper2312/README.md) is run from a repository
checkout with the package installed and is separate from production sample fits.
