[quark_gpd_gi_gt_hybrid_nlo.md](https://github.com/user-attachments/files/33740623/quark_gpd_gi_gt_hybrid_nlo.md)

<!-- lamet-agent formula cache; kernel=quark_gpd_gi_gt_hybrid_nlo; scheme=hybrid; order=NLO -->
# GI gamma^t unpolarized quark GPD hybrid-NLO kernel

This implementation retains the project's two-variable hybrid coefficient and
coupling convention, and applies the ordered regulator prescription in
[arXiv:2609.20545v1, Eqs. (9)-(10)](https://arxiv.org/html/2609.20545v1).
The paper supplies the prescription, not a new derivation of the hybrid term.

## Matrix contract and numerical prescription

`kernel(x_out, x_in, *, momentum_gev, scale_gev, zs_fm, skewness, ...)`
returns a matrix with shape `(len(x_out), len(x_in))`, acting as
`lightcone = matrix @ quasi`. The public filename and required arguments are
unchanged. The matching stage supplies momentum and the hybrid switching
distance from upstream attributes; `skewness` and the optional numerical
controls are explicit `kernel_parameters`.

Let c(x,y) denote the original off-diagonal coefficient, **including** its
alpha_s C_F/(4 pi) factor, with the inverse-matching sign applied below.
Let B_j(y) be the piecewise-linear interpolation basis on the strictly
increasing, possibly nonuniform input grid. The implementation evaluates

\[
M_{ij}=B_j(x_i)-\int_{a}^{b}dy\,\mathbf{1}_{|y-x_i|\geq\epsilon_2}
\left[c(x_i,y)B_j(y)-c(y,x_i)B_j(x_i)\right],
\quad a=y_0,\quad b=y_{N-1}.
\]

Real and virtual contributions are integrated together, using the same
cutoff and integration window. Adaptive vector quadrature splits at input
knots and y=+/-xi, and ends exactly at x_i+/-epsilon2. Thus it resolves the
cutoff independently of sample spacing. No global `dy` or output-column sum
is used. Adding, removing, or reordering other output points does not alter
a given output row.

The x_in values may include +/-xi: the apparent coefficient poles there are
combined into divided differences with removable limits. No coefficient is
evaluated at the excluded diagonal. This rearrangement preserves the
original formula away from the singular points.

## Cutoff hierarchy

Define
\[
d_{\min}=\min_i\min(|x_i-\xi|,|x_i+\xi|).
\]
Require `epsilon2 <= hierarchy_ratio * d_min`, with
`0 < hierarchy_ratio <= 0.01`. Output points exactly at +/-xi are rejected.
Their values must be obtained from one-sided limits after cutoff convergence;
the code does not extrapolate them or impose continuity or equal derivatives.

| Control | Default | Meaning |
| --- | --- | --- |
| `eps` | 1e-12 | Near-zero numerical tolerance inherited by analytic helpers |
| `epsilon2` | 1e-5 | Dimensionless exclusion radius around y=x |
| `hierarchy_ratio` | 0.01 | Maximum permitted epsilon2/d_min |
| `quad_epsabs` | 1e-8 | Absolute quadrature tolerance per row, split across both sides |
| `quad_epsrel` | 1e-6 | Relative quadrature tolerance on each side |

`epsilon2` must exceed `100*eps`. The hierarchy guard is not a convergence
proof. Reduce epsilon2 at fixed output points, then move output points toward
+/-xi and repeat the cutoff scan. Independently refine input samples and widen
the input window. Both cutoff boundaries must lie inside the input range.

## Nonuniform input example

```python
import numpy as np
from lamet_agent.kernels.quark_gpd_gi_gt_hybrid_nlo import kernel

xi = 0.3
# Base spacing 0.02; spacing 0.002 within 0.06 of each crossover.
x_in = np.unique(np.concatenate([
    np.linspace(-2.0, 2.0, 201),
    -xi + np.linspace(-0.06, 0.06, 61),
    xi + np.linspace(-0.06, 0.06, 61),
]))
x_out = np.linspace(-0.8, 0.8, 41)  # Uniform; d_min = 0.02.
matrix = kernel(
    x_out, x_in, momentum_gev=2.0, scale_gev=2.0,
    zs_fm=0.3, skewness=xi, epsilon2=1e-5,
)
# Evaluate the Fourier transform at x_in to obtain quasi_on_x_in.
# lightcone = matrix @ quasi_on_x_in
```

Refinement width and spacing are caller choices; the kernel accepts arbitrary
strictly increasing grids and does not force a particular refinement pattern.
Interpolating a coarse quasi-GPD to more points does not add measured information.

For package execution, put `skewness` and `epsilon2` in `kernel_parameters`;
keep `zs_fm` in upstream hybrid-renormalization attributes. Existing calls remain
syntactically valid, but numerical results and grid requirements change.
SciPy is already a required package dependency.

## Scope and validation limits

This is a **finite-window** realization of the convolution. Both integrals
are truncated to [a,b]; no analytic virtual tail or quasi-GPD extrapolation is
added. The paper's Eq. (9) integrates over the full real line. Widening the
window and checking tail sensitivity are therefore essential before claiming
full-line accuracy. The kernel alone does not establish the paper's physics
computability conditions or supply perturbative/systematic uncertainties.

Tests cover the original coefficient at nonsingular points, independent output
rows, invariance under inserting collinear interpolation knots, comparison to a
scalar convolution, regulator convergence for synthetic data, crossover input
knots, and invalid grids/hierarchies. These are numerical regression checks;
a comparison against 202 saved values from the author's example notebook found
maximum absolute / RMS differences of 1.68e-2 / 5.14e-3 for this kernel with
its cutoff reduced to 5e-7. The notebook itself uses cutoff 1e-5 and output
points 1e-4 from the crossovers, which this implementation's 0.01 hierarchy
guard rejects. Its other conventions differ: fixed alpha_s=0.3, cubic
interpolation, a shifted virtual window [x-2,x+2], and hbar*c=0.197 GeV fm.
A separate Python calculation using those notebook conventions agreed with
the saved output to 2.48e-5 maximum absolute error (RMS 2.44e-6), using SciPy
cubic interpolation rather than claiming an identical Mathematica rerun.
This is not an exact reproduction by the delivered kernel. Alignment of
these conventions and full-line tail validation remain open.

Adaptive matrix construction costs more than the previous point-sampling sum.
Build once per set of kinematics and reuse the matrix across resampled data.
