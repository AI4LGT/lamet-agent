# quark_pdf_cg_gt_hybrid_nlo_rgr_re

## Fixed-order input

The local `_fixed_order` function builds the NLO inverse matching matrix
using the existing coefficient and plus-prescription implementation.
See [quark_pdf_cg_gt_hybrid_nlo](quark_pdf_cg_gt_hybrid_nlo.md) for the corresponding fixed-order channel.
The RGR update does not change those coefficients or their finite-grid subtraction.
The two-loop evolution coefficient is `_p_nlo_valence`; the shared LO coefficient
is `_p_qq_lo`.

## Independent-scale RGR

For each output point, set $\mu_i=2\kappa x_i P_z$. Build the full
fixed-order matrix $B(\mu_i)$ independently at that scale, evolve it to
$\mu$, then select row $i$:

$$M_{ij}(\mu)=Z_i[U(\mu,\mu_i)B(\mu_i)]_{ij}.$$

Here $Z_i$ is 1. Each row is matched independently; higher-$x$
rows from previous matching steps are not carried into subsequent steps.
This is the package's chosen prescription, not the recursive construction
in [arXiv:2209.01236, Appendix D](https://arxiv.org/html/2209.01236v2#A4).
That paper motivates the intrinsic scale and ordered evolution, but is not
an exact validation of this Coulomb-gauge channel or its coefficients.

For column-vector PDFs, with $t=\ln\mu^2$ and $a=\alpha_s/(4\pi)$,

$$\frac{dU}{dt}=[a(t)D_0+a(t)^2D_1]U,\qquad U(t_i,t_i)=I.$$

The implementation uses 20 midpoint exponential steps. Each new step
multiplies on the left, so the result has the correct scale ordering for
both upward and downward evolution. This is a finite-step approximation.
The existing uniform-grid column-sum subtraction in $D_0,D_1$ is retained;
this update does not establish continuum or grid/range convergence.

## Domain and controls

`momentum_gev`, `scale_gev`, `kappa`, and `mu_min_gev` must be finite
and positive. The output grid must be a nonempty finite 1D array,
increasing and uniformly spaced when it contains more than one point.
The input grid retains the fixed-order builder's requirements.

Rows with $\mu_i<\mu_{\min}$ are zero. With positive momentum and
$\kappa$, this also zeros nonpositive-$x$ rows; it is not a prescription
for independently reconstructed negative-$x$ PDFs. The defaults are
`kappa=1` and `mu_min_gev=0.6`. The cutoff does not itself certify
perturbative reliability at every retained scale.

The regression checks compare the evolution and independent matching
against a high-accuracy ODE solution for the same discrete generators.
They do not establish agreement with a physical data set or the reference
notebook's different coupling, subtraction, and matching choices.
