# quark_pdf_gi_gt_hybrid_nnlo_rgr_tr

GI gamma-t hybrid unpolarized nonsinglet PDF with NNLO matching, coupled NNLO
DGLAP RGR, and the supplied notebook's NNLO threshold prescription. Select
`order: "nnlo"`, `resummation: "rgr_tr"`. Both quark and antiquark sectors are
retained, with f(-x)=-qbar(x), and independent matching at each output magnitude.
This does not include LRR, singlet/gluon mixing, or total-valence sea evolution.

## Scientific scope: finite-cut matching limitation

This implements the notebook prescription, not a certification of exact NNLO
fixed-order accuracy at finite threshold cut. Its separate cut factors leave
an O(alpha_s^2) contribution in the TR correction even when the hard/soft
ingredients are NNLO. A consistent fixed-nf weak-coupling diagnostic with a
smooth Gaussian, Pz=2, mu_h=2.8, mu_i=1.2, x=.7, and gamma=.4 gives a nonzero
correction/alpha_s(mu_h)^2 near .224 under refinement. The residual varies
with gamma and does not disappear as grid spacing is reduced. It is distinct
from the previously diagnosed pointwise-cut oscillation. The implementation
preserves this reference prescription; no additional subtraction was invented.
Do not claim that TR first enters at O(alpha_s^3) in this finite-cut operator.

The historical NLO selector remains unchanged. Comparisons across NLO/NNLO must
align cut quadrature before attributing differences to perturbative order.

## Reference and matrix order

`Proton_PDF.nb`, supplied To_Jinxin.rar, 469 input cells, SHA256
`eee85fd321d19079d8db3c0e2eb39e9622066df1cf4187dc78082aaa036ffbca`.
Zero-based cells 36/39 give HNNLO/ANNLO; 42 builds the hard matrix; 43 gives
JcoeNNLO; 51–53 give SNNLO/aGammaNNLO/aJNNLO; 54/56 build soft evolution.
The TR sections compose

    M_NNLO(mu_h) + [(cut*P(mu_h,mu_i)) @ (cut*J_NNLO(mu_i))
                   - cut*J_NNLO(mu_h)] @ (cut*H_NNLO(mu_h)).

`*` denotes elementwise cut weights and `@` denotes matrix multiplication.
Diagonals and finite-domain tails are assembled before applying cuts. Do not
force column normalization afterwards. The full smooth hybrid conversion in
the package's NNLO fixed matching and inverse I-C1+C1@C1-C2 are retained.
The hard modulus includes the notebook's two-loop hybrid normalization, and
the sine phase/cosine diagonal are left unexpanded as in the reference.
The NNLO soft polynomial is 1-alpha_s*J1+alpha_s^2*(J1^2-J2), degree four.

The star function uses the NLO implementation's exact eta=0 limit. Its first
four derivatives are evaluated analytically using the series of
sin(pi*eta/2)*Gamma(1-eta)*exp(-EulerGamma*eta)/pi, including analytic tails.
This replaces the notebook's eta0=.01/five-point finite differences; it is
not equality with its finite-regulator output.

Evolution uses the notebook's four-loop cusp polynomial, three-loop soft
anomalous dimension, and expanded four-loop-beta Sudakov expression, with its
published decimal coefficients. The Sudakov expression is evaluated by
integrating its truncated Laurent series in alpha_i through alpha_i^2, which
avoids the very long expanded polynomial. It is not untruncated numerical RG
evolution. Couplings are the existing package three-loop running values;
fixed nf=3 coefficient functions versus threshold-matched running are retained
from the reference. See [arXiv:2410.12910v2](https://arxiv.org/html/2410.12910v2),
especially the factorized cut prescription and matching discussion.

## Cut quadrature and controls

New control `cut_quadrature="fractional"` (default) uses the fraction of each
source cell inside |1-x/y|<gamma, retaining a sharp boundary. For a row x,
the allowed source interval has endpoints x/(1+gamma), x/(1-gamma), sorted
for negative x. This is partial-cell integration of the cut with center-sampled
coefficients, not smoothing of the physical cut. Distributional diagonal
weights remain one. `cut_quadrature="point"` recovers the NLO implementation's
strict pointwise mask, excluding ties within 32 machine epsilons, for reference
comparisons. Fractional weights address boundary sampling noise, not the
finite-cut O(alpha_s^2) matching limitation above.

Required: `zs_fm`. Defaults: `kappa=1`, `spectator_kappa=1`, `threshold_cut=.4`,
`mu_min_gev=.6`, `evolution_points=100`, `evolution_steps=40`, `y_max=100`,
`quadrature_order=64`, `eps=1e-12`.
The independent scales are mu_h=2*kappa*|x|*Pz and
mu_i=2*spectator_kappa*(1-|x|)*Pz. TR is off exactly for mu_i>=mu_h.
Hard scales below mu_min retain existing RGR zero rows; active spectator scales
below mu_min raise an error. Do not include |x|=1. The star prescription
requires -1<eta<1. Input is uniform, increasing, nonzero, and spans [-1,1];
output may be nonuniform. Both scale multipliers must be positive and 0<gamma<1.

Independent matching and linear transfer to the signed DGLAP grid are unchanged.
No recursive notebook reuse is introduced. Numerical grid, domain, cut, and
scale dependence must be assessed for each physical analysis. Kernel tests and
workflow completion do not establish precision convergence.
