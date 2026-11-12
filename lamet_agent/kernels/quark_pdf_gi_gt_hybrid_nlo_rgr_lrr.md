# NLO RGR and LRR for the GI gamma-t hybrid nonsinglet PDF

Kernel: `quark_pdf_gi_gt_hybrid_nlo_rgr_lrr`.

Select `order: "nlo"` and `resummation: "rgr_lrr"` in the matching stage.
The signed distribution satisfies f(-x)=-qbar(x); quark and antiquark flavor
 differences evolve together. This is unpolarized flavor-nonsinglet evolution,
not singlet/gluon or NNLO total-valence sea-term evolution.

Matching is independent at each intrinsic scale mu0=2*kappa*abs(x)*Pz.
At each scale the full LRR matrix is assembled on the square input grid and
linearly transferred to signed evolution nodes. The existing simultaneous
ordered evolution carries the state to scale_gev. No recursively matched
higher-x state is reused.

Let C1 and C2 be forward matching corrections including coupling factors,
S the finite-grid column-plus renormalon shape, r0=rnasym(0,1,mu0,3)*alpha_s,
r1=rnasym(1,1,mu0,3)*alpha_s^2 and R=dPVasym(1,mu0,3,alpha_s).

$$M_{NLO+LRR}=(I-C_1+r_0S)\exp(-RS),$$
$$M_{NNLO+LRR}=[I-C_1+C_1^2-C_2+(r_0+r_1)S-r_0C_1S
+\tfrac12r_0^2S^2]\exp(-RS).$$

The order of matrix products matters. Counterterms preserve the fixed-order
inverse through the selected order. NLO uses the existing two-loop coupling;
NNLO uses three-loop coupling and includes the full smooth hybrid conversion
and the continuous xi=-1 limit. Renormalon and splitting coefficients use nf=3;
coupling thresholds retain the existing package convention.

Controls: kappa=1, mu_min_gev=0.6, evolution_points=100, evolution_steps=40,
y_max=100, eps=1e-12, eps_m=0.005 GeV. NNLO also accepts quadrature_order=64,
with the existing convergence check. Positive finite zs_fm and eps_m are
required. Input nodes must be uniform, increasing, nonzero, and cover [-1,1].
Output nodes may be nonuniform within [-1,1]. Both signs use the same intrinsic
scale cutoff; points below it are returned as zero. A notebook grid formed by
omitting zero from an otherwise uniform grid is not supported.

The renormalon shape and principal-value prescription follow the building
blocks of [arXiv:2305.05212](https://arxiv.org/abs/2305.05212), Eqs. (12)-(17).
The matrix exponential and finite-grid subtraction retain package conventions;
grid and regulator convergence must be checked for a physical analysis.
Consistent upstream mass renormalization is required for a physical LRR result;
the matching selector does not refit the mass-renormalization parameter.

NLO preparation constructs the constant and log(mu^2/Pz^2) coefficient
matrices once, using vectorized evaluations of the existing regulated GI
gamma-t hybrid expression. The square-grid column-plus subtraction is formed
from these same entries. Both tables are reused at every intrinsic scale;
the fixed-order scalar builder remains available for numerical cross-checks.
No change to precision, independent matching, LRR or evolution is introduced.
