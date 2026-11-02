# NLO GI gamma-t hybrid nonsinglet PDF with RGR + TR

Kernel ID: `quark_pdf_gi_gt_hybrid_nlo_rgr_tr`.

Select `order: "nlo"`, `resummation: "rgr_tr"`. This kernel maps signed
quasi-PDF samples to an unpolarized flavor-difference PDF; negative x means
minus the antiquark distribution. Singlet/gluon mixing is not implemented.
For NNLO TR, select `order: "nnlo"` with `resummation: "rgr_tr"`;
for TR with LRR at either order, select `resummation: "rgr_lrr_tr"`.
Source components `re`, `im`, and `both` are supported.

## Reference and prescription

The reference is `Proton_PDF.nb` in the supplied `To_Jinxin.rar`, inspected
November 1, 2026 (469 input cells; notebook SHA256
`eee85fd321d19079d8db3c0e2eb39e9622066df1cf4187dc78082aaa036ffbca`).
Zero-based cells 35/37/42 define the NLO hybrid hard modulus, phase, and matrix;
43/45/47 define the inverse soft function; 48–50 and 54/56 define its evolution.
Cell 220 gives the cut matrix composition, also used in the later TR sections.
Use the plain NLO building blocks, without those sections' LRR additions.
See [Ji et al., arXiv:2410.12910v2](https://arxiv.org/html/2410.12910v2),
Eqs. (28)–(35), (44), and (48)–(51).

For each requested magnitude r, independently match at
`mu_h = 2*kappa*r*momentum_gev`, with spectator scale
`mu_i = 2*spectator_kappa*(1-r)*momentum_gev`. Below `mu_h <= mu_i`, use
the existing NLO+RGR kernel exactly. Otherwise the square matching matrix is

    M_NLO(mu_h) + [(cut*P(mu_h,mu_i)) @ (cut*J(mu_i))
                   - cut*J(mu_h)] @ (cut*H(mu_h)).

Here `*` is elementwise multiplication; `@` is matrix composition. The cut
is `abs(1-y_row/y_column) < threshold_cut` on each individual factor, after
assembling its diagonal and finite-domain tail. Values within 32 machine epsilons
of the strict cut boundary are excluded consistently on both signed sectors. Do not impose a new column-sum
subtraction after the cut. The equal-scale product subtraction retains the
notebook's higher-order completion and gives an exact TR-off limit.

The hard matrix contains both `cos(phase)` on the diagonal and the odd
`sin(phase)/(pi*(1-xi))` term. Its hybrid normalization uses zs in inverse GeV.
The even star distribution has an analytic finite-domain tail; it is not a
charge-conserving column-plus matrix. At eta=0 its evolution is exactly the
identity. For the NLO soft matrix we analytically evaluate
`[1-alpha_s*CF/(2*pi)*(2+pi^2/12-2*d_eta+2*d_eta^2)] B(eta)` at eta=0.
This replaces the notebook's eta0=0.01 and finite-difference step 0.001.
It removes the regulator error; it is not bitwise reproduction of its default
soft matrix. Grid-cell edges define the finite integration domain.

Evolution integrals follow notebook SNLO/aGammaNLO/aJNLO, with three-loop
cusp, two-loop soft anomalous dimension, and its expanded Sudakov integral.
They use nf=3 coefficients and the package's existing two-loop running coupling,
including its flavor-threshold convention. This mixed fixed-nf coefficient /
threshold-matched coupling convention is inherited, not an exact fixed-nf RG
solution across heavy-flavor thresholds. NLO matching order is distinct from
threshold-log accuracy. No independent certification of full NNLL accuracy is
claimed by matrix equivalence tests.

After square matching, transfer to the existing signed DGLAP grid and evolve
to `scale_gev`, using the unchanged independent nonsinglet RGR machinery.
The notebook's recursive low-x reuse is not introduced. The full package NLO
hybrid coefficient and linear grid transfer are retained.

## Controls and domain

`zs_fm` is required. Defaults: `kappa=1`, `spectator_kappa=1`,
`threshold_cut=0.4`, `mu_min_gev=0.6`, `evolution_points=100`,
`evolution_steps=40`, `y_max=100`, `eps=1e-12`.
`0 < threshold_cut < 1`; all scale multipliers must be positive.
The input grid is finite, increasing, uniform, nonzero, and covers [-1,1].
Output nodes may be nonuniform within [-1,1]. Hard scales below `mu_min_gev`
retain the existing RGR zero-row behavior. Active TR rows with spectator scale
below `mu_min_gev` raise an error, rather than silently freezing the scale or
returning zeros. Thus omit |x|=1; at the defaults require
`abs(x) <= 1-mu_min_gev/(2*momentum_gev)` in the TR region.
The implemented star prescription additionally requires `-1 < eta < 1`.

Grid, domain, threshold-cut, and scale convergence must be checked for each
physical analysis. Completion and finite matrices do not certify convergence,
especially near the spectator infrared boundary.
