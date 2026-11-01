# quark_pdf_gi_gt_hybrid_nlo_rgr

Unpolarized flavor-difference nonsinglet matching with independent matching
at each intrinsic scale mu0 = 2 kappa |x| Pz, followed by coupled evolution
to scale_gev. This accepts both quark and antiquark content; it does not
implement singlet/gluon mixing, total flavor-summed valence evolution,
helicity or transversity evolution.

Matching is computed on the quasi input grid as a square matrix, including
the full subtraction and (at NNLO) the inverse-convolution term. The complete
matched matrix is then linearly interpolated onto the signed evolution grid.
This preserves the subtraction when input and evolution nodes do not coincide.

The internal state is [q_i-q_j, antiquark_i-antiquark_j] on the uniform grid
x = 1/N, ..., 1, with N = evolution_points. The signed PDF convention is
f(-x) = -antiquark(x). The qq and q-antiquark blocks are respectively
(P_ns_plus + P_ns_minus)/2 and (P_ns_plus - P_ns_minus)/2.
The splitting matrices use nf=3 and the diagonal/tail prescription of cell 30
of the supplied Proton_PDF.nb (To_Jinxin archive), with tail cutoff y_max.
Evolution uses simultaneous ordered midpoint matrix exponentials in ln(mu^2),
with evolution_steps steps. NLO retains a P0 + a^2 P1; NNLO adds a^3 P2,
where a = alpha_s/(4 pi), using the repository's threshold-matched running
coupling at two or three loops respectively. Splitting nf remains fixed at 3;
this is the package convention, not a variable-flavor splitting implementation.

x_in must be increasing, uniformly spaced, nonzero and cover [-1,1].
x_out may be unordered and nonuniform within [-1,1]. Output uses linear
interpolation on the internal grid. Active |x| below 1/N is rejected.
x=0 and rows with mu0 below mu_min_gev are zero on both sides.
The rectangular matrix acts linearly on real or complex input samples.
Grid size, evolution steps, tail extent, and matching quadrature require
convergence checks for production analysis. No finite-grid sum-rule claim is made.

The fixed-order matching formula and its prescription are documented in
[quark_pdf_gi_gt_hybrid_nlo](quark_pdf_gi_gt_hybrid_nlo.md).
Legacy suffixed RGR kernels remain available for reproducibility.
