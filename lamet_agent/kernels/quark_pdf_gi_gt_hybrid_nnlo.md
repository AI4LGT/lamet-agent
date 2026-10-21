# Unpolarized nonsinglet quark PDF: GI gamma-t, hybrid, NNLO

This kernel maps a hybrid-renormalized quasi-PDF to the light-cone PDF through
fixed order alpha_s squared. It is for the flavor nonsinglet unpolarized
Wilson-line operator with gamma-t. The two-loop coefficient uses nf=3, as in
the existing coefficient implementation; the running coupling retains the
package's two-loop running and heavy-flavor threshold convention.

Sources: [Chen, Wang, Zhu, arXiv:2006.14825v4](https://arxiv.org/html/2006.14825v4),
including the supplemental regular MSbar coefficients, and
[Su et al., arXiv:2209.01236v2](https://arxiv.org/html/2209.01236v2),
Appendices A and C. The local proton sample notebooks supplied by the user
(`Proton_PDF_new/Sample_codes/To_Minhuan/Proton_PDF.nb` and `To_Jinxin/Proton_PDF.nb`)
provide a further implementation reference. Their two-loop color functions
agree with the pion reference used for transcription. Their inverse term is
`C2i = -C2 + C1.C1`. Their fixed-order conversion omits the smooth term retained here.

The analytic MSbar functions, hybrid-conversion functions, and public `kernel()`
are all defined together in `quark_pdf_gi_gt_hybrid_nnlo.py`.

## Direction and normalization

Write the forward discretized relation as

$$\widetilde f=(I+aK_1+a^2K_2)f,\qquad a=\alpha_s(\mu)/(2\pi).$$

The returned inverse is truncated consistently:

$$M=I-aK_1+a^2(K_1K_1-K_2).$$

The existing NLO kernel supplies the one-loop inverse without modification.
For different input/output grids the convolution is
$K_1^{\mathrm{out,in}}K_1^{\mathrm{in,in}}$, using the quasi grid as the
intermediate quadrature grid. The LO term uses the existing interpolation.
This is a finite-grid approximation; range and spacing convergence must be
checked for an analysis. Column-plus subtraction is performed on that same
input grid, preserving the package's hybrid prescription. On a square grid
both forward corrections have zero column sums.

## Full two-loop hybrid conversion

Use $\xi=x/y$, $b=|y|z_sP_z$, $d=1-\xi$,
$L_s=\ln(z_s^2\mu^2e^{2\gamma_E}/4)$ and
$S=2\mathrm{Si}(bd)/(\pi d)$. The code combines the analytic terms in
Eq. (41) before evaluation:

$$c_2^{\mathrm{hybrid}}=c_2^{\overline{\mathrm{MS}}}
+A\left[2L_s S-\frac{8b}{\pi}\Re\,{}_3F_3(1,1,1;2,2,2;ibd)\right]
+B S-C_F^2(\tfrac32 L_s+\tfrac52)c_1^{\mathrm{hybrid}}+c_{2,\mathrm{smooth}},$$

where $c_1^{\mathrm{hybrid}}$ is the existing coefficient with its overall
$C_F$ removed,

$$A=\frac{11C_AC_F}{8}-\frac{C_Fn_fT_F}{2}+\frac{9C_F^2}{8},$$
$$B=(\frac{53}{8}-\frac{\pi^2}{6})C_AC_F
 +(\frac{25}{8}+\frac{2\pi^2}{3})C_F^2-\frac52 C_Fn_fT_F.$$

All color factors are included in $c_2$. The last term of Eq. (41) is retained.
It is evaluated with the plus subtraction inside the integral, before any
numerical quadrature. Exponential substitutions for the Fourier variable
and $1-u$ give two Gauss-Laguerre integrals. A sine-product form avoids
subtracting almost equal cosines near $u=1$. Successive quadrature orders
must agree to absolute tolerance 1e-10 plus relative tolerance 1e-8;
nonconvergence through order 256 raises an error.

## Interface and limitations

Choose `order: nnlo`, with no resummation. Kernel selection accepts only
`quark/pdf/gi/gt/hybrid` for this order. Other channels and NNLO combinations
with RGR or LRR remain unsupported.

The interface retains `momentum_gev`, `scale_gev`, `zs_fm` and `eps`.
`quadrature_order` (default 64, integer 16 through 256) sets the initial smooth-term
quadrature order. Distances are converted from fm with the package constant.
Input grids remain uniform and exclude zero, as required by the existing
NLO builder. The removable regular-coefficient boundary at xi=-1 is evaluated
by symmetric limiting values. The xi=0 logarithm uses the declared epsilon
regulator; the plus singularity at xi=1 is handled by diagonal subtraction.
This does not supply small-x resummation or an error estimate for finite-grid
truncation. It is not an exact numerical reproduction of a full sample-notebook
analysis, which includes additional choices and resummations.
