# Unpolarized straight-link Wilson coefficient at NNLO

`z_msbar_pdf_nnlo` evaluates Eq. (5) of
[Holligan and Lin, arXiv:2312.10829](https://arxiv.org/abs/2312.10829).
For nonzero physical separation in fm, define
\(L=\ln[(z/\hbar c)^2\mu^2e^{2\gamma_E}/4]\) and \(a_s=\alpha_s(\mu)/(2\pi)\).

\[
C_0=1+a_s C_F(3L/2+5/2)
+a_s^2[(15/2-n_f/3)L^2+(37.1731-5n_f/3)L+51.836-4.34259n_f].
\]

The decimal coefficients follow the paper. `n_f=3` by default controls the
explicit loop coefficients; the coupling uses the project's three-loop,
threshold-matched running convention. This is fixed order, without RGR or LRR.
It applies to the gauge-invariant gamma-t unpolarized quark PDF reference,
including when that reference renormalizes collinear GPD targets. It is not
a GPD light-cone matching kernel or a gluon/DA coefficient.
