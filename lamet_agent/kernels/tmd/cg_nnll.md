# NNLL Sudakov coefficient

`cg_nnll.py` extends the existing NLL form-factor kernel with the same
`sudakov_factor` and `form_factor_kernel` arguments, defaults and broadcasting.
The shared entry point accepts `form_factor_kernel(..., order="nnll")`.

The main reference is Collins and Rogers,
[arXiv:1705.07167, Sec. VI.2, Eqs. (58)-(60)](https://arxiv.org/html/1705.07167#S6.SS2).
Our $C_{\rm Sud}$ is their **spacelike hard factor** $H^{\rm Sud}$, rather than
their collinear factor $C^{\rm Sud}$. With $a=\alpha_s/(4\pi)$, use
$\Gamma_{\rm cusp}=\gamma_K/2$ and $\gamma_V=-\gamma_j$, so that

$$
C_{\rm Sud}(Q,\mu)=
\left[1+a(Q)C_F\left(-8+\frac{\pi^2}{6}\right)\right]
\exp\left\{\int_Q^\mu\frac{du}{u}
\left[-\Gamma_{\rm cusp}(a(u))\log\frac{u^2}{Q^2}+\gamma_V(a(u))\right]\right\}.
$$

NNLL uses three-loop cusp, two-loop non-cusp, three-loop running, and the
one-loop boundary; see [arXiv:1002.2213, Table 1](https://arxiv.org/html/1002.2213#S4.T1).
The finite boundary is multiplied, not exponentiated. At $Q=\mu$ it is
$1+a(\mu)C_F(-8+\pi^2/6)$, whereas the existing NLL boundary remains one.
Expanding through $O(\alpha_s)$ recovers the complete NLO coefficient;
at $O(\alpha_s^2)$ it recovers Eq. (60)'s logarithms, but not its two-loop
boundary constant. The result is therefore NNLL, not NNLO fixed-order matching.

As in [arXiv:2504.04625v5, Eq. (18)](https://arxiv.org/html/2504.04625v5),
$H_F(x,y)=C_{\rm Sud}(2P\sqrt{xy})C_{\rm Sud}(2P\sqrt{(1-x)(1-y)})$.
The product keeps higher-order cross terms, as does the existing resummed kernel.

The evolution uses one-dimensional numerical quadrature in $\log u$ with
`epsabs=epsrel=1e-10`; the anomalous dimensions and coupling are analytic
expressions. The coupling is the existing $n_f=3$ analytic three-loop
approximation (arXiv:1002.2213, Eq. (155)), anchored at the supplied
`alpha_s = alpha_s(mu)`. The `error` and `freeze` policies follow
[cg_nll.md](cg_nll.md), including flooring the boundary scale and logarithms.
`reference_quad` is an NLL reproduction option and is rejected for NNLL.
The finite matching boundary retains its sign if evaluated at strong coupling.

This extension covers $H_F$ only. The separate TMDWF/TMDPDF hard factors
$H_\phi$ and $H_f$ retain their existing NLO/NLL choices; the complete
soft-function extraction is not promoted to NNLL by this kernel alone.
