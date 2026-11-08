# Sudakov kernel for the intrinsic soft function

Use `lamet_agent.kernels.tmd.coefficients.form_factor_kernel` to evaluate
the dimensionless coefficient $H_F(x,y;P,\mu)$ for transverse vector or
axial-vector currents. The source is
[arXiv:2504.04625v5, Eq. (18) and Appendix A.2](https://arxiv.org/html/2504.04625v5#A1.SS2).
The NLO/NLL routines are taken from upstream lamet-agent commit `c55a373`.
The NNLL extension follows Collins and Rogers,
[arXiv:1705.07167, Eqs. (58)-(60)](https://arxiv.org/html/1705.07167#S6.SS2).

For $0<x,y<1$ and positive $P,\mu$ in GeV, define

$$Q=2P\sqrt{xy},\qquad \bar Q=2P\sqrt{(1-x)(1-y)}.$$

The default `order="nlo"` uses

$$H_F^{\rm NLO}=1+\delta C_{\rm Sud}(Q)+\delta C_{\rm Sud}(\bar Q),$$

$$\delta C_{\rm Sud}(Q)=\frac{\alpha_s(\mu)C_F}{4\pi}
\left[-L^2-3L-8+\frac{\pi^2}{6}\right],\qquad
L=\log\frac{\mu^2}{Q^2},\quad C_F=\frac43.$$

`order="nll"` follows the paper's tree-boundary evolution convention:

$$H_F^{\rm NLL}=C_{\rm Sud}(Q)C_{\rm Sud}(\bar Q),\qquad
\log C_{\rm Sud}(Q)=\int_Q^\mu\frac{du}{u}
\frac{-C_F\alpha_s(u)}{4\pi}\left[4\log\frac{u^2}{Q^2}+6\right].$$

The coupling convention and infrared options are detailed in
[cg_nll.md](cg_nll.md). `alpha_s` specifies the coupling **at `mu`**.
The default NLL policy rejects scales at or below the coupling pole;
`infrared_policy="freeze", freeze_scale_gev=1.0` explicitly selects a scale
floor. NLO retains signed coefficients and is truncated before any
$O(\alpha_s^2)$ cross term. NLL uses a unit boundary at $Q=\mu$, so it does
not include the finite NLO boundary constant.

`order="nnll"` keeps the same product, scales and input conventions, using
three-loop cusp, two-loop non-cusp and the one-loop matching boundary.
It shares the NLL three-loop running coupling and `error`/`freeze` policies.
See [cg_nnll.md](cg_nnll.md) for the formula and normalization. This choice
extends only `form_factor_kernel`; TMDWF/TMDPDF `hard_factor` remains NLO/NLL.

## Evaluate a grid

```python
import numpy as np
from lamet_agent.kernels.tmd.coefficients import form_factor_kernel

x = np.linspace(0.2, 0.8, 61)
H = form_factor_kernel(
    x[:, None], x[None, :],
    momentum_gev=2.58, mu=2.0, alpha_s=0.293, order="nll",
)
assert H.shape == (61, 61)
```

Inputs follow NumPy broadcasting. Two equally sized one-dimensional arrays
produce pairwise values; the singleton axes above request an outer grid.
The result contains coefficient values only, without quadrature weights or
wave functions. For weights `w` and reduced wave function `phi` on the same
nodes, the denominator of the soft function is
`(w * phi.conj()) @ H @ (w * phi)`. Use the form-factor momentum for
`momentum_gev` and evolve the reduced wave functions to it beforehand.

Soft Analysis uses this coefficient in its intrinsic-soft integral. The
numerical API above is also callable independently of a manifest run.
