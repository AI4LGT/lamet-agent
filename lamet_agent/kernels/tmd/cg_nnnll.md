# NNNLL form-factor Sudakov coefficient

Select `order="nnnll"` in `coefficients.form_factor_kernel`. This implements
N3LL for the spacelike coefficient entering $H_F$, at fixed $n_f=3$:

$$H_F(x,y;P,\mu)=C_V(Q^2,\mu)C_V(\bar Q^2,\mu),\qquad
Q=2P\sqrt{xy},\quad \bar Q=2P\sqrt{(1-x)(1-y)}.$$

The amplitude coefficient $C_V$ is not squared separately. With
$a(u)=\alpha_s(u)/(4\pi)$,

The formal RG solution is

$$C_V(Q^2,\mu)=[1+a(Q)c_1+a(Q)^2c_2]
\exp\left\{\int_Q^\mu\frac{du}{u}
\left[\Gamma_{\rm cusp}(a(u))\ln\frac{Q^2}{u^2}+\gamma_V(a(u))\right]\right\}.$$

Here $\Gamma_{\rm cusp}=\sum_{n=0}^3\Gamma_n a^{n+1}$,
$\gamma_V=\sum_{n=0}^2\gamma_n^V a^{n+1}$ and
$da/d\ln u=-2\sum_{n=0}^3\beta_n a^{n+2}$.
The finite boundary is multiplied as a two-loop polynomial, not exponentiated:

$$c_1=C_F(-8+\pi^2/6),$$
$$\begin{aligned}
c_2={}&C_F^2\left(\frac{255}{8}+\frac{7\pi^2}{2}
-\frac{83\pi^4}{360}-30\zeta_3\right)\\
&+C_FC_A\left(-\frac{51157}{648}-\frac{337\pi^2}{108}
+\frac{11\pi^4}{45}+\frac{313\zeta_3}{9}\right)\\
&+C_FT_Fn_f\left(\frac{4085}{162}+\frac{23\pi^2}{27}+\frac{4\zeta_3}{9}\right).
\end{aligned}$$

Sources:

- [Becher, Neubert and Pecjak, hep-ph/0607228](https://arxiv.org/abs/hep-ph/0607228),
  Eqs. (36), (50), (51), (95), (96): RGE, matching boundary and anomalous dimensions.
- [von Manteuffel, Panzer and Schabinger, 2002.04617](https://arxiv.org/abs/2002.04617),
  Eq. (6): exact four-loop cusp, including both quartic color invariants.
  Their $\Gamma_4^q$ is this module's $\Gamma_3$; no Pade estimate is used.
- [van Ritbergen, Vermaseren and Larin, hep-ph/9701390](https://arxiv.org/abs/hep-ph/9701390),
  Eq. (11): four-loop beta function in MSbar.
- [2504.04625v5](https://arxiv.org/abs/2504.04625v5), Eq. (18): the $H_F$ product and scales.

For $n_f=3$, useful checks are $\Gamma_3=7035.152973936289$,
$\gamma_2^V=-738.256293050813$, $\beta_3=12090.3781308037$,
and $c_2=-94.1497083488582$.

## Numerical prescription and scope

`alpha_s` is the coupling at `mu`, and all scales are in GeV. The
four-loop beta function is solved numerically in log scale at fixed $n_f=3$.
The evolution integrals are expanded using hep-ph/0607228 (93),(94).
Write $2S-a_{\gamma_V}=E_0+E_1+E_2$, where $E_0$ contains the leading
$1/a$ and constant terms, and $E_1,E_2$ have RG orders $a,a^2$. Then

$$C_V=e^{E_0}(1+p_1+p_2),\qquad
p_1=E_1+a_Qc_1,\qquad
p_2=E_2+\tfrac12 E_1^2+a_Q^2c_2+E_1a_Qc_1.$$

The product is also truncated consistently:

$$H_F=e^{E_0(Q)+E_0(\bar Q)}
[1+p_1(Q)+p_1(\bar Q)+p_2(Q)+p_2(\bar Q)+p_1(Q)p_1(\bar Q)].$$

Thus, multiplying two `sudakov_factor` values without truncation retains
higher-order terms and is not exactly the `form_factor_kernel` prescription.
At $Q=\bar Q=\mu$, the latter is
$1+2ac_1+a^2(c_1^2+2c_2)$, rather than the unexpanded square of a polynomial.
The leading Sudakov factor stays exponentiated; this is RG-improved
perturbation theory, not fixed-order expansion of the large logarithms.

The default `infrared_policy="error"` rejects scales at or below the pole of
this four-loop beta function. For `mu=2`, `alpha_s=0.293`, that pole is about
0.586146 GeV. An explicit `infrared_policy="freeze", freeze_scale_gev=1.0`
floors the natural scale, its logarithms, and its matching boundary. The floor
must be above the pole and at or below `mu`. This is a model-dependent infrared
completion, not perturbative control near the endpoint. `reference_quad` is
not supported. No flavor thresholds are crossed in this fixed-flavor scheme.

The NNNLL selector applies only to the standalone $H_F$ API. It does not
upgrade the separate $H_\phi$, $H_f$, CS kernel, or the full soft-analysis
pipeline to NNNLL. The existing pipeline order controls remain unchanged.

```python
import numpy as np
from lamet_agent.kernels.tmd.coefficients import form_factor_kernel

x = np.linspace(0.2, 0.8, 61)
H_F = form_factor_kernel(x[:, None], x[None, :], momentum_gev=2.58,
                         mu=2.0, alpha_s=0.293, order="nnnll")
```

Tests compare independent implicit-beta quadrature, the complete published
two-loop expansion, RG scale variation, product normalization, broadcasting,
symmetries, zero coupling, pole rejection and the explicit scale floor.

## Historical DIS validation

`examples/validate_dis_nnnll.py` uses the production hard-evolution function
inside the DIS formula (63). Twenty-four vector vertices from Figs. 5 and 6
cover NLL, NNLL, and NNNLL at Q=5 and 30 GeV. The test tolerance is 0.005%
relative, reflecting PDF-coordinate precision, not theoretical uncertainty.
The benchmark uses fixed nf=5, the historical Pade cusp, a four-loop
inverse-log coupling, and alpha_s(MZ)=0.1187 inferred from the fixed-order
curves. It consistently expands the gamma-function and matching factors as
well. These historical inputs are not defaults for the TMD API. Reproducing
the curves validates the common hard evolution; it does not validate an
NNNLL DIS jet as a TMD matching coefficient.
