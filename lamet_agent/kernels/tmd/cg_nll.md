# Coulomb-gauge TMD NLL coefficients

Source: arXiv:2504.04625v5, Appendix A. The NLL coefficients have a **tree-level
boundary**. They do not exponentiate the finite NLO hard coefficient. The
fixed-order implementation in `cg_nlo.py` is independent and unchanged.

For the single-leg coefficient, with $q=2xP$, use

$$
\frac{d\log C_{\rm TMD}}{d\log\mu}
=-\Gamma_{\rm cusp}\log\frac{\mu^2}{q^2}+\gamma_C,
\quad
\Gamma_0=2C_F,\quad
\Gamma_1=2C_F\left[(67/9-\pi^2/3)C_A-(20/9)T_FN_f\right],
\quad \gamma_{C,0}=-6C_F.
$$

`tmd_coefficient` implements the analytic NLL evolution used by the reference
repositories. `hard_factor` combines it as
$H_\phi=C(xP)C((1-x)P)$ and $H_f=C(xP)^2$.

The form-factor kernel is

$$
H_F(x_1,x_2;P)=C_{\rm Sud}(2P\sqrt{x_1x_2})
\,C_{\rm Sud}(2P\sqrt{(1-x_1)(1-x_2)}),
$$

and

$$
\log C_{\rm Sud}(Q,\mu)
=\int_Q^\mu\frac{du}{u}
\frac{-C_F\alpha_s(u)}{4\pi}\left[4\log(u^2/Q^2)+6\right].
$$

The supplied `alpha_s` means $\alpha_s(\mu)$, and the coupling follows the
reference's analytic **NNLO** running approximation with $N_f=3$, including
$\beta_0,\beta_1,\beta_2$. This is an explicit running-coupling convention for
reproduction; it does not promote the tree-boundary NLL coefficient to NNLL.
For $\mu=2$ GeV and $\alpha_s=.293$, its infrared pole is
**0.391910396244829 GeV**. The coupling is negative immediately below that pole.

Three infrared policies are deliberately distinct:

- `error` is the default. It rejects scales at/below the pole before quadrature.
- `freeze` requires `freeze_scale_gev` above the pole and at/below `mu`.
  Both hard and Sudakov factors use $Q_{\rm eff}=\max(Q,Q_{\rm floor})$ in
  the boundary, coupling and logarithms. The factors plateau below the floor.
  This is a stated model-dependent infrared completion; it changes neither
  hadron momentum nor Collins–Soper rapidity evolution. The floor is recorded
  in coefficient provenance, and `scale_diagnostics` reports affected nodes.
- `reference_quad` reproduces the historical Sudakov calculation with
  `scipy.integrate.quad`, integration variable $u$ and `epsrel=1e-3`. It can
  return a finite number while crossing the pole, but this **does not define a
  convergent controlled integral**. Every unique evaluation records the pole
  crossing, warnings and estimated integration error in
  `reference_quad_diagnostics()`. Nonfinite/nonpositive coefficients still
  fail, and hard coefficients still reject nonpositive couplings.

The scalar Sudakov integrations are cached and repeated/vectorized coordinates
reuse them. The controlled policies integrate in $\log u$ with tight tolerance;
`reference_quad` intentionally retains the old variable and tolerance for
diagnosis. Errors from that prescription are separate from the statistical
uncertainty carried by `EnsembleData` replicas.
