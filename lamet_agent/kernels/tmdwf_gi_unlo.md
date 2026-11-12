# TMDWF finite-bT uNLO hard coefficient

## Source and normalization

Reference: Avkhadiev et al., Phys. Rev. D **108**, 114505,
[arXiv:2307.12359v2, Appendix C](https://arxiv.org/html/2307.12359v2),
Eqs. C3, C4, C42 and C43 (printed PDF numbering).
The supplied `TMDWF_hard_kernel.m` implements this in its final block.

**Default `convention="author"`:** the two Meijer-G terms carry `sqrt(pi)`.
This follows the supplied Mathematica code and the user's report on 2026-11-10
that the original author confirmed this correction. This private confirmation
has not been independently verified by this implementation.
The printed C43 has `1/sqrt(pi)`; `convention="printed"` is available only for
comparison. The correction multiplies the Meijer-G part by pi; it does **not**
multiply the logarithmic or imaginary terms by pi.

## Formula and units

Inputs: bT in fm; parton pz, hadron Pz, and mu in GeV; alpha_s dimensionless.
Use hbar*c = 0.1973269631 GeV fm, as in the supplied `.m`.
Define r = pz*bT/(hbar*c), z = r^2/4, b0 = 2 exp(-EulerGamma),
Lb = log[(bT*mu/(hbar*c*b0))^2], and Lz = log[(2*pz/mu)^2].

\[
C_\phi^{\pm u(1)} = C_F\left\{
 A\left[\frac{G_{2,4}^{2,2}(z\mid 3/2,3/2;3/2,3/2,0,1/2)}{r}
 -\frac r2 G_{3,5}^{2,3}(z\mid 1/2,1/2,1/2;1/2,1/2,-1/2,-1/2,0)\right]
 +\frac12 L_b^2-L_b(1-L_z)+\frac{\pi^2}{12}
 \pm i\pi\left[2\operatorname{Ei}(-r)-\frac{1-e^{-r}}r-L_z+1\right]\right\}.
\]

Here CF=4/3; A=sqrt(pi) by default, A=1/sqrt(pi) in printed mode.
`c_phi_one_loop` returns this coefficient including CF, excluding alpha_s/(4*pi).
`sign` is explicitly +1 or -1: it selects the Wilson-line orientation, **not**
the gtg5/gzg5 source channel. Do not infer it from the source operator.

With a_s=alpha_s/(4*pi), define S(x,P)=C1(xP)+C1((1-x)P).

| Prescription | C_phi | H | inverse hard factor |
|---|---|---|---|
| `exponentiated` (default, supplied .m) | exp(a_s C1) | exp(a_s S) | exp(-a_s S) |
| `strict_nlo` | 1+a_s C1 | 1+a_s S | 1-a_s S |

Exponentiating this finite one-loop term is not an implemented RGR/NLL scheme.
The two prescriptions agree through O(alpha_s) and differ beyond that order.
For strict NLO, the O(alpha_s^2) product of the two one-loop coefficients is dropped.
alpha_s is an explicit input evaluated at mu; this module does not run the coupling.
The `.m` example uses alpha_s=0.293 at mu=2 GeV.

## Three different outputs

1. `c_phi`: single-leg matching coefficient.
2. `hard_factor`: two-leg H in the diagonal approximation of C42, neglecting
   the nonlocal finite-bT remainder delta H in C41.
3. `delta_gamma`: additive perturbative correction to a CS kernel extracted
   from quasi-TMDWF ratios at two different positive momenta:

\[
\delta\gamma^{uNLO}(x,b_T;P_1,P_2)
=-\frac{a_s[S(x,P_1)-S(x,P_2)]}{\ln(P_1/P_2)}.
\]

The measured log(quasi(P1)/quasi(P2))/log(P1/P2) must be supplied separately.
The function returns the complex correction; it does not discard its imaginary part.
Default `logarithm="expanded"` evaluates the logarithm through O(alpha_s), without
branch wrapping. `logarithm="principal"` reproduces the two separate principal
`Log[Cphi(P1)/Cphi(P2)]` expressions of `.m` with exponentiated Cphi. These agree
when neither leg crosses the complex logarithm branch cut.

## Existing agent interface

This `.py`/`.md` pair is discovered by `list_kernel_ids()` and can be loaded with
`load_kernel("tmdwf_gi_unlo")`. Its `kernel(x_out,x_in,*,momentum_gev,scale_gev,...)`
has the existing matrix interface, so the existing
`perturbative_matching.physics.apply_matrix` can apply it to all bootstrap samples.
It is a diagonal inverse-H operator, with one nonzero entry per retained x.
There are no integration weights or plus prescriptions. x_out must be an exact,
increasing subset of x_in strictly inside (0,1); out-of-support input points are
not used. No endpoint clipping or interpolation is performed.

**The hard-only matrix does not have automatic full matching-stage routing.**
The current `matching_kernel_id()` only routes the existing NLO PDF/DA naming
and renormalization schemes. The CS_kernel analysis uses the additive hard
correction directly in its fourth stage.

For a later stage adapter, keep the existing `perturbative_matching` stage:

- Add an explicit TMDWF hard-only operation; select this kernel and preserve the
  actual upstream scheme (do not relabel Wilson-loop division as MSbar).
- Take Pz and bT from checked upstream metadata; require mu, alpha_s, orientation,
  prescription, convention, and an interior x window in the job parameters.
- Reuse the sample-wise matrix application and record all conventions in output attrs.
- Name this result **hard-corrected quasi-TMDWF**. A full light-cone TMDWF also
  requires consistent UV conversion, soft-factor and rapidity-evolution inputs.
  Wilson-loop subtraction alone does not provide these ingredients.
- CS extraction is implemented in the `soft_analysis` stage: one job consumes
  two compatible complex Fourier outputs with aligned bootstrap samples, then
  adds `delta_gamma` to the sample-wise log ratio. See
  `examples/cs_kernel.json` for a calculation starting at the bootstrap two-point correlators.

## Runnable example

From the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m lamet_agent run examples/cs_kernel.json --provider codex --model gpt-6.1-sol
```

The recipe paths are relative to the JSON file. Set `two_point_h5` to the C32P23
bootstrap correlator file with embedded Wilson loops, and `b_lattice` to the
requested separations in lattice units. `pairs: "all"` requests the 4/5, 4/6
and 5/6 pairs. For a single estimate, set `"pairs": [[4, 6]]`.
The agent fits the four source/direction channels at each required momentum,
applies the Wilson central factor, fits the tails, transforms both components,
and combines them using their bootstrap row ordering. One numbered stage 03
job produces the complex quasi-TMDWF consumed by each applicable stage 04 pair.
Under `runs/CS_kernel/<run>/artifacts/`, the files
`03_fourier_transform/quasi_p6_b3/output.nc` and
`04_soft_analysis/csk_p4_p6_b3/csk.csv` illustrate the intermediate and final results.
The real-part pair comparison at each separation is saved under
`04_soft_analysis/plots/`, for example `csk_real_b3.pdf`. Its band is the
pointwise standard deviation of the paired bootstrap samples.
Numerics use mpmath with local precision (default 30 digits), cached scalar
coefficients, and complex128 outputs. Increase dps to check precision convergence.
