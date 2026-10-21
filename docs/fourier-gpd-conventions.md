# Real distributions, signed separation, and GPD endpoint conventions

All PDF, DA, and GPD objects handled by this workflow are assumed real in
momentum space. Their coordinate-space data therefore obey

\[
h(-z)=h(z)^*,\qquad \operatorname{Im}h(0)=0.
\]

The nonnegative branch is authoritative. Preparation replaces any supplied
negative branch with its conjugate reflection, sample by sample. It does not
average the two branches or combine their statistical errors. This policy also
applies to direct `scan_fourier_transform` calls. The low-level explicit-symmetry
utility remains available for callers outside this workflow.

## Matrix-element definitions

All field positions below are equal-time spatial positions along the boost
direction: a position written as \(z\) means \(x^\mu=(0,0,0,z)\).
\(U(a,b)\) connects the quark field at \(b\) to the barred field at \(a\).
Momenta \(P_i,P_f,P\) denote four-momenta; the phase formulas use their
longitudinal components \(P_i^z,P_f^z,P^z\). Distances supplied to the code
are in fm and these momentum components are in GeV.

The notation suppresses spin indices and the fixed kinematic normalization
of the scalar amplitude supplied to the workflow. The real-distribution
assumption applies to that amplitude, with any conventional overall factors
already accounted for; it is not an assertion about arbitrary unprojected
spinor matrix elements. The displayed operators are gauge invariant. For
the CG implementation, the corresponding gauge-fixed bilinear replaces the
Wilson-line operator; the endpoint positions and momentum labels are unchanged.

### PDF: forward hadron matrix element

\[
h_{\rm PDF}^{\Gamma}(z;P)
=\langle H(P)|\bar\psi(z)U(z,0)\Gamma\psi(0)|H(P)\rangle,
\qquad P_i=P_f=P,\qquad \Delta=P_f-P_i=0.
\]

The incoming hadron is the ket and the outgoing hadron is the bra, both
with momentum \(P\). The barred field is at \(z\), and the quark field is
at zero. Representative structures in the asymptotic discussion are
\(\Gamma=\gamma^t\) for the unpolarized channel,
\(\gamma^z\gamma^5\) for helicity, and
\(\gamma^t\gamma^\perp\gamma^5\) for transversity. Spin labels are
understood for polarized channels; the actual input must use its authored
operator and scalar projection. There is no off-forward endpoint transfer
phase because \(\Delta=0\).

### DA: vacuum-to-meson transition

\[
h_{\rm DA}(z;P)
=\langle M(P)|\bar\psi_1(z)U(z,0)\Gamma_{\rm DA}\psi_2(0)|\Omega\rangle,
\qquad \Gamma_{\rm DA}=\gamma^z\gamma^5
\]

for the pseudoscalar-meson convention used in the asymptotic formulas.
The final state is the meson bra with momentum \(P\); the initial state
is the vacuum ket \(|\Omega\rangle\), which carries zero four-momentum.
The ordered flavor fields are \(\bar\psi_1\) at \(z\) and \(\psi_2\)
at zero. This is a vacuum-to-meson transition, not a forward hadron matrix
element. The DA controls specify the two flavor classes; they do not use
the GPD initial/final momentum attributes.

Its midpoint operator has endpoints \((z/2,-z/2)\) and matrix element
\[
h_{{\rm DA},{\rm mid}}(z;P)
=\langle M(P)|\bar\psi_1(z/2)U(z/2,-z/2)
\Gamma_{\rm DA}\psi_2(-z/2)|\Omega\rangle
=e^{+iP^zz/(2\hbar c)}h_{\rm DA}(z;P).
\]
The optional `phase_transfer_da` operation projects this rotated amplitude
onto its real part and rotates it back before fitting; it does not change
the GPD endpoint labels.

### GPD: nonforward hadron matrix element

The raw GPD input must be

\[
F_{fi}(z)=\langle H(P_f)|\bar\psi(z)U(z,0)\Gamma\psi(0)|H(P_i)\rangle.
\]

The initial/source hadron is the ket \(|H(P_i)\rangle\), and the final/sink
hadron is the bra \(\langle H(P_f)|\). The longitudinal source/ket momentum
is `initial_momentum_gev` and the longitudinal sink/bra momentum is
`final_momentum_gev`. Write
\(\Delta=P_f-P_i\), \(\Delta_z=P_f^z-P_i^z\),
\(\overline P_z=(P_f^z+P_i^z)/2\), and
\(\theta=\Delta_z z/(\hbar c)\).
For unpolarized GPDs \(\Gamma=\gamma^t\); other operator projections must
retain their corresponding input convention. All three endpoint choices
below keep the same external bra and ket momenta:

\[
\begin{aligned}
h_{\texttt{psi\_at\_0}}(z;P_f,P_i)
&=\langle H(P_f)|\bar\psi(z)U(z,0)\Gamma\psi(0)|H(P_i)\rangle,\\
h_{\texttt{mid\_at\_0}}(z;P_f,P_i)
&=\langle H(P_f)|\bar\psi(z/2)U(z/2,-z/2)
\Gamma\psi(-z/2)|H(P_i)\rangle,\\
h_{\texttt{barpsi\_at\_0}}(z;P_f,P_i)
&=\langle H(P_f)|\bar\psi(0)U(0,z)\Gamma\psi(z)|H(P_i)\rangle.
\end{aligned}
\]

## Input convention and retained labels

The existing label strings are retained and now specify literal field positions:

| Label | Endpoints \((x_{\bar\psi},x_\psi)\) | Positive-\(z\) value |
| --- | --- | --- |
| `psi_at_0` | \((z,0)\) | \(F(z)\) |
| `mid_at_0` | \((z/2,-z/2)\) | \(e^{+i\theta/2}F(z)\) |
| `barpsi_at_0` | \((0,z)\) | \(e^{-i\theta}F(z)^*\) |

The symbol \(z\) is used exactly as displayed in each operator. Thus
\(x_{\bar\psi}-x_\psi=z\) for `psi_at_0` and `mid_at_0`, whereas
\(x_{\bar\psi}-x_\psi=-z\) for `barpsi_at_0`. The barred-origin label
does not mean \(\bar\psi(0)\psi(-z)\); that would be a different signed
coordinate convention. Switching labels does not exchange \(P_i\) and \(P_f\).

The midpoint label remains the workflow default. Spatial translation gives
the positive midpoint phase. The barred-origin convention includes separation
reversal:

\[
\langle P_f|\bar\psi(0)U(0,z)\Gamma\psi(z)|P_i\rangle
=e^{-i\theta}F(-z)=e^{-i\theta}F(z)^*.
\]

These same conversions are used for input data, tail fitting, and extension.
After conversion, every label uses its own positive branch for negative-\(z\)
completion. A nonforward job no longer requires `hermitian_partner`.
An optional exchanged-flow input is accepted and checked for compatible
provenance, but its values are neither averaged nor used in the transform.

## Asymptotic forms

The base formula uses the paper's operator convention, with its \(P=P_f\) and
\(P'=P_i\). Define \(s=\operatorname{sgn}z\), \(r=|z|/(1\,\mathrm{fm})\), and

\[
C_j^{\rm LA}=A_je^{is\phi_j},\qquad
C_j^{\rm NLA}=A_je^{is\phi_j}+A'_je^{is\phi'_j}/r.
\]

For the full pion endpoint family,

\[
F^{\cal O}(z)=e^{-\Lambda|z|/(\hbar c)}
\left[C_1^{\cal O}e^{-iP_f^zz/(\hbar c)}
+C_3^{\cal O}e^{+iP_i^zz/(\hbar c)}
+C_2^{\cal O}+C_{\tilde2}^{\cal O}e^{-i\theta}\right],
\quad {\cal O}\in\{\mathrm{LA},\mathrm{NLA}\}.
\]

Pion sea and nucleon families retain only terms \(2,\tilde2\).
The CG model additionally multiplies the entire tail by \(r^{-n}\).
For every family, the three labeled tails are

\[
h_{\texttt{psi\_at\_0}}^{\cal O}=F^{\cal O},\qquad
h_{\texttt{mid\_at\_0}}^{\cal O}=e^{+i\theta/2}F^{\cal O},\qquad
h_{\texttt{barpsi\_at\_0}}^{\cal O}=e^{-i\theta}(F^{\cal O})^*.
\]

The last conjugation reverses intrinsic leading and primed phases as well as
the oscillatory frequencies. It cannot be replaced by exchanging two labels
or by adding a common frequency shift alone. Fit parameters retain the paper's
base convention for all labels.

Reference: [arXiv:2601.12189v1, Appendix E, Eqs. (282)–(287)](https://arxiv.org/html/2601.12189v1#A5).

## Compatibility and verification

This changes the numerical meaning of the two endpoint labels; old fitted
parameters and endpoint-prepared files must not be silently reused. Supply raw
data in the stated input convention and rerun preparation and fitting. Existing
negative branches and imaginary origin noise are replaced deliberately. The
midpoint positive-branch conversion is unchanged, but its negative branch now
comes from the same flow rather than an exchanged partner.

Published momentum-space output is real. The complex coordinate-space samples
remain available through preparation and extension. No GPD matching kernel is
changed by this update.

The regression tests check all three labels, nonzero and zero transfer, signed
and positive-only inputs, independence from partner values, all implemented GPD
families, LA/NLA and GI/CG tails, and a real package Fourier transform after tail
extension. Existing PDF/DA tail and Fourier tests are run alongside them.
