# Coulomb-gauge TMD fixed-order matching

Source: arXiv:2504.04625v5, Eqs. (9), (13), (14), (18)–(20), (36),
(A1), and (A6). Scales and momenta are in GeV. The default is μ = 2 GeV
and a fixed αs(μ) = 0.293; this module implements NLO without RG resummation.

With CF = 4/3 and L = log[μ²/(2xP)²],

δC(xP) = αs CF/(4π) [−L²/2 − 3L − 12 + 7π²/12].

Every composite hard coefficient is explicitly truncated at NLO:

- Hφ = 1 + δC(xP) + δC((1−x)P).
- Hf = 1 + 2δC(xP).
- δSud(Q) = αs CF/(4π) [−log²(μ²/Q²) − 3log(μ²/Q²) − 8 + π²/6].
- HF = 1 + δSud(2P√(x1x2)) + δSud(2P√((1−x1)(1−x2))).

No exponentiation of these fixed-order hard coefficients, frozen coupling,
infrared profile, or product-generated O(αs²) term is included. Dividing by
the truncated hard factor implements the stated factorization equations;
the statistical nonlinear operations themselves are not Taylor-expanded.

The physical Collins–Soper kernel is
γ = log[Hφ(P1) qφ(P2)/(Hφ(P2) qφ(P1))]/log(P2/P1).
The reduced wave function Φ=qφ/Hφ evolves to a reference momentum as
Φ(Pref)=Φ(P) exp[γ log(Pref/P)]. In the intrinsic-soft integral, HF uses
the form-factor momentum after that evolution. Endpoint powers are fitted
to the reduced wave function and only replace its endpoint regions.
The intrinsic-soft estimates from the different wave-function momenta are
averaged with equal weights inside each aligned replica, retaining their
full covariance. A regularized common-constant GLS fit is reported only as
a momentum-consistency diagnostic and does not set the published average.

Matching uses f = √SI qf/Hf exp{γ log[ζ/(2xP)²]/2} and
φ = √SI qφ/Hφ exp{γ log(Pref/P)}. The input quasi distributions and SI
must share the ratio renormalization scheme. Positive hard factors and SI
are required wherever logarithms, division, or square roots use them.
The finite-order HF may become negative near integration corners; its
signed contribution is retained and diagnosed, not clipped.

This intentionally differs from legacy scripts that exponentiate a fixed
coefficient, evolve an already reduced wave function with an uncorrected CS
kernel, or use the wave-function momentum in HF after evolution to PFF.
