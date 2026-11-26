# Residual GI TMD renormalization coefficient

Source: [arXiv:2412.19988](https://arxiv.org/abs/2412.19988), Eqs. (5)-(8), Sec. III.5.
This coefficient converts a normalized zero-momentum gamma-t reference after
division by the square root of its matched Euclidean Wilson loop.

With R²=z²+b², C=1+alpha_s CF/(2 pi) [3/2 log(R² mu² exp(2 gamma_E)/4)
-2(z/b) atan(z/b)+1/2]. The Python interface takes lengths in fm and converts
them using HBAR_C_GEV_FM; mu is in GeV. b must be positive; z=0 is allowed.

`nlo` evaluates this expression at the supplied mu and alpha_s(mu).
`nlo_rgr` evaluates the complete one-loop expression at
mu0=kappa*2 exp(-gamma_E)/R, including the boundary logarithm 2 log(kappa),
then evolves to mu using two-loop running and the two-loop noncusp anomalous dimension.

The evolution convention is d ln C/d ln mu = gamma_C = -2 gamma_j(HQET).
Using a_s=alpha_s/(4 pi),

gamma_C = 6 CF a_s + CF [CF(-5+16 pi²/3)+CA(49/3-4 pi²/3)
-20 TF nf/3] a_s².

This is 2 gamma_F in the convention of Eq. (7) of the TMD paper. The HQET
current result is Eq. (8) of [arXiv:2311.09894](https://arxiv.org/abs/2311.09894).
Beta uses d alpha_s/d ln mu = -2 alpha_s [beta0 a_s + beta1 a_s²],
beta0=11-2 nf/3 and beta1=102-38 nf/3 for SU(3).

The coupling is anchored at the authored alpha_s(mu), with fixed nf=3,4,5.
There is no threshold matching or flavor change. This choice must be recorded
when comparing numerical results; the paper's quoted ensemble factors are not
hard-coded benchmarks for arbitrary coupling choices. No coupling freezing is
performed. Scales below min_scale_gev or trajectories reaching alpha_s >= 1
raise errors. The scale floor is a domain guard, not a perturbativity guarantee.

The published NLO prescriptions and the supplied notebook NNLO extension
below are implemented. LRR is not included.
This is the single-log renormalization evolution, not the double-log hard
matching or Collins-Soper evolution.


## NNLO notebook extension

`zo_nnlo.py` ports the NNLO Fixed order module of `TMD(2).nb` v2 and
`TMD_NNLO.wl` (2026-11-14), including the numerical same-side remainder.
`order="nnlo"` uses its fixed-order coefficient; `order="nnlo_rgr"` evolves
that NNLO boundary with the three-loop beta function and three-loop endpoint
anomalous dimension (NNLL evolution). This source is separate from arXiv:2412.19988.
See `docs/tmd-zo-extraction.md` for the subtraction, numerical refinement,
fixed-flavor conventions, and massless pure-DR limitations.



The three-loop endpoint anomalous dimension is -2 times the HQET current
result in [arXiv:2311.09894v3](https://arxiv.org/pdf/2311.09894v3),
Eq. (3.3), retaining terms through alpha_s^3. The sign and factor two use
d ln C/d ln mu = -2 gamma_j. Three-loop SU(3) running adds
beta2=2857/2-5033*n_f/18+325*n_f^2/54 in the alpha_s/(4*pi) expansion.
The loop count is part of the evolution-cache key.
