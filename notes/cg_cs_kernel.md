# Note for CG CS Kernel Analysis

- inputs: 2pt, qTMDWF data for different momenta & TMD hard kernel;
- outputs: CG CS kernel in bT space, with both statistical and systematic errors;


## Ground state fit on qTMDWF

- fit SS C2pt together with qTMDWF
    - joint fit: fit C2pt and qTMDWF together, share the same fit parameters like En, z0, etc.
    - fit ratio only: take the ratio of qTMDWF / C2pt, and fit the ratio only;

## Renormalization

- since the renormalization factor is a constant in CG, it will be canceled in the ratio of qTMDWF with different momenta;

## fourier transform the qTMDWF to x space

- mostly do not need extrapolation, but could be done;

## extract the CS kernel with x dependence

- two ways to go:
    - fit qTMDWF = a * H * exp(ln(Pz / Pfix) * gamma(x, bT))
    - form the ratio as gamma(x, bT) = 1 / ln(P2 / P1) * ln( (H(P1) * qTMDWF(P2)) / (H(P2) * qTMDWF(P1)) )

## remove the x dependence of the CS kernel

- constant fit on xdep for each bT

## systematics

- momenta set for fit / range of different momenta ratios
- residual x dependence