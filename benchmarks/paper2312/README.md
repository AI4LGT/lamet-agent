# Figure 1 reproduction from digitized inputs

Source: Jack Holligan and Huey-Wen Lin, *Systematic Improvement of x-dependent
Unpolarized Nucleon Generalized Parton Distribution in Lattice-QCD Calculation*,
arXiv:2312.10829v2, Phys. Rev. D 110, 034503 (2024),
https://arxiv.org/html/2312.10829v2 and https://doi.org/10.1103/PhysRevD.110.034503.
Source figures are CC BY 4.0. The three source assets were downloaded from the
arXiv HTML page. Numerical outputs are independent fits to digitized graphics.

Run `python benchmarks/paper2312/reproduce.py` from the repository root.
Requires NumPy, SciPy, Matplotlib and the project's kernel dependencies.

## Results

The SVG encodes 21 pion points at z=0,0.09,...,1.80 fm, including marginal error
bars. Axis calibration and vector extraction are documented in reproduce.py.
No bootstrap samples or covariance matrices have been fabricated.

Fitting B exp(-delta_m*z/hbar_c) over [0.54,1.53] fm, weighted by the recovered
marginal errors, gives B=2.43474 and delta_m=0.667927 GeV. Independently fitting
the red curve vertices gives 0.667885 GeV. Both agree with the reported 0.668 GeV.
Unweighted fitting gives 0.636632 GeV: the weighting is numerically important.

The diagonal-only formal error is 0.002445 GeV, with chi2/dof=165.714/10.
Rescaling this error by sqrt(chi2/dof) gives 0.009955 GeV, numerically equal to
the published 0.010 GeV after rounding. This is a possible reconstruction of
the quoted uncertainty, not evidence of the authors' exact procedure. Missing
correlations and the imperfect exponential description prevent interpreting
this as a recovery of the full lattice statistical analysis.

For the short-distance calculation we use an explicitly chosen not-a-knot cubic
spline of h, the paper's three-point windows z+{-0.02,0,0.02} fm, and unweighted
linear fits of log(C0/h) versus z/hbar_c, subtracting fitted delta_m. At mu=2 GeV
the existing project's three-loop threshold-matched coupling is 0.29907048;
the NNLO explicit coefficients use n_f=3. The five digitized central-curve
points per order agree to within 0.00633 GeV. Pixel uncertainty is roughly
0.003-0.006 GeV. The paper does not specify all interpolation details; this is
an approximate reconstruction, not an exact pointwise equality.

The m0 source curves are the central orange lines, not midpoints of their
asymmetric scale bands. Their selected pixel positions and calibration are
stored in reproduce.py and m0_fixed_order_comparison.csv. Overlapping and
ambiguous points were omitted. The figure displays recomputed curves against
the digitized benchmarks.

## Scope

This reproduces the linear-divergence central value and approximately the
fixed-order NLO/NNLO finite-mass curves. It does not reproduce the RGR/LRR
variants, their scale bands, or final PDFs/GPDs. The new production strategy
currently uses piecewise linear interpolation; this benchmark deliberately
uses cubic interpolation and therefore does not constitute an end-to-end
validation of that production workflow against the paper.

Files: pion_digitized.csv, m0_fixed_order_comparison.csv, results.json,
figure1_reproduction.png, reproduce.py, and the three original figure assets.
