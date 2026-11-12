# NLO hybrid matching with LRR

Kernel: `quark_da_gi_gzg5_hybrid_nlo_lrr`.

This kernel maps a quasi-distribution column vector to its matched distribution.
The fixed-order operator is defined in `quark_da_gi_gzg5_hybrid_nlo.md`.

Let Y be the uniform nonzero input grid and P(X,Y) its linear interpolation
matrix. Assemble all factors on Y before transferring to output points X:

$$M_{\rm LRR}(X,Y)=P(X,Y)\,[M_{\rm NLO}(Y,Y)+r_0 S(Y,Y)]
\exp[-R_{\rm PV}S(Y,Y)].$$

For identical input/output grids the original square matrix is returned directly.
The input grid must cover every output point; extrapolation is rejected. Finite,
nonempty one-dimensional grids are required. The existing fixed-order builder
also checks uniform spacing and excludes input zero.

The shape has no factor of $N_m\mu$ inside it. For $p_z=|y|P_z$ and
$z_s=\mathtt{zs\_fm}/(\hbar c)$, its off-diagonal expression is the Abel-regulated
Fourier integral

$$C_z(\xi)={p_z\over\pi}\operatorname{Re}\int_{z_s}^{\infty} dz\,
 e^{i(\xi-1)p_z z}[z e^{-\epsilon_m z}-z_s e^{-\epsilon_m z_s}].$$

Here $\epsilon_m=0.005$ GeV. The square discretization is
$S_{ij}=\Delta y\,C_z(y_i/y_j)/|y_j|$ for $i\ne j$, with
$S_{jj}=-\sum_{i\ne j}S_{ij}$. This is a finite-domain column-plus prescription;
continuum and regulator convergence must be assessed separately.

The scalar factors use $n_f=3$ and the existing two-loop running coupling:

$$r_0=r_{\rm asym}(0,1,\mu,n_f)\alpha_s
=N_m\mu(1+c_1+c_2)\alpha_s,$$

$$R_{\rm PV}=N_m\mu e^w{-2\pi\over\beta_0}\operatorname{Re}
 [E_{1+b}(w)+c_1E_b(w)+c_2E_{b-1}(w)],\qquad
 w={-2\pi\over\beta_0\alpha_s},\quad b={\beta_1\over2\beta_0^2}.$$

The $c_1,c_2$ terms and $N_m=0.5749687262865643$ are retained from the existing
implementation. The $r_0$ term cancels the order-alpha_s contribution of the
exponential, preserving NLO matching. The ordering of these generally
noncommuting matrix factors is unchanged.

[Zhang et al., arXiv:2305.05212](https://arxiv.org/abs/2305.05212), Eqs. (12)-(17),
provides the renormalon coefficients, PV prescription and regulated hybrid
correction. The finite-grid matrix exponential is the package's existing
resummation choice; it is not a claim of an exact continuum or notebook
reproduction. Consistent mass renormalization is needed for a physical LRR
prediction. NNLO LRR and combined RGR+LRR are not provided by this kernel.

The grid-transfer regressions independently assemble the square plus matrix;
separate tests check the shape via its complex Fourier integral and the PV sum
via integration-by-parts Borel quadrature. These checks do not constitute a full
physical-data or continuum-convergence validation.

For this DA wrapper the shape density is active only for input 0 < y < 1 (with the existing epsilon cutoff). The PDF paper above does not independently validate the full DA matching prescription.
