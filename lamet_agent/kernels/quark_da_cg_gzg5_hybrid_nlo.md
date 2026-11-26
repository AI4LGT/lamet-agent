<!-- lamet-agent formula cache; kernel=quark_da_cg_gzg5_hybrid_nlo; source=coulomb-gauge-quasiDA/Coulomb-gauge_quasi-DA_one-loop_matching.tex (internal note, 2026-10-22); arxiv=none; paper_used=false; equations=Sec. 5 gamma^z gamma_5 variant plus CG hybrid Si tail -->
### CG quasi-DA matching, $\gamma^z\gamma_5$, hybrid scheme

In units of $\alpha_s C_F/(2\pi)$, $C(x,y) = \delta(x-y) + \frac{\alpha_s C_F}{2\pi} C^{(1)}(x,y)$ with

$$C^{(1)}_{\rm hybrid}(x,y) = \big[G_z(x,y) + \Delta_{\rm hyb}(x,y)\big]_+^{(-\infty,\infty)},$$

where

$$G(x,y) = \begin{cases}
\dfrac{1+y-x}{y-x}\dfrac{x}{y}\ln\dfrac{y-x}{-x} + \dfrac{1+x-y}{y-x}\dfrac{1-x}{1-y}\ln\dfrac{y-x}{1-x} + T(x,y), & x<0,\\[2mm]
\dfrac{1+y-x}{y-x}\dfrac{x}{y}\ln\dfrac{4x(y-x)P_z^2}{\mu^2} + \dfrac{1+x-y}{y-x}\dfrac{1-x}{1-y}\ln\dfrac{y-x}{1-x} - \dfrac{x}{y} + T(x,y), & 0<x<y,
\end{cases}$$

with the regions $y<x<1$ and $x>1$ obtained from the mirror map $G(x,y) = G(1-x,1-y)$. The Coulomb-gauge-specific term is

$$T(x,y) = \frac{(y-3x)\,\Theta(x,y) + (2+y-3x)\,\Theta(1-x,1-y)}{2(y-x)} - \frac{3}{2|x-y|},$$

$$\Theta(x,y) = \begin{cases} \dfrac{1}{\sqrt{y(y-2x)}}\arctan\dfrac{\sqrt{y(y-2x)}}{|x|}, & y>2x,\\[2mm] \dfrac{1}{\sqrt{y(2x-y)}}\,\mathrm{artanh}\dfrac{\sqrt{y(2x-y)}}{|x|}, & y<2x,\end{cases} \qquad \Theta \to \frac{1}{|x|}\ \text{at}\ y=2x.$$

$G = \mathcal O(1/x^2)$ as $|x|\to\infty$, and for $0<x<1$ the coefficient of $\ln\mu^2$ is $-V_0(x,y)$ (the ERBL kernel), so the IR matches the light-cone DA.

The $\gamma^z\gamma_5$ operator differs only in the vertex diagram (an extra factor $1-2\epsilon$), which adds

$$G_z(x,y) = G(x,y) + 2W(x,y)\,\theta(0<x<1), \qquad W(x,y) = \frac{x}{y}\theta(x<y) + \frac{1-x}{1-y}\theta(x>y),$$

the DA analogue of the $2(1-\xi)$ term of the $\gamma^z$ quasi-PDF kernel.

The hybrid scheme follows the CG quasi-PDF hybrid kernel: for $|z|>z_s$ the ratio normalization is replaced by the $\overline{\rm MS}$ one, which restores the $\overline{\rm MS}$ tail $1/(2|x-y|)$ cut off by a sine integral,

$$\Delta_{\rm hyb}(x,y) = \frac{1}{2}\left[\frac{1}{|x-y|} - \frac{2\,\mathrm{Si}\big(z_s P_z (y-x)\big)}{\pi (y-x)}\right],$$

so that $z_s\to\infty$ gives the ratio kernel and $z_s\to0$ gives the $\overline{\rm MS}$ tail. Coulomb gauge has no Wilson-line linear divergence, so no LRR variant is provided (as for the CG quasi-PDF kernels).

Conventions: $\tilde\phi(x) = \int_0^1 dy\, C(x,y)\,\phi(y)$, NDR $\gamma_5$, $x$ is the quark momentum fraction at the $\psi(0)$ end. The plus prescription is over the whole real line, $[g]_+ = g(x,y) - \delta(x-y)\int dx'\,g(x',y)$, and is implemented by the column-sum subtraction in `build_matching_matrix_column_plus`; columns with $y\notin(0,1)$ vanish, as for the GI DA kernels.

#### Consistency check

`G_qq_cg_t` in `implementation.py` agrees with the reference script `coulomb-gauge-quasiDA/scripts/kernel.py` (`Fhat` minus $1/(2|x-y|)$) to $10^{-14}$, reproduces $x^2 G(x,0.3)\to\pm0.0665$ at $x=\pm10^4$, satisfies $\partial G/\partial\ln\mu^2 = -V_0$ for $0<x<1$ and $G(x,y)=G(1-x,1-y)$. Caveat from the note: the hybrid $z_s$ term is the PDF-analogue Si term and was not derived independently for the DA; no published CG quasi-DA kernel was available for comparison.
