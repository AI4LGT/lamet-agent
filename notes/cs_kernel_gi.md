# GI CSK Extraction Workflow

**Inputs:** nonlocal and local two-point correlators at several $P^z$, rectangular Wilson loops, and the perturbative hard kernel $H$.

## 1. Bare matrix element: one-/two-state fit

Form the correlator ratio $R(t)=C_2^{\mathrm{nonlocal}}(t)/C_2^{\mathrm{local}}(t)$ and fit it using

$$
\begin{aligned}
\text{One state:}\quad &R(t)=\Phi^0,\\
\text{Two states:}\quad &R(t)=\Phi^0\bigl(1+c\,e^{-\Delta E\,t}\bigr).
\end{aligned}
$$

The fitted $\Phi^0(z,b_\perp,P^z,L)$ is the bare ground-state matrix element.

## 2. Wilson-loop fit and subtraction

For each transverse size $n_\perp=b_\perp/a$, fit the Wilson loop versus its longitudinal size $n_z$:

$$
W(n_\perp,n_z)=c_0\,e^{-E n_z}\bigl(1+c_1e^{-\Delta E_W n_z}\bigr).
$$

Evaluate the fit at $n_z=(2L+|z|)/a$ and subtract the Wilson-line divergence:

$$
\Phi^{\mathrm{WL}}(z,b_\perp,P^z,L)=
\frac{\Phi^0(z,b_\perp,P^z,L)}{\sqrt{W_{\mathrm{fit}}(b_\perp/a,(2L+|z|)/a)}}.
$$

Check large-$L$ saturation using a constant fit, $\Phi^{\mathrm{WL}}(L)=\Phi_\infty$; use the plateau value or a stable $L$. The momentum-independent $Z_O$ factor is omitted because it cancels in the CSK momentum ratio.

## 3. Large-$\lambda$ fit and extrapolation

With $\lambda=zP^z$, fit the reliable positive-$\lambda$ tail ($\lambda\geq\lambda_L$) to

$$
\Phi_{\mathrm{tail}}(\lambda)=
\left[\frac{d_1}{(-i\lambda)^{n_1}}+
 e^{i\lambda}\frac{d_2}{(i\lambda)^{n_2}}\right]
 e^{-\lambda/\lambda_0},\qquad \lambda>0.
$$

**Our TMDWF convention:** both parts are even under $\lambda\to-\lambda$:

$$
\mathrm{Re}\Phi(-\lambda)=\mathrm{Re}\Phi(\lambda),\qquad
\mathrm{Im}\Phi(-\lambda)=\mathrm{Im}\Phi(\lambda).
$$

Extend the negative-$\lambda$ region by **even reflection of both parts**, not complex conjugation.

## 4. Fourier transform

Combine lattice points and the fitted tail to obtain

$$
\widetilde\Psi(x,b_\perp,P^z)
=\int_{-\infty}^{\infty}\frac{d\lambda}{2\pi}
 e^{i(x-1/2)\lambda}\Phi(\lambda,b_\perp,P^z)
=\frac{1}{\pi}\int_0^\infty d\lambda\,
 \cos\!\bigl[(x-1/2)\lambda\bigr]\Phi(\lambda,b_\perp,P^z).
$$

Keep the complex-valued result.

## 5. CSK with hard matching

For two hadron momenta $P_1^z$ and $P_2^z$, use the corresponding hard kernels $H_1$ and $H_2$ (e.g., $b_\perp$-unexpanded NLO):

$$
K(x,b_\perp,\mu;P_1^z,P_2^z)
=\frac{1}{\ln(P_1^z/P_2^z)}
\ln\ \left|\frac{H_2 \ \widetilde\Psi(x,b_\perp,P_1^z)}
{H_1 \ \widetilde\Psi(x,b_\perp,P_2^z)}\right|.
$$

Propagate statistical uncertainties throughout. Further physical-limit extrapolations are not included.

