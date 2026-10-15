<!-- lamet-agent formula cache; kernel=quark_gpd_gi_gt_hybrid_nlo; source=project kernels_gpd.py; scheme=hybrid; order=NLO -->
# GI γ^t unpolarized quark GPD hybrid-NLO kernel

This kernel is the current-contract wrapper of
`GI_gt_quark_GPD_hybrid_NLO` from
`/Users/zhaodianjun/Desktop/combine_watermelon2.0/kernels_gpd.py`.
It keeps the original two-variable coefficient at fixed skewness,

\[
 x_1=\xi+x,\quad x_2=\xi-x,\quad y_1=\xi+y,\quad y_2=\xi-y,
\]

including the logarithmic terms, the hybrid sine-integral term
\(6\,\mathrm{Si}[(x-y)z_sP_z]/[\pi(x-y)]\), and the fixed-\(y\)
column-wise plus prescription.  The matrix acts as

\[
 F(x)=\sum_y C^{-1}(x,y,\xi,\mu/P_z,z_sP_z)\,\widetilde F(y).
\]

The public filename stem encodes gauge-invariant (`gi`), \(\gamma^t\)
(`gt`), quark GPD (`quark_gpd`), hybrid scheme, and NLO order.  `skewness`
is intentionally an explicit `kernel_parameters` value because it is a
GPD-kinematic input rather than a generic matching-stage parameter.

