# quark_pdf_gi_gt_hybrid_nnlo_rgr_lrr_tr

GI gamma-t hybrid nonsinglet PDF with NNLO matching, LRR, DGLAP RGR,
and the supplied Proton_PDF.nb threshold prescription. Select order: "nnlo"
and resummation: "rgr_lrr_tr". Includes quark and antiquark sectors with
f(-x)=-qbar(x). Independent matching at each output magnitude is retained.

The operator is M_LRR(mu_h) + [(cut*P)@(cut*J_LRR(mu_i)) - cut*J_LRR(mu_h)]
@(cut*H_LRR(mu_h)), followed by the existing signed DGLAP evolution.
The notebook LRR modifies fixed matching, hard modulus and phase, and soft
matrices. Exponentials and counterterms are applied before the threshold cut.
The hard modulus uses exp(zs*PV), subtracting r0 at NLO and adding the NNLO
-r1-r0*H1+r0^2/2 counterterms. The phase adds 2*(PV-r0[-r1])/abs(y*Pz).
The soft matrix uses (J+r0*C) exp(-PV*C) at NLO; at NNLO its prefactor also
contains r1*C+r0*J1@C+r0^2*C@C/2, where J1 is the first-order part, without
identity. The noncommuting matrix order follows the notebook.

Required zs_fm. Defaults: kappa=1, spectator_kappa=1, threshold_cut=.4,
cut_quadrature="fractional", mu_min_gev=.6, evolution_points=100,
evolution_steps=40, y_max=100, eps=1e-12. NNLO quadrature_order=64.
Fractional weights integrate partial source cells at the sharp cut;
"point" retains strict center sampling. Historical non-LRR NLO TR is unchanged
and defaults to point sampling; align this control for order/LRR comparisons.

Fixed-matching eps_m=.005 GeV preserves the existing RGR+LRR baseline.
The soft LRR defaults soft_eps_m=.01 GeV and soft_zs_gev_inv=.01 GeV^-1
follow the notebook's separate MCzJ regulator and small-distance parameter.
The notebook fixed-matching regulator is .01 GeV; use eps_m=.01 to align it.
soft_zs_gev_inv is not zs_fm and must not be converted as a length in fm.
These finite regulators require convergence studies for physical precision.

mu_h=2*kappa*abs(x)*Pz; mu_i=2*spectator_kappa*(1-abs(x))*Pz.
TR switches off exactly for mu_i>=mu_h, recovering the existing RGR+LRR
operator. Hard scales below mu_min retain zero rows; active spectator scales
below mu_min raise an error. Omit abs(x)=1. The star exponent requires
-1<eta<1. Input must be uniform, increasing, nonzero and span [-1,1].

Soft derivatives use the exact analytic eta=0 limit instead of notebook
eta=.01 finite differences. Couplings and full fixed matching are those of
the package. Formula comparisons do not reproduce the full notebook analysis.
The separately cut product prescription has a known finite-cut O(alpha_s^2)
residual in the NNLO non-LRR case; adding LRR does not certify exact finite-cut
NNLO accuracy. No extra finite-cut subtraction is introduced. No singlet/gluon
extension or claim of continuum/systematic uncertainty control.
