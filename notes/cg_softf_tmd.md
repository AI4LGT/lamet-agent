# Note for CG soft function and TMD analysis

- inputs: 2pt, qTMDWF data for different momenta, qTMDPDF 3pt, form factor 3pt, local SP 2pt & TMD hard kernel & sudakov kernel;
- outputs: intrinsic soft function & TMDPDF & TMDWF


## Ground state fit

- fit SS C2pt together with qTMDWF
    - joint fit: fit C2pt and qTMDWF together, share the same fit parameters like En, z0, etc.
    - fit ratio only: take the ratio of qTMDWF / C2pt, and fit the ratio only;

- fit qTMDPDF 3pt together with 2pt

- fit FF ratio or joint fit FF ratio + sum, the ratio itself will converge to F0, the bare FF;


## Renormalization

- the renormalization factor ZA / ZV is a constant close to 1 in CG 

## extraction of intrinsic soft function

- soft function = FF / integral, where the integral is the 2D integral of sudakov kernel and reduced quasi-TMDWF (qTMDWF / TMD hard kernel);
- integral needs cut off near the boundary region where x~0 or x~1;
- reduced qTMDWF needs to be extrapolated in the end point regions;
- 2D integral is slow, especially for each sample;
- if FF and qTMDWF have different momenta, needs an evolution with cs kernel;


## extraction of TMDPDF and TMDWF

- combine quasi distributions, cs kernel, intrinsic soft function, and TMD hard kernel;
- extrapolation in the large bT region, gaussian or exponential decay?
- 2D FT to the kT space;