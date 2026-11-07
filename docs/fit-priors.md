# Configurable fit priors

Fit jobs accept an optional `priors` mapping in their stage defaults or job
parameters. Each entry is a string accepted directly by `gvar.gvar()`:

```json
{
  "priors": {
    "parameter_name": "0.5(0.2)"
  }
}
```

For example, `"1.0(0.3)"` and `"1.00(30)"` both describe 1.0 ± 0.3.
Scientific notation follows gvar syntax, for example `"2.00(30)e-08"`.
Parsed means must be finite and standard deviations finite and positive.
Numbers, arrays and the former `{mean, sdev}` input objects are rejected.
Diagnostics retain numeric `mean`/`sdev` fields for machine-readable output.

Each model first constructs its existing default priors, then replaces the
parameters named in this mapping. Missing parameters keep their defaults.
Extra names are ignored and recorded in `inactive_prior_overrides`, so one map
can be reused across state counts, fit stages, LA/NLA orders, and extrapolation
ansatze. Unused entries must still have valid Gaussian definitions. Parameter
names are exact strings; a misspelling is inactive rather than an error.

The existing variable transformations, prior-width factors, chained posterior
transfers, and replica-fit policies apply after this initial override. In
particular, a later posterior-derived prior may replace an initial override.
This option does not change those policies. Omitting overrides retains the
existing fit behavior.

## Parameter names and existing conventions

| Fit | Names | Input convention |
| --- | --- | --- |
| Correlator matrix elements / qDA | `log(E0)`, `log(dE1)`, `z0`, `O00_re`, etc. | Existing logarithmic or raw fit-parameter units, before numerical conditioning. |
| Direct two-point spectrum | `E0`, `E1`, `z0`, etc. | Lattice-unit energies and raw overlaps; missing entries come from the existing recommendation. |
| Self-renormalization reference | `g0.1`, `f10.1`, etc. | Exact `g{z}` / `f1{z}` keys, with z in fm; g is dimensionless and f1 is in GeV. |
| Self-renormalization short-distance matching | `m0`, `b` | Existing fit convention: m0 in inverse fm and b dimensionless. |
| Fourier tails | `A2`, `phi2`, `Lambda`, `n`, and NLA names such as `A2p` | Mean in the named physical parameter; sdev in its bounded internal coordinate, as in the existing fitter. |
| Extrapolation | `h0` and correction names such as `a`, `a2`, `inv_p2` | Each coefficient's existing native units. |
| Correlator report dispersion fit | `m2`, `k2`, `k3` | m2 in GeV²; k2 and k3 dimensionless in E²=m2+k2 p²+k3 a²p⁴ with a in GeV⁻¹. |

## Correlator analysis

Least-squares correlator jobs (`analysis_method: "lsqfit"`) accept separate
controls for replica fits and chained fits. Put these fields in
`stages.correlator_analysis.defaults` or directly on a job. The examples below
illustrate the configuration; choose numerical priors and fit windows for
your data.

The lists below describe the base priors constructed for each candidate, before
chained posterior transfer. `N` is the concrete state count for the named atom,
not the complete candidate list. Indices start at zero; braces in parameter
patterns are placeholders, not characters in the JSON key.

### Controls and defaults

| Field | Manifest default | Meaning |
| --- | --- | --- |
| `priors` | `{}` | Partial overrides of named model parameters, each a gvar string such as `1.0(0.3)`. |
| `prior_width` | `[1.0]` | Candidate factors multiplying every base or authored prior standard deviation exactly once. |
| `sample_prior_mode` | `"center_posterior"` | Selects `"original"` or `"center_posterior"` priors for replica fits. |
| `posterior_prior_error_scale` | `10.0` | Width factor for replica priors derived from the ensemble-center posterior; also the fallback for chained transfers. |
| `chained_prior_error_scale` | Omitted | Optional independent width factor for transferring the preceding fit-stage posterior. |

With no overrides and `prior_width: [1.0]`, existing base prior means and
standard deviations are preserved. Multiple widths scan multiple candidates;
they do not apply successive multiplications. Width and error-scale factors
must be finite and positive.

### Direct two-point spectrum

For `fit_scope: ["2pt"]`, the complete input parameter list is:

| State count | Prior names |
| --- | --- |
| 1 | `E0`, `z0` |
| 2 | `E0`, `E1`, `z0`, `z1` |
| N | `E0` through `E{N-1}`; `z0` through `z{N-1}` |

Energies use lattice units, and overlaps use raw input-correlator units. Energy
means must be positive and strictly increasing; overlap means must be positive.
The internal `log(E0)` / `log(dE1)` transformation is performed by the fitter;
those internal keys are not input-prior names for this direct-spectrum path.
Missing priors come from the existing spectrum recommendation.

### Shared spectrum in matrix-element, FH and qDA fits

Every atom in these models constructs a spectral parameter block:

| Fitting form | Prior names |
| --- | --- |
| `Breit` | `log(E0)`; `log(dE1)` through `log(dE{N-1})`; `z0` through `z{N-1}` |
| `NonBreit`, initial/source spectrum | `log(E0_i)`; `log(dE1_i)` through `log(dE{N-1}_i)`; `z0_i` through `z{N-1}_i` |
| `NonBreit`, final/sink spectrum | `log(E0_f)`; `log(dE1_f)` through `log(dE{N-1}_f)`; `z0_f` through `z{N-1}_f` |

A joint stage uses the union of the blocks, so its spectral block extends to
the largest state count among its atoms. There is no `log(dE0)` parameter.
In this path, `E0` and `dE1` without `log(...)` are derived variables, not
independent input-prior keys.

For example, `"log(E0)": "-0.9(0.2)"` specifies a Gaussian
on the logarithm of the energy. It does not specify an energy of `-0.9`.
There is no automatic conversion of an ordinary-fit `E0` key to `log(E0)`.

### Ordinary three-point models

Add the following parameters to the corresponding shared spectral block:

| `fit_scope` atom | `fitting_form` | Additional prior names |
| --- | --- | --- |
| `2pt`, as part of a matrix-element pipeline | Either | None |
| `3pt` or `3pt_over_2pt` | `Breit` | `O{i}{j}_re`, `O{i}{j}_im`, for `0 <= i <= j < N` |
| `3pt` or `3pt_over_2pt` | `NonBreit` | `O{i}{j}_re`, `O{i}{j}_im`, for every `0 <= i,j < N` |
| `3pt_over_3pt` | `Breit` | The same Breit `O` block, plus `D{i}{j}_re` for `0 <= i <= j < N`, excluding `(0,0)` |
| `3pt_over_3pt` | `NonBreit` | The same NonBreit `O` block, plus `D{i}{j}_re` for every `0 <= i,j < N`, excluding `(0,0)` |

For example, a two-state Breit `3pt` or `3pt_over_2pt` model has exactly these
base keys:

```text
log(E0), log(dE1), z0, z1,
O00_re, O00_im, O01_re, O01_im, O11_re, O11_im
```

The two-state NonBreit model instead has:

```text
log(E0_i), log(dE1_i), z0_i, z1_i,
log(E0_f), log(dE1_f), z0_f, z1_f,
O00_re, O00_im, O01_re, O01_im,
O10_re, O10_im, O11_re, O11_im
```

For a two-state self-ratio (`3pt_over_3pt`), add `D01_re`, `D11_re` in Breit,
or `D01_re`, `D10_re`, `D11_re` in NonBreit. The denominator ground element is
fixed to one; neither `D00_re` nor any `D..._im` is an independent prior.
NonBreit matrix indices are sink then source; only its spectral parameters
carry `_i` / `_f` suffixes.

### Feynman–Hellmann (`FH`)

`FH` uses `Breit`. Add these names to its shared spectral block:

| State count | Additional prior names |
| --- | --- |
| 1 | `O00_re`, `O00_im` |
| Greater than 1 | `O00_re`, `O00_im`, `sum_re_excited_coeff`, `sum_im_excited_coeff`, `sum_re_offset`, `sum_im_offset`, `sum_re_exp_offset`, `sum_im_exp_offset`, `sum_den_exp_coeff` |

The current FH likelihood uses `E0` and, for any state count greater than one,
only the first gap `dE1` with the summed-ratio coefficients above. Increasing
the FH state count beyond two does not introduce further FH exponential terms.
The common prior builder still includes spectral overlaps and higher gaps;
they can be constrained by an accompanying two-point block, but do not enter
an FH-only likelihood.

### qDA models

Both `qda` and `wf_over_2pt` use `Breit` and the shared spectral block. The
denominator kind is determined by the supplied correlators:

| Atom / denominator | Additional prior names |
| --- | --- |
| `qda` or `wf_over_2pt`, with an explicit two-point denominator | `O0{j}_re`, `O0{j}_im`, for `0 <= j < N` |
| `qda` or `wf_over_2pt`, using the qDA `z=0` denominator | The same `O0{j}` block, plus `zprime0` through `zprime{N-1}` |
| Accompanying `2pt`, with an explicit two-point input | None |
| Accompanying `2pt`, using the qDA `z=0` denominator | `zprime0` through `zprime{N_2pt-1}`, where `N_2pt` is the two-point atom's state count |

For example, two-state qDA with an explicit two-point denominator has:

```text
log(E0), log(dE1), z0, z1, O00_re, O00_im, O01_re, O01_im
```

With a qDA `z=0` denominator, add `zprime0`, `zprime1`. There is no `O11` in
the qDA model: its matrix-element names have a fixed first index of zero.
If the two-point and qDA atoms use different state counts, take the union of
their spectral and denominator blocks.

### Components, combinations and defaults

The correlator prior builder retains both `_re` and `_im` entries even for a
single-component fit. `component` chooses which data channels enter the
likelihood. Consequently, a key being present in `effective_priors` does not
guarantee that the selected likelihood constrains it; `inactive_prior_overrides`
lists keys absent from the prior dictionary, not every unconstrained parameter.

For a joint scope such as `["2pt+3pt_over_2pt"]`, take the union of the atom
lists. For a chain such as `["2pt", "3pt_over_2pt"]`, each stage starts from
its own list and transfers the preceding posterior under the existing policy.
Shared keys occur only once in a joint model. State indices are determined
separately by each atom's `nstate` choice.

Before the candidate's `prior_width` factor, the existing matrix-element/qDA
defaults are:

| Parameter group | Default mean ± sdev |
| --- | --- |
| `log(E0)` and `_i` / `_f` versions | 0 ± 3 |
| `log(dE{j})` and `_i` / `_f` versions | 0 ± 1 |
| `z{j}`, `z{j}_i`, `z{j}_f`, `zprime{j}` | `(1 ± 10) / 3**j` |
| Ordinary `O..._re`, `O..._im`; independent `D..._re`; FH `O00_re`, `O00_im` | 1 ± 10 |
| qDA `O0{j}_re` | 1 ± 10 |
| qDA `O0{j}_im`; all `sum_...` coefficients | 0 ± 10 |

Built-in matrix-element/qDA defaults describe the conditioned fit variables;
authored overlap and qDA-amplitude overrides are transformed from raw units.
Direct-spectrum defaults are supplied by the recommendation, not this table.

Authored values are supplied before numerical conditioning. The fitter applies
the same conditioning to the priors and data: linear parameters have their
means and standard deviations scaled together, while logarithmic parameters
have their means shifted by the logarithm of the conditioning factor.

### Replica priors and chained fits

`sample_prior_mode: "original"` reuses the base/authored priors, including the
chosen `prior_width`, for each replica in a standalone or initial fit stage.
`"center_posterior"` instead uses the ensemble-center posterior means and
standard deviations multiplied by `posterior_prior_error_scale` for those
replica fits. The choice does not change resampling or the data covariance.

For example, an authored `O00_re` prior of `1.0 ± 0.2` becomes `1.0 ± 0.4`
with `prior_width: [2.0]`. If that fit yields a center posterior standard
deviation of `0.03`, a `center_posterior` replica prior with
`posterior_prior_error_scale: 10.0` has standard deviation `0.3`.
`prior_width` is not applied again to the posterior-derived width.

Ordered entries in `fit_scope` denote chained stages. A later stage receives
the preceding center posterior for its center fit and the corresponding
preceding replica posterior for each replica fit. This transfer occurs in
both sample-prior modes; parameters newly introduced by the later stage
retain their base/authored priors.

For a chain that preserves the preceding posterior standard deviations, use:

```json
{
  "fit_scope": ["2pt", "3pt_over_2pt"],
  "nstate": {"2pt": [2], "3pt_over_2pt": [2]},
  "sample_prior_mode": "original",
  "posterior_prior_error_scale": 10.0,
  "chained_prior_error_scale": 1.0
}
```

If `chained_prior_error_scale` is omitted, chained transfers use
`posterior_prior_error_scale` for compatibility with existing manifests.
Setting it explicitly separates chain transfers from the replica-prior
choice. `prior_width` does not additionally multiply transferred widths.

The direct Python function `fit_spectrum_samples` retains its historical
default `sample_prior_mode="original"`. Manifest-driven direct spectrum jobs
use the manifest default `"center_posterior"`; set `"original"` explicitly
when you want the direct Python behavior.

For existing manifests, ordinary and qDA fits with `prior_width: [1.0]` and
no overrides retain their numerical prior behavior. At other widths, the
extra multiplication of posterior-derived replica widths by `prior_width`
has been removed intentionally. Direct spectrum manifest jobs previously
used original priors for every replica; set `sample_prior_mode: "original"`
to retain that behavior under the shared manifest default.

### Ordinary joint-fit example

This `defaults` fragment fits a shared two-state spectrum and matrix element.
Unspecified priors retain the model defaults. It can be used with the
correlator inputs in `examples/pion_pdf_cg_manifest.json`.

```json
{
  "analysis_method": "lsqfit",
  "component": "re",
  "fit_scope": ["2pt+3pt_over_2pt"],
  "nstate": {"2pt": [2], "3pt_over_2pt": [2]},
  "fitting_form": "Breit",
  "model_average": false,
  "pt2_windows": [{"tmin": 3, "tmax": 12}],
  "pt3_windows": [{"tsep_ls": [8, 10, 12], "tau_cut": 3}],
  "priors": {
    "log(E0)": "-0.9(0.3)",
    "log(dE1)": "-0.7(0.5)",
    "O00_re": "1.0(0.2)"
  },
  "prior_width": [1.0],
  "sample_prior_mode": "original",
  "posterior_prior_error_scale": 10.0,
  "svdcut": 0.000001,
  "q_min": 0.05
}
```

Two-point `tmax` is exclusive: this window uses times 3 through 11. Ordinary
matrix-element jobs still obtain their tuning coordinates through the
parameter-recommendation workflow.

### Fully authored direct two-point fit

For a direct spectrum fit, supplying all physical `E0`, `E1`, … and `z0`,
`z1`, … priors through the largest declared state count, together with
explicit two-point windows, bypasses the initial parameter recommendation.
The workflow scans the Cartesian product of windows, state counts, and prior
widths. Set `metadata.parameter_recommendation_retries` to `0` to prevent
later recommendation calls as well. If no candidate reaches `q_min`, the
workflow can still select a finite candidate under its fallback policy.

This complete manifest can be saved at the repository root and uses the
existing example descriptor. Its referenced correlator data must be available
at the paths specified by that descriptor.

```json
{
  "metadata": {
    "run_id": "spectrum_priors",
    "root_directory": ".",
    "artifacts_directory": "runs/spectrum_priors/artifacts",
    "random_seed": 1984,
    "workers": 4,
    "target_observable": "pdf",
    "parton": "quark",
    "resample_mode": "jackknife",
    "bin_size": 1,
    "sample_error_mode": "covariance",
    "parameter_recommendation_retries": 0
  },
  "stages": {
    "correlator_analysis": {
      "defaults": {
        "analysis_method": "lsqfit",
        "component": "re",
        "fit_scope": ["2pt"],
        "nstate": {"2pt": [1, 2]},
        "fitting_form": "Breit",
        "model_average": false,
        "pt2_windows": [{"tmin": 3, "tmax": 12}],
        "pt3_windows": [],
        "priors": {
          "E0": "0.4(0.2)",
          "E1": "0.9(0.3)",
          "z0": "1.0(0.5)",
          "z1": "0.5(0.3)"
        },
        "prior_width": [1.0],
        "sample_prior_mode": "original",
        "posterior_prior_error_scale": 10.0,
        "svdcut": 0.000001,
        "q_min": 0.05
      },
      "jobs": [
        {
          "id": "spectrum_p5",
          "inputs": {
            "correlators": [
              {"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_2pt"}
            ]
          }
        }
      ]
    }
  },
  "systematics": {}
}
```

A partial direct-spectrum `priors` map remains valid: the recommendation
supplies missing base priors, and explicit overrides remain authoritative on
initial and retry fits. Setting retries to zero alone does not bypass an
initial recommendation that is still needed.

### Fit diagnostics

Fit diagnostics record `effective_priors` and `inactive_prior_overrides` so
you can inspect each candidate. Ordinary-fit effective priors are in
conditioned fit units. Direct spectrum diagnostics report physical effective
priors and a separate `fit_priors` map for the internal energy transformation.

Source: `matrix_element_prior`, `matrix_element_fcn`, `_summed_ratio_model`, and
`fit_spectrum_samples` in
[correlator physics](../lamet_agent/stages/correlator_analysis/physics.py).

## Self-renormalization

For `strategy: "self_renormalization"`, `type: "fit"`, the two fits have these
complete parameter lists:

| Fit | Prior names | Default gvar string |
| --- | --- | --- |
| Reference fit, finite term at each z | `g{z}` for every input coordinate | `"0.0(20.0)"` |
| Reference fit, lattice-spacing term at each z | `f1{z}` for every input coordinate | `"0.0(5.0)"` |
| Short-distance matching slope | `m0` | `"0.0(20.0)"` |
| Short-distance matching intercept | `b` | `"0.0(100.0)"` |

For an input grid `[0.1, 0.2, 0.3]` fm, the reference-fit keys are exactly
`g0.1`, `f10.1`, `g0.2`, `f10.2`, `g0.3`, `f10.3`. These are coordinate
suffixes, not state indices. `mu`, `LambdaQCD_gev`, and `d` are separate job
settings; they are not fitted prior parameters.

One map is passed to both reference and short-distance fits. For example:

```json
{"priors": {"g0.1": "0.0(5.0)", "m0": "0.2(0.1)"}}
```

The reference fit uses `g0.1`; the short-distance fit uses `m0`. Each ignores
the other entry. The remaining defaults stay g=0±20, f1=0±5, m0=0±20, b=0±100.
Coordinate keys use the fitter's string representation of each input z value;
the effective-prior diagnostics list the exact keys.

Source: `_fit_factor_result` in
[renormalization physics](../lamet_agent/stages/renormalization/physics.py).

## Fourier tails

The tail implementation is `gi_nla` for input `gfix="GI"` and `cg_nla` for
`gfix="CG"`; both support `scheme_scan.order` values `LA` and `NLA` despite
the `_nla` suffix in their implementation names.

The following table gives the complete amplitude/phase block at **LA**. The
flavor-pair column is ordered as `(psi1_flavor_class, psi2_flavor_class)`.
Here, "nucleon family" means input hadron `nucleon`, `proton`, `lambda`, or `xi0`.

| Observable | Hadron / flavor pair | `scheme_scan.sector` | LA amplitude/phase prior names |
| --- | --- | --- | --- |
| PDF | `pion` | `valence` | `A2`, `A1`, `phi1` |
| PDF | `pion` | `singlet`, `full` | `A2`, `phi2`, `A1`, `phi1`, `A3`, `phi3` |
| PDF | Nucleon family | `valence`, `singlet`, `full` | `A2`, `phi2` |
| DA | `pion`, `(light, light)` | `full` | `A1`, `phi1` |
| DA | Other hadrons with `(light, light)`, or any hadron with `(heavy, heavy)` | `full` | `A1`, `phi1`, `A2`, `phi2` |
| DA | `(light, heavy)` | `full` | `A2`, `phi2` |
| DA | `(heavy, light)` | `full` | `A1`, `phi1` |
| GPD | `pion` | `sea` | `A2`, `phi2`, `At2`, `phit2` |
| GPD | `pion` | `valence`, `singlet`, `full` | `A1`, `phi1`, `A3`, `phi3`, `A2`, `phi2`, `At2`, `phit2` |
| GPD | Nucleon family | `sea`, `valence`, `singlet`, `full` | `A2`, `phi2`, `At2`, `phit2` |

Build the complete prior list from the selected row using these rules:

| Model / order | Complete prior list |
| --- | --- |
| `gi_nla`, LA | LA amplitude/phase block + `Lambda` |
| `gi_nla`, NLA | LA block + a literal `p` suffix for every name in that block + `Lambda` |
| `cg_nla`, LA | LA amplitude/phase block + `Lambda`, `n` |
| `cg_nla`, NLA | LA block + a literal `p` suffix for every name in that block + `Lambda`, `n` |

Thus `A2` adds `A2p`, `phi2` adds `phi2p`, `At2` adds `At2p`, and `phit2`
adds `phit2p`. There are no independent `Lambdap` or `np` parameters.
For a nucleon PDF, the four complete lists are:

```text
GI, LA:  A2, phi2, Lambda
GI, NLA: A2, phi2, A2p, phi2p, Lambda
CG, LA:  A2, phi2, Lambda, n
CG, NLA: A2, phi2, A2p, phi2p, Lambda, n
```

`source_component` does not change this list. Use these physical parameter
names in `priors`; positional internal names such as `u0` are not input keys.
`Lambda0_gev`, the order, sector, hadron, and flavor classes are separate
settings/provenance that select the model, not fitted prior parameters.

The first-pass defaults are: first amplitude 1, remaining amplitudes 0.1,
phases 0, `Lambda=max(0.5-Lambda0_gev, 0.05)`, and CG `n=0.5`. "First" follows
the order in the table, with NLA suffixes appended afterward. Every initial
standard deviation is 3 in the internal coordinate.

For example, `"Lambda": "0.6(0.2)"` gives the existing
internal log-Lambda parameter a mean of log(0.6) and width 0.2. Lambda is the
fitted GeV decay scale before adding `scheme_scan.Lambda0_gev`. This is not a
Gaussian width of 0.2 GeV on Lambda. Amplitudes, phases and the CG power use
their existing bounded coordinate maps. Explicit means must lie strictly
inside those bounds (amplitudes: -20 to 20; phases: -pi to pi; Lambda: positive;
CG n: 0 to 4), and are rejected instead of being silently clipped.
Unspecified first-pass widths remain
3 in those internal coordinates.

Overrides apply during range selection and every subsequent model fit.
`scheme_scan.posterior_prior_error_scale` still controls the subsequent
posterior-derived widths. Diagnostics expose the input convention,
`base_fit_priors` before posterior transfer, and the final `fit_priors`.

Source: `_tail_family`, `_tail_parameter_names`, `_tail_bounds`, and
`_scan_tail_priors` in
[Fourier physics](../lamet_agent/stages/fourier_transform/physics.py).

## Extrapolation

For `operation: "fit"`, every model contains `h0`. Its remaining parameter
names are exactly the union of `x_independent_terms` and `x_dependent_terms`:

| Prior name | Contribution multiplied by its coefficient |
| --- | --- |
| `h0` | Intercept / extrapolated distribution `h0(x)` |
| `a` | `a_s` |
| `a2` | `a_s**2` |
| `a4` | `a_s**4` |
| `ap2` | `(a_s * P)**2` |
| `ap4` | `(a_s * P)**4` |
| `exp_mpi_L` | `exp(-m_pi * L / HBAR_C_GEV_FM)` |
| `exp_sqrt2_mpi_L` | `exp(-sqrt(2) * m_pi * L / HBAR_C_GEV_FM)` |
| `mpi2` | `m_pi**2 - m_pi_phys**2` |
| `mpi4_log_mpi2` | `m_pi**4 * log(m_pi**2) - m_pi_phys**4 * log(m_pi_phys**2)` |
| `inv_p2` | `1 / P**2` |
| `inv_p4` | `1 / P**4` |

Here `a_s` and `L` are numerical values in fm, and P and masses are in GeV.
The `ap2`/`ap4` bases use those numerical values without an hbar-c conversion;
priors on their coefficients retain that existing convention.

For example, `x_independent_terms: ["a2"]` and
`x_dependent_terms: ["inv_p2", "mpi2"]` give exactly
`h0`, `a2`, `inv_p2`, `mpi2`. Use these names directly, without a `c_` prefix,
`_re`/`_im` suffix, or an x-coordinate suffix. `h0` is always x-dependent;
the term lists determine whether each other coefficient is shared across x or
pointwise. `x_covariance` changes the solver's covariance treatment, not the
parameter names. Unselected correction names are ignored by that candidate.

```json
{"priors": {"h0": "0.8(0.5)", "inv_p2": "0.0(1.0)"}}
```

Missing coefficients keep N(0,3²). For an x-dependent coefficient, the supplied
mean and width apply independently at each x; for an x-independent coefficient,
they define its one shared prior. Both covariance modes use the same mapping.

A common string `"priors": "0.0(3.0)"` applies to every active coefficient.
Use either that form or a mapping of parameter names to strings. The former
`{mean, sdev}` input objects are no longer accepted. The linear solver and center-posterior
replica policy are unchanged.

Source: `basis_terms` and `fit_candidate` in
[extrapolation physics](../lamet_agent/stages/extrapolation/physics.py), and
`_EXTRAPOLATION_TERMS` in its [contract](../lamet_agent/stages/extrapolation/contract.py).

## Dispersion report

The report uses one model, `E² = m2 + k2 p² + k3 a² p⁴`, with exactly:

| Prior name | Meaning | Default mean ± sdev |
| --- | --- | --- |
| `m2` | Squared mass / intercept, in GeV² | Minimum input mean E² ± 10 |
| `k2` | Coefficient of p² | 1 ± 10 |
| `k3` | Coefficient of a²p⁴, with a in GeV⁻¹ | 0 ± 10 |

Include `m2`, `k2`, or `k3` in the correlator jobs' `priors` (normally in stage
defaults). Correlator fits ignore these names; the report uses them when fitting
the corresponding ensemble's momenta. Definitions supplied by multiple jobs
in one ensemble must agree for each parameter. Conflicting definitions omit
that fit band with an explanation in the report. Effective dispersion priors
are recorded beside the plot. The usual aligned-sample and momentum-count
requirements still apply.

Source: `_dispersion_model` and `_dispersion_lines` in
[correlator reporting](../lamet_agent/stages/correlator_analysis/reporting.py).

## Steps without configurable fit priors

| Stage / mode | Prior parameters |
| --- | --- |
| Correlator `analysis_method="lanczos"` | None; the Lanczos algorithm does not use Gaussian fit priors. |
| Renormalization `type="apply"`, including external denominators | None; the supplied factors and application settings are not prior parameters. |
| Extrapolation `operation="systematics_budget"` | None; it combines completed results without a fit. |
| Matching, for all matching kernels | None; kernel options belong in `kernel_parameters`. |
| Review | None. |
| Soft Analysis | No implemented numerical models or prior parameters. |
