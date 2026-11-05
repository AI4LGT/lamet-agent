# Correlator fit priors

Least-squares correlator jobs (`analysis_method: "lsqfit"`) accept explicit
Gaussian priors and separate controls for replica fits and chained fits. Put
these fields in `stages.correlator_analysis.defaults` or directly on a job.
The examples below illustrate the configuration; choose numerical priors and
fit windows for your data.

## Controls and defaults

| Field | Manifest default | Meaning |
| --- | --- | --- |
| `priors` | `{}` | Partial overrides of named model parameters, each with `mean` and `sdev`. |
| `prior_width` | `[1.0]` | Candidate factors multiplying every base or authored prior standard deviation exactly once. |
| `sample_prior_mode` | `"center_posterior"` | Selects `"original"` or `"center_posterior"` priors for replica fits. |
| `posterior_prior_error_scale` | `10.0` | Width factor for replica priors derived from the ensemble-center posterior; also the fallback for chained transfers. |
| `chained_prior_error_scale` | Omitted | Optional independent width factor for transferring the preceding fit-stage posterior. |

With no overrides and `prior_width: [1.0]`, existing base prior means and
standard deviations are preserved. Multiple widths scan multiple candidates;
they do not apply successive multiplications. Width and error-scale factors
must be finite and positive.

`priors` replaces only the parameters it names. Each entry must contain
exactly a finite scalar `mean` and a positive finite scalar `sdev`. Arrays,
booleans, unknown names, and misspelled names are rejected. The allowed names
are the union of the declared model and state-count candidates. For example,
an override for an excited-state parameter is accepted when scanning one and
two states; it is reported as inactive for the one-state candidate.

## Parameter names and units

Use the parameter names of the selected model:

| Fit | Example prior names | Units |
| --- | --- | --- |
| Ordinary Breit matrix-element fit, including joint or chained two-point stages | `log(E0)`, `log(dE1)`, `z0`, `z1`, `O00_re` | Energy and gap priors use logarithmic units; overlaps and matrix elements use raw fit-parameter units. |
| Direct two-point spectrum, `fit_scope: ["2pt"]` | `E0`, `E1`, `z0`, `z1` | Energies in lattice units (`a_t E`) and raw overlap amplitudes, before the internal energy transformation. |

For example, `"log(E0)": {"mean": -0.9, "sdev": 0.2}` specifies a Gaussian
on the logarithm of the energy. It does not specify an energy of `-0.9`.
There is no automatic conversion of an ordinary-fit `E0` key to `log(E0)`.
NonBreit models distinguish source and sink parameters with `_i` and `_f`
suffixes, such as `log(E0_i)` and `z0_f`; use the actual model names.

Authored values are supplied before numerical conditioning. The fitter applies
the same conditioning to the priors and data: linear parameters have their
means and standard deviations scaled together, while logarithmic parameters
have their means shifted by the logarithm of the conditioning factor.

Direct spectrum energy means must be positive and strictly ordered, and
overlap means must be positive. The fitter converts the physical energy
priors to its internal ground-energy and positive-gap parameterization.

## Ordinary joint-fit example

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
    "log(E0)": {"mean": -0.9, "sdev": 0.3},
    "log(dE1)": {"mean": -0.7, "sdev": 0.5},
    "O00_re": {"mean": 1.0, "sdev": 0.2}
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

## Replica priors and chained fits

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

## Fully authored direct two-point fit

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
          "E0": {"mean": 0.4, "sdev": 0.2},
          "E1": {"mean": 0.9, "sdev": 0.3},
          "z0": {"mean": 1.0, "sdev": 0.5},
          "z1": {"mean": 0.5, "sdev": 0.3}
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

Fit diagnostics record `effective_priors` and `inactive_prior_overrides` so
you can inspect each candidate. Ordinary-fit effective priors are in
conditioned fit units. Direct spectrum diagnostics report physical effective
priors and a separate `fit_priors` map for the internal energy transformation.
