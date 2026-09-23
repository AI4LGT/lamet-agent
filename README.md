# lamet-agent

<p align="right">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="docs/lamet-agent-demo.gif" alt="lamet-agent demo" width="800" />
</p>

`lamet-agent` is a Python-first framework for reproducible **La**rge **M**omentum **E**ffective **T**heory (LaMET) and lattice QCD
analysis workflows.

## Quick Start

Requires a logged-in Codex CLI on this machine. Codex does not use
`--api-key-file`.

```bash
git clone https://github.com/AI4LGT/lamet-agent.git && cd lamet-agent
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip && python3 -m pip install -e ".[codex]"
wget --user=download --password=protonpdf -r -np -nH --no-check-certificate \
  https://149.28.115.134:43999/data_pion_pdf_cg.zip && unzip data_pion_pdf_cg.zip
lamet-agent run examples/pion_pdf_cg_manifest.json \
  --provider codex --model gpt-5.6-luna
```

To compare the run against the reference result:

```bash
cd runs/pion_pdf_cg/ && wget --user=download --password=protonpdf -r -np -nH --no-check-certificate \
  https://149.28.115.134:43999/plot_pion_pdf_compare.py && python plot_pion_pdf_compare.py
```

The archive unpacks to `data_pion_pdf_cg/` at the repository root.
The same files can be downloaded in a browser at
[https://149.28.115.134:43999](https://149.28.115.134:43999)
with user `download` and password `protonpdf`.
The server has no trusted certificate: `wget` uses
`--no-check-certificate`, and a browser must trust the self-signed
certificate.

Alternatively with `uv`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[codex]"
```

### Other providers

API providers (`openai`, `anthropic`, `gemini`, `grok`, `deepseek`, or a custom
HTTP(S) OpenAI-compatible URL) need `python3 -m pip install -e .` (no `[codex]`
extra) and a key via `--api-key-file` or the provider's environment variable.
See [Providers and models](#providers-and-models) for details.

```bash
lamet-agent run examples/pion_pdf_cg_manifest.json \
  --provider openai --model gpt-5.6-luna \
  --api-key-file api.key
```

### Example manifests

| Manifest                                     | Workflow                                                       | Data reference |
| -------------------------------------------- | -------------------------------------------------------------- | -------------- |
| `examples/pion_pdf_cg_manifest.json`         | Coulomb-gauge pion PDF, least-squares correlator analysis.     | [^1]           |
| `examples/pion_pdf_cg_lanczos_manifest.json` | Coulomb-gauge pion PDF with nested-bootstrap Lanczos analysis. | [^1]           |
| `examples/pion_pdf_gi_manifest.json`         | Gauge-invariant pion PDF.                                      | [^1]           |
| `examples/pion_da_gi_manifest.json`          | Gauge-invariant pion DA with systematic variants.              | [^2]           |
| `examples/kaon_da_gi_manifest.json`          | Gauge-invariant kaon DA with systematic variants.              | [^2]           |

Other example archives are on the same host; pick the zip that matches the
`data_*` directory used by that manifest (`data_pion_pdf_cg`,
`data_pion_pdf_gi`, `data_pion_da_gi`, `data_kaon_da_gi`).

[^1]: Xiang Gao, Wei-Yang Liu, and Yong Zhao, [*Parton Distributions from Boosted Fields in the Coulomb Gauge*](https://arxiv.org/pdf/2306.14960), arXiv:2306.14960.
[^2]: Jun Hua et al., [*Pion and Kaon Distribution Amplitudes from Lattice QCD*](https://arxiv.org/pdf/2201.09173), arXiv:2201.09173.

## Command Line

`lamet-agent` and `python -m lamet_agent` expose the same interface:

```text
lamet-agent {validate,plan,run} ...
```

### Validate

```bash
lamet-agent validate MANIFEST
```

Validate reads JSON or JSONC and performs no LLM communication. It checks the
manifest envelope, stage contracts, systematics declarations, job DAG, paths,
correlator descriptors, and kernel parameters.

### Plan

```bash
lamet-agent plan MANIFEST \
  --provider PROVIDER \
  [--model MODEL] \
  [--api-key-file FILE] \
  [--output FILE | --in-place]
```

Plan completes an incomplete manifest through an interactive LLM conversation.
It validates proposed changes and presents a final natural-language summary for
explicit user confirmation.

Options:

- `--provider`: registered provider name or an HTTP(S) OpenAI-compatible URL;
- `--model`: model ID; prompts for selection when omitted or unavailable;
- `--api-key-file`: text file containing only the API key;
- `--output`: output path;
- `--in-place`: overwrite the source after explicit acceptance.

`--output` and `--in-place` are mutually exclusive. A planned output must remain
beside its source manifest so relative input paths preserve their meaning.
If neither option is set, Plan first asks for an output filename, resolved relative
to the source manifest's directory. The terminal displays that directory as a fixed
prefix relative to the current working directory. Only the filename is editable,
prefilled with `<manifest>.planned.json`;
edit it or press Enter to accept. Clear the input to save in place; in plain-text
mode, empty input also selects in-place. Any output pointing to the source file
requires a separate overwrite confirmation (default No), including `--in-place`.
Saving still requires final acceptance.

Terminal controls include `/show`, `/issues`, `/undo`, `/edit`, `/save`,
`/help`, and `/quit`. `Enter` submits, `Shift+Enter` inserts a newline, and
`Ctrl+C` cancels.

Standalone Plan writes the accepted manifest and exits without running analysis
stages.
Plan does not save LLM transcripts by default. Pass `--plan-log-dir DIRECTORY`
to `plan` or `run` to enable them. The directory is resolved relative to the CLI
working directory; each Plan session gets a manifest-name and timestamp subdirectory.
The UI prints a relative path to the transcript. It records requests, responses,
rejected replies, failure reasons, and durations without overwriting earlier sessions.

### Run

```bash
lamet-agent run MANIFEST \
  --provider PROVIDER \
  [--model MODEL] \
  [--api-key-file FILE] \
  [--progress {auto,stage,job,none}]
```

Run validates before executing numerical stages. If validation fails, it enters
Plan with the selected provider. After the user accepts a valid repaired
manifest, numerical execution continues automatically.

The default progress mode is `auto`:

- `auto`: stage-level job progress when systematics are declared; otherwise
  progress is owned by each numerical job;
- `stage`: one job counter for each stage;
- `job`: stage-specific numerical progress;
- `none`: disable progress bars.

### Providers and models

#### Codex CLI

The `codex` provider uses the optional `openai-codex>=0.147` package and the cached
Codex login on the current machine. It does not use an API key. `--model` is
optional; available models are discovered from the Codex app server.

#### OpenAI-compatible APIs

The registered API providers are `openai`, `anthropic`, `gemini`, `grok`, and
`deepseek`. Each reads its API key from `--api-key-file` or the corresponding
environment variable:

| Provider  | Environment variable |
| --------- | -------------------- |
| OpenAI    | `OPENAI_API_KEY`     |
| Anthropic | `ANTHROPIC_API_KEY`  |
| Gemini    | `GEMINI_API_KEY`     |
| Grok      | `GROK_API_KEY`       |
| DeepSeek  | `DEEPSEEK_API_KEY`   |

All providers discover their available models before selection: API providers use
`/models`, Codex uses app-server `model/list`, and Claude Code uses the SDK server
initialization catalog (including aliases and resolved model IDs). Omit `--model`
or supply an unavailable ID to choose a model by number or name in the shared UI.
No model is selected automatically, even when only one is available. Library
callers can pass a `select_model` callback to `create_backend`; without one,
missing or unavailable models raise an error listing the available choices.
HTTP(S) OpenAI-compatible base URLs can also be passed directly as the provider.

### Model Context Protocol

`lamet-agent` ships an MCP server, so a coding agent can validate, plan, and run
manifests as ordinary tool calls instead of shelling out. It speaks stdio
JSON-RPC and needs no extra dependency:

```bash
lamet-agent mcp                      # serve on stdio; the client launches this
lamet-agent install-mcp --dry-run    # report what registration would change
lamet-agent install-mcp              # register with every detected harness
```

`install-mcp` writes one server entry for each harness it finds and is safe to
re-run: an entry it wrote is refreshed in place, and an entry it did not write
is reported rather than overwritten. Use `--target codex|claude|dsh` to limit
it and `--server-name` to register under a different name.

| Harness | Where the entry goes |
| --- | --- |
| Codex | `[mcp_servers.<name>]` in `~/.codex/config.toml`, plus the MCP feature flag |
| Claude Code | registered under the `user` scope through the `claude` CLI |
| DeepSeek Harness | a loader row in `~/.dsh/profiles/*/cordis.patch.yml` |

**Codex defers MCP tools behind tool search.** Registering the server and
enabling the feature flag are both necessary but still not sufficient to see the
tools: Codex keeps MCP tools out of the model's function list, so a question
like "which MCP tools do you have?" truthfully answers that there are none. The
tools load on demand instead:

```text
tool_search("validate_manifest")  ->  mcp__lamet__validate_manifest
```

Because of this, ask Codex to *search* for the tool, or name the tool you want
and let it search:

> Use tool_search for "validate_manifest", then call the lamet tool it returns
> on examples/pion_pdf_cg_manifest.json and report the raw output.

Diagnose the three layers separately when something looks missing:

```bash
codex mcp list | grep lamet                    # 1. server registered?
codex features list | grep mcp_2026_07_28      # 2. MCP exposure enabled?
# 3. tools load on demand -- use tool_search rather than listing
```

A tool that is merely deferred is not a broken tool. Note also that a call is
still subject to the harness's approval and sandbox policy, which is independent
of registration.

The command is registered with an absolute launch path, because these harnesses
spawn the server with a scrubbed environment where a `PATH` lookup is not
reliable. Codex and the DeepSeek Harness read their server list at startup, so
restart or reload them afterwards; Claude Code picks it up per session. The
DeepSeek Harness entry raises its per-call timeout, since a real analysis runs
far longer than the 60-second default.

The exposed tools are `validate_manifest`, `plan_manifest`, `run_manifest`,
`read_run`, and `list_correlators`. They report the CLI's own verdicts; none of
them re-implements validation. `plan_manifest` reports an error rather than
hanging, because Plan ends with an interactive acceptance review that a tool
call cannot answer — use `run_manifest`, which repairs and executes in one step.

## Core Idea

The manifest contains run metadata and an ordered mapping of stage job lists.
Job ids form a DAG: correlator jobs select raw records, and later jobs consume
earlier outputs through role-named inputs such as `target`, `denominator`,
`input`, and `quasi`.

Expected agent behavior:

- Validate the complete authored workflow before numerical execution.
- Run numerical stages deterministically once their parameters are known.
- Consult the LLM only when a workflow needs fit or range recommendations.
- Write intermediate NetCDF data, diagnostics, plots, and stage reports so the
  complete analysis path remains inspectable.
- Base the final Review on numerical evidence, consistency checks, and selected
  literature.

Implemented stage families, normally authored in this order, are:

1. `correlator_analysis`
2. `renormalization`
3. `fourier_transform`
4. `perturbative_matching`
5. `extrapolation`
6. `review`

A partial workflow may omit unneeded stages. The order of keys under `stages`
is the execution order; there is no separate `metadata.stages` list.

Architecture, file ownership, and contributor workflows are documented in
[`DEVELOPMENT.md`](DEVELOPMENT.md).

## Intermediate Data (NetCDF)

Stage-to-stage numerical artifacts are stored as NetCDF files. Every array has:

- a leading `resample` dimension;
- a sampling mode: `raw`, `jackknife`, `bootstrap`, or `gvar`;
- physical dimensions and coordinates such as `t`, `tsep`, `tau`, `z`, `x`,
  `a`, or momentum;
- ensemble and stage provenance stored as attributes.

A job may also use an external `{ "file": ".../output.nc" }` artifact as its
input.

Typical per-job files are:

| File                  | Purpose                                               |
| --------------------- | ----------------------------------------------------- |
| `output.nc`           | Primary sample-bearing numerical result.              |
| `summary.json`        | Decisions, diagnostics, and declared artifacts.       |
| `llm_transcript.md`   | Recorded LLM requests and responses, when applicable. |
| `diagnostics/*`       | Candidate tables and numerical diagnostics.           |
| `plots/*`             | PDF/SVG result and fit-quality figures.               |

Stage directories also receive an aggregate `report.md`; job directories do not
write report files. Review writes its final `review.md`, `review_bundle.json`,
and consistency/literature evidence.

### Inspect or read without lamet-agent

NetCDF is self-describing and can be inspected with `ncdump`, Panoply, or
xarray:

```python
import xarray as xr

array = xr.load_dataarray("output.nc", auto_complex=True)
print(array.dims)
print(array.coords)
print(array.attrs)
```

The first dimension is always `resample`; the remaining dimensions describe the
physical layout documented by the corresponding stage report.

## Manifest Example

The loader accepts JSON and JSONC comments. The current manifest envelope is:

```json
{
  "metadata": {
    "run_id": "pion_pdf_cg",
    "root_directory": "..",
    "artifacts_directory": "runs/pion_pdf_cg/artifacts",
    "random_seed": 1984,
    "workers": 4,
    "target_observable": "pdf",
    "parton": "quark",
    "resample_mode": "jackknife",
    "bin_size": 1,
    "sample_error_mode": "covariance"
  },
  "stages": {
    "correlator_analysis": {
      "defaults": {},
      "jobs": [
        {
          "id": "ca_p5",
          "inputs": {
            "correlators": [
              {"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_2pt"},
              {"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_3pt"}
            ]
          }
        }
      ]
    }
  },
  "systematics": {}
}
```

The three top-level objects are:

- `metadata`: run-wide paths, target identity, resampling, errors, seed, and
  worker count;
- `stages`: the ordered stage/job graph;
- `systematics`: optional stage-owned variant declarations.

Each stage contains shared `defaults` and an ordered `jobs` list. A job contains
its global `id`, role-named `inputs`, and parameter overrides directly on the
job. Stage defaults fill omitted job fields; explicit job values remain
authoritative.

Input values may be:

- an earlier job id;
- `{ "file": "path/to/output.nc" }`;
- `{ "json": "descriptor.json", "id": "correlator_record" }`;
- a numeric constant where the receiving contract permits one;
- an ordered list where the receiving role permits multiple sources.

For correlator inputs that use a descriptor JSON record, see
[Standard Correlator HDF5 Format](#standard-correlator-hdf5-format) for the
input-file conventions.

Unknown fields, invalid choices, broken input roles, duplicate ids, forward job
references, missing paths, and cross-parameter inconsistencies are rejected.

Run-wide metadata fields include:

- required: `run_id`, `root_directory`, `artifacts_directory`, `random_seed`,
  `workers`, `target_observable`, `resample_mode`, `sample_error_mode`, and
  `bin_size`;
- `parton`, currently `quark`, with that value as its default;
- `samples`, required only for bootstrap mode;
- `parameter_recommendation_retries`, defaulting to one extra attempt per job.

`target_observable` accepts `pdf`, `da`, and `gpd`. `sample_error_mode` accepts
`covariance`, `variance`, and bootstrap-only `one_sigma`.

Systematic variants are currently supported for Fourier, matching, and
extrapolation. They are expanded into concrete jobs before execution and saved
in `resolved_manifest.json`.

## Standard Correlator HDF5 Format

Correlator jobs select a record from a descriptor JSON:

```json
{"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_3pt"}
```

### Descriptor example

This complete example defines one two-point correlator:

```json
{
  "correlators": [
    {
      "id": "p5_2pt",
      "ensemble": {"series": "HISQ", "id": "HISQa060_X", "a_s": 0.06,
                   "a_t": 0.06, "L_s": 48, "L_t": 64, "m_pi": 0.3},
      "count": 109,
      "format": "hdf5",
      "path": "correlators/pion_2pt.h5",
      "dataset": "g5/g5/PX5PY0PZ0",
      "dataset_dims": ["t", "configuration"],
      "dims": ["configuration", "t"],
      "coords": {"t": [0, 1, 2, 3]},
      "selectors": {"source_operator": "g5", "sink_operator": "g5",
                    "momentum": "PX5PY0PZ0", "gfix": "CG"},
      "correlator_type": "two_point",
      "hadron": {"name": "pion"},
      "source_momentum": [5, 0, 0],
      "sink_momentum": [5, 0, 0],
      "current": null,
      "source_sink_separation": null
    }
  ]
}
```

`path` locates the HDF5 file relative to the descriptor JSON; `dataset` locates
the array inside it:

```text
pion_2pt.h5
└── g5
    └── g5
        └── PX5PY0PZ0     dataset, shape (4, 109)
```

The leaf axes are `(t, configuration)`, as declared by `dataset_dims`.

### Dataset paths and dimensions

`dims` begins with `configuration`, while `coords` supplies every other axis.
`dataset_dims` gives the axis order stored in each HDF5 leaf. A `dataset`
template can place coordinate values in the HDF5 path:

```json
{
  "dataset": "g5/g5/gT_nonlocal/PX5PY0PZ0/tsep{tsep}/bT0/bz{z}",
  "dataset_dims": ["tau", "configuration"],
  "dims": ["configuration", "tsep", "tau", "z"],
  "coords": {"tsep": [8, 10, 12], "tau": [0, 1, 2, 3], "z": [0, 1, 2]}
}
```

This expands `tsep` and `z` into separate leaves:

```text
pion_3pt.h5
└── g5/g5/gT_nonlocal/PX5PY0PZ0
    ├── tsep8/bT0/{bz0,bz1,bz2}
    ├── tsep10/bT0/{bz0,bz1,bz2}
    └── tsep12/bT0/{bz0,bz1,bz2}
```

Each `bz*` leaf stores `(tau, configuration)`. The assembled output is ordered
as `(configuration, tsep, tau, z)`. The group names themselves are unrestricted;
only the `dataset` template defines the hierarchy.

### Correlator types

| Type          | Common dimensions             | Additional requirements                                                                   |
| ------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| `two_point`   | `configuration, t`            | `current` must be `null`.                                                                 |
| `three_point` | `configuration, tsep, tau, z` | `current` is required; either provide a `tsep` dimension or set `source_sink_separation`. |
| `qda`         | `configuration, t, z`         | `current` is required.                                                                    |

Momenta are integer triples. A non-null `current` contains exactly
`kernel_operator`, `parton`, and `renormalization_scheme`. The descriptor is
authoritative for all coordinates and provenance fields. Selected records in
one job must share the same ensemble and configuration count.

See `examples/pion_pdf_cg_correlators.json` and
`examples/pion_da_gi_correlators.json` for complete descriptors.

## Development

Install the development dependencies with either uv or pip:

```bash
uv pip install -e ".[dev]"
# or
python -m pip install -e ".[dev]"
```

Architecture, file ownership, testing, and contributor workflows are documented
in [`DEVELOPMENT.md`](DEVELOPMENT.md).

## Related Links

- [LQCD_Master](https://github.com/sjtu-sai-agents/LQCD_Master) ([arXiv:2607.15001](https://arxiv.org/abs/2607.15001))
