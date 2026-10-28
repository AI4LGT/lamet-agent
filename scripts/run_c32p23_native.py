"""Run native fitting/tail/Fourier jobs with a real LLM backend, then combine components."""

import argparse
import json
from pathlib import Path

from lamet_agent.agent import create_session
from lamet_agent.data import EnsembleData
from lamet_agent.llm import create_backend
from lamet_agent.manifest import load_manifest
from lamet_agent.stages.tmdwf_analysis.native import combine_transforms
from lamet_agent.stages.tmdwf_analysis.workflow import plot
from lamet_agent.ui import PlainUi, log


class FixedSuggestions:
    """Offline integration fixture, explicitly not an LLM or scientific selection."""

    identity = "offline-fixed-suggestions-no-llm"

    def complete(self, *, response_schema=None, **kwargs):
        from lamet_agent.llm import _AssistantResponse

        values = {"tune_z_values": [1.0, 6.0, 12.0],
                  "pt2_windows": [{"tmin": t, "tmax": 15} for t in (8, 10, 12)],
                  "zmin_fm": [0.7371, 0.9477], "zmax_fm": [1.2636]}
        if response_schema is None or set(response_schema["schema"]["properties"]) - set(values):
            raise RuntimeError("Offline fixture received an unsupported recommendation request")
        response = {key: values[key] for key in response_schema["schema"]["properties"]}
        return _AssistantResponse(text=json.dumps(response), structured=response)

    def close(self):
        pass


def combine(folder):
    stage = folder / "03_fourier_transform"
    real = EnsembleData.from_netcdf(stage / "fourier_real/output.nc")
    imag = EnsembleData.from_netcdf(stage / "fourier_imag/output.nc")
    result = combine_transforms(real, imag)
    result.to_netcdf(folder / "quasi_tmdwf.nc")
    plot(result, folder / "quasi_tmdwf.png")
    (folder / "combination.json").write_text(json.dumps({
        "inputs": ["03_fourier_transform/fourier_real/output.nc", "03_fourier_transform/fourier_imag/output.nc"],
        "formula": "real + 1j * imag (paired samples)", "sample_count": result.n_sample,
        "x_count": len(result.coords["x"]), "resample_id": result.attrs["resample_id"],
        "matching_applied": False, "factor_uncertainty": result.attrs["factor_uncertainty"],
    }, indent=2), encoding="utf-8")
    log(str(folder / "quasi_tmdwf.nc"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--provider", default="codex")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--artifacts-directory", help="Output path relative to the manifest root")
    parser.add_argument("--fixed-suggestions", action="store_true", help="Local workflow test; does not call an LLM")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.artifacts_directory:
        manifest.document["metadata"]["artifacts_directory"] = args.artifacts_directory
    issues = manifest.validate()
    if issues:
        raise ValueError("Invalid manifest: " + str(issues))
    meta = manifest.document["metadata"]
    if args.fixed_suggestions:
        meta["artifacts_directory"] += "_offline"
    root = (args.manifest.resolve().parent / meta["root_directory"]).resolve()
    folder = root / meta["artifacts_directory"]
    if not args.combine_only:
        backend = FixedSuggestions() if args.fixed_suggestions else create_backend(args.provider, args.model)
        try:
            session = create_session(backend, ui=PlainUi(), progress_mode="stage")
            session.run_manifest(manifest)
        finally:
            backend.close()
    combine(folder)


if __name__ == "__main__":
    main()
