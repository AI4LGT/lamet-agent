"""Run a validated, deterministic TMDWF-only manifest through the real agent loop."""

import argparse
from pathlib import Path

from lamet_agent.agent import create_session
from lamet_agent.manifest import load_manifest
from lamet_agent.ui import PlainUi


class NoLlmBackend:
    identity = "deterministic-tmdwf-no-llm"

    def complete(self, **kwargs):
        raise RuntimeError("This fixed-parameter runner must never request an LLM")

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifacts-directory", help="New output path relative to the manifest root")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.artifacts_directory:
        manifest.document["metadata"]["artifacts_directory"] = args.artifacts_directory
    if set(manifest.document["stages"]) != {"tmdwf_analysis"}:
        parser.error("Only the deterministic tmdwf_analysis stage is supported by this runner")
    session = create_session(NoLlmBackend(), ui=PlainUi(), progress_mode="stage")
    session.run_manifest(manifest)


if __name__ == "__main__":
    main()
