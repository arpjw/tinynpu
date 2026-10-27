"""Export an allowlisted static viewer from a verified experiment manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from tinynpu.evidence import ROOT, sha256, write_json

PUBLIC_FILES = {
    "index.html",
    "app.js",
    "style.css",
    "demo.json",
    "provenance.json",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "vercel.json",
}


def export_site(run: Path, output: Path) -> dict:
    """Verify frozen evidence, then copy only browser assets and public provenance."""
    run = run.resolve()
    if output.is_symlink():
        raise ValueError("Export directory must not be a symlink")
    output = output.resolve()
    if output == run or output in run.parents or run in output.parents:
        raise ValueError("Export directory must be separate from experiment evidence")
    if output.exists():
        unexpected = {p.name for p in output.iterdir()} - PUBLIC_FILES - {".vercel", ".gitignore"}
        if unexpected or any((output / name).is_symlink() for name in PUBLIC_FILES):
            raise ValueError("Export directory contains unexpected files or symlinks")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "passed":
        raise ValueError("Only passed experiments can be exported")
    for name, record in manifest["files"].items():
        path = run / name
        if path.resolve().parent != run or not path.is_file():
            raise ValueError("Invalid manifest path")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Manifest verification failed: {name}")
    required = {"demo.json", "summary.json", "freeze.json"}
    if not required <= manifest["files"].keys():
        raise ValueError("Manifest is missing required evidence")
    demo = json.loads((run / "demo.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    freeze = json.loads((run / "freeze.json").read_text())
    source_id = hashlib.sha256(
        json.dumps(freeze["source_files"], sort_keys=True).encode()
    ).hexdigest()
    if not (source_id == manifest["source_id"] == summary["source_id"] == freeze["source_id"]):
        raise ValueError("Evidence source identifiers disagree")
    if demo["summary"] != summary or summary["rtl_mismatches"] != 0:
        raise ValueError("Demo summary does not match verified results")
    if not summary["source_unchanged"] or not summary["frozen_inputs_unchanged"]:
        raise ValueError("Experiment inputs changed during evaluation")
    output.mkdir(parents=True, exist_ok=True)
    html = (
        (ROOT / "viewer/index.html")
        .read_text()
        .replace('content="../artifacts/run/demo.json"', 'content="demo.json"')
        .replace(
            'href="https://github.com/arpjw/tinynpu/blob/main/THIRD_PARTY_NOTICES.md"',
            'href="THIRD_PARTY_NOTICES.md"',
        )
    )
    (output / "index.html").write_text(html)
    for name in ("app.js", "style.css"):
        (output / name).write_bytes((ROOT / "viewer" / name).read_bytes())
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        (output / name).write_bytes((ROOT / name).read_bytes())
    # JSON whitespace is removed; every recorded value remains unchanged.
    (output / "demo.json").write_text(
        json.dumps(demo, separators=(",", ":"), allow_nan=False) + "\n"
    )
    write_json(
        output / "vercel.json",
        {
            "$schema": "https://openapi.vercel.sh/vercel.json",
            "framework": None,
            "buildCommand": None,
            "installCommand": None,
            "headers": [
                {
                    "source": "/(.*)",
                    "headers": [
                        {"key": "X-Content-Type-Options", "value": "nosniff"},
                        {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"},
                        {
                            "key": "Content-Security-Policy",
                            "value": (
                                "default-src 'self'; script-src 'self'; style-src 'self'; "
                                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                                "base-uri 'none'; frame-ancestors 'none'"
                            ),
                        },
                    ],
                }
            ],
        },
    )
    provenance = {
        "schema_version": 1,
        "experiment_source_id": source_id,
        "completed_at": manifest["completed_at"],
        "original_demo_sha256": manifest["files"]["demo.json"]["sha256"],
        "files": {
            name: sha256(output / name) for name in sorted(PUBLIC_FILES - {"provenance.json"})
        },
    }
    write_json(output / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    provenance = export_site(args.run, args.output)
    print(f"Exported {len(PUBLIC_FILES)} public files; source {provenance['experiment_source_id']}")


if __name__ == "__main__":
    main()
