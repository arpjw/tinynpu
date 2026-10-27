"""Publishing must reject corrupted evidence and exclude private run metadata."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tinynpu.evidence import ROOT, sha256, write_json

spec = importlib.util.spec_from_file_location("export_site", ROOT / "scripts/export_site.py")
assert spec and spec.loader
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.fixture
def run(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    sources = {"rtl/pe.sv": "test-fixture"}
    source_id = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    summary = {
        "source_id": source_id,
        "rtl_mismatches": 0,
        "source_unchanged": True,
        "frozen_inputs_unchanged": True,
    }
    write_json(path / "summary.json", summary)
    write_json(path / "demo.json", {"schema_version": 1, "summary": summary, "examples": []})
    write_json(
        path / "freeze.json",
        {"source_id": source_id, "source_files": sources, "resolved_output": "/private/example"},
    )
    (path / "Varray").write_bytes(b"private executable")
    files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in path.iterdir()}
    write_json(
        path / "manifest.json",
        {"status": "passed", "source_id": source_id, "completed_at": "fixture", "files": files},
    )
    return path


def test_export_allowlist_and_provenance(run: Path, tmp_path: Path) -> None:
    output = tmp_path / "site"
    provenance = export.export_site(run, output)
    assert {p.name for p in output.iterdir()} == export.PUBLIC_FILES
    assert json.loads((output / "demo.json").read_text()) == json.loads(
        (run / "demo.json").read_text()
    )
    assert 'content="demo.json"' in (output / "index.html").read_text()
    assert all(sha256(output / name) == digest for name, digest in provenance["files"].items())
    assert "/private/example" not in (output / "provenance.json").read_text()


def test_corrupt_evidence_is_rejected_before_export(run: Path, tmp_path: Path) -> None:
    (run / "demo.json").write_text("{}")
    output = tmp_path / "site"
    with pytest.raises(ValueError, match="Manifest verification failed"):
        export.export_site(run, output)
    assert not output.exists()


def test_unexpected_output_files_are_not_published(run: Path, tmp_path: Path) -> None:
    output = tmp_path / "site"
    output.mkdir()
    (output / ".env").write_text("private fixture")
    with pytest.raises(ValueError, match="unexpected"):
        export.export_site(run, output)
    assert (output / ".env").read_text() == "private fixture"


def test_evidence_cannot_be_overwritten(run: Path) -> None:
    with pytest.raises(ValueError, match="separate"):
        export.export_site(run, run)
