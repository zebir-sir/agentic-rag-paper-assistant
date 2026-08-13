from pathlib import Path
import asyncio

import pytest

pytest.importorskip("fastapi")

from agent import ingestion_jobs


@pytest.mark.asyncio
async def test_ingest_saved_pdf_file_uses_isolated_single_file_directory(monkeypatch):
    workspace_tmp = Path("D:/agent/agentic_rag_project-main2/.tmp")
    source_dir = workspace_tmp / "single-file-source"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = source_dir / "target.pdf"
    sibling_pdf = source_dir / "other.pdf"
    target_pdf.write_bytes(b"%PDF-1.7 target")
    sibling_pdf.write_bytes(b"%PDF-1.7 other")

    observed = {}

    class _Process:
        returncode = 0

        async def wait(self):
            return 0

        def terminate(self):
            return None

    async def fake_exec(*command, **_kwargs):
        import_dir = Path(command[command.index("--documents") + 1])
        observed["import_dir"] = import_dir
        observed["fast"] = "--no-images" in command
        observed["files"] = sorted(path.name for path in import_dir.glob("*.pdf"))
        process = _Process()
        process.stdout = asyncio.StreamReader()
        process.stderr = asyncio.StreamReader()
        process.stdout.feed_eof()
        process.stderr.feed_eof()
        return process

    monkeypatch.setattr(ingestion_jobs.asyncio, "create_subprocess_exec", fake_exec)

    result = await ingestion_jobs.ingest_saved_pdf_file(str(target_pdf), fast=True)

    assert observed["import_dir"] != source_dir
    assert observed["files"] and len(observed["files"]) == 1
    assert observed["files"][0].endswith("_target.pdf")
    assert result["returncode"] == 0
