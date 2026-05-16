from pathlib import Path
from unittest.mock import AsyncMock

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

    async def fake_run(import_dir: Path, *, fast: bool = True):
        observed["import_dir"] = import_dir
        observed["fast"] = fast
        files = sorted([p.name for p in import_dir.glob("*.pdf")])
        observed["files"] = files

        class _P:
            returncode = 0
            stdout = "Saved document to PostgreSQL with ID: 11111111-1111-1111-1111-111111111111"
            stderr = ""

        return _P()

    monkeypatch.setattr(ingestion_jobs, "_run_ingestion_for_directory", fake_run)

    result = await ingestion_jobs.ingest_saved_pdf_file(str(target_pdf), fast=True)

    assert observed["import_dir"] != source_dir
    assert observed["files"] and len(observed["files"]) == 1
    assert observed["files"][0].endswith("_target.pdf")
    assert result["document_id"] == "11111111-1111-1111-1111-111111111111"
