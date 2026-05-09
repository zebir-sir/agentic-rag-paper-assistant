import asyncio
import base64
import importlib.util
import shutil
import sys
import types
import uuid
from pathlib import Path

import pytest

if importlib.util.find_spec("fastapi") is None:
    fake_fastapi = types.ModuleType("fastapi")

    class _HTTPException(Exception):
        def __init__(self, status_code: int, detail=None):
            super().__init__(str(detail))
            self.status_code = status_code
            self.detail = detail

    fake_fastapi.HTTPException = _HTTPException
    sys.modules["fastapi"] = fake_fastapi

from agent import ingestion_jobs


@pytest.mark.parametrize("fast_flag", [True, False])
def test_start_upload_ingestion_job_keeps_fast_flag(monkeypatch, fast_flag: bool):
    ingestion_jobs.INGESTION_JOBS.clear()

    workspace_tmp = Path("D:/agent/agentic_rag_project-main2/.tmp")
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    temp_root = workspace_tmp / f"ingest-fast-{uuid.uuid4().hex[:8]}"
    import_dir = temp_root / "uploads"
    import_dir.mkdir(parents=True, exist_ok=True)

    def fake_make_upload_paths(filename_raw: str):
        safe_name = "paper.pdf"
        target_path = import_dir / safe_name
        return safe_name, import_dir, target_path

    scheduled = {"called": 0}

    def fake_create_task(coro):
        scheduled["called"] += 1
        coro.close()

        class DummyTask:
            def done(self):
                return True

        return DummyTask()

    monkeypatch.setattr(ingestion_jobs, "_make_upload_paths", fake_make_upload_paths)
    monkeypatch.setattr(ingestion_jobs.asyncio, "create_task", fake_create_task)

    payload = {
        "filename": "paper.pdf",
        "content_base64": base64.b64encode(b"%PDF-1.7 test").decode("ascii"),
        "fast": fast_flag,
    }

    result = asyncio.run(ingestion_jobs.start_upload_ingestion_job(payload))

    assert result["status"] == "accepted"
    assert result["job_id"]
    assert result["filename"] == "paper.pdf"
    assert result["fast"] is fast_flag
    assert scheduled["called"] == 1

    job_id = result["job_id"]
    assert job_id in ingestion_jobs.INGESTION_JOBS
    assert ingestion_jobs.INGESTION_JOBS[job_id].fast is fast_flag

    shutil.rmtree(temp_root, ignore_errors=True)
