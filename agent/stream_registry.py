from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class StreamRun:
    run_id: str
    session_id: str
    task: asyncio.Task
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cancelled_by_user: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


_ACTIVE_RUNS: dict[str, StreamRun] = {}
_LOCK = asyncio.Lock()


async def register_stream_run(
    run_id: str,
    session_id: str,
    task: asyncio.Task,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    async with _LOCK:
        _ACTIVE_RUNS[run_id] = StreamRun(
            run_id=run_id,
            session_id=session_id,
            task=task,
            metadata=dict(metadata or {}),
        )


async def unregister_stream_run(run_id: str) -> None:
    async with _LOCK:
        _ACTIVE_RUNS.pop(run_id, None)


async def get_stream_run(run_id: str) -> Optional[StreamRun]:
    async with _LOCK:
        return _ACTIVE_RUNS.get(run_id)


async def cancel_stream_run(run_id: str) -> dict[str, Any]:
    async with _LOCK:
        run = _ACTIVE_RUNS.get(run_id)
        if run is None:
            return {"status": "not_found", "run_id": run_id}
        if run.task.done():
            _ACTIVE_RUNS.pop(run_id, None)
            return {"status": "already_finished", "run_id": run_id}

        run.cancelled_by_user = True
        run.task.cancel()
        return {"status": "cancelled", "run_id": run_id}
