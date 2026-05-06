import json
from typing import Any, AsyncIterator

from fastapi.responses import StreamingResponse


def sse_event(event_type: str, **payload: Any) -> str:
    data = {"type": event_type, **payload}
    return f"data: {json.dumps(data)}\n\n"


def stream_response(generator: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
