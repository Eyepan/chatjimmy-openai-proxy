"""OpenAI-compatible HTTP adapter for chatjimmy.ai."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from chatjimmy.client import ChatJimmy, ChatResponse, Stats


def _error(message: str, *, status_code: int, param: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "api_error",
                "param": param,
                "code": None,
            }
        },
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
                raise ValueError("Only text message content is supported")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("Text content parts must include a string 'text' field")
            parts.append(text)
        return "".join(parts)
    raise ValueError("Message content must be a string or text content parts")


def _messages(body: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("'messages' must be a non-empty array")

    messages: list[dict[str, str]] = []
    system_parts: list[str] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("Every message must be an object")
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError("Message role must be system, developer, user, assistant, or tool")
        content = _content_to_text(message.get("content"))
        if role in {"system", "developer"}:
            system_parts.append(content)
        elif role == "tool":
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("At least one non-system message is required")
    return messages, "\n\n".join(system_parts)


def _usage(stats: Stats | None) -> dict[str, int]:
    prompt_tokens = stats.prefill_tokens if stats else 0
    completion_tokens = stats.decode_tokens if stats else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": stats.total_tokens if stats else prompt_tokens + completion_tokens,
    }


def _completion(response: ChatResponse, *, model: str, completion_id: str, created: int) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.text},
                "finish_reason": response.stats.done_reason if response.stats and response.stats.done_reason else "stop",
            }
        ],
        "usage": _usage(response.stats),
    }


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _stream(response: ChatResponse, *, model: str, completion_id: str, created: int) -> Iterator[str]:
    base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model}
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    if response.text:
        yield _sse({**base, "choices": [{"index": 0, "delta": {"content": response.text}, "finish_reason": None}]})
    finish_reason = response.stats.done_reason if response.stats and response.stats.done_reason else "stop"
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
    yield "data: [DONE]\n\n"


def create_app(client: ChatJimmy | None = None) -> FastAPI:
    """Create the adapter app; pass a client to replace the upstream in tests."""
    upstream = client or ChatJimmy()
    app = FastAPI(title="chatjimmy OpenAI-compatible API", version="0.1.0")

    @app.get("/v1/models", response_model=None)
    def models() -> dict[str, Any] | JSONResponse:
        try:
            return {"object": "list", "data": [model.__dict__ for model in upstream.models()]}
        except requests.RequestException as error:
            return _error(f"Upstream model request failed: {error}", status_code=502)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> dict[str, Any] | JSONResponse | StreamingResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error("Request body must be valid JSON", status_code=400)
        if not isinstance(body, dict):
            return _error("Request body must be a JSON object", status_code=400)
        if body.get("tools"):
            return _error("Tool calling is not supported by this model", status_code=400, param="tools")
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return _error("'model' must be a non-empty string", status_code=400, param="model")
        try:
            messages, system_prompt = _messages(body)
            top_k = body.get("top_k", 8)
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError("'top_k' must be a positive integer")
            response = upstream.chat(messages, model=model, system_prompt=system_prompt, top_k=top_k)
        except ValueError as error:
            return _error(str(error), status_code=400, param="messages")
        except requests.RequestException as error:
            return _error(f"Upstream chat request failed: {error}", status_code=502)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if body.get("stream") is True:
            return StreamingResponse(
                _stream(response, model=model, completion_id=completion_id, created=created),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return _completion(response, model=model, completion_id=completion_id, created=created)

    return app


app = create_app()


def main() -> None:
    uvicorn.run("chatjimmy.server:app", host="127.0.0.1", port=8000)
