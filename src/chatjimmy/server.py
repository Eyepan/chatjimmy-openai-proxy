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

MAX_SYSTEM_PROMPT_CHARS = 12_000


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


def _tool_instruction(tools: Any) -> tuple[str, dict[str, dict[str, str]]]:
    if not isinstance(tools, list):
        return "", {}

    manifest = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        parameters = function.get("parameters")
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        arguments = {
            name: schema.get("type", "value")
            for name, schema in properties.items()
            if isinstance(name, str) and isinstance(schema, dict)
        }
        manifest.append(
            {
                "name": function["name"],
                "description": str(function.get("description", ""))[:160],
                "arguments": arguments,
            }
        )

    if not manifest:
        return "", {}
    tool_arguments = {tool["name"]: tool["arguments"] for tool in manifest}
    return (
        "You can call tools. When a tool is needed, respond with ONLY valid JSON: "
        '{"tool_calls":[{"name":"tool_name","arguments":{"argument":"value"}}]}. '
        "Do not use markdown. Only call these tools:\n"
        + json.dumps(manifest, separators=(",", ":")),
        tool_arguments,
    )


def _parse_tool_payload(text: str) -> dict[str, Any] | None:
    payload = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        # Some small models emit a second tool request before they have seen
        # the first result. Process only the first envelope so OpenCode can
        # complete that call and supply its real result in the next turn.
        parsed, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError:
        if not payload.startswith("{"):
            return None
        stack: list[str] = []
        in_string = False
        escaped = False
        for character in payload:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "{":
                stack.append("}")
            elif character == "[":
                stack.append("]")
            elif stack and character == stack[-1]:
                stack.pop()
        if in_string:
            return None
        try:
            parsed = json.loads(payload + "".join(reversed(stack)))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_arguments(arguments: dict[str, Any], schema: dict[str, str]) -> dict[str, Any]:
    coerced = {}
    for name, value in arguments.items():
        if name not in schema:
            continue
        if schema[name] == "integer" and isinstance(value, str) and value.lstrip("-").isdigit():
            value = int(value)
        elif schema[name] == "number" and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        elif schema[name] == "boolean" and isinstance(value, str) and value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        coerced[name] = value
    return coerced


def _tool_calls(text: str, tool_arguments: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    if not tool_arguments:
        return []
    parsed = _parse_tool_payload(text)
    if parsed is None:
        return []
    raw_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    if not isinstance(raw_calls, list) or not raw_calls:
        return []

    calls = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict) or call.get("name") not in tool_arguments:
            return []
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            return []
        arguments = _coerce_arguments(arguments, tool_arguments[call["name"]])
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(arguments, separators=(",", ":"))},
                "index": index,
            }
        )
    return calls


def _messages(body: dict[str, Any]) -> tuple[list[dict[str, str]], str, dict[str, dict[str, str]]]:
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
        content = message.get("content")
        if content is None and role == "assistant" and message.get("tool_calls"):
            content = "The previous assistant requested tool calls."
        else:
            content = _content_to_text(content)
        if role in {"system", "developer"}:
            system_parts.append(content)
        elif role == "tool":
            messages.append({"role": "user", "content": f"Tool result: {content}"})
        else:
            messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("At least one non-system message is required")
    tool_prompt, tool_arguments = _tool_instruction(body.get("tools"))
    # chatjimmy silently returns an empty response beyond its ~6k-token input
    # limit. Keep the compact tool protocol before OpenCode's longer prompt.
    system_messages = "\n\n".join(system_parts)
    system_prompt = f"{tool_prompt}\n\n{system_messages}" if tool_prompt else system_messages
    return messages, system_prompt[:MAX_SYSTEM_PROMPT_CHARS], tool_arguments


def _usage(stats: Stats | None) -> dict[str, int]:
    prompt_tokens = stats.prefill_tokens if stats else 0
    completion_tokens = stats.decode_tokens if stats else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": stats.total_tokens if stats else prompt_tokens + completion_tokens,
    }


def _completion(
    response: ChatResponse, *, model: str, completion_id: str, created: int, tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None if tool_calls else response.text,
                    **(
                        {"tool_calls": [{key: value for key, value in call.items() if key != "index"} for call in tool_calls]}
                        if tool_calls
                        else {}
                    ),
                },
                "finish_reason": "tool_calls" if tool_calls else response.stats.done_reason if response.stats and response.stats.done_reason else "stop",
            }
        ],
        "usage": _usage(response.stats),
    }


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _stream(
    response: ChatResponse, *, model: str, completion_id: str, created: int, tool_calls: list[dict[str, Any]]
) -> Iterator[str]:
    base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model}
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    if tool_calls:
        yield _sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}]})
    elif response.text:
        yield _sse({**base, "choices": [{"index": 0, "delta": {"content": response.text}, "finish_reason": None}]})
    finish_reason = "tool_calls" if tool_calls else response.stats.done_reason if response.stats and response.stats.done_reason else "stop"
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
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return _error("'model' must be a non-empty string", status_code=400, param="model")
        try:
            messages, system_prompt, tool_arguments = _messages(body)
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
        tool_calls = _tool_calls(response.text, tool_arguments)
        if body.get("stream") is True:
            return StreamingResponse(
                _stream(response, model=model, completion_id=completion_id, created=created, tool_calls=tool_calls),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return _completion(response, model=model, completion_id=completion_id, created=created, tool_calls=tool_calls)

    return app


app = create_app()


def main() -> None:
    uvicorn.run("chatjimmy.server:app", host="127.0.0.1", port=8000)
