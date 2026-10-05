import unittest

from fastapi.testclient import TestClient

from chatjimmy.client import ChatResponse, Model, Stats
from chatjimmy.server import create_app


class FakeChatJimmy:
    def models(self):
        return [Model(id="llama3.1-8B", created=1, owned_by="Taalas Inc.")]

    def chat(self, messages, model, system_prompt, top_k):
        self.messages = messages
        self.model = model
        self.system_prompt = system_prompt
        self.top_k = top_k
        return ChatResponse("Hello", Stats(prefill_tokens=3, decode_tokens=1, total_tokens=4, done_reason="stop"))


class OpenAIContractTests(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeChatJimmy()
        self.client = TestClient(create_app(self.upstream))

    def test_models_match_openai_shape(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "llama3.1-8B")

    def test_chat_completion_maps_system_messages_and_usage(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"], {"role": "assistant", "content": "Hello"})
        self.assertEqual(payload["usage"]["total_tokens"], 4)
        self.assertEqual(self.upstream.system_prompt, "Be brief.")

    def test_stream_uses_sse_and_done_sentinel(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
        ) as response:
            body = "".join(response.iter_text())
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        self.assertIn('"object":"chat.completion.chunk"', body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    def test_tool_declarations_are_ignored_for_text_only_model(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "function", "function": {"name": "read_file"}}],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "Hello")

    def test_long_system_prompts_are_capped_for_upstream_limit(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [
                    {"role": "system", "content": "x" * 20_000},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.upstream.system_prompt), 12_000)
