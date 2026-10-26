"""Exercise the built-in sync moderation fallback with a real local HTTP pool."""

from __future__ import annotations

import importlib
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import openai
import pytest

import guardrails
from guardrails.context import GuardrailsContext, clear_context, set_context

http = importlib.import_module(openai.DefaultHttpxClient.__mro__[1].__module__.split(".")[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_errors", [False, True])
async def test_repeated_sync_moderation_fallback(monkeypatch: pytest.MonkeyPatch, raise_errors: bool) -> None:
    """Every repeated public call must finish its built-in moderation check."""
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        """Return deterministic moderation results over persistent HTTP/1.1."""

        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            """Keep synthetic request metadata out of test output."""
            return None

        def do_POST(self) -> None:
            """Consume one request and return a safe moderation response."""
            self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(self.path)
            payload = json.dumps(
                {"id": "mod-test", "model": "omni-moderation-latest", "results": [{"flagged": False, "categories": {}, "category_scores": {}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    original_connect = socket.socket.connect

    def connect(sock: socket.socket, address: Any) -> None:
        if address != ("127.0.0.1", server.server_port):
            raise AssertionError("Only the test's loopback server may receive connections")
        original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")

    def provider(request: Any) -> Any:
        if "/moderations" in request.url.path:
            return http.Response(404, json={"error": {"message": "Unavailable", "type": "invalid_request_error"}})
        return http.Response(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "model": "test-model",
                "output": [
                    {
                        "type": "message",
                        "id": "msg-test",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
            },
        )

    config = {"version": 1, "output": {"version": 1, "guardrails": [{"name": "Moderation", "config": {}}]}}
    kwargs = {
        "api_key": "test-key",
        "azure_endpoint": "https://unit-test.openai.azure.com",
        "api_version": "2025-04-01-preview",
        "http_client": http.Client(transport=http.MockTransport(provider), trust_env=False),
    }
    thread.start()
    try:
        with openai.AzureOpenAI(**kwargs) as moderation_client:
            set_context(GuardrailsContext(moderation_client))
            with guardrails.GuardrailsAzureOpenAI(config=config, raise_guardrail_errors=raise_errors, **kwargs) as client:
                for _ in range(3):
                    result = client.responses.create(model="test-model", input="hello")
                    assert isinstance(result, guardrails.GuardrailsResponse)
                    assert len(result.guardrail_results.output) == 1
                    assert not result.guardrail_results.output[0].execution_failed
                    assert not result.guardrail_results.output[0].tripwire_triggered
        assert requests == ["/v1/moderations"] * 3
    finally:
        clear_context()
        server.shutdown()
        server.server_close()
        thread.join()
