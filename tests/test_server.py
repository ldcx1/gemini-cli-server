"""Unit tests for the server endpoints (uses mocked CLI runner)."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from gemini_cli_server.config import ServerConfig
from gemini_cli_server.models import ModelRegistry
from tests.conftest import MockRunner


# ── Root & Health ─────────────────────────────────────────────────────────────


class TestRootAndHealth:
    @pytest.mark.asyncio
    async def test_root(self, client: AsyncClient) -> None:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "gemini-cli-server"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["gemini_cli"]["healthy"] is True
        assert data["gemini_cli"]["command"] == "gemini"

    @pytest.mark.asyncio
    async def test_health_degraded_when_cli_missing(
        self, config: ServerConfig, registry: ModelRegistry
    ) -> None:
        """When check_health reports failure, /health should return 503."""
        unhealthy_runner = MockRunner()
        # Override check_health to simulate a missing CLI.
        from gemini_cli_server.cli_runner import HealthCheckResult

        async def _unhealthy() -> HealthCheckResult:
            return HealthCheckResult(healthy=False, command="gemini", error="not found")
        unhealthy_runner.check_health = _unhealthy  # type: ignore[assignment]

        from gemini_cli_server.server import create_app
        app = create_app(config=config, runner=unhealthy_runner, registry=registry)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["gemini_cli"]["healthy"] is False
        assert "error" in data["gemini_cli"]


# ── Models ────────────────────────────────────────────────────────────────────


class TestModelEndpoints:
    @pytest.mark.asyncio
    async def test_list_models(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0

    @pytest.mark.asyncio
    async def test_get_known_model(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/models/gemini-2.5-pro")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "gemini-2.5-pro"
        assert data["object"] == "model"

    @pytest.mark.asyncio
    async def test_get_unknown_model_404(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/models/no-such-model")
        assert resp.status_code == 404


# ── Chat Completions (non-streaming) ─────────────────────────────────────────


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_basic_completion(self, client: AsyncClient, mock_runner: MockRunner) -> None:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-pro",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "gemini-2.5-pro"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Hello from Gemini!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_prompt_built_from_messages(
        self, client: AsyncClient, mock_runner: MockRunner
    ) -> None:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        assert mock_runner.last_prompt is not None
        assert "[System]: Be helpful." in mock_runner.last_prompt
        assert "[User]: Hello" in mock_runner.last_prompt

    @pytest.mark.asyncio
    async def test_model_passthrough(
        self, client: AsyncClient, mock_runner: MockRunner
    ) -> None:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert mock_runner.last_model == "claude-3-5-sonnet"

    @pytest.mark.asyncio
    async def test_multimodal_base64_image(
        self, client: AsyncClient, mock_runner: MockRunner
    ) -> None:
        import base64
        test_img = base64.b64encode(b"fake_image_data").decode("utf-8")
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_img}"}}
                        ]
                    }
                ],
            },
        )
        assert resp.status_code == 200
        assert mock_runner.last_prompt is not None
        assert "[User]: What is this?\\n[User uploaded image saved at file path:" in mock_runner.last_prompt
        assert ".png" in mock_runner.last_prompt



# ── Working Directory ─────────────────────────────────────────────────────────


class TestWorkingDirectory:
    @pytest.mark.asyncio
    async def test_working_dir_from_body(
        self, client: AsyncClient, mock_runner: MockRunner
    ) -> None:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "working_dir": "/tmp/myproject",
            },
        )
        assert mock_runner.last_working_dir == "/tmp/myproject"

    @pytest.mark.asyncio
    async def test_working_dir_from_header(
        self, client: AsyncClient, mock_runner: MockRunner
    ) -> None:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
            },
            headers={"X-Working-Dir": "/opt/code"},
        )
        assert mock_runner.last_working_dir == "/opt/code"

    @pytest.mark.asyncio
    async def test_body_takes_precedence_over_header(
        self, client: AsyncClient, mock_runner: MockRunner
    ) -> None:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "working_dir": "/from-body",
            },
            headers={"X-Working-Dir": "/from-header"},
        )
        assert mock_runner.last_working_dir == "/from-body"


# ── Error Handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_cli_failure_returns_502(self, client: AsyncClient, mock_runner: MockRunner) -> None:
        mock_runner.return_code = 1  # Will cause RuntimeError
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_missing_messages_returns_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "m"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_messages_returns_422(self, client: AsyncClient) -> None:
        """Empty messages list should fail validation (min_length=1)."""
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_cli_timeout_returns_504(
        self, config: ServerConfig, registry: ModelRegistry
    ) -> None:
        """TimeoutError from the runner should map to HTTP 504."""
        timeout_runner = MockRunner()

        async def _timeout(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise TimeoutError("timed out after 10s")
        timeout_runner.run = _timeout  # type: ignore[assignment]

        from gemini_cli_server.server import create_app
        app = create_app(config=config, runner=timeout_runner, registry=registry)
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
            )
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]


# ── Streaming ─────────────────────────────────────────────────────────────────


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_response(self, client: AsyncClient, mock_runner: MockRunner) -> None:
        mock_runner.response = "Streamed response"
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events.
        raw = resp.text
        events = [
            line.removeprefix("data: ")
            for line in raw.strip().split("\n")
            if line.startswith("data: ")
        ]

        # There should be multiple chunks + [DONE].
        assert events[-1] == "[DONE]"

        # First chunk should have role.
        first = json.loads(events[0])
        assert first["choices"][0]["delta"]["role"] == "assistant"

        # Collect content from all chunks.
        content_parts = []
        for ev in events[:-1]:  # skip [DONE]
            parsed = json.loads(ev)
            c = parsed["choices"][0]["delta"].get("content")
            if c:
                content_parts.append(c)
        full = "".join(content_parts)
        assert full == "Streamed response"

    @pytest.mark.asyncio
    async def test_stream_error_yields_error_object(
        self, config: ServerConfig, registry: ModelRegistry
    ) -> None:
        """When the CLI fails during streaming, the SSE stream should
        contain an OpenAI-compatible error object — not error-as-content."""
        error_runner = MockRunner(return_code=1)

        from gemini_cli_server.server import create_app
        app = create_app(config=config, runner=error_runner, registry=registry)
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        events = [
            line.removeprefix("data: ")
            for line in resp.text.strip().split("\n")
            if line.startswith("data: ")
        ]
        assert events[-1] == "[DONE]"

        # One of the events should contain an error object.
        found_error = False
        for ev in events[:-1]:
            parsed = json.loads(ev)
            if "error" in parsed:
                found_error = True
                assert "message" in parsed["error"]
                assert parsed["error"]["type"] == "server_error"
                break
        assert found_error, "Expected an error object in the SSE stream"
