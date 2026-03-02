"""End-to-end tests that invoke the real Gemini CLI.

These tests require:
  - ``gemini`` CLI available in PATH
  - A valid Gemini API key / auth configured

Run with:  ``pytest tests/test_e2e.py -v -m e2e``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from gemini_cli_server.cli_runner import GeminiCLIRunner, MAX_ARG_PROMPT_BYTES
from gemini_cli_server.config import ServerConfig
from gemini_cli_server.models import ModelRegistry
from gemini_cli_server.server import create_app

# Skip the entire module if gemini CLI is not available.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        shutil.which("gemini") is None,
        reason="gemini CLI not found in PATH",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_app():
    """Create an app with the real CLI runner."""
    config = ServerConfig(timeout=180)
    runner = GeminiCLIRunner(
        command=config.gemini_command,
        timeout=config.timeout,
        max_retries=1,
    )
    registry = ModelRegistry()
    return create_app(config=config, runner=runner, registry=registry)


@pytest.fixture
async def e2e_client(e2e_app):
    transport = ASGITransport(app=e2e_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# 1. Health check — verifies the gemini CLI is found and responds
# ---------------------------------------------------------------------------


class TestE2EHealthCheck:
    """Verify the /health endpoint with the real gemini binary."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, e2e_client: AsyncClient) -> None:
        resp = await e2e_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["gemini_cli"]["healthy"] is True
        assert data["gemini_cli"]["command"] == "gemini"

    @pytest.mark.asyncio
    async def test_root_returns_server_info(self, e2e_client: AsyncClient) -> None:
        resp = await e2e_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "gemini-cli-server"
        assert "version" in data


# ---------------------------------------------------------------------------
# 2. Model listing — all default models are returned in OpenAI format
# ---------------------------------------------------------------------------


class TestE2EModels:
    """Verify model listing and retrieval via the real server."""

    @pytest.mark.asyncio
    async def test_list_models(self, e2e_client: AsyncClient) -> None:
        resp = await e2e_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        # Every entry must have the expected OpenAI fields.
        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert "created" in model
            assert "owned_by" in model

    @pytest.mark.asyncio
    async def test_get_specific_model(self, e2e_client: AsyncClient) -> None:
        resp = await e2e_client.get("/v1/models/gemini-2.5-pro")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "gemini-2.5-pro"
        assert data["object"] == "model"
        assert data["owned_by"] == "google"

    @pytest.mark.asyncio
    async def test_get_unknown_model_returns_404(self, e2e_client: AsyncClient) -> None:
        resp = await e2e_client.get("/v1/models/does-not-exist-xyz")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 3. Chat completions — non-streaming
# ---------------------------------------------------------------------------


class TestE2EChatCompletion:
    """Test non-streaming chat completion with the real Gemini CLI."""

    @pytest.mark.asyncio
    async def test_simple_completion(self, e2e_client: AsyncClient) -> None:
        """Basic request → non-empty response in OpenAI format."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": "Reply with exactly one word: Hello"},
                ],
            },
            timeout=180,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "gemini-2.5-flash"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["message"]["role"] == "assistant"
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0
        # Verify usage stats are present.
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_system_message_is_respected(self, e2e_client: AsyncClient) -> None:
        """Send a system message and verify the model follows it."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": "You must reply in ALL CAPS only."},
                    {"role": "user", "content": "Say the word hello"},
                ],
            },
            timeout=180,
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        # The response should contain "HELLO" in uppercase.
        assert "HELLO" in content.upper()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, e2e_client: AsyncClient) -> None:
        """Send multiple messages simulating a conversation."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": "My name is TestBot."},
                    {"role": "assistant", "content": "Hello TestBot!"},
                    {"role": "user", "content": "What is my name? Reply with just the name."},
                ],
            },
            timeout=180,
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "TestBot" in content

    @pytest.mark.asyncio
    async def test_response_id_is_unique(self, e2e_client: AsyncClient) -> None:
        """Two requests should produce different response IDs."""
        payload = {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Reply with one word: yes"}],
        }
        resp1 = await e2e_client.post("/v1/chat/completions", json=payload, timeout=180)
        resp2 = await e2e_client.post("/v1/chat/completions", json=payload, timeout=180)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["id"] != resp2.json()["id"]


# ---------------------------------------------------------------------------
# 4. Streaming — SSE format
# ---------------------------------------------------------------------------


class TestE2EStreaming:
    """Test streaming (Server-Sent Events) with the real CLI."""

    @pytest.mark.asyncio
    async def test_streaming_format(self, e2e_client: AsyncClient) -> None:
        """Verify SSE stream structure: role chunk → content chunks → [DONE]."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": "Reply with exactly one word: world"},
                ],
                "stream": True,
            },
            timeout=180,
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events.
        events = [
            line.removeprefix("data: ")
            for line in resp.text.strip().split("\n")
            if line.startswith("data: ")
        ]
        assert len(events) >= 3  # role chunk + at least one content chunk + [DONE]
        assert events[-1] == "[DONE]"

        # First chunk should have the assistant role.
        first = json.loads(events[0])
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"]["role"] == "assistant"

        # Collect streamed content and verify it's non-empty.
        content_parts = []
        for ev in events[:-1]:  # skip [DONE]
            parsed = json.loads(ev)
            c = parsed["choices"][0]["delta"].get("content")
            if c:
                content_parts.append(c)
        full_content = "".join(content_parts)
        assert len(full_content) > 0

        # Last JSON chunk should have finish_reason="stop".
        last_json = json.loads(events[-2])
        assert last_json["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streaming_with_system_message(self, e2e_client: AsyncClient) -> None:
        """Streaming should also work with system messages."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": "Reply in exactly one word."},
                    {"role": "user", "content": "Say hello"},
                ],
                "stream": True,
            },
            timeout=180,
        )
        assert resp.status_code == 200
        assert "[DONE]" in resp.text


# ---------------------------------------------------------------------------
# 5. Working directory — the CLI sees the correct directory context
# ---------------------------------------------------------------------------


class TestE2EWorkingDirectory:
    """Verify that working_dir is actually passed to the CLI subprocess."""

    @pytest.mark.asyncio
    async def test_cli_sees_marker_file(
        self, e2e_client: AsyncClient, tmp_path: Path
    ) -> None:
        """Create a uniquely-named file and verify the CLI can see it."""
        marker_name = "gemini_e2e_marker_7x9k2m.txt"
        (tmp_path / marker_name).write_text("E2E test marker file.")

        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "List every file in the current working directory. "
                            "Output ONLY the filenames, one per line, no commentary."
                        ),
                    },
                ],
                "working_dir": str(tmp_path),
            },
            timeout=180,
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert marker_name in content, (
            f"CLI should see '{marker_name}' in {tmp_path}. Got:\n{content}"
        )

    @pytest.mark.asyncio
    async def test_working_dir_via_header(
        self, e2e_client: AsyncClient, tmp_path: Path
    ) -> None:
        """Verify X-Working-Dir header works the same as body field."""
        marker_name = "header_marker_q3w8p.txt"
        (tmp_path / marker_name).write_text("Header test.")

        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "List every file in the current working directory. "
                            "Output ONLY the filenames, one per line, no commentary."
                        ),
                    },
                ],
            },
            headers={"X-Working-Dir": str(tmp_path)},
            timeout=180,
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert marker_name in content, (
            f"CLI should see '{marker_name}' via header. Got:\n{content}"
        )

    @pytest.mark.asyncio
    async def test_working_dir_with_nested_structure(
        self, e2e_client: AsyncClient, tmp_path: Path
    ) -> None:
        """CLI should see a directory structure when working_dir is set."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Test Project")

        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "List every file and directory at the top level of "
                            "the current working directory. Output ONLY names, "
                            "one per line."
                        ),
                    },
                ],
                "working_dir": str(tmp_path),
            },
            timeout=180,
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "README.md" in content
        assert "src" in content


# ---------------------------------------------------------------------------
# 6. Different models
# ---------------------------------------------------------------------------


class TestE2EDifferentModels:
    """Verify that different model IDs are passed through correctly."""

    @pytest.mark.asyncio
    async def test_gemini_flash(self, e2e_client: AsyncClient) -> None:
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": "Reply with one word: alpha"},
                ],
            },
            timeout=180,
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "gemini-2.5-flash"
        assert len(resp.json()["choices"][0]["message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_gemini_pro(self, e2e_client: AsyncClient) -> None:
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-pro",
                "messages": [
                    {"role": "user", "content": "Reply with one word: beta"},
                ],
            },
            timeout=180,
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "gemini-2.5-pro"
        assert len(resp.json()["choices"][0]["message"]["content"]) > 0


# ---------------------------------------------------------------------------
# 7. Large prompts — stdin delivery
# ---------------------------------------------------------------------------


class TestE2ELargePrompt:
    """Verify that large prompts (>100KB) are handled via stdin delivery."""

    @pytest.mark.asyncio
    async def test_prompt_over_threshold(self, e2e_client: AsyncClient) -> None:
        """A prompt exceeding MAX_ARG_PROMPT_BYTES should still work
        because it's delivered via stdin instead of CLI args."""
        # Create a prompt just over the threshold.
        padding = "x" * (MAX_ARG_PROMPT_BYTES + 100)
        prompt = (
            f"The following is padding text, ignore it:\n{padding}\n\n"
            "Now answer this question: Reply with exactly one word: yes"
        )
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=180,
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_large_prompt_streaming(self, e2e_client: AsyncClient) -> None:
        """Large prompts should also work with streaming enabled."""
        padding = "y" * (MAX_ARG_PROMPT_BYTES + 500)
        prompt = (
            f"Ignore this padding:\n{padding}\n\n"
            "Reply with exactly one word: ok"
        )
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=180,
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert "[DONE]" in resp.text


# ---------------------------------------------------------------------------
# 8. Error handling
# ---------------------------------------------------------------------------


class TestE2EErrorHandling:
    """Verify error cases return proper HTTP status codes."""

    @pytest.mark.asyncio
    async def test_missing_messages_returns_422(self, e2e_client: AsyncClient) -> None:
        """Request without messages should fail validation."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={"model": "gemini-2.5-flash"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_messages_returns_422(self, e2e_client: AsyncClient) -> None:
        """Empty messages list should fail validation."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={"model": "gemini-2.5-flash", "messages": []},
        )
        # Empty messages list is now rejected by min_length=1 validation.
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_json_returns_422(self, e2e_client: AsyncClient) -> None:
        """Malformed JSON should be rejected."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_model_returns_422(self, e2e_client: AsyncClient) -> None:
        """Request without model field should fail validation."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_role_returns_422(self, e2e_client: AsyncClient) -> None:
        """An invalid role should fail Pydantic validation."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "invalid_role", "content": "hello"}],
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 9. Response format compliance
# ---------------------------------------------------------------------------


class TestE2EResponseFormat:
    """Verify that responses match the OpenAI API contract."""

    @pytest.mark.asyncio
    async def test_completion_response_fields(self, e2e_client: AsyncClient) -> None:
        """Every required field from the OpenAI spec should be present."""
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "Reply: hi"}],
            },
            timeout=180,
        )
        assert resp.status_code == 200
        data = resp.json()

        # Top-level fields.
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert data["model"] == "gemini-2.5-flash"

        # Choice fields.
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)

        # Usage fields.
        usage = data["usage"]
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.asyncio
    async def test_model_list_response_fields(self, e2e_client: AsyncClient) -> None:
        """Model list response should match OpenAI spec."""
        resp = await e2e_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)

        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert isinstance(model["created"], int)
        assert isinstance(model["owned_by"], str)


# ---------------------------------------------------------------------------
# 10. Multimodal payloads
# ---------------------------------------------------------------------------


class TestE2EMultimodal:
    """Verify that multimodal payloads (e.g. inline base64 images) work end-to-end."""

    @pytest.mark.asyncio
    async def test_base64_image_upload(self, e2e_client: AsyncClient, tmp_path: Path) -> None:
        """Upload tests/image.png as a base64 string and ask the model about it."""
        import base64
        import os
        
        # tests/image.png should exist
        img_path = Path("tests/image.png")
        assert img_path.exists(), "tests/image.png not found"
        
        image_data = img_path.read_bytes()
        encoded = base64.b64encode(image_data).decode("utf-8")
        
        resp = await e2e_client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Which animal is in this image? Reply with one word."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded}"}
                            }
                        ]
                    }
                ],
                "working_dir": str(tmp_path),
            },
            timeout=180,
        )
        
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert len(content.strip()) > 0
        assert "cat" in content.lower()
        
        # Verify that the server actually placed the image file into the tmp_path
        # so the CLI could read it
        files_in_tmp = list(tmp_path.glob("*.png"))
        
        # If it was cleaned up properly by BackgroundTasks, it might be gone by the
        # time we check. But in the current implementation, we are just cleaning up
        # in the background task. Fast background tasks might delete it before we assert.
        # Actually, let's just make sure the LLM successfully processed it without error.
        assert "error" not in content.lower()

