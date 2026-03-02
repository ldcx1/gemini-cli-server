"""FastAPI application exposing OpenAI-compatible endpoints."""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import time
import uuid
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse

from gemini_cli_server import __version__
from gemini_cli_server.api_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    MessageContentImage,
    MessageContentText,
    StreamChoice,
    UsageInfo,
)
from gemini_cli_server.cli_runner import GeminiCLIRunner
from gemini_cli_server.config import ServerConfig
from gemini_cli_server.models import ModelRegistry

logger = logging.getLogger(__name__)


def create_app(
    config: ServerConfig | None = None,
    runner: GeminiCLIRunner | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    All dependencies can be injected for testing.
    """
    config = config or ServerConfig.from_env()
    runner = runner or GeminiCLIRunner(
        command=config.gemini_command,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )
    registry = registry or ModelRegistry()

    app = FastAPI(
        title="Gemini CLI Server",
        description="OpenAI-compatible API server wrapping the Gemini CLI",
        version=__version__,
    )

    # Store deps on app state so they can be accessed in tests.
    app.state.config = config
    app.state.runner = runner
    app.state.registry = registry

    # ── routes ────────────────────────────────────────────────────────────

    @app.get("/")
    async def root() -> dict:
        """Server information."""
        return {
            "name": "gemini-cli-server",
            "version": __version__,
            "description": "OpenAI-compatible API wrapping the Gemini CLI",
        }

    @app.get("/health")
    async def health() -> JSONResponse:
        """Health-check endpoint.

        Verifies that the Gemini CLI is installed and responsive by
        running ``gemini --help``.
        """
        cli_health = await runner.check_health()
        status = "ok" if cli_health["healthy"] else "degraded"
        payload = {"status": status, "gemini_cli": cli_health}
        code = 200 if cli_health["healthy"] else 503
        return JSONResponse(content=payload, status_code=code)

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """List available models (OpenAI-compatible)."""
        return JSONResponse(content=registry.to_openai_list())

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str) -> JSONResponse:
        """Get a single model's details."""
        model = registry.get(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return JSONResponse(content=model.to_openai_dict())

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
        background_tasks: BackgroundTasks,
        x_working_dir: str | None = Header(None),
    ) -> StreamingResponse | JSONResponse:
        """Create a chat completion (OpenAI-compatible).

        The working directory for the Gemini CLI can be set via:
        1. ``working_dir`` field in the request body, **or**
        2. ``X-Working-Dir`` HTTP header.

        Body field takes precedence over the header.
        """
        # Resolve working directory: body > header > config default
        working_dir = body.working_dir or x_working_dir or config.default_working_dir
        actual_cwd = working_dir or os.getcwd()

        # Build a single prompt by concatenating all messages. Handle temp files.
        prompt, temp_files = _build_prompt(body, actual_cwd)

        # Register cleanup for temp files
        for tmp_file in temp_files:
            background_tasks.add_task(_cleanup_file, tmp_file)

        # Resolve model — allow any model name (Gemini CLI will validate).
        model = body.model

        if body.stream:
            return StreamingResponse(
                _stream_response(runner, prompt, model, actual_cwd, temp_files),
                media_type="text/event-stream",
            )

        # Non-streaming path.
        try:
            result = await runner.run(
                prompt=prompt,
                model=model,
                working_dir=actual_cwd,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        content = result.stdout.strip()

        response = ChatCompletionResponse(
            model=model,
            choices=[
                Choice(message=ChoiceMessage(content=content)),
            ],
            usage=_estimate_usage(prompt, content),
        )
        return JSONResponse(content=response.model_dump())

    return app


# ── helpers ───────────────────────────────────────────────────────────────────


def _cleanup_file(filepath: str) -> None:
    """Safely delete a file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as exc:
        logger.warning("Failed to clean up temp file %s: %s", filepath, exc)


def _build_prompt(body: ChatCompletionRequest, working_dir: str) -> tuple[str, list[str]]:
    """Concatenate chat messages into a single prompt string.

    System messages are prepended, then user/assistant messages are laid out
    in order so the CLI receives the full conversational context.

    Returns the constructed prompt and a list of temporary file paths created
    from multimodal image attachments.
    """
    parts: list[str] = []
    temp_files: list[str] = []

    for msg in body.messages:
        role_label = msg.role.capitalize()
        
        if isinstance(msg.content, str):
            parts.append(f"[{role_label}]: {msg.content}")
        else:
            # Handle list of multimodal content blocks
            content_parts: list[str] = []
            for item in msg.content:
                if isinstance(item, MessageContentText):
                    content_parts.append(item.text)
                elif isinstance(item, MessageContentImage):
                    url = item.image_url.url
                    # Handle base64 encoded images
                    if url.startswith("data:image/"):
                        try:
                            # Format: data:image/jpeg;base64,...
                            header, encoded = url.split(",", 1)
                            ext = header.split(";")[0].split("/")[1]
                            image_data = base64.b64decode(encoded)
                            
                            # Write to a temp file in the working_dir
                            fd, filepath = tempfile.mkstemp(dir=working_dir, suffix=f".{ext}")
                            with os.fdopen(fd, "wb") as f:
                                f.write(image_data)
                            
                            temp_files.append(filepath)
                            content_parts.append(f"[User uploaded image saved at file path: {filepath}. You must use your tools to analyze this file.]")
                        except Exception as exc:
                            logger.error("Failed to decode base64 image: %s", exc)
                            content_parts.append("[Attached Image: <failed to decode>]")
                    else:
                        # Passing raw URLs currently relies on the CLI to fetch it
                        content_parts.append(f"[User uploaded image URL: {url}. You must use your tools to analyze this URL.]")
            
            combined_content = "\\n".join(content_parts)
            parts.append(f"[{role_label}]: {combined_content}")

    return "\n\n".join(parts), temp_files


def _estimate_usage(prompt: str, completion: str) -> UsageInfo:
    """Rough token-count estimate (4 chars ≈ 1 token).

    These are **approximate** values — the Gemini CLI does not report
    actual token counts.
    """
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, len(completion) // 4)
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


async def _stream_response(
    runner: GeminiCLIRunner,
    prompt: str,
    model: str,
    working_dir: str | None,
    temp_files: list[str],
) -> AsyncGenerator[str, None]:
    """Stream the CLI response as SSE events.

    Because the Gemini CLI returns the full response at once, we simulate
    streaming by splitting the output into chunks.
    """
    try:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Send initial role chunk.
        first_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model,
            choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield f"data: {json.dumps(first_chunk.model_dump())}\n\n"

        try:
            # Stream directly from the runner.
            async for chunk_text in runner.run_stream(
                prompt=prompt,
                model=model,
                working_dir=working_dir,
            ):
                if not chunk_text:
                    continue
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    model=model,
                    choices=[StreamChoice(delta=DeltaMessage(content=chunk_text))],
                )
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        except (TimeoutError, RuntimeError) as exc:
            # Yield an OpenAI-compatible error object so clients can detect
            # failures instead of interpreting the error as assistant content.
            error_payload = {
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "code": "cli_error",
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Final chunk with finish_reason.
        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model,
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
        )
        yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        for tmp_file in temp_files:
            _cleanup_file(tmp_file)
