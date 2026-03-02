"""Pydantic models for OpenAI-compatible request/response types."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────


class ImageUrl(BaseModel):
    url: str
    detail: str | None = None


class MessageContentImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


class MessageContentText(BaseModel):
    type: Literal["text"]
    text: str


MessageContent = str | list[MessageContentText | MessageContentImage]


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["system", "user", "assistant"]
    content: MessageContent


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible ``POST /v1/chat/completions`` request body.

    The ``working_dir`` field is a non-standard extension that sets the
    working directory for the Gemini CLI subprocess.
    """

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # Non-standard extension — also accepted via ``X-Working-Dir`` header.
    working_dir: str | None = None


# ── Responses ─────────────────────────────────────────────────────────────────


class UsageInfo(BaseModel):
    """Token usage statistics (estimated for CLI-based responses)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    """A message within a chat completion choice."""

    role: str = "assistant"
    content: str


class Choice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Streaming ─────────────────────────────────────────────────────────────────


class DeltaMessage(BaseModel):
    """Delta content for streamed responses."""

    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """A single streaming chunk choice."""

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A single SSE chunk in a streaming chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
