"""Unit tests for API type validation and serialisation."""

from __future__ import annotations

import pytest

from gemini_cli_server.api_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    StreamChoice,
    UsageInfo,
)


class TestChatCompletionRequest:
    """Test request body validation."""

    def test_minimal_request(self) -> None:
        req = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert req.model == "gemini-2.5-pro"
        assert len(req.messages) == 1
        assert req.stream is False
        assert req.working_dir is None

    def test_with_working_dir(self) -> None:
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            working_dir="/tmp/project",
        )
        assert req.working_dir == "/tmp/project"

    def test_with_stream(self) -> None:
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            stream=True,
        )
        assert req.stream is True

    def test_multiple_messages(self) -> None:
        req = ChatCompletionRequest(
            model="m",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi!"),
                ChatMessage(role="user", content="How are you?"),
            ],
        )
        assert len(req.messages) == 4

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(Exception):
            ChatMessage(role="invalid", content="x")  # type: ignore[arg-type]


class TestChatCompletionResponse:
    """Test response serialisation."""

    def test_response_has_expected_fields(self) -> None:
        resp = ChatCompletionResponse(
            model="gemini-2.5-pro",
            choices=[Choice(message=ChoiceMessage(content="Hello"))],
        )
        d = resp.model_dump()
        assert d["object"] == "chat.completion"
        assert d["model"] == "gemini-2.5-pro"
        assert len(d["choices"]) == 1
        assert d["choices"][0]["message"]["role"] == "assistant"
        assert d["choices"][0]["message"]["content"] == "Hello"
        assert d["choices"][0]["finish_reason"] == "stop"
        assert "id" in d
        assert "created" in d
        assert "usage" in d

    def test_id_is_unique(self) -> None:
        r1 = ChatCompletionResponse(model="m", choices=[])
        r2 = ChatCompletionResponse(model="m", choices=[])
        assert r1.id != r2.id


class TestUsageInfo:
    """Test usage stats model."""

    def test_defaults_to_zero(self) -> None:
        u = UsageInfo()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0


class TestStreamingTypes:
    """Test SSE chunk models."""

    def test_chunk_serialisation(self) -> None:
        chunk = ChatCompletionChunk(
            model="m",
            choices=[StreamChoice(delta=DeltaMessage(content="Hi"))],
        )
        d = chunk.model_dump()
        assert d["object"] == "chat.completion.chunk"
        assert d["choices"][0]["delta"]["content"] == "Hi"
        assert d["choices"][0]["finish_reason"] is None

    def test_final_chunk_with_finish_reason(self) -> None:
        chunk = ChatCompletionChunk(
            model="m",
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
        )
        d = chunk.model_dump()
        assert d["choices"][0]["finish_reason"] == "stop"
