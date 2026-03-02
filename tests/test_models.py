"""Unit tests for the model registry."""

from __future__ import annotations

import time

import pytest

from gemini_cli_server.models import DEFAULT_MODELS, ModelInfo, ModelRegistry


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_to_openai_dict_fields(self) -> None:
        m = ModelInfo(id="test-model", name="Test", owned_by="test-org", created=1000)
        d = m.to_openai_dict()
        assert d["id"] == "test-model"
        assert d["object"] == "model"
        assert d["created"] == 1000
        assert d["owned_by"] == "test-org"

    def test_created_defaults_to_current_time(self) -> None:
        m = ModelInfo(id="x", name="X")
        d = m.to_openai_dict()
        # created=0 in the dataclass → to_openai_dict fills with current time
        assert abs(d["created"] - int(time.time())) < 5


class TestModelRegistry:
    """Test ModelRegistry class."""

    def test_default_models_loaded(self) -> None:
        reg = ModelRegistry()
        models = reg.list()
        assert len(models) == len(DEFAULT_MODELS)

    def test_get_existing_model(self) -> None:
        reg = ModelRegistry()
        m = reg.get("gemini-2.5-pro")
        assert m is not None
        assert m.id == "gemini-2.5-pro"

    def test_get_missing_model(self) -> None:
        reg = ModelRegistry()
        assert reg.get("nonexistent") is None

    def test_has(self) -> None:
        reg = ModelRegistry()
        assert reg.has("gemini-2.5-pro") is True
        assert reg.has("nonexistent") is False

    def test_register_custom_model(self) -> None:
        reg = ModelRegistry(models=[])
        assert len(reg.list()) == 0
        custom = ModelInfo(id="my-model", name="My Model", owned_by="me")
        reg.register(custom)
        assert reg.has("my-model")
        assert reg.get("my-model") == custom

    def test_register_replaces_existing(self) -> None:
        reg = ModelRegistry(models=[])
        m1 = ModelInfo(id="x", name="First")
        m2 = ModelInfo(id="x", name="Second")
        reg.register(m1)
        reg.register(m2)
        assert reg.get("x").name == "Second"  # type: ignore[union-attr]
        assert len(reg.list()) == 1

    def test_to_openai_list_format(self) -> None:
        reg = ModelRegistry()
        result = reg.to_openai_list()
        assert result["object"] == "list"
        assert isinstance(result["data"], list)
        assert len(result["data"]) == len(DEFAULT_MODELS)
        # Each entry must have the required OpenAI fields.
        for item in result["data"]:
            assert "id" in item
            assert item["object"] == "model"
            assert "created" in item
            assert "owned_by" in item

    def test_empty_registry(self) -> None:
        reg = ModelRegistry(models=[])
        assert reg.list() == []
        assert reg.to_openai_list() == {"object": "list", "data": []}

    def test_default_models_are_google(self) -> None:
        """All default models should be owned by Google."""
        reg = ModelRegistry()
        for m in reg.list():
            assert m.owned_by == "google"
            assert "gemini" in m.id
