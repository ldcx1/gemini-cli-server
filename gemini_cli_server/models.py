"""Model registry — tracks available models and produces OpenAI-format responses."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a single model."""

    id: str
    name: str
    owned_by: str = "google"
    created: int = 0  # unix timestamp; 0 → auto-filled at registration time

    def to_openai_dict(self) -> dict:
        """Serialise to the format returned by ``GET /v1/models``."""
        return {
            "id": self.id,
            "object": "model",
            "created": self.created or int(time.time()),
            "owned_by": self.owned_by,
        }


# Default models known to the Gemini CLI.
DEFAULT_MODELS: list[ModelInfo] = [
    # Gemini models
    ModelInfo(id="gemini-3-pro-preview", name="Gemini 3 Pro Preview", owned_by="google"),
    ModelInfo(id="gemini-3-flash-preview", name="Gemini 3 Flash Preview", owned_by="google"),
    ModelInfo(id="gemini-2.5-pro", name="Gemini 2.5 Pro", owned_by="google"),
    ModelInfo(id="gemini-2.5-flash", name="Gemini 2.5 Flash", owned_by="google"),
    ModelInfo(id="gemini-2.5-flash-lite", name="Gemini 2.5 Flash Lite", owned_by="google"),
]


class ModelRegistry:
    """In-memory registry of available models."""

    def __init__(self, models: list[ModelInfo] | None = None) -> None:
        self._models: dict[str, ModelInfo] = {}
        for m in (DEFAULT_MODELS if models is None else models):
            self.register(m)

    def register(self, model: ModelInfo) -> None:
        """Add or replace a model in the registry."""
        self._models[model.id] = model

    def list(self) -> list[ModelInfo]:
        """Return all registered models."""
        return list(self._models.values())

    def get(self, model_id: str) -> ModelInfo | None:
        """Look up a model by ID, returning ``None`` if not found."""
        return self._models.get(model_id)

    def has(self, model_id: str) -> bool:
        """Check whether a model is registered."""
        return model_id in self._models

    def to_openai_list(self) -> dict:
        """Return the full model list in OpenAI ``/v1/models`` format."""
        return {
            "object": "list",
            "data": [m.to_openai_dict() for m in self._models.values()],
        }
