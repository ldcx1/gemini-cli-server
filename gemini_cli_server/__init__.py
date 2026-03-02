"""Gemini CLI Server — OpenAI-compatible API wrapping the Gemini CLI."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gemini-cli-server")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback for editable installs without metadata.
