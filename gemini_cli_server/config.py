"""Server configuration with environment variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for the Gemini CLI Server.

    Values can be overridden with ``GEMINI_CLI_*`` environment variables.
    """

    host: str = "0.0.0.0"
    port: int = 11435
    default_model: str = "gemini-2.5-pro"
    gemini_command: str = "gemini"
    default_working_dir: str | None = None
    timeout: int = 300
    max_retries: int = 2
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Build configuration from environment variables.

        Recognised variables (all optional):
            ``GEMINI_CLI_HOST``
            ``GEMINI_CLI_PORT``
            ``GEMINI_CLI_DEFAULT_MODEL``
            ``GEMINI_CLI_COMMAND``
            ``GEMINI_CLI_WORKING_DIR``
            ``GEMINI_CLI_TIMEOUT``
            ``GEMINI_CLI_MAX_RETRIES``
            ``GEMINI_CLI_LOG_LEVEL``
        """
        return cls(
            host=os.environ.get("GEMINI_CLI_HOST", cls.host),
            port=int(os.environ.get("GEMINI_CLI_PORT", str(cls.port))),
            default_model=os.environ.get("GEMINI_CLI_DEFAULT_MODEL", cls.default_model),
            gemini_command=os.environ.get("GEMINI_CLI_COMMAND", cls.gemini_command),
            default_working_dir=os.environ.get("GEMINI_CLI_WORKING_DIR", cls.default_working_dir),
            timeout=int(os.environ.get("GEMINI_CLI_TIMEOUT", str(cls.timeout))),
            max_retries=int(os.environ.get("GEMINI_CLI_MAX_RETRIES", str(cls.max_retries))),
            log_level=os.environ.get("GEMINI_CLI_LOG_LEVEL", cls.log_level),
        )
