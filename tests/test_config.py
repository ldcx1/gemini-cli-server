"""Unit tests for configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gemini_cli_server.config import ServerConfig


class TestServerConfigDefaults:
    """Test default configuration values."""

    def test_defaults(self) -> None:
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 11435
        assert cfg.default_model == "gemini-2.5-pro"
        assert cfg.gemini_command == "gemini"
        assert cfg.default_working_dir is None
        assert cfg.timeout == 300
        assert cfg.max_retries == 2
        assert cfg.log_level == "info"


class TestServerConfigFromEnv:
    """Test environment variable overrides."""

    def test_all_overrides(self) -> None:
        env = {
            "GEMINI_CLI_HOST": "127.0.0.1",
            "GEMINI_CLI_PORT": "9999",
            "GEMINI_CLI_DEFAULT_MODEL": "gemini-3-flash-preview",
            "GEMINI_CLI_COMMAND": "/usr/local/bin/gemini",
            "GEMINI_CLI_WORKING_DIR": "/projects",
            "GEMINI_CLI_TIMEOUT": "60",
            "GEMINI_CLI_MAX_RETRIES": "5",
            "GEMINI_CLI_LOG_LEVEL": "debug",
        }
        with patch.dict("os.environ", env, clear=False):
            cfg = ServerConfig.from_env()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9999
        assert cfg.default_model == "gemini-3-flash-preview"
        assert cfg.gemini_command == "/usr/local/bin/gemini"
        assert cfg.default_working_dir == "/projects"
        assert cfg.timeout == 60
        assert cfg.max_retries == 5
        assert cfg.log_level == "debug"

    def test_partial_overrides(self) -> None:
        with patch.dict("os.environ", {"GEMINI_CLI_PORT": "8080"}, clear=False):
            cfg = ServerConfig.from_env()
        assert cfg.port == 8080
        # Other values remain at defaults.
        assert cfg.host == "0.0.0.0"
        assert cfg.default_model == "gemini-2.5-pro"

    def test_no_env_vars(self) -> None:
        # Ensure from_env works even when no GEMINI_CLI_* vars exist.
        cfg = ServerConfig.from_env()
        assert cfg.port == 11435
