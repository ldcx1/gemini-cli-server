"""Shared pytest fixtures."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gemini_cli_server.cli_runner import CLIResult, GeminiCLIRunner, HealthCheckResult
from gemini_cli_server.config import ServerConfig
from gemini_cli_server.models import ModelRegistry
from gemini_cli_server.server import create_app


class MockRunner(GeminiCLIRunner):
    """A runner that returns canned responses without touching the filesystem."""

    def __init__(self, response: str = "Hello from Gemini!", return_code: int = 0) -> None:
        super().__init__(command="gemini", timeout=10, max_retries=0)
        self.response = response
        self.return_code = return_code
        self.last_prompt: str | None = None
        self.last_model: str | None = None
        self.last_working_dir: str | None = None

    async def run(
        self,
        prompt: str,
        model: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> CLIResult:
        self.last_prompt = prompt
        self.last_model = model
        self.last_working_dir = working_dir
        if self.return_code != 0:
            raise RuntimeError(f"CLI failed with code {self.return_code}")
        return CLIResult(
            stdout=self.response,
            stderr="",
            return_code=0,
            elapsed=0.1,
        )

    async def run_stream(
        self,
        prompt: str,
        model: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> AsyncGenerator[str, None]:
        self.last_prompt = prompt
        self.last_model = model
        self.last_working_dir = working_dir
        if self.return_code != 0:
            raise RuntimeError(f"CLI failed with code {self.return_code}")
        
        chunk_size = 5
        for i in range(0, len(self.response), chunk_size):
            yield self.response[i : i + chunk_size]

    async def check_health(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True, command="gemini")


@pytest.fixture
def mock_runner() -> MockRunner:
    return MockRunner()


@pytest.fixture
def config() -> ServerConfig:
    return ServerConfig(
        host="127.0.0.1",
        port=11435,
        default_model="gemini-2.5-pro",
        timeout=10,
    )


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry()


@pytest.fixture
def app(config: ServerConfig, mock_runner: MockRunner, registry: ModelRegistry) -> FastAPI:
    return create_app(config=config, runner=mock_runner, registry=registry)


@pytest.fixture
async def client(app: object) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
