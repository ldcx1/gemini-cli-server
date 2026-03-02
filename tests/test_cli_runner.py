"""Unit tests for the Gemini CLI runner."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gemini_cli_server.cli_runner import (
    GeminiCLIRunner,
    CLIResult,
    MAX_ARG_PROMPT_BYTES,
    TRANSIENT_ERROR_PATTERNS,
)


class TestBuildArgs:
    """Test argument construction."""

    def test_basic_args(self) -> None:
        runner = GeminiCLIRunner()
        args = runner._build_args("Hello", "gemini-2.5-pro")
        assert args == ["-m", "gemini-2.5-pro", "-p", "Hello"]

    def test_model_substitution(self) -> None:
        runner = GeminiCLIRunner()
        args = runner._build_args("Hi", "claude-3-5-sonnet")
        assert "-m" in args
        assert "claude-3-5-sonnet" in args

    def test_prompt_with_special_chars(self) -> None:
        runner = GeminiCLIRunner()
        prompt = 'What is "hello" in French?'
        args = runner._build_args(prompt, "gemini-2.5-pro")
        assert prompt in args

    def test_stdin_args_have_no_prompt(self) -> None:
        runner = GeminiCLIRunner()
        args = runner._build_args_stdin("gemini-2.5-pro")
        assert args == ["-m", "gemini-2.5-pro"]
        assert "-p" not in args


class TestCleanEnv:
    """Test environment sanitisation."""

    def test_removes_claudecode(self) -> None:
        with patch.dict("os.environ", {"CLAUDECODE": "1", "PATH": "/usr/bin"}, clear=True):
            env = GeminiCLIRunner._clean_env()
            assert "CLAUDECODE" not in env
            assert "PATH" in env

    def test_preserves_other_vars(self) -> None:
        with patch.dict("os.environ", {"HOME": "/home/user", "LANG": "en"}, clear=True):
            env = GeminiCLIRunner._clean_env()
            assert env["HOME"] == "/home/user"
            assert env["LANG"] == "en"


class TestTransientDetection:
    """Test transient error pattern matching."""

    @pytest.mark.parametrize(
        "stderr",
        [
            "503 Service Temporarily Unavailable: overload",
            "Error 429: rate limit exceeded",
            "connection reset by peer",
            "connection refused",
            "service unavailable",
        ],
    )
    def test_detects_transient(self, stderr: str) -> None:
        assert GeminiCLIRunner._is_transient(stderr) is True

    @pytest.mark.parametrize(
        "stderr",
        [
            "invalid model name",
            "authentication failed",
            "permission denied",
            "",
        ],
    )
    def test_non_transient(self, stderr: str) -> None:
        assert GeminiCLIRunner._is_transient(stderr) is False


class TestRun:
    """Test the high-level run() method."""

    @pytest.mark.asyncio
    async def test_successful_run(self) -> None:
        runner = GeminiCLIRunner(command="echo", timeout=5, max_retries=0)
        # Patch _execute to return a canned result.
        canned = CLIResult(stdout="Hello!", stderr="", return_code=0, elapsed=0.1)
        runner._execute = AsyncMock(return_value=canned)  # type: ignore[assignment]

        result = await runner.run("test", model="gemini-2.5-pro")
        assert result.stdout == "Hello!"
        assert result.return_code == 0

    @pytest.mark.asyncio
    async def test_working_dir_passed(self) -> None:
        runner = GeminiCLIRunner(timeout=5, max_retries=0)
        canned = CLIResult(stdout="ok", stderr="", return_code=0, elapsed=0.1)
        runner._execute = AsyncMock(return_value=canned)  # type: ignore[assignment]

        await runner.run("test", model="m", working_dir="/tmp")
        # The second positional arg to _execute is cwd.
        call_args = runner._execute.call_args
        assert call_args[0][1] == "/tmp"

    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        runner = GeminiCLIRunner(timeout=1, max_retries=0)
        runner._execute = AsyncMock(side_effect=asyncio.TimeoutError)  # type: ignore[assignment]

        with pytest.raises(TimeoutError, match="timed out"):
            await runner.run("test", model="m")

    @pytest.mark.asyncio
    async def test_non_zero_exit_raises(self) -> None:
        runner = GeminiCLIRunner(timeout=5, max_retries=0)
        canned = CLIResult(stdout="", stderr="bad model", return_code=1, elapsed=0.1)
        runner._execute = AsyncMock(return_value=canned)  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="bad model"):
            await runner.run("test", model="m")

    @pytest.mark.asyncio
    async def test_retry_on_transient(self) -> None:
        runner = GeminiCLIRunner(timeout=5, max_retries=1)
        fail = CLIResult(stdout="", stderr="503 overload", return_code=1, elapsed=0.1)
        ok = CLIResult(stdout="recovered", stderr="", return_code=0, elapsed=0.2)
        runner._execute = AsyncMock(side_effect=[fail, ok])  # type: ignore[assignment]

        result = await runner.run("test", model="m")
        assert result.stdout == "recovered"
        assert runner._execute.call_count == 2  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises(self) -> None:
        runner = GeminiCLIRunner(timeout=5, max_retries=1)
        fail = CLIResult(stdout="", stderr="503 overload", return_code=1, elapsed=0.1)
        still_fail = CLIResult(stdout="", stderr="503 overload again", return_code=1, elapsed=0.1)
        runner._execute = AsyncMock(side_effect=[fail, still_fail])  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="503 overload again"):
            await runner.run("test", model="m")


class TestCheckHealth:
    """Test the check_health() method."""

    @pytest.mark.asyncio
    async def test_healthy_when_command_succeeds(self) -> None:
        runner = GeminiCLIRunner(command="echo")
        result = await runner.check_health()
        assert result["healthy"] is True
        assert result["command"] == "echo"
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_unhealthy_when_command_not_found(self) -> None:
        runner = GeminiCLIRunner(command="this_command_does_not_exist_xyz")
        result = await runner.check_health()
        assert result["healthy"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_unhealthy_when_command_fails(self) -> None:
        runner = GeminiCLIRunner(command="false")  # `false` always exits 1
        result = await runner.check_health()
        assert result["healthy"] is False
        assert "error" in result


class TestRunFileNotFound:
    """Test that a missing command gives a clear error."""

    @pytest.mark.asyncio
    async def test_missing_command_raises_runtime_error(self) -> None:
        runner = GeminiCLIRunner(command="this_command_does_not_exist_xyz", max_retries=0)
        with pytest.raises(RuntimeError, match="not found in PATH"):
            await runner.run("test", model="m")


class TestLargePromptStdin:
    """Test stdin-based delivery for large prompts."""

    @pytest.mark.asyncio
    async def test_large_prompt_uses_stdin(self) -> None:
        """Prompts exceeding MAX_ARG_PROMPT_BYTES should be piped via stdin."""
        runner = GeminiCLIRunner(timeout=5, max_retries=0)
        canned = CLIResult(stdout="ok", stderr="", return_code=0, elapsed=0.1)
        runner._execute = AsyncMock(return_value=canned)  # type: ignore[assignment]

        large_prompt = "x" * (MAX_ARG_PROMPT_BYTES + 1)
        await runner.run(large_prompt, model="m")

        call_kwargs = runner._execute.call_args
        # args should NOT contain -p (stdin delivery)
        args_passed = call_kwargs[0][0]
        assert "-p" not in args_passed
        assert args_passed == ["-m", "m"]
        # stdin_data should contain the prompt
        assert call_kwargs.kwargs["stdin_data"] == large_prompt

    @pytest.mark.asyncio
    async def test_small_prompt_uses_args(self) -> None:
        """Prompts under the threshold should be passed as -p arg."""
        runner = GeminiCLIRunner(timeout=5, max_retries=0)
        canned = CLIResult(stdout="ok", stderr="", return_code=0, elapsed=0.1)
        runner._execute = AsyncMock(return_value=canned)  # type: ignore[assignment]

        await runner.run("small prompt", model="m")

        call_kwargs = runner._execute.call_args
        args_passed = call_kwargs[0][0]
        assert "-p" in args_passed
        assert "small prompt" in args_passed
        # stdin_data should be None
        assert call_kwargs.kwargs.get("stdin_data") is None


class TestRunStream:
    """Test the run_stream() async generator."""

    @pytest.mark.asyncio
    async def test_stream_collects_full_output(self) -> None:
        """run_stream should yield chunks that reassemble into the full output."""
        runner = GeminiCLIRunner(command="echo", timeout=5, max_retries=0)
        chunks: list[str] = []
        async for chunk in runner.run_stream("Hello streaming!", model="m"):
            chunks.append(chunk)
        full = "".join(chunks).strip()
        # echo passes prompt via -p flag, so output includes args
        assert len(full) > 0

    @pytest.mark.asyncio
    async def test_stream_timeout_raises(self) -> None:
        """Timeout during streaming should raise TimeoutError."""
        runner = GeminiCLIRunner(command="bash", timeout=1, max_retries=0)
        # Use bash -c to ignore the extra flags and just sleep
        original_build = runner._build_args

        def _slow_args(prompt: str, model: str) -> list[str]:
            return ["-c", "sleep 30"]

        runner._build_args = _slow_args  # type: ignore[assignment]
        with pytest.raises(TimeoutError, match="timed out"):
            async for _ in runner.run_stream("ignored", model="m"):
                pass

    @pytest.mark.asyncio
    async def test_stream_missing_command_raises(self) -> None:
        """A missing command should raise RuntimeError."""
        runner = GeminiCLIRunner(
            command="this_command_does_not_exist_xyz", timeout=5, max_retries=0,
        )
        with pytest.raises((RuntimeError, FileNotFoundError)):
            async for _ in runner.run_stream("test", model="m"):
                pass
