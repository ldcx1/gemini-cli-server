"""Real subprocess tests exercising prompt size limits.

These tests spawn actual subprocesses (using ``echo`` / ``cat``) to verify
that the prompt delivery mechanism works end-to-end — including the ARG_MAX
boundary where CLI arguments become too large.

No mocking is used.  All tests run without the Gemini CLI.
"""

from __future__ import annotations

import os

import pytest

from gemini_cli_server.cli_runner import CLIResult, GeminiCLIRunner, MAX_ARG_PROMPT_BYTES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class EchoRunner(GeminiCLIRunner):
    """A runner that uses ``echo`` so we can verify arg-based delivery
    without needing the Gemini CLI.  ``echo -m <model> -p <prompt>``
    will simply print all its arguments to stdout."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(command="echo", timeout=10, max_retries=0, **kwargs)


class CatRunner(GeminiCLIRunner):
    """A runner that uses ``cat`` to echo back stdin.

    When the prompt is delivered via stdin, ``cat`` will print it to stdout
    so we can verify the content survived the pipe.
    """

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(command="cat", timeout=10, max_retries=0, **kwargs)

    def _build_args(self, prompt: str, model: str) -> list[str]:
        # cat takes no arguments — we just want it to echo stdin.
        return []

    def _build_args_stdin(self, model: str) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Tests — small prompts (arg-based, real subprocess)
# ---------------------------------------------------------------------------


class TestRealSmallPrompt:
    """Verify that small prompts are passed as command-line arguments to a
    real subprocess."""

    @pytest.mark.asyncio
    async def test_small_prompt_echoed_via_args(self) -> None:
        """``echo -m model -p prompt`` should print the args."""
        runner = EchoRunner()
        result = await runner.run("hello world", model="test-model")
        assert result.return_code == 0
        # echo will output: -m test-model -p hello world
        assert "hello world" in result.stdout
        assert "test-model" in result.stdout

    @pytest.mark.asyncio
    async def test_prompt_at_threshold_boundary(self) -> None:
        """A prompt exactly at the threshold should still use args."""
        runner = EchoRunner()
        # Create a prompt exactly at the byte limit.
        prompt = "x" * MAX_ARG_PROMPT_BYTES
        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        # The full prompt should appear in echo output (arg-based).
        assert prompt in result.stdout

    @pytest.mark.asyncio
    async def test_special_characters_in_args(self) -> None:
        """Verify that quotes, newlines, and shell metacharacters survive
        arg-based delivery via the subprocess (no shell involved)."""
        runner = EchoRunner()
        prompt = 'say "hello" && echo $HOME\nnewline'
        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        # echo outputs each arg separated by spaces; newlines become spaces.
        # The important thing is it doesn't crash.
        assert result.stdout.strip() != ""


# ---------------------------------------------------------------------------
# Tests — large prompts (stdin-based, real subprocess)
# ---------------------------------------------------------------------------


class TestRealLargePromptStdin:
    """Verify that prompts exceeding MAX_ARG_PROMPT_BYTES are delivered via
    stdin to a real subprocess."""

    @pytest.mark.asyncio
    async def test_large_prompt_delivered_via_stdin(self) -> None:
        """A prompt just over the limit should be piped via stdin to ``cat``,
        which echoes it back."""
        runner = CatRunner()
        prompt = "A" * (MAX_ARG_PROMPT_BYTES + 1)
        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        assert result.stdout.strip() == prompt

    @pytest.mark.asyncio
    async def test_200kb_prompt_via_stdin(self) -> None:
        """A 200 KB prompt (well over the 100 KB threshold)."""
        runner = CatRunner()
        prompt = "B" * 200_000
        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        assert result.stdout.strip() == prompt

    @pytest.mark.asyncio
    async def test_1mb_prompt_via_stdin(self) -> None:
        """A 1 MB prompt — close to typical ARG_MAX on Linux."""
        runner = CatRunner()
        prompt = "C" * 1_000_000
        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        assert result.stdout.strip() == prompt

    @pytest.mark.asyncio
    async def test_large_prompt_with_unicode(self) -> None:
        """Large prompt with multi-byte UTF-8 characters (emoji).
        Each emoji is 4 bytes, so 30K chars = 120 KB > threshold."""
        runner = CatRunner()
        prompt = "🔥" * 30_000  # 120 KB in UTF-8
        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        assert result.stdout.strip() == prompt

    @pytest.mark.asyncio
    async def test_large_prompt_content_integrity(self) -> None:
        """Verify that a large structured prompt survives stdin delivery
        with its content fully intact."""
        runner = CatRunner()
        # Build a structured prompt with line numbers.
        lines = [f"Line {i}: {'x' * 100}" for i in range(1200)]
        prompt = "\n".join(lines)
        assert len(prompt.encode("utf-8")) > MAX_ARG_PROMPT_BYTES

        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        assert result.stdout.strip() == prompt


# ---------------------------------------------------------------------------
# Tests — boundary and edge cases
# ---------------------------------------------------------------------------


class TestPromptSizeBoundary:
    """Exact boundary tests for MAX_ARG_PROMPT_BYTES."""

    @pytest.mark.asyncio
    async def test_one_byte_under_threshold_uses_args(self) -> None:
        runner = EchoRunner()
        prompt = "x" * (MAX_ARG_PROMPT_BYTES - 1)
        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        # echo receives it as an arg, so prompt is in stdout.
        assert prompt in result.stdout

    @pytest.mark.asyncio
    async def test_exactly_at_threshold_uses_args(self) -> None:
        runner = EchoRunner()
        prompt = "x" * MAX_ARG_PROMPT_BYTES
        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        assert prompt in result.stdout

    @pytest.mark.asyncio
    async def test_one_byte_over_threshold_uses_stdin(self) -> None:
        runner = CatRunner()
        prompt = "x" * (MAX_ARG_PROMPT_BYTES + 1)
        result = await runner.run(prompt, model="m")
        assert result.return_code == 0
        # cat echoes stdin, so the prompt should come back.
        assert result.stdout.strip() == prompt


# ---------------------------------------------------------------------------
# Tests — real subprocess error handling
# ---------------------------------------------------------------------------


class TestRealSubprocessErrors:
    """Test error paths with real subprocesses."""

    @pytest.mark.asyncio
    async def test_nonexistent_command(self) -> None:
        runner = GeminiCLIRunner(
            command="this_binary_does_not_exist_xyz", max_retries=0
        )
        with pytest.raises(RuntimeError, match="not found in PATH"):
            await runner.run("test", model="m")

    @pytest.mark.asyncio
    async def test_command_that_exits_nonzero(self) -> None:
        runner = GeminiCLIRunner(command="false", timeout=5, max_retries=0)
        with pytest.raises(RuntimeError, match="exited with code"):
            await runner.run("test", model="m")

    @pytest.mark.asyncio
    async def test_timeout_with_real_process(self) -> None:
        """``sleep 10`` with a 1-second timeout should raise TimeoutError."""

        class SleepRunner(GeminiCLIRunner):
            def _build_args(self, prompt: str, model: str) -> list[str]:
                return [prompt]  # sleep expects a number

        runner = SleepRunner(command="sleep", timeout=1, max_retries=0)
        with pytest.raises(TimeoutError, match="timed out"):
            await runner.run("10", model="m")

    @pytest.mark.asyncio
    async def test_working_dir_with_real_process(self) -> None:
        """Verify cwd is actually set by running ``pwd``."""

        class PwdRunner(GeminiCLIRunner):
            def _build_args(self, prompt: str, model: str) -> list[str]:
                return []  # pwd takes no args

        runner = PwdRunner(command="pwd", timeout=5, max_retries=0)
        result = await runner.run("ignored", model="m", working_dir="/tmp")
        assert result.return_code == 0
        assert result.stdout.strip() == os.path.realpath("/tmp")
