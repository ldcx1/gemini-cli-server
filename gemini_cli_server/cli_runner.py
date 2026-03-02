"""Gemini CLI subprocess runner with retry logic and timeout handling."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import AsyncGenerator, TypedDict

logger = logging.getLogger(__name__)


class HealthCheckResult(TypedDict, total=False):
    """Return type for :meth:`GeminiCLIRunner.check_health`."""

    healthy: bool
    command: str
    error: str  # only present when healthy is False


# Transient error patterns that warrant automatic retry.
TRANSIENT_ERROR_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"503.*overload",
        r"503.*over capacity",
        r"503.*too many requests",
        r"429.*rate limit",
        r"temporarily unavailable",
        r"service unavailable",
        r"connection.*reset",
        r"connection.*refused",
    ]
]

# Maximum prompt size (in bytes) that can be passed as a CLI argument.
# Prompts exceeding this are piped via stdin to avoid ARG_MAX limits.
MAX_ARG_PROMPT_BYTES: int = 100_000  # 100 KB


@dataclass
class CLIResult:
    """Result of a single CLI invocation."""

    stdout: str
    stderr: str
    return_code: int
    elapsed: float  # seconds


class GeminiCLIRunner:
    """Execute the Gemini CLI in non-interactive (prompt) mode.

    Usage::

        runner = GeminiCLIRunner()
        result = await runner.run("Hello!", model="gemini-2.5-pro")
        print(result.stdout)
    """

    def __init__(
        self,
        command: str = "gemini",
        timeout: int = 300,
        max_retries: int = 2,
    ) -> None:
        self.command = command
        self.timeout = timeout
        self.max_retries = max_retries

    # ── public API ────────────────────────────────────────────────────────

    async def check_health(self) -> HealthCheckResult:
        """Verify that the Gemini CLI is installed and responsive.

        Runs ``gemini --help`` and checks for a zero exit code.

        Returns:
            A :class:`HealthCheckResult` with ``"healthy"`` (bool),
            ``"command"`` (str), and optionally ``"error"`` (str) if the
            check failed.
        """
        try:
            env = self._clean_env()
            process = await asyncio.create_subprocess_exec(
                self.command,
                "--help",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=10,
            )
            if process.returncode == 0:
                return HealthCheckResult(healthy=True, command=self.command)
            else:
                error = stderr_bytes.decode("utf-8", errors="replace").strip()
                return HealthCheckResult(
                    healthy=False,
                    command=self.command,
                    error=error or f"exit code {process.returncode}",
                )
        except FileNotFoundError:
            return HealthCheckResult(
                healthy=False,
                command=self.command,
                error=f"'{self.command}' not found in PATH",
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                healthy=False,
                command=self.command,
                error="health check timed out",
            )

    async def run(
        self,
        prompt: str,
        model: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> CLIResult:
        """Run the Gemini CLI and return the result.

        Args:
            prompt: Prompt text sent via ``-p`` (or stdin for large prompts).
            model: Model identifier sent via ``-m``.
            working_dir: Working directory for the subprocess (``cwd``).
            timeout: Per-invocation timeout override (seconds).

        Returns:
            A :class:`CLIResult` with stdout, stderr, return code, and elapsed
            time.

        Raises:
            TimeoutError: If the CLI does not respond within *timeout* seconds.
            RuntimeError: If the CLI exits with a non-zero return code after
                exhausting retries.
        """
        effective_timeout = timeout or self.timeout
        cwd = working_dir or os.getcwd()
        prompt_bytes = len(prompt.encode("utf-8"))

        # NOTE: prompts are NEVER logged to prevent leaking sensitive content.
        logger.info(
            "Running gemini CLI | model=%s cwd=%s timeout=%ss prompt_len=%d",
            model,
            cwd,
            effective_timeout,
            prompt_bytes,
        )

        # Decide delivery method based on prompt size.
        use_stdin = prompt_bytes > MAX_ARG_PROMPT_BYTES
        if use_stdin:
            args = self._build_args_stdin(model)
            logger.info(
                "Prompt exceeds %d bytes (%d bytes), using stdin delivery",
                MAX_ARG_PROMPT_BYTES,
                prompt_bytes,
            )
        else:
            args = self._build_args(prompt, model)

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._execute(
                    args, cwd, effective_timeout,
                    stdin_data=prompt if use_stdin else None,
                )

                if result.return_code != 0:
                    if self._is_transient(result.stderr) and attempt < self.max_retries:
                        wait = 2**attempt
                        logger.warning(
                            "Transient error (attempt %d/%d) — retrying in %ds",
                            attempt + 1,
                            self.max_retries + 1,
                            wait,
                        )
                        last_error = result.stderr
                        await asyncio.sleep(wait)
                        continue

                    raise RuntimeError(
                        f"Gemini CLI exited with code {result.return_code}: "
                        f"{result.stderr[:200]}"
                    )

                if attempt > 0:
                    logger.info("Succeeded on retry attempt %d", attempt + 1)
                return result

            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Gemini CLI timed out after {effective_timeout}s"
                ) from None
            except FileNotFoundError:
                raise RuntimeError(
                    f"Gemini CLI command '{self.command}' not found in PATH"
                ) from None

        # All retries exhausted (should not normally reach here).
        raise RuntimeError(
            f"Gemini CLI failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    async def run_stream(
        self,
        prompt: str,
        model: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Run the Gemini CLI and yield output as it's streamed."""
        effective_timeout = timeout or self.timeout
        cwd = working_dir or os.getcwd()
        prompt_bytes = len(prompt.encode("utf-8"))

        use_stdin = prompt_bytes > MAX_ARG_PROMPT_BYTES
        args = self._build_args_stdin(model) if use_stdin else self._build_args(prompt, model)

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            env = self._clean_env()
            process = await asyncio.create_subprocess_exec(
                self.command,
                *args,
                stdin=asyncio.subprocess.PIPE if use_stdin else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            if use_stdin and process.stdin:
                process.stdin.write(prompt.encode("utf-8"))
                process.stdin.close()
                await process.stdin.wait_closed()

            try:
                # Read the first chunk to detect immediate failures or content.
                assert process.stdout is not None
                first_chunk = await asyncio.wait_for(process.stdout.read(1024), timeout=effective_timeout)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
                raise TimeoutError(f"Gemini CLI stream timed out after {effective_timeout}s") from None

            if not first_chunk:
                # Process exited without writing to stdout.
                await process.wait()
                stderr_bytes = await (process.stderr.read() if process.stderr else asyncio.sleep(0, result=b""))
                stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()
                
                if process.returncode != 0:
                    if self._is_transient(stderr_str) and attempt < self.max_retries:
                        wait = 2**attempt
                        logger.warning("Transient error (attempt %d/%d) — retrying in %ds", attempt + 1, self.max_retries + 1, wait)
                        last_error = stderr_str
                        await asyncio.sleep(wait)
                        continue
                    raise RuntimeError(f"Gemini CLI exited with code {process.returncode}: {stderr_str[:200]}")
                else:
                    return

            # Yield the first chunk.
            yield first_chunk.decode("utf-8", errors="replace")

            while True:
                try:
                    chunk = await asyncio.wait_for(process.stdout.read(1024), timeout=effective_timeout)
                    if not chunk:
                        break
                    yield chunk.decode("utf-8", errors="replace")
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
                    raise TimeoutError(f"Gemini CLI stream timed out after {effective_timeout}s") from None

            await process.wait()
            if process.returncode != 0:
                stderr_bytes = await (process.stderr.read() if process.stderr else asyncio.sleep(0, result=b""))
                stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Gemini CLI failed mid-stream with code {process.returncode}: {stderr_str[:200]}")

            return

        raise RuntimeError(f"Gemini CLI failed after {self.max_retries + 1} attempts. Last error: {last_error}")

    # ── internals ─────────────────────────────────────────────────────────

    def _build_args(self, prompt: str, model: str) -> list[str]:
        """Construct the CLI argument list (prompt as -p arg)."""
        return ["-m", model, "-p", prompt]

    def _build_args_stdin(self, model: str) -> list[str]:
        """Construct the CLI argument list for stdin delivery (no -p)."""
        return ["-m", model]

    async def _execute(
        self,
        args: list[str],
        cwd: str,
        timeout: int,
        stdin_data: str | None = None,
    ) -> CLIResult:
        """Spawn the subprocess and wait for completion."""
        env = self._clean_env()
        start = time.monotonic()

        process = await asyncio.create_subprocess_exec(
            self.command,
            *args,
            stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        input_bytes = stdin_data.encode("utf-8") if stdin_data else None
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=input_bytes), timeout=timeout
            )
        except asyncio.TimeoutError:
            # Ensure the child process is cleaned up to prevent zombies.
            try:
                process.kill()
            except ProcessLookupError:
                pass  # Process already exited.
            await process.wait()
            raise

        elapsed = time.monotonic() - start
        return CLIResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            return_code=process.returncode if process.returncode is not None else -1,
            elapsed=elapsed,
        )

    @staticmethod
    def _clean_env() -> dict[str, str]:
        """Return a copy of ``os.environ`` with problematic keys removed."""
        return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    @staticmethod
    def _is_transient(stderr: str) -> bool:
        """Return True if *stderr* matches a transient error pattern."""
        return any(pat.search(stderr) for pat in TRANSIENT_ERROR_PATTERNS)
