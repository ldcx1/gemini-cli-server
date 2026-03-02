<p align="center">
  <h1 align="center">Gemini CLI Server</h1>
  <p align="center">
    <strong>OpenAI-compatible API server wrapping the <a href="https://github.com/google-gemini/gemini-cli">Gemini CLI</a></strong><br>
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/tests-74%20unit%20%2B%2025%20e2e-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-86%25-brightgreen?style=flat-square" alt="Coverage">
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **OpenAI-compatible API** | Works with any client that speaks the OpenAI chat completions format |
| **Drop-in Ollama replacement** | Change your base URL and you're done |
| **True streaming** | Real-time SSE streaming — tokens are emitted as the CLI generates them |
| **Multiple models** | Supports all Gemini models available through the CLI |
| **Working directory control** | Point the CLI at any directory for context-aware code exploration |
| **Large prompt handling** | Prompts over 100 KB are automatically piped via stdin |
| **Multimodal support** | Base64-encoded images via OpenAI's vision format |
| **Privacy-first logging** | User prompts are never logged |
| **Zero configuration** | Sensible defaults, everything configurable via env vars |

> [!TIP]
> For image analysis tasks, use `gemini-2.5-pro` or higher. `gemini-2.5-flash` may hallucinate that it cannot read image files through the CLI.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- [**Gemini CLI**](https://github.com/google-gemini/gemini-cli) installed and authenticated

### Installation

```bash
# Install
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### Start the Server

```bash
# Default: http://0.0.0.0:11435
python -m gemini_cli_server

# Or use the CLI command
gemini-cli-server
```

### Make a Request

```bash
# List models
curl http://localhost:11435/v1/models

# Chat completion
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# With a working directory (for code exploration)
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [{"role": "user", "content": "Explain the architecture of this project"}],
    "working_dir": "/path/to/your/project"
  }'

# Streaming
curl -N -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Write a poem"}],
    "stream": true
  }'
```

---

## 🔄 Drop-in Replacement

### OpenAI Python SDK

```python
# Before (Ollama)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

# After (Gemini CLI Server) — just change the port
client = OpenAI(base_url="http://localhost:11435/v1", api_key="unused")

response = client.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### LiteLLM

```python
import litellm

response = litellm.completion(
    model="openai/gemini-2.5-pro",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="http://localhost:11435/v1",
    api_key="unused",
)
```

---

## 📡 API Reference

### Endpoints

| Method | Path | Description |
|:------:|------|-------------|
| `GET` | `/` | Server info |
| `GET` | `/health` | Health check (verifies CLI is responsive) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/{id}` | Get model details |
| `POST` | `/v1/chat/completions` | Create chat completion |

### Chat Completion Request

```jsonc
{
  "model": "gemini-2.5-pro",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,            // Enable SSE streaming
  "working_dir": "/path/to/project"  // Non-standard extension
}
```

> The `working_dir` field sets the working directory for the CLI subprocess — enabling context-aware responses for code exploration. Also accepted via the `X-Working-Dir` HTTP header.

### Available Models

| Model ID | Description |
|----------|-------------|
| `gemini-3-pro-preview` | Gemini 3 Pro Preview |
| `gemini-3-flash-preview` | Gemini 3 Flash Preview |
| `gemini-2.5-pro` | Gemini 2.5 Pro *(default)* |
| `gemini-2.5-flash` | Gemini 2.5 Flash |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite |

> Any model identifier accepted by the Gemini CLI can be used — the server passes it through directly.

---

## ⚙️ Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_CLI_HOST` | `0.0.0.0` | Server bind address |
| `GEMINI_CLI_PORT` | `11435` | Server port |
| `GEMINI_CLI_DEFAULT_MODEL` | `gemini-2.5-pro` | Default model |
| `GEMINI_CLI_COMMAND` | `gemini` | Path to the Gemini CLI binary |
| `GEMINI_CLI_WORKING_DIR` | *(cwd)* | Default working directory |
| `GEMINI_CLI_TIMEOUT` | `300` | CLI timeout in seconds |
| `GEMINI_CLI_MAX_RETRIES` | `2` | Max retries on transient errors |
| `GEMINI_CLI_LOG_LEVEL` | `info` | Log level |

---

## 🛡️ Security Considerations

<details>
<summary><strong>Working directory access</strong></summary>

The `working_dir` parameter is passed directly to the CLI subprocess — **there is no path validation or sandboxing**. Any accessible path on the server can be specified.

This is by design for local and trusted-network use. If exposing to untrusted clients:
- Run the server in a container or restricted environment
- Add a reverse proxy with path validation
- Implement an allowlist of permitted directories

</details>

<details>
<summary><strong>Prompt size limits</strong></summary>

- Prompts **under 100 KB** → passed as CLI arguments (fast, simple)
- Prompts **over 100 KB** → automatically piped via stdin (avoids `ARG_MAX` limits)

</details>

<details>
<summary><strong>Privacy</strong></summary>

**User prompts are never logged.** Only metadata is recorded (model name, working directory, prompt size, timing).

</details>

---

## 🧪 Development

```bash
make install-dev     # Install dev dependencies
make test            # Run unit tests
make test-cov        # Run with coverage
make test-e2e        # Run E2E tests (requires gemini CLI)
make lint            # Lint
make format          # Format
```

### Project Structure

```
gemini-cli-server/
├── gemini_cli_server/
│   ├── __init__.py            # Package version
│   ├── __main__.py            # CLI entry point
│   ├── api_types.py           # OpenAI-compatible Pydantic models
│   ├── cli_runner.py          # Subprocess executor + streaming
│   ├── config.py              # Configuration with env overrides
│   ├── models.py              # Model registry
│   └── server.py              # FastAPI application + SSE streaming
├── tests/
│   ├── conftest.py            # Shared fixtures & mock runner
│   ├── test_api_types.py      # API type validation
│   ├── test_cli_runner.py     # CLI runner unit tests
│   ├── test_config.py         # Configuration tests
│   ├── test_e2e.py            # End-to-end tests with real CLI
│   ├── test_models.py         # Model registry tests
│   ├── test_real_subprocess.py# Real subprocess tests
│   └── test_server.py         # Server endpoint tests
├── Makefile
├── pyproject.toml
└── README.md
```

**Test suite**: 74 unit/integration tests + 25 E2E tests — **86% code coverage**.

---

## ⚠️ Disclaimer

This project wraps the [Gemini CLI](https://github.com/google-gemini/gemini-cli), which is subject to **Google's usage limits, quotas, and Terms of Service**. By using this server, you agree to comply with all applicable [Google Terms of Service](https://policies.google.com/terms) and [Gemini API Terms](https://ai.google.dev/gemini-api/terms). It is your responsibility to review and adhere to these policies, including any rate limits or usage restrictions that may apply.

---

## 📄 License

[MIT](LICENSE)
