"""Entry-point: ``python -m gemini_cli_server``."""

from __future__ import annotations

import logging

import uvicorn

from gemini_cli_server.config import ServerConfig
from gemini_cli_server.server import create_app


def main() -> None:
    """Start the server."""
    config = ServerConfig.from_env()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )


if __name__ == "__main__":
    main()
