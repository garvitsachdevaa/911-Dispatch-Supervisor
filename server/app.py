"""OpenEnv server shim for validators.

The canonical FastAPI app lives in src.server.app.
This module exists to satisfy OpenEnv's expected multi-mode layout.
"""

from src.server.app import app


def main() -> None:
    """Run the OpenEnv FastAPI server."""
    from src.server.app import main as _main

    _main()


if __name__ == "__main__":
    main()
