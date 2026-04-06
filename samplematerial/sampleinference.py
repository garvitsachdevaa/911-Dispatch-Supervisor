"""Sample inference launcher for the 911 dispatch project.

Use this file as a runnable reference from samplematerial.
For submission, the authoritative script is the root-level inference.py.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from inference import main as run_inference

    return asyncio.run(run_inference())


if __name__ == "__main__":
    raise SystemExit(main())