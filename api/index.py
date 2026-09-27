"""Vercel entrypoint: exports the FastAPI ``app``."""

import sys
from pathlib import Path

try:
    from hoa_qa.web.app import app
except ModuleNotFoundError:  # the project itself isn't installed in the bundle
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hoa_qa.web.app import app

__all__ = ["app"]
