"""Integration tests reuse the QA and web test doubles."""

import sys
from pathlib import Path

TESTS = Path(__file__).parents[1]
for helpers in ("qa", "web"):
    path = str(TESTS / helpers)
    if path not in sys.path:
        sys.path.insert(0, path)
