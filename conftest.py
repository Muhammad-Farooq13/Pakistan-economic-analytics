# conftest.py
"""
Root-level pytest configuration.
Adds the project root to sys.path so that `config` and `src` are
importable regardless of how pytest is invoked (locally or in CI).
"""
import sys
from pathlib import Path

# Ensure project root is always on the path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
