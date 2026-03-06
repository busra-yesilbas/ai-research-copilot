"""Root-level pytest configuration.

Ensures the project root is on ``sys.path`` so that ``from app.xxx import yyy``
works regardless of how pytest is invoked (e.g. from IDE run configurations).

Also defines shared fixtures available across the entire test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root is importable.
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
