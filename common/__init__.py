"""common package — shared utilities for the HITL PR review agent.

This file also configures the rich Console to be safe on Windows consoles
that default to cp1252 (where Unicode glyphs like '→' / '✓' would crash).
"""

from __future__ import annotations

import os
import sys

# Force UTF-8 stdio so Rich/Streamlit can print Unicode glyphs on Windows
# (default Windows console encoding is cp1252 → UnicodeEncodeError).
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# Avoid Rich trying to use the legacy Windows console renderer (which is
# what triggers the cp1252 encode error).  When running under the modern
# Windows Terminal or under streamlit-run, this is a no-op.
try:  # pragma: no cover — defensive
    import rich.console as _rc

    _orig_console_init = _rc.Console.__init__

    def _patched_console_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("legacy_windows", False)
        kwargs.setdefault("force_terminal", None)
        return _orig_console_init(self, *args, **kwargs)

    _rc.Console.__init__ = _patched_console_init  # type: ignore[assignment]
except Exception:
    pass
