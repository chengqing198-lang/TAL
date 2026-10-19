"""Local web UI for TAL: model access, prompt libraries, chat probing, runs.

The UI is deliberately dependency-free (stdlib HTTP server + vanilla JS) so it
works on the same offline machine as `tal demo`.
"""

from __future__ import annotations

__all__ = ["serve"]


def serve(*args, **kwargs):  # pragma: no cover - thin re-export
    from .server import serve as _serve

    return _serve(*args, **kwargs)
