"""Optional Gmail connector.

Imports are intentionally lazy: the Google client libraries are an optional
dependency, and the demo inbox must work on a machine that has never installed them.
"""

from __future__ import annotations

from .client import (
    SCOPES,
    GmailError,
    create_draft,
    fetch_recent,
    get_service,
    gmail_available,
)

__all__ = [
    "SCOPES",
    "GmailError",
    "create_draft",
    "fetch_recent",
    "get_service",
    "gmail_available",
]
