"""Optional Gmail connector.

Imports are intentionally lazy: the Google client libraries are an optional
dependency, and the demo inbox must work on a machine that has never installed them.
"""

from __future__ import annotations

from .client import (
    GOOGLE_LIBS_MISSING,
    SCOPES,
    GmailError,
    build_reply_headers,
    create_draft,
    create_reply_draft,
    extract_address,
    fetch_recent,
    get_service,
    gmail_available,
    normalize_reply_subject,
)

__all__ = [
    "GOOGLE_LIBS_MISSING",
    "SCOPES",
    "GmailError",
    "build_reply_headers",
    "create_draft",
    "create_reply_draft",
    "extract_address",
    "fetch_recent",
    "get_service",
    "gmail_available",
    "normalize_reply_subject",
]
