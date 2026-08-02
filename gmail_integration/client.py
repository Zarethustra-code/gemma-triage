"""Gmail OAuth client: fetch recent mail, create drafts.

Scopes are deliberately minimal:

* ``gmail.readonly`` -- read messages so the agent can triage them.
* ``gmail.compose``  -- create *drafts* only. This client never sends mail; a human
  always presses send.

Credentials are read from a local ``credentials.json`` (path configurable via
``GOOGLE_CREDENTIALS_FILE``) and the resulting token is cached in ``token.json``
(``GOOGLE_TOKEN_FILE``). Both are gitignored. Nothing here is required for the demo
inbox path, and every import of the Google libraries is lazy so the app runs without them.
"""

from __future__ import annotations

import base64
import html
import os
import re
from email.message import EmailMessage
from typing import Any, Dict, List, Optional

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

CREDENTIALS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
TOKEN_FILE = os.environ.get("GOOGLE_TOKEN_FILE", "token.json")

MAX_BODY_CHARS = 8000


class GmailError(RuntimeError):
    """Raised for any recoverable Gmail problem, with a message safe to show a user."""


def gmail_available() -> bool:
    """True when the optional Google client libraries are importable."""
    try:
        import google.auth  # noqa: F401
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def get_service(interactive: bool = True) -> Any:
    """Build an authorised Gmail API service object.

    Uses a cached token when present, refreshes it when expired, and otherwise runs
    the local OAuth consent flow (which needs a browser, hence ``interactive``).
    """
    if not gmail_available():
        raise GmailError(
            "Google API libraries are not installed. Run: "
            "pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
        )

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds: Optional[Credentials] = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as exc:  # noqa: BLE001 - a corrupt cache should not be fatal
            raise GmailError(f"Could not read the cached token at {TOKEN_FILE}: {exc}") from exc

    if creds and creds.valid:
        return build("gmail", "v1", credentials=creds)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
            return build("gmail", "v1", credentials=creds)
        except Exception:  # noqa: BLE001 - fall through to a fresh consent flow
            creds = None

    if not interactive:
        raise GmailError("No valid cached Gmail token and interactive login is disabled.")

    if not os.path.exists(CREDENTIALS_FILE):
        raise GmailError(
            f"OAuth client file not found at '{CREDENTIALS_FILE}'. Create a Desktop-app "
            "OAuth client in Google Cloud Console, download the JSON, and point "
            "GOOGLE_CREDENTIALS_FILE at it."
        )

    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
    except Exception as exc:  # noqa: BLE001
        raise GmailError(f"OAuth flow failed: {exc}") from exc

    _save_token(creds)
    return build("gmail", "v1", credentials=creds)


def _save_token(creds: Any) -> None:
    """Cache the token locally. The path is gitignored -- never commit it."""
    try:
        with open(TOKEN_FILE, "w", encoding="utf-8") as handle:
            handle.write(creds.to_json())
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass  # a read-only filesystem just means re-authenticating next time


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def _decode(data: Optional[str]) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>", "\n", raw)
    return html.unescape(_TAG_RE.sub(" ", raw))


def _extract_body(payload: Dict[str, Any]) -> str:
    """Walk the MIME tree, preferring text/plain and falling back to stripped HTML."""
    plain_parts: List[str] = []
    html_parts: List[str] = []

    def walk(part: Dict[str, Any]) -> None:
        mime = part.get("mimeType", "")
        body = part.get("body", {}) or {}
        if mime == "text/plain":
            plain_parts.append(_decode(body.get("data")))
        elif mime == "text/html":
            html_parts.append(_decode(body.get("data")))
        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)

    text = "\n".join(p for p in plain_parts if p).strip()
    if not text:
        text = _html_to_text("\n".join(html_parts)).strip()

    text = _WS_RE.sub("\n\n", "\n".join(line.rstrip() for line in text.splitlines()))
    return text[:MAX_BODY_CHARS].strip()


def _header(headers: List[Dict[str, str]], name: str) -> str:
    target = name.lower()
    for header in headers:
        if header.get("name", "").lower() == target:
            return header.get("value", "")
    return ""


def fetch_recent(n: int = 10, query: str = "in:inbox", service: Any = None) -> List[Dict[str, Any]]:
    """Fetch the ``n`` most recent messages as plain dicts the agent understands.

    Returned shape matches ``data/sample_inbox.json`` exactly, so the demo path and
    the Gmail path feed identical data into :class:`~agent.triage.TriageAgent`.
    """
    n = max(1, min(int(n), 50))
    service = service or get_service()

    try:
        listing = (
            service.users()
            .messages()
            .list(userId="me", maxResults=n, q=query)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise GmailError(f"Could not list Gmail messages: {exc}") from exc

    emails: List[Dict[str, Any]] = []
    for stub in listing.get("messages", []) or []:
        try:
            message = (
                service.users()
                .messages()
                .get(userId="me", id=stub["id"], format="full")
                .execute()
            )
        except Exception:  # noqa: BLE001 - skip an unreadable message, keep the rest
            continue

        payload = message.get("payload", {}) or {}
        headers = payload.get("headers", []) or []
        emails.append(
            {
                "id": message.get("id", stub.get("id", "")),
                "thread_id": message.get("threadId", ""),
                "from": _header(headers, "From"),
                "to": _header(headers, "To"),
                "subject": _header(headers, "Subject") or "(no subject)",
                "date": _header(headers, "Date"),
                "body": _extract_body(payload) or message.get("snippet", ""),
            }
        )

    return emails


# ---------------------------------------------------------------------------
# Writing (drafts only -- never send)
# ---------------------------------------------------------------------------


def create_draft(
    to: str,
    subject: str,
    body: str,
    thread_id: Optional[str] = None,
    service: Any = None,
) -> Dict[str, Any]:
    """Create a Gmail draft. Sending stays a deliberate human action."""
    if not to:
        raise GmailError("A recipient address is required to create a draft.")

    service = service or get_service()

    message = EmailMessage()
    message.set_content(body or "")
    message["To"] = to
    message["Subject"] = subject or "(no subject)"

    encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
    draft_body: Dict[str, Any] = {"message": {"raw": encoded}}
    if thread_id:
        draft_body["message"]["threadId"] = thread_id

    try:
        created = service.users().drafts().create(userId="me", body=draft_body).execute()
    except Exception as exc:  # noqa: BLE001
        raise GmailError(f"Could not create the Gmail draft: {exc}") from exc

    return {"draft_id": created.get("id", ""), "to": to, "subject": subject}
