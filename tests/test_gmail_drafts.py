"""Gmail draft creation, address validation and reply threading.

Every test here runs against a fake service object. Nothing in this file performs
network I/O, reads a credential file, or triggers an OAuth flow -- ``create_draft``
takes an injected ``service``, and the fake records every call made against it so the
"never sends" guarantee can be asserted rather than assumed.
"""

from __future__ import annotations

import base64
import sys
from email import message_from_bytes
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gmail_integration import (  # noqa: E402
    GOOGLE_LIBS_MISSING,
    GmailError,
    build_reply_headers,
    create_draft,
    create_reply_draft,
    extract_address,
    fetch_recent,
    normalize_reply_subject,
)
from gmail_integration import client as gmail_client  # noqa: E402

#: Captured before the autouse fixture below replaces it, so the real implementation
#: is still reachable for the one test that needs to exercise it.
REAL_GET_SERVICE = gmail_client.get_service


# ---------------------------------------------------------------------------
# Fake Gmail service
# ---------------------------------------------------------------------------


class FakeExecutable:
    def __init__(self, log, name, kwargs, result, error=None):
        self._log = log
        self._name = name
        self._kwargs = kwargs
        self._result = result
        self._error = error

    def execute(self):
        self._log.append((self._name, self._kwargs))
        if self._error is not None:
            raise self._error
        return self._result


class FakeDrafts:
    def __init__(self, log, error=None):
        self._log = log
        self._error = error

    def create(self, userId, body):
        return FakeExecutable(
            self._log, "drafts.create", {"userId": userId, "body": body},
            {"id": "draft-123"}, self._error,
        )

    def send(self, *args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("drafts().send must never be called")


class FakeMessages:
    def __init__(self, log, listing=None, messages=None):
        self._log = log
        self._listing = listing or {}
        self._messages = messages or {}

    def list(self, **kwargs):
        return FakeExecutable(self._log, "messages.list", kwargs, self._listing)

    def get(self, **kwargs):
        return FakeExecutable(
            self._log, "messages.get", kwargs, self._messages.get(kwargs.get("id"), {})
        )

    def send(self, *args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("messages().send must never be called")


class FakeUsers:
    def __init__(self, log, draft_error=None, listing=None, messages=None):
        self._log = log
        self._draft_error = draft_error
        self._listing = listing
        self._messages = messages

    def drafts(self):
        return FakeDrafts(self._log, self._draft_error)

    def messages(self):
        return FakeMessages(self._log, self._listing, self._messages)


class FakeService:
    """Records every API call made against it."""

    def __init__(self, draft_error=None, listing=None, messages=None):
        self.log: list = []
        self._draft_error = draft_error
        self._listing = listing
        self._messages = messages

    def users(self):
        return FakeUsers(self.log, self._draft_error, self._listing, self._messages)

    @property
    def calls(self):
        return [name for name, _ in self.log]


@pytest.fixture(autouse=True)
def never_authenticate(monkeypatch):
    """Any code path that reaches real OAuth is a test failure, not a slow test."""

    def explode(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("get_service() was called; the test should inject a service")

    monkeypatch.setattr(gmail_client, "get_service", explode)


def decode_draft(service: FakeService):
    """Pull the RFC 5322 message back out of the recorded drafts.create call."""
    (_, kwargs), = [entry for entry in service.log if entry[0] == "drafts.create"]
    raw = kwargs["body"]["message"]["raw"]
    return message_from_bytes(base64.urlsafe_b64decode(raw)), kwargs["body"]["message"]


ORIGINAL = {
    "id": "m1",
    "thread_id": "t-77",
    "from": "Daniel Okafor <daniel.okafor@brightpath-partners.example>",
    "subject": "Contract review",
    "message_id": "<abc123@mail.example>",
    "references": "<root@mail.example>",
    "body": "Please send the signed copy.",
}


# ---------------------------------------------------------------------------
# Address extraction and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("daniel@b.example", "daniel@b.example"),
        ("Daniel Okafor <daniel.okafor@b.example>", "daniel.okafor@b.example"),
        ('"Okafor, Daniel" <d@b.co.example>', "d@b.co.example"),
        ("  <spaced@b.example>  ", "spaced@b.example"),
        ("first+tag@sub.b.example", "first+tag@sub.b.example"),
    ],
)
def test_extract_address_accepts_real_addresses(raw, expected):
    assert extract_address(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        None,
        "   ",
        "not-an-email",
        "Name <>",
        "@b.example",
        "daniel@",
        "daniel@localhost",
        "daniel@b",
        "two@b.example, other@b.example",
        "daniel@b.example\nBcc: attacker@evil.example",
        "daniel@b.example\r\nTo: attacker@evil.example",
    ],
)
def test_extract_address_rejects_everything_else(raw):
    """Malformed input returns None -- including header-injection attempts."""
    assert extract_address(raw) is None


# ---------------------------------------------------------------------------
# Subject normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subject,expected",
    [
        ("Contract review", "Re: Contract review"),
        ("Re: Contract review", "Re: Contract review"),
        ("Re: Re: Re: Contract review", "Re: Contract review"),
        ("RE: re:  RE: Contract review", "Re: Contract review"),
        ("re:Contract review", "Re: Contract review"),
        ("", "Re: (no subject)"),
        (None, "Re: (no subject)"),
        ("Re:", "Re: (no subject)"),
    ],
)
def test_normalize_reply_subject_keeps_exactly_one_prefix(subject, expected):
    assert normalize_reply_subject(subject) == expected


def test_normalize_reply_subject_is_idempotent():
    once = normalize_reply_subject("Contract review")
    assert normalize_reply_subject(once) == once


# ---------------------------------------------------------------------------
# Threading headers
# ---------------------------------------------------------------------------


def test_build_reply_headers_chains_message_id_onto_references():
    headers = build_reply_headers(ORIGINAL)
    assert headers["In-Reply-To"] == "<abc123@mail.example>"
    assert headers["References"] == "<root@mail.example> <abc123@mail.example>"


def test_build_reply_headers_without_prior_references():
    headers = build_reply_headers({"message_id": "<solo@mail.example>"})
    assert headers["In-Reply-To"] == "<solo@mail.example>"
    assert headers["References"] == "<solo@mail.example>"


def test_build_reply_headers_without_a_message_id():
    """A missing Message-ID means no In-Reply-To -- never a fabricated one."""
    assert build_reply_headers({"references": "<root@mail.example>"}) == {
        "References": "<root@mail.example>"
    }
    assert build_reply_headers({}) == {}
    assert build_reply_headers({"message_id": "", "references": ""}) == {}


def test_build_reply_headers_drops_values_carrying_newlines():
    assert build_reply_headers({"message_id": "<a@b>\nX-Evil: yes"}) == {}


# ---------------------------------------------------------------------------
# Draft creation
# ---------------------------------------------------------------------------


def test_create_reply_draft_produces_a_threaded_unsent_draft():
    service = FakeService()

    result = create_reply_draft(ORIGINAL, "Happy to sign it today.", service=service)

    message, payload = decode_draft(service)
    assert message["To"] == "daniel.okafor@brightpath-partners.example"
    assert message["Subject"] == "Re: Contract review"
    assert message["In-Reply-To"] == "<abc123@mail.example>"
    assert message["References"] == "<root@mail.example> <abc123@mail.example>"
    assert "Happy to sign it today." in message.get_payload()
    assert payload["threadId"] == "t-77"

    assert result["draft_id"] == "draft-123"
    assert result["to"] == "daniel.okafor@brightpath-partners.example"
    assert result["threaded"] is True
    assert service.calls == ["drafts.create"], "exactly one API call, and it is a draft"


def test_create_reply_draft_uses_the_edited_text_verbatim():
    service = FakeService()
    edited = "I rewrote this reply entirely before approving it."

    create_reply_draft(ORIGINAL, edited, service=service)

    message, _ = decode_draft(service)
    assert edited in message.get_payload()


def test_create_reply_draft_rejects_a_malformed_recipient_without_calling_gmail():
    service = FakeService()
    broken = dict(ORIGINAL, **{"from": "Nobody At All"})

    with pytest.raises(GmailError) as excinfo:
        create_reply_draft(broken, "text", service=service)

    assert service.log == [], "no draft may be created for an unaddressable reply"
    assert "valid recipient" in str(excinfo.value)
    assert "Nobody At All" not in str(excinfo.value), "errors must not echo raw input"


@pytest.mark.parametrize("bad", ["", None, "   ", "<>"])
def test_create_draft_rejects_empty_recipients(bad):
    service = FakeService()
    with pytest.raises(GmailError):
        create_draft(to=bad, subject="s", body="b", service=service)
    assert service.log == []


def test_create_reply_draft_without_a_thread_id_omits_it():
    service = FakeService()
    no_thread = {k: v for k, v in ORIGINAL.items() if k != "thread_id"}

    create_reply_draft(no_thread, "text", service=service)

    _, payload = decode_draft(service)
    assert "threadId" not in payload, "an absent thread id must not become an empty one"


def test_create_reply_draft_without_a_message_id_still_creates_a_draft():
    """Imperfect threading beats no draft at all."""
    service = FakeService()
    no_id = {k: v for k, v in ORIGINAL.items() if k not in ("message_id", "references")}

    result = create_reply_draft(no_id, "text", service=service)

    message, payload = decode_draft(service)
    assert message["In-Reply-To"] is None
    assert message["References"] is None
    assert payload["threadId"] == "t-77", "the thread id alone still threads it"
    assert result["threaded"] is False


def test_a_gmail_api_failure_becomes_a_user_safe_error():
    service = FakeService(draft_error=RuntimeError("500 backend error"))

    with pytest.raises(GmailError) as excinfo:
        create_reply_draft(ORIGINAL, "text", service=service)

    assert "Could not create the Gmail draft" in str(excinfo.value)


def test_nothing_is_ever_sent():
    """The fake raises on any send; a passing draft flow proves none was attempted."""
    service = FakeService()
    create_reply_draft(ORIGINAL, "text", service=service)
    assert service.calls == ["drafts.create"]


def test_no_send_call_exists_anywhere_in_the_connector():
    """Belt and braces: the capability is absent from the source, not merely unused.

    Parsed rather than grepped, so the guarantee is about executable code and is not
    satisfied -- or broken -- by prose in a docstring that happens to say "send".
    """
    import ast

    source = (REPO_ROOT / "gmail_integration" / "client.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "send" not in called, "the connector must contain no send call"

    assert "gmail.send" not in source, "the send scope must never be requested"
    assert gmail_client.SCOPES == [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
    ]


# ---------------------------------------------------------------------------
# Fetch stores what threading later needs
# ---------------------------------------------------------------------------


def test_fetch_recent_stores_the_message_id_and_references():
    """A reply drafted later cannot thread without headers captured at fetch time."""
    message = {
        "id": "m1",
        "threadId": "t-77",
        "snippet": "hello",
        "payload": {
            "mimeType": "text/plain",
            "body": {"data": base64.urlsafe_b64encode(b"Body text").decode()},
            "headers": [
                {"name": "From", "value": "Daniel <daniel@b.example>"},
                {"name": "To", "value": "me@b.example"},
                {"name": "Subject", "value": "Contract review"},
                {"name": "Date", "value": "Sat, 1 Aug 2026 10:00:00 +0000"},
                {"name": "Message-ID", "value": "<abc123@mail.example>"},
                {"name": "References", "value": "<root@mail.example>"},
            ],
        },
    }
    service = FakeService(listing={"messages": [{"id": "m1"}]}, messages={"m1": message})

    (email,) = fetch_recent(5, service=service)

    assert email["message_id"] == "<abc123@mail.example>"
    assert email["references"] == "<root@mail.example>"
    assert email["thread_id"] == "t-77"

    # And the captured headers are enough to thread a reply.
    draft_service = FakeService()
    create_reply_draft(email, "reply text", service=draft_service)
    threaded, _ = decode_draft(draft_service)
    assert threaded["In-Reply-To"] == "<abc123@mail.example>"


def test_fetch_recent_tolerates_a_message_with_no_threading_headers():
    message = {
        "id": "m2",
        "threadId": "t-1",
        "snippet": "hi",
        "payload": {"headers": [{"name": "From", "value": "a@b.example"}]},
    }
    service = FakeService(listing={"messages": [{"id": "m2"}]}, messages={"m2": message})

    (email,) = fetch_recent(5, service=service)

    assert email["message_id"] == ""
    assert email["references"] == ""


# ---------------------------------------------------------------------------
# The missing-libraries message has exactly one home
# ---------------------------------------------------------------------------


def test_the_missing_libraries_message_is_defined_once_and_reused():
    assert "pip install google-api-python-client" in GOOGLE_LIBS_MISSING

    app_source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    client_source = (REPO_ROOT / "gmail_integration" / "client.py").read_text(encoding="utf-8")

    assert "GOOGLE_LIBS_MISSING" in app_source, "the UI must reuse the constant"
    assert app_source.count("pip install google-api-python-client") == 0, (
        "the install instruction must not be duplicated into the UI layer"
    )
    assert client_source.count("pip install google-api-python-client") == 1


def test_get_service_raises_the_shared_message_when_libraries_are_absent(monkeypatch):
    """The one place the sentence is raised from, verified against the one definition."""
    monkeypatch.setattr(gmail_client, "gmail_available", lambda: False)

    with pytest.raises(GmailError) as excinfo:
        REAL_GET_SERVICE(interactive=False)

    assert str(excinfo.value) == GOOGLE_LIBS_MISSING


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
