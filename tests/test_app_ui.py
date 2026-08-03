"""UI-layer behaviour: the draft button, the status banner, and error hygiene.

Skipped when Gradio is not installed, so a lean environment can still run the rest of
the suite. Gmail is faked throughout -- these tests assert that classification never
touches Gmail and that a draft happens only on an explicit click.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("gradio", reason="the UI layer needs Gradio installed")

import app  # noqa: E402
import gmail_integration  # noqa: E402
from agent.tools import STATUS_CREATED, STATUS_FAILED, STATUS_PREPARED  # noqa: E402
from gmail_integration import GmailError  # noqa: E402

EMAIL = {
    "id": "m1",
    "thread_id": "t-1",
    "from": "Daniel Okafor <daniel@brightpath.example>",
    "subject": "Contract review",
    "message_id": "<abc@mail.example>",
    "body": "Please send the signed copy.",
}


@pytest.fixture
def gmail(monkeypatch):
    """Fake connector that records draft calls and never reaches the network."""
    calls: list = []

    def fake_create_reply_draft(email, body, service=None):
        calls.append({"email": email, "body": body})
        return {
            "draft_id": "draft-9",
            "to": "daniel@brightpath.example",
            "subject": "Re: Contract review",
            "thread_id": email.get("thread_id", ""),
            "threaded": True,
        }

    monkeypatch.setattr(gmail_integration, "gmail_available", lambda: True)
    monkeypatch.setattr(gmail_integration, "create_reply_draft", fake_create_reply_draft)
    return calls


# ---------------------------------------------------------------------------
# The draft button
# ---------------------------------------------------------------------------


def test_a_click_creates_exactly_one_draft_from_the_edited_text(gmail):
    edited = "I rewrote this before approving it."

    message = app.create_gmail_draft([EMAIL], 0, edited)

    assert len(gmail) == 1
    assert gmail[0]["body"] == edited, "the draft must use the edited text, not the original"
    assert STATUS_CREATED in message
    assert "daniel@brightpath.example" in message
    assert "not** been sent" in message or "not been sent" in message


def test_an_empty_reply_creates_no_draft(gmail):
    for text in ("", "   ", None):
        message = app.create_gmail_draft([EMAIL], 0, text)
        assert "❌" in message
    assert gmail == [], "an empty reply must never reach Gmail"


@pytest.mark.parametrize("index", [-1, 1, 99])
def test_a_stale_row_index_creates_no_draft(gmail, index):
    message = app.create_gmail_draft([EMAIL], index, "text")
    assert "❌" in message
    assert gmail == []


def test_a_missing_display_state_creates_no_draft(gmail):
    assert "❌" in app.create_gmail_draft(None, 0, "text")
    assert "❌" in app.create_gmail_draft([], 0, "text")
    assert gmail == []


def test_an_unaddressable_sender_is_reported_and_no_draft_is_created(monkeypatch):
    """The real connector refuses; the UI surfaces the refusal verbatim."""
    monkeypatch.setattr(gmail_integration, "gmail_available", lambda: True)
    broken = dict(EMAIL, **{"from": "Nobody At All"})

    message = app.create_gmail_draft([broken], 0, "text")

    assert STATUS_FAILED in message
    assert "valid recipient" in message


def test_a_gmail_failure_is_reported_without_internal_detail(monkeypatch):
    monkeypatch.setattr(gmail_integration, "gmail_available", lambda: True)

    def explode(email, body, service=None):
        raise GmailError("Could not create the Gmail draft: quota exceeded")

    monkeypatch.setattr(gmail_integration, "create_reply_draft", explode)

    message = app.create_gmail_draft([EMAIL], 0, "text")

    assert STATUS_FAILED in message
    assert "quota exceeded" in message, "a user-safe GmailError message is shown"


def test_an_unexpected_exception_does_not_leak_internals(monkeypatch, capsys):
    monkeypatch.setattr(gmail_integration, "gmail_available", lambda: True)

    def explode(email, body, service=None):
        raise RuntimeError("/home/someone/token.json contains hf_SECRET")

    monkeypatch.setattr(gmail_integration, "create_reply_draft", explode)

    message = app.create_gmail_draft([EMAIL], 0, "text")

    assert STATUS_FAILED in message
    assert "hf_SECRET" not in message
    assert "token.json" not in message
    assert "Traceback" not in message


def test_missing_google_libraries_reuse_the_shared_message(monkeypatch):
    monkeypatch.setattr(gmail_integration, "gmail_available", lambda: False)
    message = app.create_gmail_draft([EMAIL], 0, "text")
    assert gmail_integration.GOOGLE_LIBS_MISSING in message


def test_running_triage_never_creates_a_draft(monkeypatch):
    """Classification is a read-only operation. Nothing about it touches Gmail."""
    calls: list = []
    monkeypatch.setattr(gmail_integration, "gmail_available", lambda: True)
    monkeypatch.setattr(
        gmail_integration,
        "create_reply_draft",
        lambda *a, **k: calls.append(a) or {},
    )

    emails = json.loads((REPO_ROOT / "data" / "sample_inbox.json").read_text(encoding="utf-8"))
    for _ in app.run_triage(emails["emails"][:4]):
        pass

    assert calls == [], "triage must not create drafts"


# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------


class FakeLLM:
    requested_backend = "auto"
    model_id = "google/gemma-4-E4B-it"
    notes: list = []

    def __init__(self, backend, allow_remote=False):
        self.active_backend = backend
        self.allow_remote = allow_remote

    @property
    def is_local(self):
        return self.active_backend != "hf_api"

    @property
    def is_remote(self):
        return self.active_backend == "hf_api"

    @property
    def backend_label(self):
        return f"label for {self.active_backend}"


def banner(monkeypatch, backend, allow_remote=False):
    monkeypatch.setattr(app.ENGINE, "agent", type("A", (), {"llm": FakeLLM(backend, allow_remote)})())
    return app.status_bar_html()


def test_the_banner_says_local_for_on_device_backends(monkeypatch):
    for backend in ("transformers", "heuristic"):
        html = banner(monkeypatch, backend)
        assert "LOCAL processing" in html
        assert "REMOTE processing" not in html


def test_the_banner_says_remote_for_hosted_inference(monkeypatch):
    html = banner(monkeypatch, "hf_api")
    assert "REMOTE processing" in html
    assert "external" in html


def test_the_banner_states_whether_remote_is_permitted(monkeypatch):
    assert "Remote inference is disabled" in banner(monkeypatch, "transformers", allow_remote=False)
    assert "Remote inference is enabled" in banner(monkeypatch, "transformers", allow_remote=True)


def test_the_banner_makes_no_unconditional_privacy_claim(monkeypatch):
    for backend, allow in (("transformers", False), ("heuristic", False), ("hf_api", True)):
        html = banner(monkeypatch, backend, allow).lower()
        assert "never leaves" not in html
        assert "fully private" not in html
        assert "always local" not in html


# ---------------------------------------------------------------------------
# Action status vocabulary in the rendered detail
# ---------------------------------------------------------------------------


def test_prepared_actions_are_labelled_prepared_not_created(monkeypatch):
    from agent import GemmaLLM, TriageAgent

    agent = TriageAgent(llm=GemmaLLM(backend="heuristic"))
    emails = json.loads((REPO_ROOT / "data" / "sample_inbox.json").read_text(encoding="utf-8"))
    processed = agent.process_inbox(emails["emails"])

    with_actions = [p for p in processed if p.tasks or p.events]
    assert with_actions, "expected the demo inbox to produce prepared actions"

    for item in with_actions:
        html = app.detail_html(item)
        assert "Prepared actions" in html
        assert STATUS_PREPARED in html
        assert "Executed tool calls" not in html
        for task in item.tasks:
            assert task.payload["status"] == STATUS_PREPARED
        for event in item.events:
            assert event.payload["status"] == STATUS_PREPARED


def test_nothing_in_a_triage_run_claims_an_external_record_was_created(monkeypatch):
    from agent import GemmaLLM, TriageAgent

    agent = TriageAgent(llm=GemmaLLM(backend="heuristic"))
    emails = json.loads((REPO_ROOT / "data" / "sample_inbox.json").read_text(encoding="utf-8"))
    processed = agent.process_inbox(emails["emails"])

    blob = json.dumps([p.to_dict(include_raw=True) for p in processed])
    assert STATUS_CREATED not in blob


# ---------------------------------------------------------------------------
# Output plumbing
# ---------------------------------------------------------------------------


def test_build_outputs_matches_the_component_graph():
    """Gradio needs a fixed-width output tuple; a row-shape change must fail loudly.

    A row is 11 wide: group, header, accordion, detail, reply, tone, regenerate button,
    regenerate status, draft button, draft status, row state. Both fillers have to
    agree with that number and with each other.
    """
    assert app.ROW_WIDGETS == 11

    outputs = app.build_outputs([], 0, "status")
    assert len(outputs) == 4 + app.MAX_ROWS * app.ROW_WIDGETS
    assert len(app.blank_row()) == app.ROW_WIDGETS


def test_build_outputs_exposes_the_displayed_email_order():
    from agent import GemmaLLM, TriageAgent

    agent = TriageAgent(llm=GemmaLLM(backend="heuristic"))
    emails = json.loads((REPO_ROOT / "data" / "sample_inbox.json").read_text(encoding="utf-8"))
    processed = agent.process_inbox(emails["emails"][:5])

    _, order, *_ = app.build_outputs(processed, 5, "status")

    assert [e["id"] for e in order] == [p.email["id"] for p in processed], (
        "the draft button resolves its row through this list; it must match display order"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
