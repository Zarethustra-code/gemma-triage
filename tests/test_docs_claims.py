"""The documentation is part of the product, so its claims are tested too.

A submission is judged partly on what it says about itself. These tests fail when the
README, the environment template or the app copy promises something the code does not
deliver -- an unconditional privacy guarantee, or an action described as executed when
it was only prepared.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.tools import (  # noqa: E402
    ACTION_STATUSES,
    STATUS_APPROVED,
    STATUS_CREATED,
    STATUS_FAILED,
    STATUS_PREPARED,
    STATUS_SUGGESTED,
)

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
ENV_EXAMPLE = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
APP = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
WRITEUP = (REPO_ROOT / "WRITEUP.md").read_text(encoding="utf-8")

#: Every prose file a judge might read. Scanning the whole set, rather than the two or
#: three that happened to get edited, is the point: WRITEUP.md kept a claim the README
#: had already dropped, and nothing caught it.
PROSE_FILES = sorted(
    p
    for p in [
        REPO_ROOT / "README.md",
        REPO_ROOT / "WRITEUP.md",
        REPO_ROOT / "app.py",
        REPO_ROOT / ".env.example",
        REPO_ROOT / "docs" / "README.md",
        REPO_ROOT / "agent" / "__init__.py",
        REPO_ROOT / "prompts" / "triage_prompt.txt",
        REPO_ROOT / "prompts" / "reply_prompt.txt",
    ]
    if p.exists()
)

#: Claims that were true only for one backend, stated as if they were universal.
OVERCLAIMS = (
    "never leaves your machine",
    "never leaves the machine",
    "never leaves this device",
    "always local",
    "fully private by default",
    "runs entirely on your device",
    "no email data leaves",
)


# ---------------------------------------------------------------------------
# Processing-location wording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PROSE_FILES, ids=lambda p: p.name)
def test_no_shipped_file_makes_an_unconditional_privacy_claim(path):
    """Checked across every prose file, not just the ones this pass happened to edit."""
    text = path.read_text(encoding="utf-8").lower()
    for claim in OVERCLAIMS:
        assert claim not in text, (
            f"{path.relative_to(REPO_ROOT)} still claims {claim!r} unconditionally"
        )


def test_the_writeup_states_the_processing_model():
    """The submission write-up is read by judges who may never open the README."""
    lowered = " ".join(WRITEUP.split()).lower()
    assert "supports fully local email processing through the transformers backend" in lowered
    assert "may be sent to the configured external inference provider" in lowered
    assert "always shown in the interface" in lowered


def test_the_writeup_describes_the_backend_ladder_accurately():
    lowered = " ".join(WRITEUP.split()).lower()
    assert "allow_remote_inference=true" in lowered, "the write-up must name the gate"
    assert "transformers → heuristic" in lowered or "transformers -> heuristic" in lowered


def test_the_writeup_does_not_describe_prepared_payloads_as_executed():
    lowered = " ".join(WRITEUP.split()).lower()
    for phrase in ("executes them for real", "executed into validated", "scheduled artefacts"):
        assert phrase not in lowered, f"WRITEUP.md still says {phrase!r}"


def test_the_readme_states_the_processing_model_exactly():
    """The wording the whole privacy story rests on, verbatim."""
    required = (
        "Gemma-Triage supports fully local email processing through the Transformers\n"
        "backend. When hosted inference is enabled, email content may be sent to the "
        "configured external\ninference provider. The active processing mode is always "
        "shown in the interface."
    )
    normalised = " ".join(README.split())
    assert " ".join(required.split()) in normalised


def test_the_space_front_matter_does_not_promise_on_device_processing():
    """The Space card is read by people who never open the app, and the hosted Space
    runs `hf_api`. Its one-line description must not claim otherwise."""
    (line,) = [ln for ln in README.splitlines() if ln.startswith("short_description:")]
    description = line.split(":", 1)[1].strip()

    assert len(description) <= 60, "Hugging Face truncates beyond 60 characters"
    assert "on-device" not in description.lower()
    assert "privacy-first" not in description.lower()


def test_the_package_docstring_matches_the_processing_model():
    docstring = (REPO_ROOT / "agent" / "__init__.py").read_text(encoding="utf-8").lower()
    assert "on-device email triage agent" not in docstring
    assert "hosted inference is opt-in" in docstring


def test_the_readme_covers_all_four_processing_claims():
    lowered = README.lower()
    assert "transformers backend keeps email content on the user's device" in lowered
    assert "may send email content to an external provider" in lowered
    assert "remote inference requires explicit permission" in lowered
    assert "the active backend is always shown in the ui" in lowered


# ---------------------------------------------------------------------------
# Action status vocabulary
# ---------------------------------------------------------------------------


def test_the_status_vocabulary_is_ordered_and_complete():
    assert ACTION_STATUSES == (
        STATUS_SUGGESTED,
        STATUS_PREPARED,
        STATUS_APPROVED,
        STATUS_CREATED,
        STATUS_FAILED,
    )
    assert STATUS_SUGGESTED == "Suggested action"
    assert STATUS_PREPARED == "Prepared payload"
    assert STATUS_APPROVED == "Approved"
    assert STATUS_CREATED == "Created (external)"
    assert STATUS_FAILED == "Failed"


def test_the_readme_documents_the_status_vocabulary():
    for status in ACTION_STATUSES:
        assert status in README, f"the README must define {status!r}"


def test_tasks_and_events_are_never_described_as_executed_or_created():
    for text, label in ((README, "README"), (APP, "app.py")):
        lowered = text.lower()
        for phrase in (
            "executes them for real",
            "executed tool calls",
            "creates the calendar event",
            "files the task",
        ):
            assert phrase not in lowered, f"{label} still says {phrase!r}"


def test_only_the_gmail_draft_can_reach_created_external():
    """The one action with a real external side effect, and it needs a click.

    Parsed, not grepped: the tool layer *defines* the constant (it owns the
    vocabulary) but must never *use* it, because nothing it does creates a remote
    record. Only the UI awards it, and only after a Gmail call returns.
    """
    import ast

    tree = ast.parse((REPO_ROOT / "agent" / "tools.py").read_text(encoding="utf-8"))

    # The ACTION_STATUSES tuple names every status; that is the vocabulary being
    # declared, not a status being applied to anything. Everything else counts.
    vocabulary = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "ACTION_STATUSES" for t in node.targets)
    ]
    declared = {id(n) for block in vocabulary for n in ast.walk(block)}

    uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "STATUS_CREATED"
        and isinstance(node.ctx, ast.Load)
        and id(node) not in declared
    ]
    assert uses == [], (
        "the tool layer prepares payloads; it must never claim an external record"
    )

    assert vocabulary, "but it does own the vocabulary"
    assert "STATUS_CREATED" in APP, "the UI awards it, after a successful Gmail call"


def test_every_documented_status_is_actually_reachable():
    """A vocabulary the README says the UI 'uses literally' must have no dead words."""
    tools_source = (REPO_ROOT / "agent" / "tools.py").read_text(encoding="utf-8")
    reachable = APP + tools_source

    for name, value in (
        ("STATUS_SUGGESTED", STATUS_SUGGESTED),
        ("STATUS_PREPARED", STATUS_PREPARED),
        ("STATUS_APPROVED", STATUS_APPROVED),
        ("STATUS_CREATED", STATUS_CREATED),
        ("STATUS_FAILED", STATUS_FAILED),
    ):
        # Two mentions are the definition and the ACTION_STATUSES tuple. A third means
        # it is applied somewhere; fewer means the word is documented but never shown.
        assert reachable.count(name) > 2, f"{value!r} is documented but never emitted"


# ---------------------------------------------------------------------------
# Environment template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    ["GEMMA_BACKEND=transformers", "ALLOW_REMOTE_INFERENCE=false", "MAX_ACTIONS_PER_EMAIL=5"],
)
def test_env_example_documents_the_new_variables(line):
    assert line in ENV_EXAMPLE


@pytest.mark.parametrize(
    "name", ["APP_MODE", "AUTH_REQUIRED", "TOKEN_STORAGE_DIR", "LOG_LEVEL"]
)
def test_no_out_of_scope_variables_were_introduced(name):
    """The live demo stays login-free; identity and auth are explicitly out of scope."""
    assert name not in ENV_EXAMPLE
    assert name not in APP
    assert name not in README


def test_the_env_example_contains_no_real_secret():
    for line in ENV_EXAMPLE.splitlines():
        if line.startswith("HF_TOKEN="):
            assert set(line.split("=", 1)[1].strip("hf_")) <= {"x"}, "placeholder only"


def test_the_readme_documents_the_new_variables():
    assert "`ALLOW_REMOTE_INFERENCE`" in README
    assert "`MAX_ACTIONS_PER_EMAIL`" in README


def test_the_readme_explains_why_the_hosted_space_may_be_remote():
    """The one deliberate exception has to be justified where a judge will read it."""
    lowered = README.lower()
    assert "allow_remote_inference=true" in lowered
    assert "synthetic" in lowered
    assert "sample_inbox.json" in lowered


# ---------------------------------------------------------------------------
# The demo is documented and recordable
# ---------------------------------------------------------------------------


def test_the_readme_has_a_local_demo_recording_checklist():
    assert "## Local Demo Recording Checklist" in README

    section = README.split("## Local Demo Recording Checklist", 1)[1].split("\n## ", 1)[0]
    lowered = section.lower()

    for beat in (
        "transformers",       # show the active local backend
        "run triage",         # classify
        "suggested reply",    # review the reply
        "create gmail draft", # explicit approval
        "drafts",             # show it in Gmail
        "msg-013",            # the injection email
    ):
        assert beat in lowered, f"the checklist must cover: {beat}"

    assert "prepared payload" in lowered, "the checklist must show the status vocabulary"
    assert "no email was sent" in lowered or "nothing is sent" in lowered


def test_the_injection_email_the_checklist_names_actually_exists():
    import json

    emails = json.loads((REPO_ROOT / "data" / "sample_inbox.json").read_text(encoding="utf-8"))
    email = next((e for e in emails["emails"] if e["id"] == "msg-013"), None)

    assert email is not None, "the checklist points at msg-013; it has to be there"
    assert "###SYSTEM###" in email["body"]
    assert len({e["id"] for e in emails["emails"]}) == len(emails["emails"])


def test_no_author_or_personal_name_is_present():
    """Submission hygiene: the repo carries no attribution to a real person."""
    import json
    import re

    emails = json.loads((REPO_ROOT / "data" / "sample_inbox.json").read_text(encoding="utf-8"))
    fictional = {e["from"].split("<")[0].strip() for e in emails["emails"]}

    for text, label in ((README, "README"), (APP, "app.py")):
        for match in re.findall(r"(?im)^\s*(?:author|by|created by|written by)\s*[:=]\s*(.+)$", text):
            assert not match.strip(), f"{label} carries an author attribution: {match!r}"

    # Every human name in the repo belongs to the synthetic inbox.
    assert all("example" in e["from"] for e in emails["emails"]), (
        "demo senders must stay on .example domains"
    )
    assert fictional, "sanity: the demo inbox has senders"


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
