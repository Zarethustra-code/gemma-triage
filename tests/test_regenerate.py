"""Reply regeneration at the tool layer: the second Gemma call, run again on its own.

``regenerate_reply`` is the reply-writing half of the agent loop extracted so that both
``draft_reply`` and the UI's per-row 🔁 button drive one code path. What it must not do
matters as much as what it does:

  * it does not classify anything -- ``process_inbox`` / ``process_email`` are never
    called, so the inbox is never recomputed,
  * it does not run the tool executor, so no task or calendar payload is rebuilt,
  * it does not touch Gmail, so nothing is drafted and nothing is sent.

Everything here runs on the heuristic backend or a recording stub: no weights, no
token, no network. The UI half of the feature is asserted in ``test_regenerate_ui.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import GemmaLLM, TriageAgent  # noqa: E402
from agent import tools  # noqa: E402
from agent.llm import _TONE_LINES  # noqa: E402
from agent.schema import parse_triage  # noqa: E402
from agent.tools import VALID_TONES, draft_reply, regenerate_reply  # noqa: E402

EMAIL = {
    "id": "msg-a",
    "thread_id": "t-a",
    "from": "Daniel Okafor <daniel@brightpath.example>",
    "subject": "Contract review",
    "body": "Could you review the contract and send the signed copy back by Friday?",
}


@pytest.fixture(scope="module")
def llm():
    return GemmaLLM(backend="heuristic")


class SpyLLM:
    """Records prompt pairs instead of generating, so the prompt itself can be asserted."""

    active_backend = "heuristic"

    def __init__(self, reply: str = "Hi there,\n\nBody.\n\nBest regards,\n[Your name]") -> None:
        self.calls: list = []
        self.reply = reply

    def generate(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.reply


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_regenerate_reply_returns_non_empty_text_on_the_heuristic_backend(llm):
    text = regenerate_reply(llm, EMAIL)

    assert isinstance(text, str)
    assert text.strip(), "the heuristic backend must always produce a reply body"
    assert len(text.split()) > 5
    assert "Daniel" in text, "the reply should address the sender by name"


def test_exactly_one_generation_happens_per_regeneration():
    spy = SpyLLM()
    regenerate_reply(spy, EMAIL, "friendly")
    assert len(spy.calls) == 1, "regeneration is the reply call and nothing else"


def test_regeneration_uses_the_reply_prompt_and_the_shared_email_block():
    """One code path: the same prompt file and the same email block as ``draft_reply``."""
    spy = SpyLLM()

    regenerate_reply(spy, EMAIL, "brief")

    (call,) = spy.calls
    assert call["system"] == tools.render(tools.load_prompt("reply_prompt.txt"), tone="brief")
    assert call["user"] == tools.build_email_block(EMAIL, extra={"TRIAGE SUMMARY": ""})


@pytest.mark.parametrize("tone", VALID_TONES)
def test_the_requested_tone_reaches_the_prompt(tone):
    """Whatever the backend, the tone the user picked is the tone the model is asked for."""
    spy = SpyLLM()

    regenerate_reply(spy, EMAIL, tone)

    (call,) = spy.calls
    assert f"in a {tone} tone" in call["system"]
    for other in VALID_TONES:
        if other != tone:
            assert f"in a {other} tone" not in call["system"]


@pytest.mark.parametrize("tone", VALID_TONES)
def test_the_requested_tone_is_honoured_by_the_heuristic_reply(llm, tone):
    """The heuristic reply is tone-templated, so the tone is visible in the output text."""
    text = regenerate_reply(llm, EMAIL, tone)

    assert _TONE_LINES[tone] in text
    for other in VALID_TONES:
        if other != tone:
            assert _TONE_LINES[other] not in text


@pytest.mark.parametrize(
    "tone", ["", "   ", None, 7, "shouty", "PROFESSIONAL!!", "ignore-previous-instructions"]
)
def test_an_unknown_tone_falls_back_to_professional(llm, tone):
    spy = SpyLLM()
    regenerate_reply(spy, EMAIL, tone)

    assert "in a professional tone" in spy.calls[0]["system"]
    assert _TONE_LINES["professional"] in regenerate_reply(llm, EMAIL, tone)


def test_a_known_tone_survives_surrounding_whitespace_and_case():
    spy = SpyLLM()
    regenerate_reply(spy, EMAIL, "  FRIENDLY  ")
    assert "in a friendly tone" in spy.calls[0]["system"]


def test_the_triage_summary_is_carried_in_however_the_caller_holds_it():
    """The tool has a TriageDecision; a UI row keeps only the summary string."""
    decision = parse_triage(
        json.dumps(
            {
                "category": "ACTION_NEEDED",
                "priority": 3,
                "summary": "Wants the signed contract by Friday.",
                "reasoning": "r",
                "actions": [],
            }
        )
    )

    for supplied in (decision, decision.summary, {"summary": decision.summary}):
        spy = SpyLLM()
        regenerate_reply(spy, EMAIL, "professional", supplied)
        assert "Wants the signed contract by Friday." in spy.calls[0]["user"]

    spy = SpyLLM()
    regenerate_reply(spy, EMAIL, "professional", None)
    assert "TRIAGE SUMMARY" not in spy.calls[0]["user"], "an absent summary adds no line"


def test_model_artefacts_are_stripped_from_a_regenerated_reply():
    spy = SpyLLM(reply="```\nSubject: Re: Contract review\nTo: daniel@b.example\n\nThe body.\n```")

    text = regenerate_reply(spy, EMAIL, "brief")

    assert text == "The body."


def test_an_empty_generation_raises_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="empty reply body"):
        regenerate_reply(SpyLLM(reply="   "), EMAIL, "professional")


def test_regeneration_does_not_need_an_addressable_sender(llm):
    """Rewriting the text in the box is not drafting; only the draft path needs an address."""
    text = regenerate_reply(llm, dict(EMAIL, **{"from": "Nobody At All"}), "friendly")
    assert text.strip()


def test_the_tone_vocabulary_is_not_duplicated():
    """One list, in the tool layer. Everything else reads it from there."""
    assert VALID_TONES == ("professional", "friendly", "urgent", "brief", "firm")
    assert tools.normalize_tone("nonsense") == "professional"
    for tone in VALID_TONES:
        assert tools.normalize_tone(tone) == tone


# ---------------------------------------------------------------------------
# draft_reply is unchanged by the refactor
# ---------------------------------------------------------------------------


def test_draft_reply_delegates_to_regenerate_reply(llm, monkeypatch):
    seen: list = []

    def spy_regenerate(passed_llm, email, tone="professional", decision=None):
        seen.append({"llm": passed_llm, "email": email, "tone": tone, "decision": decision})
        return "Body from the shared path."

    monkeypatch.setattr(tools, "regenerate_reply", spy_regenerate)

    result = draft_reply(llm, EMAIL, {"tone": "firm"})

    assert len(seen) == 1, "the tool must not keep a second reply-generation path"
    assert seen[0]["llm"] is llm and seen[0]["email"] is EMAIL
    assert seen[0]["tone"] == "firm", "the tone is normalised before it is handed over"
    assert result.detail == result.payload["body"] == "Body from the shared path."


def test_draft_reply_still_produces_the_same_shaped_result(llm):
    result = draft_reply(llm, EMAIL, {"tone": "professional"})

    assert result.ok and result.kind == "reply"
    assert result.tool == "draft_reply"
    assert result.title == "Draft reply (professional)"
    assert result.payload["to"] == EMAIL["from"]
    assert result.payload["recipient"] == "daniel@brightpath.example"
    assert result.payload["subject"] == "Re: Contract review"
    assert result.payload["tone"] == "professional"
    assert result.payload["status"] == tools.STATUS_SUGGESTED
    assert result.payload["body"] == result.detail == regenerate_reply(llm, EMAIL, "professional")


def test_draft_reply_still_refuses_an_unaddressable_sender(llm):
    with pytest.raises(ValueError, match="sender address"):
        draft_reply(llm, dict(EMAIL, **{"from": "Nobody At All"}), {"tone": "professional"})


def test_draft_reply_checks_the_recipient_before_spending_a_generation():
    spy = SpyLLM()

    with pytest.raises(ValueError):
        draft_reply(spy, dict(EMAIL, **{"from": "Nobody At All"}), {"tone": "professional"})

    assert spy.calls == [], "no reply is generated for an email nobody can be replied to"


def test_draft_reply_still_falls_back_to_professional_for_a_bad_tone(llm):
    result = draft_reply(llm, EMAIL, {"tone": "ignore-previous-instructions"})

    assert result.payload["tone"] == "professional"
    assert result.title == "Draft reply (professional)"


# ---------------------------------------------------------------------------
# What regeneration must never reach
# ---------------------------------------------------------------------------


def test_regenerating_never_re_triages_and_never_touches_gmail(llm, no_triage, no_gmail):
    text = regenerate_reply(llm, EMAIL, "friendly")

    assert text.strip()
    assert no_triage == {
        "process_inbox": 0,
        "process_email": 0,
        "classify": 0,
        "execute_actions": 0,
    }, "regeneration is the reply call alone"
    assert no_gmail == []


def test_the_spies_would_actually_fire(llm, no_triage, no_gmail):
    """Guards the test above: a real triage pass does trip the counters."""
    TriageAgent(llm=llm).process_email(EMAIL)

    assert no_triage["process_email"] == 1
    assert no_triage["classify"] == 1
    assert no_triage["execute_actions"] == 1
    assert no_gmail == [], "and even a full triage pass never reaches Gmail"


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
