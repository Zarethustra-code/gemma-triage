"""End-to-end tests for the triage contract, run against the heuristic backend.

The heuristic backend is deliberately the system under test: it needs no weights,
no token and no network, so CI (and a judge with a fresh clone) exercises the exact
same schema validation, tool execution and sorting code that Gemma 4 drives in production.

Run:  python -m pytest tests/ -v
  or: python tests/test_triage.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import GemmaLLM, TriageAgent  # noqa: E402
from agent.llm import _find_date, _find_time  # noqa: E402
from agent.schema import CATEGORIES, KNOWN_TOOLS, extract_json, parse_triage  # noqa: E402
from agent.tools import (  # noqa: E402
    VALID_TONES,
    build_email_block,
    create_calendar_event,
    create_task,
    execute_actions,
    normalize_date,
    normalize_time,
)
from agent.triage import inbox_stats, sort_results  # noqa: E402

SAMPLE_INBOX = REPO_ROOT / "data" / "sample_inbox.json"
REFERENCE_TODAY = date(2026, 8, 3)  # a Monday, so weekday maths is deterministic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def emails():
    data = json.loads(SAMPLE_INBOX.read_text(encoding="utf-8"))
    return data["emails"]


@pytest.fixture(scope="module")
def agent():
    return TriageAgent(llm=GemmaLLM(backend="heuristic"))


@pytest.fixture(scope="module")
def processed(agent, emails):
    return agent.process_inbox(emails, today=REFERENCE_TODAY)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


def test_sample_inbox_is_usable(emails):
    assert len(emails) >= 8, "the demo inbox must show breadth"
    for email in emails:
        for key in ("id", "from", "to", "subject", "date", "body"):
            assert key in email and str(email[key]).strip(), f"{email.get('id')} missing {key}"
    assert len({e["id"] for e in emails}) == len(emails), "email ids must be unique"


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def test_heuristic_backend_selected_explicitly():
    llm = GemmaLLM(backend="heuristic")
    assert llm.active_backend == "heuristic"
    assert llm.is_local, "the heuristic engine must count as on-device"
    assert "heuristic" in llm.backend_label.lower()


def test_unknown_backend_name_falls_back_to_auto():
    llm = GemmaLLM(backend="not-a-backend")
    assert llm.requested_backend == "auto"
    assert llm.active_backend in ("transformers", "hf_api", "heuristic")


# ---------------------------------------------------------------------------
# JSON / schema contract
# ---------------------------------------------------------------------------


def test_extract_json_survives_fences_prose_and_trailing_commas():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 1} Hope that helps.') == {"a": 1}
    assert extract_json('{"a": 1, "b": [2, 3,],}') == {"a": 1, "b": [2, 3]}
    assert extract_json('[{"a": 1}]') == {"a": 1}
    assert extract_json('{"note": "a } brace inside a string", "a": 1}')["a"] == 1
    assert extract_json("not json at all") is None


def test_parse_triage_never_raises_on_garbage():
    decision = parse_triage("the model rambled and produced no JSON")
    assert decision.parse_failed is True
    assert decision.category in CATEGORIES
    assert 1 <= decision.priority <= 5
    assert decision.actions == []


def test_parse_triage_normalises_aliases_and_clamps_priority():
    decision = parse_triage(
        json.dumps(
            {
                "category": "action needed",
                "priority": 99,
                "summary": "s",
                "reasoning": "r",
                "suggested_reply": "hi",
                "actions": [
                    {"tool": "reply", "args": {"tone": "friendly"}},
                    {"tool": "not_a_real_tool", "args": {}},
                    {"tool": "create_event", "args": {"title": "t", "date": "2026-08-05"}},
                ],
            }
        )
    )
    assert decision.category == "ACTION_NEEDED"
    assert decision.priority == 5
    tools = [a.tool for a in decision.actions]
    assert tools == ["draft_reply", "create_calendar_event"], "unknown tools must be dropped"


def test_spam_never_gets_a_reply_drafted():
    decision = parse_triage(
        json.dumps(
            {
                "category": "SPAM",
                "priority": 5,
                "summary": "s",
                "reasoning": "r",
                "suggested_reply": "Dear scammer,",
                "actions": [{"tool": "draft_reply", "args": {"tone": "friendly"}}],
            }
        )
    )
    assert decision.suggested_reply is None
    assert decision.actions == []
    assert decision.priority <= 2


def test_urgent_priority_is_floored():
    decision = parse_triage(
        json.dumps({"category": "URGENT", "priority": 1, "summary": "s", "reasoning": "r"})
    )
    assert decision.priority >= 4


# ---------------------------------------------------------------------------
# Date / time normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("2026-08-11", "2026-08-11"),
        ("today", "2026-08-03"),
        ("tomorrow", "2026-08-04"),
        ("Friday", "2026-08-07"),
        ("next Monday", "2026-08-10"),
        ("in 3 days", "2026-08-06"),
        ("Aug 20", "2026-08-20"),
        ("20 August", "2026-08-20"),
    ],
)
def test_normalize_date(phrase, expected):
    iso, display = normalize_date(phrase, REFERENCE_TODAY)
    assert iso == expected
    assert display


def test_normalize_date_picks_the_earliest_weekday_mentioned():
    """"Thursday or Friday" means Thursday, whatever order the lookup table is in."""
    iso, _ = normalize_date("Thursday or Friday", REFERENCE_TODAY)
    assert iso == "2026-08-06"
    iso, _ = normalize_date("Friday or Thursday", REFERENCE_TODAY)
    assert iso == "2026-08-07"


def test_find_date_prefers_the_first_date_mentioned():
    """The proposed date leads; a date mentioned in passing later must not win."""
    text = "half the team is out on wednesday, could we move it to thursday at 2pm?"
    assert _find_date(text) == "wednesday"
    # Callers pass "subject\nbody", so a headline date outranks the aside.
    assert _find_date("move the sync to thursday?\n" + text) == "thursday"
    assert _find_time("the window is 07:00 to 11:00, or 2pm") == "07:00"
    assert _find_date("no dates here at all") is None


def test_normalize_date_preserves_unknown_phrases():
    iso, display = normalize_date("whenever you get a chance", REFERENCE_TODAY)
    assert iso is None
    assert display == "whenever you get a chance", "never silently invent a date"
    assert normalize_date(None, REFERENCE_TODAY) == (None, "no due date")


@pytest.mark.parametrize(
    "phrase,expected",
    [("2pm", "14:00"), ("10:30", "10:30"), ("9 a.m.", "09:00"), ("12am", "00:00"), ("noon", None)],
)
def test_normalize_time(phrase, expected):
    assert normalize_time(phrase) == expected


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_create_task_emits_a_google_tasks_payload():
    result = create_task(
        {"id": "x", "from": "a@b.example", "subject": "Contract"},
        {"title": "Sign the contract", "due": "Friday"},
        today=REFERENCE_TODAY,
    )
    assert result.ok and result.kind == "task"
    assert result.payload["due_date"] == "2026-08-07"
    assert result.payload["api_payload"]["title"] == "Sign the contract"
    assert result.payload["api_payload"]["due"].startswith("2026-08-07T")


def test_create_calendar_event_emits_a_google_calendar_payload():
    result = create_calendar_event(
        {"id": "x", "from": "a@b.example", "subject": "Interview"},
        {"title": "Technical interview", "date": "2026-08-11", "time": "10:30"},
        today=REFERENCE_TODAY,
    )
    assert result.ok and result.kind == "event"
    start = result.payload["api_payload"]["start"]
    assert start["dateTime"].startswith("2026-08-11T10:30")
    assert result.payload["duration_minutes"] == 30


def test_create_calendar_event_rejects_an_unresolvable_date():
    with pytest.raises(ValueError):
        create_calendar_event({"id": "x"}, {"title": "t", "date": "sometime soonish"})


def test_a_failing_tool_does_not_abort_the_remaining_actions(agent):
    """A bad calendar date must be reported, not raised, and later tools still run."""
    email = {"id": "e", "from": "Sam <sam@b.example>", "subject": "Sync", "body": "Let's meet."}
    decision = parse_triage(
        json.dumps(
            {
                "category": "ACTION_NEEDED",
                "priority": 3,
                "summary": "s",
                "reasoning": "r",
                "actions": [
                    {"tool": "create_calendar_event", "args": {"title": "t", "date": "gibberish"}},
                    {"tool": "create_task", "args": {"title": "Still runs", "due": None}},
                ],
            }
        )
    )
    results = execute_actions(agent.llm, email, decision, today=REFERENCE_TODAY)
    assert len(results) == 2
    assert results[0].ok is False and results[0].error
    assert results[1].ok is True and results[1].title == "Still runs"


def test_draft_reply_calls_the_model_again_and_produces_a_sendable_draft(agent):
    email = {
        "id": "e",
        "from": "Daniel Okafor <daniel@b.example>",
        "subject": "Contract review",
        "body": "Could you review it and send the signed copy back by Friday?",
    }
    decision = parse_triage(
        json.dumps(
            {
                "category": "ACTION_NEEDED",
                "priority": 3,
                "summary": "s",
                "reasoning": "r",
                "actions": [{"tool": "draft_reply", "args": {"tone": "professional"}}],
            }
        )
    )
    (result,) = execute_actions(agent.llm, email, decision, today=REFERENCE_TODAY)
    assert result.ok and result.kind == "reply"
    assert result.payload["subject"] == "Re: Contract review"
    assert result.payload["to"] == email["from"]
    assert result.payload["tone"] in VALID_TONES
    assert "Daniel" in result.detail, "the reply should address the sender by name"
    assert len(result.detail.split()) > 5


def test_build_email_block_is_parseable_by_the_heuristic_backend():
    block = build_email_block({"from": "a@b.example", "subject": "Hi", "body": "Body text"})
    for marker in ("TODAY:", "FROM:", "SUBJECT:", "BODY:"):
        assert marker in block


# ---------------------------------------------------------------------------
# Full inbox
# ---------------------------------------------------------------------------


def test_every_email_yields_a_contract_valid_decision(processed, emails):
    assert len(processed) == len(emails)
    for item in processed:
        decision = item.decision
        assert decision.category in CATEGORIES
        assert isinstance(decision.priority, int) and 1 <= decision.priority <= 5
        assert decision.summary.strip(), f"{item.email['id']} produced an empty summary"
        assert decision.reasoning.strip()
        assert decision.parse_failed is False, "the heuristic backend must emit valid JSON"
        for action in decision.actions:
            assert action.tool in KNOWN_TOOLS
            assert isinstance(action.args, dict)


def test_decision_round_trips_through_json(processed):
    for item in processed:
        payload = json.loads(json.dumps(item.to_dict(include_raw=True)))
        assert payload["decision"]["category"] in CATEGORIES
        assert isinstance(payload["tool_results"], list)


def test_inbox_is_sorted_urgent_first(processed):
    keys = [item.sort_key for item in processed]
    assert keys == sorted(keys), "URGENT and high-priority email must float to the top"
    assert processed[0].decision.category in ("URGENT", "ACTION_NEEDED")
    assert processed[-1].decision.category in ("FYI", "SPAM")


def test_sort_results_is_idempotent(processed):
    assert [p.email["id"] for p in sort_results(list(processed))] == [
        p.email["id"] for p in processed
    ]


def test_the_obvious_incident_email_is_urgent_and_flagged(processed):
    incident = next(p for p in processed if p.email["id"] == "msg-001")
    assert incident.decision.category == "URGENT"
    assert incident.decision.priority >= 4
    assert incident.flags, "an urgent incident should raise a flag_urgent call"
    assert incident.reply_text, "an urgent incident should get a draft reply"


def test_the_obvious_scam_emails_are_spam_and_get_no_reply(processed):
    for email_id in ("msg-009", "msg-010"):
        item = next(p for p in processed if p.email["id"] == email_id)
        assert item.decision.category == "SPAM", f"{email_id} should be filed as spam"
        assert item.reply_text is None, "never draft a reply to spam"
        assert item.results == []


def test_the_workflow_actually_produces_work(processed):
    stats = inbox_stats(processed)
    assert stats["total"] == len(processed)
    assert stats["URGENT"] >= 1
    assert stats["ACTION_NEEDED"] >= 2
    assert stats["SPAM"] >= 2
    assert stats["replies"] >= 3, "the agent should draft several replies"
    assert stats["tasks"] >= 3, "the agent should extract several follow-up tasks"
    assert stats["events"] >= 1, "the agent should extract at least one calendar event"
    assert stats["errors"] == 0, "the demo inbox must run clean"


def test_the_scheduling_email_books_the_day_it_actually_proposes(processed):
    """msg-004 mentions Wednesday in passing but proposes Thursday 2pm."""
    item = next(p for p in processed if p.email["id"] == "msg-004")
    assert item.decision.category == "ACTION_NEEDED", "a direct scheduling question needs action"
    (event,) = item.events
    assert event.payload["date"] == "2026-08-06", "Thursday, not the Wednesday mentioned in passing"
    assert date.fromisoformat(event.payload["date"]).strftime("%A") == "Thursday"
    assert event.payload["time"] == "14:00"


def test_calendar_events_are_not_invented_from_passing_mentions(processed):
    """'Thanks for the call last week' must not book anything."""
    contract = next(p for p in processed if p.email["id"] == "msg-003")
    assert contract.events == [], "a mention of a past call is not a meeting request"
    assert contract.tasks, "but it should still produce a follow-up task"


def test_extracted_events_carry_a_resolvable_iso_date(processed):
    events = [event for item in processed for event in item.events]
    assert events, "expected at least one calendar event across the demo inbox"
    for event in events:
        assert date.fromisoformat(event.payload["date"])
        assert event.payload["api_payload"]["summary"]


def test_streaming_and_batch_paths_agree(agent, emails):
    streamed = sort_results(list(agent.stream_inbox(emails[:5], today=REFERENCE_TODAY)))
    batched = agent.process_inbox(emails[:5], today=REFERENCE_TODAY)
    assert [p.email["id"] for p in streamed] == [p.email["id"] for p in batched]
    assert [p.decision.category for p in streamed] == [p.decision.category for p in batched]


def test_empty_and_malformed_emails_do_not_crash(agent):
    for email in ({}, {"subject": "", "body": ""}, {"from": None, "subject": None, "body": None}):
        item = agent.process_email(email, today=REFERENCE_TODAY)
        assert item.decision.category in CATEGORIES
        assert item.decision.summary


def test_progress_callback_is_invoked_and_a_broken_one_is_survivable(agent, emails):
    seen: list[int] = []
    agent.process_inbox(emails[:3], on_progress=lambda i, t, e: seen.append(i), today=REFERENCE_TODAY)
    assert seen == [0, 1, 2]

    def explode(i, t, e):
        raise RuntimeError("callback blew up")

    result = agent.process_inbox(emails[:3], on_progress=explode, today=REFERENCE_TODAY)
    assert len(result) == 3, "a broken progress callback must not stop triage"


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
