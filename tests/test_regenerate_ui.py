"""The per-row 🔁 Regenerate reply control: handler, row state, and wiring.

Pressing 🔁 must re-run only the reply-writing Gemma call for the one email in that
row. The isolation is asserted twice over, because good manners in the handler are not
a guarantee:

  * at the handler -- a row's email arrives in that row's own ``gr.State``, so there is
    nothing to look up in a shared list and no way to address another row,
  * at the component graph -- each button's inputs and outputs are disjoint from every
    other row's, and neither the inbox state nor the displayed-order state appears on
    any of them, so a click cannot re-triage or re-sort anything.

Skipped when Gradio is not installed. Gmail is blocked outright throughout.
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
from agent import GemmaLLM, TriageAgent  # noqa: E402
from agent.llm import _TONE_LINES  # noqa: E402
from agent.tools import VALID_TONES  # noqa: E402

EMAIL = {
    "id": "msg-a",
    "thread_id": "t-a",
    "from": "Daniel Okafor <daniel@brightpath.example>",
    "subject": "Contract review",
    "body": "Could you review the contract and send the signed copy back by Friday?",
}

OTHER_EMAIL = {
    "id": "msg-b",
    "thread_id": "t-b",
    "from": "Priya Raman <priya@northwind.example>",
    "subject": "Server outage",
    "body": "Production is down and customers are affected. Please respond immediately.",
}

#: Index of each per-row component inside the flat row tuple, mirroring ``ROW_WIDGETS``.
TONE, REGEN_BTN, REGEN_STATUS, STATE = 5, 6, 7, 10


@pytest.fixture
def heuristic_engine(monkeypatch):
    """Pin the app's engine to the on-device heuristic backend for handler tests."""
    agent = TriageAgent(llm=GemmaLLM(backend="heuristic"))
    monkeypatch.setattr(app.ENGINE, "agent", agent)
    monkeypatch.setattr(app.ENGINE, "error", None)
    return agent


def state_for(email, summary="Summary."):
    return {"email": email, "decision_summary": summary}


def processed_rows(n=6):
    agent = TriageAgent(llm=GemmaLLM(backend="heuristic"))
    emails = json.loads((REPO_ROOT / "data" / "sample_inbox.json").read_text(encoding="utf-8"))
    return agent.process_inbox(emails["emails"][:n])


def row_of(outputs, index):
    """Slice one row out of the flat ``build_outputs`` tuple (4 leading elements)."""
    start = 4 + index * app.ROW_WIDGETS
    return outputs[start : start + app.ROW_WIDGETS]


# ---------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------


def test_the_handler_returns_regenerated_text_and_a_status_line(heuristic_engine):
    text, status = app.regenerate_handler(state_for(EMAIL), "friendly", "the old reply")

    assert text != "the old reply"
    assert text.strip() and "Daniel" in text
    assert "friendly" in status and "heuristic" in status
    assert "sent" in status.lower(), "the status keeps saying nothing was drafted or sent"


@pytest.mark.parametrize("tone", VALID_TONES)
def test_the_handler_honours_every_tone_in_the_dropdown(heuristic_engine, tone):
    text, status = app.regenerate_handler(state_for(EMAIL), tone, "old")

    assert _TONE_LINES[tone] in text
    assert f"in a {tone} tone" in status


def test_the_handler_falls_back_to_professional_for_an_unknown_tone(heuristic_engine):
    text, status = app.regenerate_handler(state_for(EMAIL), "shouty", "old")

    assert _TONE_LINES["professional"] in text
    assert "professional" in status and "shouty" not in status


@pytest.mark.parametrize("state", [None, {}, [], "", 0, {"email": None}, {"decision_summary": "s"}])
def test_a_row_with_no_email_keeps_its_text_and_says_so(heuristic_engine, state):
    text, status = app.regenerate_handler(state, "friendly", "text the user typed")

    assert text == "text the user typed", "a blank row must never lose the user's text"
    assert status == "Nothing to regenerate yet."


def test_a_broken_engine_keeps_the_text_and_reports_itself(monkeypatch):
    monkeypatch.setattr(app.ENGINE, "agent", None)
    monkeypatch.setattr(app.ENGINE, "error", "RuntimeError: no weights")

    text, status = app.regenerate_handler(state_for(EMAIL), "brief", "keep me")

    assert text == "keep me"
    assert "Engine unavailable" in status and "no weights" in status


def test_a_failed_generation_keeps_the_users_text(heuristic_engine, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("/home/someone/token.json says hf_SECRET")

    monkeypatch.setattr(app.tools, "regenerate_reply", explode)

    text, status = app.regenerate_handler(state_for(EMAIL), "brief", "keep me")

    assert text == "keep me", "a failed rewrite must not wipe the textbox"
    assert "❌" in status
    assert "hf_SECRET" not in status and "token.json" not in status


def test_the_handler_never_re_triages_and_never_touches_gmail(
    heuristic_engine, no_triage, no_gmail
):
    text, _ = app.regenerate_handler(state_for(EMAIL), "urgent", "old")

    assert text.strip()
    assert set(no_triage.values()) == {0}, f"regeneration re-entered triage: {no_triage}"
    assert no_gmail == []


def test_the_handler_is_deterministic_for_the_same_row_and_tone(heuristic_engine):
    first, _ = app.regenerate_handler(state_for(EMAIL), "brief", "old")
    second, _ = app.regenerate_handler(state_for(EMAIL), "brief", first)
    assert first == second


def test_regenerating_one_row_leaves_the_other_row_alone(heuristic_engine):
    """Row isolation at the handler: row B is neither an input nor an output."""
    row_a, row_b = state_for(EMAIL), state_for(OTHER_EMAIL)
    reply_a, reply_b = "Row A's original reply.", "Row B's original reply."

    new_a, _ = app.regenerate_handler(row_a, "friendly", reply_a)

    assert new_a != reply_a
    assert "Daniel" in new_a and "Priya" not in new_a
    assert reply_b == "Row B's original reply.", "row B's text is untouched"

    new_b, _ = app.regenerate_handler(row_b, "urgent", reply_b)

    assert "Priya" in new_b and "Daniel" not in new_b
    assert app.regenerate_handler(row_a, "friendly", reply_a)[0] == new_a, (
        "row B's regeneration did not disturb row A"
    )


# ---------------------------------------------------------------------------
# Row state and rendering
# ---------------------------------------------------------------------------


def test_a_rendered_row_carries_its_own_email_and_summary():
    processed = processed_rows(3)
    outputs = app.build_outputs(processed, len(processed), "status")

    for i, item in enumerate(processed):
        state = row_of(outputs, i)[STATE]
        assert state["email"] is item.email
        assert state["decision_summary"] == item.decision.summary

    for i in range(len(processed), app.MAX_ROWS):
        assert row_of(outputs, i)[STATE] is None, "a blank row owns no email"


def test_no_two_rows_share_an_email():
    processed = processed_rows(4)
    outputs = app.build_outputs(processed, len(processed), "status")

    ids = [row_of(outputs, i)[STATE]["email"]["id"] for i in range(len(processed))]
    assert len(set(ids)) == len(ids)


def test_a_rendered_row_can_be_regenerated_straight_from_its_state(heuristic_engine):
    """What the row renders is exactly what the button hands to the handler."""
    processed = [p for p in processed_rows(8) if p.reply_text]
    assert processed, "expected the demo inbox to draft at least one reply"

    outputs = app.build_outputs(processed, len(processed), "status")
    state = row_of(outputs, 0)[STATE]

    text, status = app.regenerate_handler(state, "brief", processed[0].reply_text)

    assert text.strip() and "brief" in status


def test_the_tone_dropdown_offers_the_shared_vocabulary_and_a_valid_default():
    assert app.DEFAULT_TONE in VALID_TONES
    assert app.blank_row()[TONE]["value"] == app.DEFAULT_TONE


def test_the_tone_dropdown_starts_on_the_tone_gemma_chose():
    processed = [p for p in processed_rows(8) if p.reply_payload]
    assert processed, "expected the demo inbox to produce a reply payload"

    row = app.render_row(processed[0], 1)

    assert row[TONE]["value"] == processed[0].reply_payload["tone"]
    assert row[TONE]["value"] in VALID_TONES


def test_the_regenerate_controls_appear_only_where_there_is_a_reply():
    processed = processed_rows(8)
    with_reply = [p for p in processed if p.reply_text]
    without_reply = [p for p in processed if not p.reply_text]
    assert with_reply and without_reply, "expected the demo inbox to contain both"

    shown = app.render_row(with_reply[0], 1)
    assert shown[TONE]["visible"] is True
    assert shown[REGEN_BTN]["visible"] is True
    assert shown[REGEN_STATUS]["value"] == "", "a fresh row starts with no stale status"

    hidden = app.render_row(without_reply[0], 2)
    assert hidden[TONE]["visible"] is False
    assert hidden[REGEN_BTN]["visible"] is False

    blank = app.blank_row()
    assert blank[TONE]["visible"] is False
    assert blank[REGEN_BTN]["visible"] is False
    assert blank[STATE] is None


def test_the_reply_textbox_stays_editable():
    """Regeneration replaces the value; the user can still type over it."""
    processed = [p for p in processed_rows(8) if p.reply_text]
    reply_update = app.render_row(processed[0], 1)[4]

    assert reply_update["visible"] is True
    assert reply_update.get("interactive") is not False


def test_running_triage_still_emits_one_value_per_component():
    """The regenerate controls joined the streamed output tuple; it stays fixed-width."""
    for outputs in app.run_triage([EMAIL, OTHER_EMAIL]):
        assert len(outputs) == 4 + app.MAX_ROWS * app.ROW_WIDGETS


# ---------------------------------------------------------------------------
# The component graph
# ---------------------------------------------------------------------------


def dependencies():
    fns = app.demo.fns
    return list(fns.values()) if hasattr(fns, "values") else list(fns)


def named(name):
    return [d for d in dependencies() if getattr(d.fn, "__name__", "") == name]


def test_every_row_has_its_own_regenerate_binding():
    bindings = named("regenerate_handler")

    assert len(bindings) == app.MAX_ROWS
    for binding in bindings:
        assert binding.fn is app.regenerate_handler, "one handler, no wrappers"
        assert not getattr(binding, "js", None), "and no client-side side effect either"


def test_a_regenerate_binding_reads_and_writes_only_its_own_row():
    """The guardrail lives in the wiring, not in the handler's good manners."""
    (triage,) = named("run_triage")
    triage_outputs = {id(c) for c in triage.outputs}

    footprints: list = []
    for binding in named("regenerate_handler"):
        assert len(binding.inputs) == 3, "row state, tone, reply textbox"
        assert len(binding.outputs) == 2, "reply textbox and row status — nothing else"

        # The textbox read is the textbox written, so the user's edits are the input.
        assert binding.inputs[2] is binding.outputs[0]

        # Two of the row's own components, never the whole triage tuple.
        assert {id(c) for c in binding.outputs} < triage_outputs

        footprints.append({id(c) for c in binding.inputs} | {id(c) for c in binding.outputs})

    for i, row in enumerate(footprints):
        for j, other in enumerate(footprints):
            if i != j:
                assert row.isdisjoint(other), f"rows {i} and {j} share a component"


def test_no_regenerate_binding_can_reach_the_inbox_or_the_displayed_order():
    """Neither state that drives a whole run may appear on a per-row rewrite."""
    (triage,) = named("run_triage")
    inbox_state = {id(c) for c in triage.inputs}
    display_state = {id(triage.outputs[1])}

    for binding in named("regenerate_handler"):
        touched = {id(c) for c in binding.inputs} | {id(c) for c in binding.outputs}
        assert touched.isdisjoint(inbox_state), "regeneration must not read the inbox"
        assert touched.isdisjoint(display_state), "regeneration must not touch display order"


def test_the_regenerate_button_is_not_wired_to_anything_else():
    """One click, one handler: no chained draft, no second effect.

    Gradio records a trigger as ``(component_id, event_name)``, so this compares the
    ids the framework itself stored against the buttons that were built.
    """
    triggers = [t for b in named("regenerate_handler") for t in b.targets]
    regenerate_buttons = {component_id for component_id, _event in triggers}

    assert len(triggers) == app.MAX_ROWS
    assert len(regenerate_buttons) == app.MAX_ROWS, "each row triggers on its own button"
    assert {event for _id, event in triggers} == {"click"}

    for binding in dependencies():
        if getattr(binding.fn, "__name__", "") == "regenerate_handler":
            continue
        for component_id, event in binding.targets:
            assert component_id not in regenerate_buttons, (
                f"a regenerate button also triggers "
                f"{getattr(binding.fn, '__name__', binding.fn)} on {event}"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
