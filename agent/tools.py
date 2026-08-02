"""Tool layer: the half of the agent that turns a decision into work.

Gemma 4 emits function calls; this module executes them. Two properties matter:

* ``draft_reply`` calls **back into Gemma** to write the reply body. That makes the
  run genuinely multi-step -- reason, choose a tool, then invoke the model again in
  service of that tool -- rather than a single classification pass.
* Nothing here raises into the caller. Every tool is wrapped, and a failed tool
  becomes a :class:`ToolResult` with ``ok=False`` so the remaining actions still run.

``create_task`` and ``create_calendar_event`` emit both a human-readable view and an
``api_payload`` already shaped for the Google Tasks / Google Calendar REST APIs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gmail_integration.client import extract_address, normalize_reply_subject

from .schema import TriageDecision

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

#: IANA timezone written into Google Calendar payloads.
CALENDAR_TIMEZONE = os.environ.get("GOOGLE_CALENDAR_TIMEZONE", "UTC")

#: Default meeting length when the email does not state one.
DEFAULT_EVENT_MINUTES = 30

VALID_TONES = ("professional", "friendly", "urgent", "brief", "firm")

# ---------------------------------------------------------------------------
# Action status vocabulary
# ---------------------------------------------------------------------------
#
# The lifecycle of every action this agent produces, in order. Used verbatim in UI
# status strings so a viewer is never left guessing whether something actually
# happened somewhere outside this app:
#
#   Suggested action  -> the model asked for it
#   Prepared payload  -> validated and shaped for the external API, not sent anywhere
#   Approved          -> a human clicked
#   Created (external)-> an external API call returned success. Only ever set after one.
#   Failed            -> it was attempted and did not succeed
#
# Nothing in this module can reach "Created (external)": it only prepares payloads.

STATUS_SUGGESTED = "Suggested action"
STATUS_PREPARED = "Prepared payload"
STATUS_APPROVED = "Approved"
STATUS_CREATED = "Created (external)"
STATUS_FAILED = "Failed"

ACTION_STATUSES = (
    STATUS_SUGGESTED,
    STATUS_PREPARED,
    STATUS_APPROVED,
    STATUS_CREATED,
    STATUS_FAILED,
)

#: Declarative catalogue, kept in sync with prompts/triage_prompt.txt. Used for
#: argument validation here and rendered in the README.
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "draft_reply": {
        "description": "Write a reply body for the user to review and send.",
        "args": {"tone": f"one of {', '.join(VALID_TONES)}"},
        "kind": "reply",
    },
    "create_task": {
        "description": "Capture a follow-up action item, ready for Google Tasks.",
        "args": {"title": "string", "due": "YYYY-MM-DD, a date phrase, or null"},
        "kind": "task",
    },
    "create_calendar_event": {
        "description": "Create a calendar entry, ready for Google Calendar.",
        "args": {"title": "string", "date": "YYYY-MM-DD or phrase", "time": "HH:MM or null"},
        "kind": "event",
    },
    "flag_urgent": {
        "description": "Raise a visible alarm on the email.",
        "args": {"reason": "string"},
        "kind": "flag",
    },
}


# ---------------------------------------------------------------------------
# Prompt loading (shared with triage.py)
# ---------------------------------------------------------------------------

_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Read ``prompts/<name>`` once and memoise it."""
    if name not in _PROMPT_CACHE:
        path = PROMPTS_DIR / name
        _PROMPT_CACHE[name] = path.read_text(encoding="utf-8")
    return _PROMPT_CACHE[name]


def render(template: str, **values: str) -> str:
    """Substitute ``{{KEY}}`` placeholders. Brace-safe, unlike ``str.format`` on JSON."""
    for key, value in values.items():
        template = template.replace("{{" + key.upper() + "}}", str(value))
    return template


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Outcome of one executed tool call."""

    tool: str
    ok: bool
    kind: str = "unknown"
    title: str = ""
    detail: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "payload": self.payload,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Date / time normalisation
# ---------------------------------------------------------------------------

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_RE = re.compile(r"(20\d{2})-(\d{1,2})-(\d{1,2})")
_MONTH_DAY_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})\b", re.I
)
_DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I
)
_IN_N_DAYS_RE = re.compile(r"\bin (\d{1,2}) days?\b", re.I)
_TIME_12H_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", re.I)
_TIME_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def normalize_date(value: Any, today: Optional[date] = None) -> tuple[Optional[str], str]:
    """Resolve a date phrase to ``(iso_or_None, display_text)``.

    Understands ISO dates, "today"/"tomorrow", weekday names, "next <weekday>",
    "in N days", "next week", and "Aug 5" / "5 August" forms. An unrecognised
    phrase is preserved verbatim as the display text with ``None`` as the ISO
    value, so nothing is silently invented.
    """
    today = today or date.today()
    if value is None:
        return None, "no due date"
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "-"}:
        return None, "no due date"

    low = text.lower()

    iso = _ISO_RE.search(text)
    if iso:
        try:
            resolved = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            return resolved.isoformat(), resolved.strftime("%a %d %b %Y")
        except ValueError:
            pass

    if "today" in low:
        return today.isoformat(), f"today ({today.strftime('%a %d %b')})"
    if "tomorrow" in low:
        resolved = today + timedelta(days=1)
        return resolved.isoformat(), f"tomorrow ({resolved.strftime('%a %d %b')})"

    in_days = _IN_N_DAYS_RE.search(low)
    if in_days:
        resolved = today + timedelta(days=int(in_days.group(1)))
        return resolved.isoformat(), resolved.strftime("%a %d %b %Y")

    # Earliest-mentioned weekday wins, so "Thursday or Friday" resolves to Thursday
    # regardless of dict ordering.
    weekday_hits = [(low.find(name), name, index) for name, index in _WEEKDAYS.items() if name in low]
    if weekday_hits:
        _, _, index = min(weekday_hits)
        delta = (index - today.weekday()) % 7
        if "next" in low:
            delta = delta + 7 if delta else 7
        resolved = today + timedelta(days=delta)
        return resolved.isoformat(), resolved.strftime("%a %d %b %Y")

    if "next week" in low:
        resolved = today + timedelta(days=7 - today.weekday())
        return resolved.isoformat(), f"next week (from {resolved.strftime('%a %d %b')})"
    if "end of week" in low or "eow" in low:
        resolved = today + timedelta(days=(4 - today.weekday()) % 7)
        return resolved.isoformat(), f"end of week ({resolved.strftime('%a %d %b')})"

    month_day = _MONTH_DAY_RE.search(text) or None
    day_month = None if month_day else _DAY_MONTH_RE.search(text)
    if month_day or day_month:
        if month_day:
            month, day = _MONTHS[month_day.group(1).lower()], int(month_day.group(2))
        else:
            day, month = int(day_month.group(1)), _MONTHS[day_month.group(2).lower()]
        try:
            resolved = date(today.year, month, day)
            if resolved < today:  # a bare "Mar 3" seen in December means next year
                resolved = date(today.year + 1, month, day)
            return resolved.isoformat(), resolved.strftime("%a %d %b %Y")
        except ValueError:
            pass

    return None, text


def normalize_time(value: Any) -> Optional[str]:
    """Resolve a time phrase to 24-hour ``HH:MM``, or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "-"}:
        return None

    twelve = _TIME_12H_RE.search(text)
    if twelve:
        hour, minute = int(twelve.group(1)), int(twelve.group(2) or 0)
        meridiem = twelve.group(3).replace(".", "").lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    twenty_four = _TIME_24H_RE.search(text)
    if twenty_four:
        return f"{int(twenty_four.group(1)):02d}:{twenty_four.group(2)}"
    return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def draft_reply(
    llm: Any,
    email: Dict[str, Any],
    args: Dict[str, Any],
    decision: Optional[TriageDecision] = None,
) -> ToolResult:
    """Second Gemma call: write the reply body in the tone the model chose."""
    tone = str(args.get("tone") or "professional").strip().lower()
    if tone not in VALID_TONES:
        tone = "professional"

    # No recipient, no draft. Checked before spending a generation on a reply that
    # could never be addressed to anyone.
    recipient = extract_address(email.get("from"))
    if not recipient:
        raise ValueError("the sender address is missing or malformed, so no reply was drafted")

    system = render(load_prompt("reply_prompt.txt"), tone=tone)
    user = build_email_block(email, extra={"TRIAGE SUMMARY": (decision.summary if decision else "")})

    body = (llm.generate(system, user) or "").strip()
    body = _strip_reply_artifacts(body)

    if not body:
        raise ValueError("the model returned an empty reply body")

    return ToolResult(
        tool="draft_reply",
        ok=True,
        kind="reply",
        title=f"Draft reply ({tone})",
        detail=body,
        payload={
            "to": email.get("from", ""),
            "recipient": recipient,
            "subject": normalize_reply_subject(email.get("subject")),
            "body": body,
            "tone": tone,
            # A drafted reply is a suggestion until a human presses the button.
            "status": STATUS_SUGGESTED,
        },
    )


_REPLY_ARTIFACT_RE = re.compile(r"^\s*(subject|to|from|cc|bcc)\s*:.*$", re.I | re.M)


def _strip_reply_artifacts(body: str) -> str:
    """Remove header lines and markdown fences a model sometimes prepends."""
    body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", body.strip())
    body = _REPLY_ARTIFACT_RE.sub("", body)
    return body.strip()


def create_task(email: Dict[str, Any], args: Dict[str, Any], today: Optional[date] = None) -> ToolResult:
    """Validate a follow-up task and shape it for the Google Tasks API."""
    title = str(args.get("title") or "").strip() or f"Follow up: {email.get('subject', '(no subject)')}"
    title = title[:200]

    due_iso, due_display = normalize_date(args.get("due"), today)

    api_payload: Dict[str, Any] = {
        "title": title,
        "notes": f"From: {email.get('from', 'unknown')}\nSubject: {email.get('subject', '')}",
    }
    if due_iso:
        # Google Tasks expects an RFC 3339 timestamp for `due`.
        api_payload["due"] = f"{due_iso}T00:00:00.000Z"

    return ToolResult(
        tool="create_task",
        ok=True,
        kind="task",
        title=title,
        detail=f"Due: {due_display}",
        payload={
            "title": title,
            "due_date": due_iso,
            "due_display": due_display,
            "source_email": email.get("id", ""),
            "api_payload": api_payload,
            "api_target": "Google Tasks — tasks.tasks.insert",
            # Shaped for the API. Nothing has been sent to Google.
            "status": STATUS_PREPARED,
        },
    )


def create_calendar_event(
    email: Dict[str, Any], args: Dict[str, Any], today: Optional[date] = None
) -> ToolResult:
    """Validate a calendar entry and shape it for the Google Calendar API."""
    title = str(args.get("title") or "").strip() or str(email.get("subject") or "Meeting")
    title = title[:200]

    date_iso, date_display = normalize_date(args.get("date"), today)
    if not date_iso:
        raise ValueError(f"could not resolve a concrete date from {args.get('date')!r}")

    time_hhmm = normalize_time(args.get("time"))
    all_day = time_hhmm is None

    if all_day:
        api_payload = {
            "summary": title,
            "start": {"date": date_iso},
            "end": {"date": (date.fromisoformat(date_iso) + timedelta(days=1)).isoformat()},
        }
        when = f"{date_display} (all day)"
        start_iso = date_iso
    else:
        start = datetime.fromisoformat(f"{date_iso}T{time_hhmm}:00")
        end = start + timedelta(minutes=DEFAULT_EVENT_MINUTES)
        api_payload = {
            "summary": title,
            "start": {"dateTime": start.isoformat(), "timeZone": CALENDAR_TIMEZONE},
            "end": {"dateTime": end.isoformat(), "timeZone": CALENDAR_TIMEZONE},
        }
        when = f"{date_display} at {time_hhmm} ({CALENDAR_TIMEZONE})"
        start_iso = start.isoformat()

    api_payload["description"] = f"Created by Gemma-Triage from an email by {email.get('from', 'unknown')}."

    return ToolResult(
        tool="create_calendar_event",
        ok=True,
        kind="event",
        title=title,
        detail=when,
        payload={
            "title": title,
            "date": date_iso,
            "time": time_hhmm,
            "all_day": all_day,
            "start_iso": start_iso,
            "duration_minutes": None if all_day else DEFAULT_EVENT_MINUTES,
            "source_email": email.get("id", ""),
            "api_payload": api_payload,
            "api_target": "Google Calendar — events.insert",
            # Shaped for the API. Nothing has been sent to Google.
            "status": STATUS_PREPARED,
        },
    )


def flag_urgent(email: Dict[str, Any], args: Dict[str, Any]) -> ToolResult:
    """Raise a visible alarm on an email."""
    reason = str(args.get("reason") or "Flagged as urgent by the triage agent.").strip()[:300]
    return ToolResult(
        tool="flag_urgent",
        ok=True,
        kind="flag",
        title="Flagged urgent",
        detail=reason,
        payload={"flagged": True, "reason": reason, "source_email": email.get("id", "")},
    )


# ---------------------------------------------------------------------------
# Email rendering + executor
# ---------------------------------------------------------------------------

MAX_BODY_CHARS = 6000


def build_email_block(email: Dict[str, Any], extra: Optional[Dict[str, str]] = None) -> str:
    """Render an email into the structured block every prompt (and the heuristic) reads."""
    body = str(email.get("body") or "").strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n[... truncated ...]"

    lines = [
        f"TODAY: {date.today().isoformat()}",
        f"FROM: {email.get('from', 'unknown')}",
        f"TO: {email.get('to', 'me')}",
        f"SUBJECT: {email.get('subject', '(no subject)')}",
        f"DATE: {email.get('date', 'unknown')}",
    ]
    for key, value in (extra or {}).items():
        if value:
            lines.append(f"{key}: {value}")
    lines.append("BODY:")
    lines.append(body or "(empty)")
    return "\n".join(lines)


def execute_actions(
    llm: Any,
    email: Dict[str, Any],
    decision: TriageDecision,
    today: Optional[date] = None,
) -> List[ToolResult]:
    """Run every tool call Gemma requested. One bad call never stops the rest."""
    results: List[ToolResult] = []

    dispatch: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
        "draft_reply": lambda a: draft_reply(llm, email, a, decision),
        "create_task": lambda a: create_task(email, a, today),
        "create_calendar_event": lambda a: create_calendar_event(email, a, today),
        "flag_urgent": lambda a: flag_urgent(email, a),
    }

    for call in decision.actions:
        handler = dispatch.get(call.tool)
        if handler is None:  # unreachable after schema validation, but cheap to guard
            results.append(
                ToolResult(tool=call.tool, ok=False, error=f"unknown tool {call.tool!r}")
            )
            continue
        try:
            results.append(handler(call.args))
        except Exception as exc:  # noqa: BLE001 - a failed tool must not abort the run
            spec = TOOL_SPECS.get(call.tool, {})
            results.append(
                ToolResult(
                    tool=call.tool,
                    ok=False,
                    kind=str(spec.get("kind", "unknown")),
                    title=f"{call.tool} failed",
                    error=f"{type(exc).__name__}: {exc}",
                    payload={"status": STATUS_FAILED},
                )
            )

    return results
