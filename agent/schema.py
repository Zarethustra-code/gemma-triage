"""Structured-output contract for the triage step.

Gemma 4 is asked to emit exactly one JSON object per email. Real language models
occasionally wrap that object in prose or a markdown fence, emit a trailing comma,
or use a near-miss category label. Everything in this module exists to turn that
messy-but-recoverable reality into a validated :class:`TriageDecision` without ever
raising -- the UI must not crash on a malformed generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Ordered by descending operational urgency. Order matters: it drives inbox sorting.
CATEGORIES: tuple[str, ...] = ("URGENT", "ACTION_NEEDED", "FYI", "SPAM")

#: Rank used when sorting an inbox (lower == floats to the top).
CATEGORY_RANK: Dict[str, int] = {name: i for i, name in enumerate(CATEGORIES)}

#: Tools the model is allowed to call. Anything else is dropped during validation.
KNOWN_TOOLS: frozenset[str] = frozenset(
    {"draft_reply", "create_task", "create_calendar_event", "flag_urgent"}
)

#: Common near-misses mapped back onto the canonical vocabulary.
_CATEGORY_ALIASES: Dict[str, str] = {
    "URGENT": "URGENT",
    "CRITICAL": "URGENT",
    "HIGH": "URGENT",
    "ACTION_NEEDED": "ACTION_NEEDED",
    "ACTION-NEEDED": "ACTION_NEEDED",
    "ACTION NEEDED": "ACTION_NEEDED",
    "ACTIONNEEDED": "ACTION_NEEDED",
    "ACTION": "ACTION_NEEDED",
    "ACTION_REQUIRED": "ACTION_NEEDED",
    "TODO": "ACTION_NEEDED",
    "FYI": "FYI",
    "INFO": "FYI",
    "INFORMATIONAL": "FYI",
    "NEWSLETTER": "FYI",
    "LOW": "FYI",
    "SPAM": "SPAM",
    "JUNK": "SPAM",
    "PHISHING": "SPAM",
    "PROMOTIONAL": "SPAM",
}

#: Aliases for tool names, so a slightly-off generation still executes.
_TOOL_ALIASES: Dict[str, str] = {
    "draft_reply": "draft_reply",
    "reply": "draft_reply",
    "draft": "draft_reply",
    "send_reply": "draft_reply",
    "create_task": "create_task",
    "task": "create_task",
    "add_task": "create_task",
    "todo": "create_task",
    "create_calendar_event": "create_calendar_event",
    "calendar_event": "create_calendar_event",
    "create_event": "create_event",
    "add_event": "create_calendar_event",
    "schedule": "create_calendar_event",
    "flag_urgent": "flag_urgent",
    "flag": "flag_urgent",
    "mark_urgent": "flag_urgent",
}
# `create_event` is itself an alias target above by mistake-proofing; normalise it here.
_TOOL_ALIASES["create_event"] = "create_calendar_event"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single function call requested by Gemma 4."""

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "args": self.args}


@dataclass
class TriageDecision:
    """Validated triage verdict for one email."""

    category: str = "FYI"
    priority: int = 3
    summary: str = ""
    reasoning: str = ""
    suggested_reply: Optional[str] = None
    actions: List[ToolCall] = field(default_factory=list)

    #: Raw model text, kept for the "agent trace" panel in the UI.
    raw_output: str = ""
    #: True when the model's output could not be parsed and defaults were used.
    parse_failed: bool = False

    @property
    def rank(self) -> int:
        """Sort key component: lower means more urgent."""
        return CATEGORY_RANK.get(self.category, len(CATEGORIES))

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        data["actions"] = [a.to_dict() for a in self.actions]
        if not include_raw:
            data.pop("raw_output", None)
        return data


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _iter_balanced_objects(text: str):
    """Yield every top-level ``{...}`` span in *text*, ignoring braces inside strings."""
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort recovery of one JSON object from raw model output.

    Strategy, in order: markdown fences first (models love them), then any balanced
    top-level object, each attempted verbatim and then with trailing commas repaired.
    Returns ``None`` when nothing usable is found.
    """
    if not text:
        return None

    candidates: List[str] = []
    candidates.extend(m.group(1).strip() for m in _FENCE_RE.finditer(text))
    candidates.extend(_iter_balanced_objects(text))
    candidates.append(text.strip())

    for candidate in candidates:
        if not candidate:
            continue
        for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
            try:
                parsed = json.loads(attempt)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
            # A model may wrap the object in a single-element list.
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
    return None


# ---------------------------------------------------------------------------
# Field coercion
# ---------------------------------------------------------------------------


def _coerce_category(value: Any) -> str:
    if not isinstance(value, str):
        return "FYI"
    key = value.strip().upper().replace("-", "_").replace(" ", "_")
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    # Substring rescue for things like "CATEGORY: URGENT".
    for name in CATEGORIES:
        if name in key:
            return name
    return "FYI"


def _coerce_priority(value: Any, category: str) -> int:
    try:
        priority = int(round(float(value)))
    except (TypeError, ValueError):
        priority = {"URGENT": 5, "ACTION_NEEDED": 3, "FYI": 2, "SPAM": 1}.get(category, 3)
    return max(1, min(5, priority))


def _coerce_text(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()[:limit]


def _coerce_optional_text(value: Any, limit: int = 4000) -> Optional[str]:
    if value is None:
        return None
    text = _coerce_text(value, limit)
    if not text or text.lower() in {"null", "none", "n/a", "na", "-"}:
        return None
    return text


def _coerce_actions(value: Any) -> List[ToolCall]:
    """Normalise the ``actions`` array, dropping anything unusable."""
    if not isinstance(value, list):
        return []

    calls: List[ToolCall] = []
    for item in value[:8]:  # a hard cap keeps a runaway generation from ballooning the UI
        if isinstance(item, str):
            item = {"tool": item, "args": {}}
        if not isinstance(item, dict):
            continue

        name = item.get("tool") or item.get("name") or item.get("function")
        if not isinstance(name, str):
            continue
        canonical = _TOOL_ALIASES.get(name.strip().lower())
        if canonical not in KNOWN_TOOLS:
            continue

        args = item.get("args") or item.get("arguments") or item.get("parameters") or {}
        if isinstance(args, str):
            # Some models stringify the argument object.
            args = extract_json(args) or {}
        if not isinstance(args, dict):
            args = {}
        # Keys must be strings for downstream ** expansion and JSON round-tripping.
        args = {str(k): v for k, v in args.items()}

        calls.append(ToolCall(tool=canonical, args=args))

    # De-duplicate identical calls while preserving order.
    seen: set[str] = set()
    deduped: List[ToolCall] = []
    for call in calls:
        key = json.dumps(call.to_dict(), sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            deduped.append(call)
    return deduped


def parse_triage(text: str) -> TriageDecision:
    """Parse raw model output into a validated :class:`TriageDecision`.

    Never raises. On unrecoverable output, returns a safe ``FYI`` decision with
    ``parse_failed=True`` so the UI can surface the degradation honestly.
    """
    raw = text or ""
    payload = extract_json(raw)

    if payload is None:
        return TriageDecision(
            category="FYI",
            priority=3,
            summary=_coerce_text(raw, 240) or "Could not parse the model response.",
            reasoning="The model did not return valid JSON; a neutral default was applied.",
            suggested_reply=None,
            actions=[],
            raw_output=raw,
            parse_failed=True,
        )

    category = _coerce_category(payload.get("category"))
    decision = TriageDecision(
        category=category,
        priority=_coerce_priority(payload.get("priority"), category),
        summary=_coerce_text(payload.get("summary"), 600),
        reasoning=_coerce_text(payload.get("reasoning"), 1200),
        suggested_reply=_coerce_optional_text(payload.get("suggested_reply")),
        actions=_coerce_actions(payload.get("actions")),
        raw_output=raw,
        parse_failed=False,
    )

    if not decision.summary:
        decision.summary = "(no summary returned)"

    # Consistency guard: an email the model flagged URGENT should never sit at the
    # bottom of the inbox because of a low priority number.
    if decision.category == "URGENT":
        decision.priority = max(decision.priority, 4)
    if decision.category == "SPAM":
        decision.priority = min(decision.priority, 2)
        # Never auto-draft a reply to spam, whatever the model asked for.
        decision.actions = [a for a in decision.actions if a.tool != "draft_reply"]
        decision.suggested_reply = None

    return decision
