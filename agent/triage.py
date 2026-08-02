"""The agent loop: classify -> execute tool calls -> prioritise.

``TriageAgent.process_inbox`` is the whole product in one method. For each email it

  1. asks Gemma 4 for a strict-JSON triage decision,
  2. validates that decision into a :class:`TriageDecision`,
  3. executes every tool call the model requested (which may call Gemma again to
     write a reply), and
  4. sorts the finished inbox so what matters floats to the top.

Nothing in here raises on a single bad email -- a failure is captured on that
email's :class:`ProcessedEmail` and the run continues.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, Iterator, List, Optional

from .llm import GemmaLLM
from .schema import TriageDecision, parse_triage
from .tools import ToolResult, build_email_block, execute_actions, load_prompt

#: Called as ``on_progress(index, total, email)`` before each email is processed.
ProgressCallback = Callable[[int, int, Dict[str, Any]], None]


@dataclass
class ProcessedEmail:
    """One email after the full triage + execution pass."""

    email: Dict[str, Any]
    decision: TriageDecision
    results: List[ToolResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    # --- convenience views used by the UI -----------------------------

    @property
    def reply_text(self) -> Optional[str]:
        """The best available draft: the tool-generated body, else the inline suggestion."""
        for result in self.results:
            if result.kind == "reply" and result.ok:
                return result.detail
        return self.decision.suggested_reply

    @property
    def reply_payload(self) -> Optional[Dict[str, Any]]:
        for result in self.results:
            if result.kind == "reply" and result.ok:
                return result.payload
        return None

    @property
    def tasks(self) -> List[ToolResult]:
        return [r for r in self.results if r.kind == "task" and r.ok]

    @property
    def events(self) -> List[ToolResult]:
        return [r for r in self.results if r.kind == "event" and r.ok]

    @property
    def flags(self) -> List[ToolResult]:
        return [r for r in self.results if r.kind == "flag" and r.ok]

    @property
    def errors(self) -> List[str]:
        return [r.error or f"{r.tool} failed" for r in self.results if not r.ok]

    @property
    def sort_key(self) -> tuple[int, int]:
        """URGENT first, then by descending priority."""
        return (self.decision.rank, -self.decision.priority)

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        return {
            "email": {
                "id": self.email.get("id", ""),
                "from": self.email.get("from", ""),
                "subject": self.email.get("subject", ""),
                "date": self.email.get("date", ""),
            },
            "decision": self.decision.to_dict(include_raw=include_raw),
            "tool_results": [r.to_dict() for r in self.results],
            "elapsed_s": round(self.elapsed_s, 2),
        }


class TriageAgent:
    """Runs the triage workflow over an inbox using a :class:`GemmaLLM` backend."""

    def __init__(self, llm: Optional[GemmaLLM] = None) -> None:
        self.llm = llm or GemmaLLM()
        self._system_prompt = load_prompt("triage_prompt.txt")

    # ------------------------------------------------------------------
    # Single email
    # ------------------------------------------------------------------

    def classify(self, email: Dict[str, Any]) -> TriageDecision:
        """Ask Gemma 4 to triage one email and validate the response."""
        user = build_email_block(email)
        try:
            raw = self.llm.generate(self._system_prompt, user)
        except Exception as exc:  # noqa: BLE001 - GemmaLLM already guards, belt and braces
            return TriageDecision(
                category="FYI",
                priority=3,
                summary=f"Triage failed for '{email.get('subject', '(no subject)')}'.",
                reasoning=f"The model backend raised {type(exc).__name__}: {exc}",
                raw_output="",
                parse_failed=True,
            )
        return parse_triage(raw)

    def process_email(self, email: Dict[str, Any], today: Optional[date] = None) -> ProcessedEmail:
        """Classify one email, then execute the tool calls it produced."""
        started = time.perf_counter()
        decision = self.classify(email)
        try:
            results = execute_actions(self.llm, email, decision, today=today)
        except Exception as exc:  # noqa: BLE001 - executor already guards per tool
            results = [ToolResult(tool="executor", ok=False, error=f"{type(exc).__name__}: {exc}")]
        return ProcessedEmail(
            email=email,
            decision=decision,
            results=results,
            elapsed_s=time.perf_counter() - started,
        )

    # ------------------------------------------------------------------
    # Whole inbox
    # ------------------------------------------------------------------

    def stream_inbox(
        self, emails: List[Dict[str, Any]], today: Optional[date] = None
    ) -> Iterator[ProcessedEmail]:
        """Yield results one email at a time, in input order, for live UI updates."""
        for email in emails:
            yield self.process_email(email, today=today)

    def process_inbox(
        self,
        emails: List[Dict[str, Any]],
        on_progress: Optional[ProgressCallback] = None,
        today: Optional[date] = None,
    ) -> List[ProcessedEmail]:
        """Process an entire inbox and return it sorted by urgency then priority."""
        total = len(emails)
        processed: List[ProcessedEmail] = []

        for index, email in enumerate(emails):
            if on_progress is not None:
                try:
                    on_progress(index, total, email)
                except Exception:  # noqa: BLE001 - a broken callback must not stop triage
                    pass
            processed.append(self.process_email(email, today=today))

        return sort_results(processed)


def sort_results(processed: List[ProcessedEmail]) -> List[ProcessedEmail]:
    """URGENT/high-priority float to the top; original order breaks ties."""
    return sorted(processed, key=lambda p: p.sort_key)


def inbox_stats(processed: List[ProcessedEmail]) -> Dict[str, int]:
    """Counts for the dashboard header."""
    stats: Dict[str, int] = {
        "total": len(processed),
        "URGENT": 0,
        "ACTION_NEEDED": 0,
        "FYI": 0,
        "SPAM": 0,
        "replies": 0,
        "tasks": 0,
        "events": 0,
        "errors": 0,
    }
    for item in processed:
        stats[item.decision.category] = stats.get(item.decision.category, 0) + 1
        stats["replies"] += 1 if item.reply_text else 0
        stats["tasks"] += len(item.tasks)
        stats["events"] += len(item.events)
        stats["errors"] += len(item.errors)
    return stats
