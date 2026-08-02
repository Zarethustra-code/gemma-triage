"""Gemma-Triage agent package.

A privacy-first, on-device email triage agent built on Gemma 4.

Public surface:
    GemmaLLM        -- backend abstraction over Gemma 4 (transformers / hf_api / heuristic)
    TriageAgent     -- classify -> execute tool calls -> prioritise
    TriageDecision  -- validated structured output contract
"""

from __future__ import annotations

from .schema import CATEGORIES, KNOWN_TOOLS, ToolCall, TriageDecision, parse_triage
from .llm import GemmaLLM
from .tools import ToolResult, execute_actions
from .triage import ProcessedEmail, TriageAgent

__all__ = [
    "CATEGORIES",
    "KNOWN_TOOLS",
    "ToolCall",
    "TriageDecision",
    "parse_triage",
    "GemmaLLM",
    "ToolResult",
    "execute_actions",
    "ProcessedEmail",
    "TriageAgent",
]

__version__ = "1.0.0"
