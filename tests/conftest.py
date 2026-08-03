"""Test-suite environment and shared spies. Imported by pytest before any test module.

The suite must never touch a real network. That is not automatic: ``app.py`` builds
its module-level ``ENGINE`` at import time, and ``GemmaLLM`` resolves its backend from
the *ambient* environment. A developer who has followed the README (``export
HF_TOKEN=...``, ``export GEMMA_BACKEND=hf_api``) would otherwise have ``import app``
alone send sample-inbox email bodies to the Hugging Face Inference API, and be billed
for it.

So the environment is pinned here, at module scope rather than in a fixture, because
conftest is imported before the test modules that do ``import app``. Individual tests
that need a different environment (``tests/test_backend.py``) override it with
``monkeypatch``, which restores these values afterwards.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The one backend that needs no weights, no token and no network.
os.environ["GEMMA_BACKEND"] = "heuristic"
os.environ["ALLOW_REMOTE_INFERENCE"] = "false"

# Any credential the ambient shell is carrying is not this suite's to spend.
for _name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MAX_ACTIONS_PER_EMAIL"):
    os.environ.pop(_name, None)

# Belt and braces: even a code path that ignored the settings above cannot reach the
# Hub, and neither Gradio nor huggingface_hub may phone home with usage telemetry.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("DO_NOT_TRACK", "1")


# ---------------------------------------------------------------------------
# Shared spies
# ---------------------------------------------------------------------------
#
# Used by the reply-regeneration tests, on both sides of the Gradio import boundary:
# a per-row reply rewrite must reach neither classification nor Gmail, and that has to
# be asserted from the tool layer and from the UI handler. The fixtures live here so
# both files spy on exactly the same call sites.


@pytest.fixture
def no_triage(monkeypatch):
    """Count every entry into classification or tool execution.

    Wraps rather than blocks, so a test that *should* trip a counter still works — see
    the guard test that proves these spies fire on a real triage run.
    """
    from agent import tools, triage as triage_module
    from agent.triage import TriageAgent

    counts = {"process_inbox": 0, "process_email": 0, "classify": 0, "execute_actions": 0}

    def counting(key, original):
        def wrapper(*args, **kwargs):
            counts[key] += 1
            return original(*args, **kwargs)

        return wrapper

    for name in ("process_inbox", "process_email", "classify"):
        monkeypatch.setattr(TriageAgent, name, counting(name, getattr(TriageAgent, name)))

    # `agent.triage` imported the executor by name, so both bindings need the spy.
    for module in (tools, triage_module):
        monkeypatch.setattr(
            module, "execute_actions", counting("execute_actions", module.execute_actions)
        )

    return counts


@pytest.fixture
def no_gmail(monkeypatch):
    """Make any attempt to reach Gmail fail loudly, and record it if it somehow does not."""
    import gmail_integration
    from gmail_integration import client as gmail_client

    attempts: list = []

    def forbidden(label):
        def call(*args, **kwargs):
            attempts.append(label)
            raise AssertionError(f"{label} must never be called here")

        return call

    for module in (gmail_integration, gmail_client):
        for name in ("create_reply_draft", "create_draft", "get_service", "fetch_recent"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden(f"{module.__name__}.{name}"))

    return attempts
