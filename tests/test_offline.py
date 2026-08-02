"""The test suite must not touch the network. This is the test that says so.

Deliberately its own module: ``tests/test_backend.py`` has an autouse fixture that
clears the very variables asserted here, because it needs to test what happens when
they are absent. The pinning itself belongs somewhere that fixture cannot reach.

The regression being guarded is specific. ``app.py`` builds its module-level ``ENGINE``
at import time, and ``GemmaLLM`` resolves its backend from the ambient environment. A
developer who has followed the README -- ``export HF_TOKEN=...``, ``export
GEMMA_BACKEND=hf_api`` -- would otherwise find that merely importing ``app`` in a UI
test sends sample-inbox email bodies to the Hugging Face Inference API, on their token.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.llm import GemmaLLM  # noqa: E402


def test_the_environment_is_pinned_to_the_offline_backend():
    assert os.environ["GEMMA_BACKEND"] == "heuristic"
    assert os.environ["ALLOW_REMOTE_INFERENCE"] == "false"


def test_ambient_hugging_face_credentials_are_cleared():
    """Whatever token the developer's shell is carrying, it is not ours to spend."""
    assert "HF_TOKEN" not in os.environ
    assert "HUGGING_FACE_HUB_TOKEN" not in os.environ


#: Both libraries rewrite their own flags to a canonical spelling on import, so these
#: assertions check the meaning rather than the exact string conftest.py wrote.
TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}


def test_hub_and_telemetry_are_disabled():
    assert os.environ["HF_HUB_OFFLINE"] in TRUTHY
    assert os.environ["TRANSFORMERS_OFFLINE"] in TRUTHY
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"].lower() in TRUTHY
    assert os.environ["GRADIO_ANALYTICS_ENABLED"].lower() in FALSY


def test_the_default_engine_resolves_offline():
    llm = GemmaLLM()
    assert llm.active_backend == "heuristic"
    assert llm.is_local is True
    assert llm.is_remote is False
    assert llm.allow_remote is False


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly on any outbound connection, and report where it was going."""
    attempts: list = []
    real_connect = socket.socket.connect

    def spy(self, address, *args, **kwargs):
        attempts.append(address)
        raise AssertionError(f"the test suite attempted to connect to {address!r}")

    monkeypatch.setattr(socket.socket, "connect", spy)
    yield attempts
    monkeypatch.setattr(socket.socket, "connect", real_connect)


def test_building_the_engine_opens_no_connection(no_network):
    llm = GemmaLLM(backend="auto")
    assert llm.active_backend == "heuristic"
    assert llm._client is None
    assert no_network == []


def test_triaging_the_whole_demo_inbox_opens_no_connection(no_network):
    import json

    from agent import TriageAgent

    emails = json.loads((REPO_ROOT / "data" / "sample_inbox.json").read_text(encoding="utf-8"))
    processed = TriageAgent(llm=GemmaLLM()).process_inbox(emails["emails"])

    assert len(processed) == len(emails["emails"])
    assert no_network == []


def test_importing_the_app_opens_no_connection(no_network):
    """The exact path that leaked: a bare `import app` in a UI test module."""
    pytest.importorskip("gradio", reason="the UI layer needs Gradio installed")

    import app

    assert app.ENGINE.llm is not None
    assert app.ENGINE.llm.active_backend == "heuristic"
    assert no_network == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
