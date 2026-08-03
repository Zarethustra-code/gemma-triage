"""The Ollama tier: real Gemma 4, on this device, without torch.

Ollama is a server on localhost, so it belongs on the on-device side of every decision
this file checks: it is never gated behind ``ALLOW_REMOTE_INFERENCE``, it is labelled
LOCAL, and ``auto`` reaches it before the hosted tier. It also must never *become* a
dependency — the transport is ``urllib`` from the standard library.

No test here touches a real server. ``conftest.py`` pins the probe off for the whole
suite (a developer with Ollama running would otherwise change which backend the
selection tests resolve to), so this module captures the genuine method at import time
and re-installs it where the real code path is under test, driving it with a fake
``urlopen``.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import llm as llm_module  # noqa: E402
from agent.llm import GemmaLLM, remote_inference_allowed  # noqa: E402

#: The genuine probe, captured before any fixture replaces it.
REAL_PROBE = GemmaLLM._try_init_ollama


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start from a known, offline environment, as the backend tests do."""
    for name in (
        "ALLOW_REMOTE_INFERENCE",
        "GEMMA_BACKEND",
        "GEMMA_MODEL_ID",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def live_probe(monkeypatch):
    """Undo the suite-wide stub so the real ``_try_init_ollama`` runs."""
    monkeypatch.setattr(GemmaLLM, "_try_init_ollama", REAL_PROBE)


class FakeResponse(io.BytesIO):
    """Minimal stand-in for what ``urlopen`` returns: a context manager with a status."""

    def __init__(self, body: str = "", status: int = 200):
        super().__init__(body.encode("utf-8"))
        self.status = status

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def urlopen(monkeypatch):
    """Record every request and reply with a queued response. No socket is opened."""

    class Recorder:
        def __init__(self):
            self.calls: list = []
            self.response = FakeResponse(json.dumps({"models": [{"name": "gemma4:12b"}]}))
            self.error = None

        def __call__(self, request, timeout=None):
            self.calls.append({"request": request, "timeout": timeout})
            if self.error is not None:
                raise self.error
            return self.response

        @property
        def urls(self):
            return [
                c["request"] if isinstance(c["request"], str) else c["request"].full_url
                for c in self.calls
            ]

    recorder = Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    return recorder


def chat_response(content: str) -> FakeResponse:
    return FakeResponse(json.dumps({"choices": [{"message": {"content": content}}]}))


def ollama_llm(model: str = "gemma4:12b", url: str = "http://localhost:11434/v1") -> GemmaLLM:
    """A heuristic-built instance with the Ollama fields grafted on, no probe involved."""
    llm = GemmaLLM(backend="heuristic")
    llm._active_backend = "ollama"
    llm._ollama_model = model
    llm._ollama_url = url
    return llm


# ---------------------------------------------------------------------------
# Vocabulary and privacy posture
# ---------------------------------------------------------------------------


def test_ollama_is_a_valid_backend():
    assert "ollama" in llm_module.VALID_BACKENDS
    assert GemmaLLM(backend="ollama").requested_backend == "ollama"


def test_ollama_counts_as_on_device():
    assert set(llm_module.LOCAL_BACKENDS) == {"transformers", "ollama", "heuristic"}

    llm = ollama_llm()
    assert llm.is_local is True
    assert llm.is_remote is False


def test_ollama_is_never_labelled_remote():
    label = ollama_llm().backend_label
    assert label == "Gemma 4 on-device via Ollama (gemma4:12b)"
    assert "REMOTE" not in label


def test_the_label_names_the_model_actually_configured():
    assert "gemma4:27b" in ollama_llm(model="gemma4:27b").backend_label


def test_the_defaults_point_at_a_local_ollama():
    assert llm_module.OLLAMA_BASE_URL.startswith("http://localhost:11434")
    assert llm_module.OLLAMA_BASE_URL.endswith("/v1"), "the OpenAI-compatible surface"
    assert llm_module.OLLAMA_MODEL == "gemma4:12b"


def test_the_transport_adds_no_dependency():
    """`urllib` only. A pip install must never stand between a judge and the demo."""
    source = (REPO_ROOT / "agent" / "llm.py").read_text(encoding="utf-8")
    for package in ("import requests", "import httpx", "import aiohttp", "from openai", "import ollama"):
        assert package not in source, f"{package!r} would be a new dependency"
    assert "import urllib.request" in source


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def test_a_healthy_server_is_accepted_and_recorded(urlopen):
    llm = GemmaLLM(backend="heuristic")

    assert REAL_PROBE(llm) is True
    assert llm._ollama_model == "gemma4:12b"
    assert llm._ollama_url == "http://localhost:11434/v1"
    assert any("Ollama" in note for note in llm.notes)


def test_the_probe_is_a_get_to_the_native_tags_endpoint(urlopen):
    REAL_PROBE(GemmaLLM(backend="heuristic"))

    assert urlopen.urls == ["http://localhost:11434/api/tags"]
    assert urlopen.calls[0]["timeout"] == llm_module.OLLAMA_PROBE_TIMEOUT_S
    assert llm_module.OLLAMA_PROBE_TIMEOUT_S <= 5, "a missing server must not stall start-up"


def test_the_probe_follows_a_relocated_server(monkeypatch, urlopen):
    monkeypatch.setattr(llm_module, "OLLAMA_BASE_URL", "http://192.168.1.9:11434/v1")
    llm = GemmaLLM(backend="heuristic")

    assert REAL_PROBE(llm) is True
    assert urlopen.urls == ["http://192.168.1.9:11434/api/tags"]
    assert llm._ollama_url == "http://192.168.1.9:11434/v1"


def test_a_non_200_answer_is_refused(urlopen):
    urlopen.response = FakeResponse("nope", status=503)
    llm = GemmaLLM(backend="heuristic")

    assert REAL_PROBE(llm) is False
    assert llm._ollama_model == ""
    assert any("503" in note for note in llm.notes)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        OSError("network unreachable"),
    ],
)
def test_a_missing_server_is_reported_not_raised(urlopen, error):
    urlopen.error = error
    llm = GemmaLLM(backend="heuristic")

    assert REAL_PROBE(llm) is False
    assert any("No local Ollama server" in note for note in llm.notes)


def test_the_probe_loads_nothing_and_sends_nothing(urlopen):
    """A health check, not a generation: one GET, no body."""
    REAL_PROBE(GemmaLLM(backend="heuristic"))

    (call,) = urlopen.calls
    assert isinstance(call["request"], str), "a bare URL is a GET"


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_requesting_ollama_explicitly_selects_it(live_probe, urlopen):
    llm = GemmaLLM(backend="ollama")

    assert llm.active_backend == "ollama"
    assert llm.is_local is True
    assert "requested explicitly" in " ".join(llm.notes)


def test_requesting_ollama_without_a_server_degrades_to_heuristic(live_probe, urlopen):
    urlopen.error = urllib.error.URLError("connection refused")

    llm = GemmaLLM(backend="ollama")

    assert llm.active_backend == "heuristic"
    assert llm.is_local is True


def test_ollama_is_not_gated_behind_remote_inference(live_probe, urlopen, monkeypatch):
    """Reaching localhost is not leaving the device, so no permission is required."""
    monkeypatch.setenv("ALLOW_REMOTE_INFERENCE", "false")
    assert remote_inference_allowed() is False

    llm = GemmaLLM(backend="ollama")

    assert llm.active_backend == "ollama"


def test_auto_reaches_ollama_before_the_hosted_tier(live_probe, urlopen, monkeypatch):
    monkeypatch.setenv("ALLOW_REMOTE_INFERENCE", "true")
    monkeypatch.setenv("HF_TOKEN", "hf_this_must_never_be_used")
    monkeypatch.setattr(GemmaLLM, "_try_init_transformers", lambda self: False)

    hosted: list = []
    monkeypatch.setattr(GemmaLLM, "_try_init_hf_api", lambda self: hosted.append(1) or True)

    llm = GemmaLLM(backend="auto")

    assert llm.active_backend == "ollama"
    assert hosted == [], "the hosted tier must not even be initialised"
    assert llm.is_remote is False


def test_auto_still_prefers_transformers_over_ollama(live_probe, urlopen, monkeypatch):
    monkeypatch.setattr(GemmaLLM, "_try_init_transformers", lambda self: True)

    llm = GemmaLLM(backend="auto")

    assert llm.active_backend == "transformers"
    assert urlopen.calls == [], "no probe once an on-device tier is already loaded"


def test_auto_falls_past_a_missing_ollama_to_the_hosted_tier(live_probe, urlopen, monkeypatch):
    monkeypatch.setenv("ALLOW_REMOTE_INFERENCE", "true")
    monkeypatch.setattr(GemmaLLM, "_try_init_transformers", lambda self: False)
    monkeypatch.setattr(GemmaLLM, "_try_init_hf_api", lambda self: True)
    urlopen.error = urllib.error.URLError("connection refused")

    llm = GemmaLLM(backend="auto")

    assert llm.active_backend == "hf_api"


def test_auto_ends_at_the_heuristic_when_nothing_local_is_available(live_probe, urlopen, monkeypatch):
    monkeypatch.setattr(GemmaLLM, "_try_init_transformers", lambda self: False)
    urlopen.error = urllib.error.URLError("connection refused")

    llm = GemmaLLM(backend="auto")

    assert llm.active_backend == "heuristic"
    assert "ALLOW_REMOTE_INFERENCE" in " ".join(llm.notes)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generation_posts_an_openai_shaped_chat_completion(urlopen):
    urlopen.response = chat_response("  the model's answer  ")
    llm = ollama_llm()

    assert llm._generate_ollama("SYS", "USR") == "the model's answer"

    (call,) = urlopen.calls
    request = call["request"]
    assert request.full_url == "http://localhost:11434/v1/chat/completions"
    assert request.method == "POST"
    assert request.headers["Content-type"] == "application/json"

    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "gemma4:12b"
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    assert body["temperature"] == 0, "triage must be reproducible"
    assert body["stream"] is False, "the whole reply must arrive as one JSON document"


def test_generation_allows_time_for_a_cold_model_load(urlopen):
    urlopen.response = chat_response("answer")
    ollama_llm()._generate_ollama("SYS", "USR")

    assert urlopen.calls[0]["timeout"] == llm_module.OLLAMA_TIMEOUT_S
    assert llm_module.OLLAMA_TIMEOUT_S >= 120, "the first call has to load the weights"


def test_an_empty_content_field_becomes_an_empty_string(urlopen):
    urlopen.response = FakeResponse(json.dumps({"choices": [{"message": {"content": None}}]}))
    assert ollama_llm()._generate_ollama("SYS", "USR") == ""


def test_generate_routes_the_ollama_backend_to_the_ollama_call(urlopen):
    urlopen.response = chat_response("real Gemma output")
    llm = ollama_llm()

    assert llm.generate("SYS", "USR") == "real Gemma output"
    assert urlopen.urls == ["http://localhost:11434/v1/chat/completions"]


def test_a_failed_generation_falls_back_to_the_heuristic_engine(urlopen):
    """The demo must not die because the server went away mid-run."""
    urlopen.error = urllib.error.URLError("connection reset")
    llm = ollama_llm()

    output = llm.generate("Answer in a professional tone", "FROM: Sam <s@b.example>\nBODY: hi")

    assert output.strip(), "the heuristic reply stands in"
    assert llm._degraded is True
    assert any("ollama generation failed" in note.lower() for note in llm.notes)


def test_a_dead_server_stops_being_retried(urlopen):
    urlopen.error = urllib.error.URLError("connection refused")
    llm = ollama_llm()

    for _ in range(llm_module.MAX_CONSECUTIVE_FAILURES):
        llm.generate("SYS", "USR")

    assert llm.active_backend == "heuristic", "no point paying for a failing round trip"
    assert "degraded" in llm.backend_label


def test_a_successful_call_clears_the_failure_count(urlopen):
    llm = ollama_llm()
    urlopen.error = urllib.error.URLError("blip")
    llm.generate("SYS", "USR")
    assert llm._consecutive_failures == 1

    urlopen.error = None
    urlopen.response = chat_response("back again")
    assert llm.generate("SYS", "USR") == "back again"
    assert llm._consecutive_failures == 0
    assert llm.active_backend == "ollama"


# ---------------------------------------------------------------------------
# The UI knows about it
# ---------------------------------------------------------------------------


def test_the_backend_dropdown_offers_ollama():
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert '"auto", "transformers", "ollama", "hf_api", "heuristic"' in source


def test_the_status_bar_calls_an_ollama_run_local():
    pytest.importorskip("gradio", reason="the UI layer needs Gradio installed")
    import app

    class FakeLLM:
        active_backend = "ollama"
        requested_backend = "ollama"
        model_id = "gemma4:12b"
        allow_remote = False
        is_local = True
        is_remote = False
        backend_label = "Gemma 4 on-device via Ollama (gemma4:12b)"
        notes: list = []

    original = app.ENGINE.agent
    app.ENGINE.agent = type("A", (), {"llm": FakeLLM()})()
    try:
        html = app.status_bar_html()
        notes = app.engine_notes_md()
    finally:
        app.ENGINE.agent = original

    assert "LOCAL processing" in html
    assert "REMOTE processing" not in html
    assert "Ollama" in html and "keyword engine" not in html
    assert "on this device" in notes


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
