"""Backend selection and the local-inference path.

Two things are pinned here.

**Where email content goes.** ``auto`` must never reach a hosted provider on its own.
Every test that touches the hosted tier substitutes a fake client -- nothing in this
file may perform network I/O, and the spies assert as much.

**Whether the local tier actually loads.** Real Gemma degrades to the heuristic engine
silently when a chat template rejects a separate ``system`` role or list-style content
blocks. The shape-probe tests exist so that regression is caught in CI rather than on
a judge's laptop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.llm import GemmaLLM, remote_inference_allowed  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from a known, offline environment."""
    for name in (
        "ALLOW_REMOTE_INFERENCE",
        "GEMMA_BACKEND",
        "GEMMA_MODEL_ID",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


class Spy:
    """Records calls and returns a fixed value."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.result


def patch_backends(monkeypatch, *, local_ok: bool, remote_ok: bool):
    """Replace both initialisers with spies. Returns ``(local_spy, remote_spy)``."""
    local, remote = Spy(local_ok), Spy(remote_ok)
    monkeypatch.setattr(GemmaLLM, "_try_init_transformers", local)
    monkeypatch.setattr(GemmaLLM, "_try_init_hf_api", remote)
    return local, remote


# ---------------------------------------------------------------------------
# ALLOW_REMOTE_INFERENCE
# ---------------------------------------------------------------------------


def test_remote_inference_is_disabled_by_default():
    assert remote_inference_allowed() is False
    assert GemmaLLM(backend="heuristic").allow_remote is False


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("TRUE", True), (" True ", True), ("false", False), ("1", False), ("", False)],
)
def test_allow_remote_inference_parses_strictly(monkeypatch, value, expected):
    """Only a literal 'true' opts in. Anything ambiguous stays off."""
    monkeypatch.setenv("ALLOW_REMOTE_INFERENCE", value)
    assert remote_inference_allowed() is expected


def test_auto_never_calls_the_hosted_api_when_remote_is_disabled(monkeypatch):
    local, remote = patch_backends(monkeypatch, local_ok=False, remote_ok=True)

    llm = GemmaLLM(backend="auto")

    assert local.calls == 1, "the local tier must still be tried"
    assert remote.calls == 0, "the hosted backend must not even be initialised"
    assert llm.active_backend == "heuristic"
    assert llm.is_local is True and llm.is_remote is False


def test_auto_with_local_failure_and_remote_enabled_uses_the_hosted_api(monkeypatch):
    monkeypatch.setenv("ALLOW_REMOTE_INFERENCE", "true")
    local, remote = patch_backends(monkeypatch, local_ok=False, remote_ok=True)

    llm = GemmaLLM(backend="auto")

    assert local.calls == 1 and remote.calls == 1
    assert llm.active_backend == "hf_api"
    assert llm.is_remote is True and llm.is_local is False


def test_auto_prefers_the_local_tier_even_when_remote_is_enabled(monkeypatch):
    monkeypatch.setenv("ALLOW_REMOTE_INFERENCE", "true")
    local, remote = patch_backends(monkeypatch, local_ok=True, remote_ok=True)

    llm = GemmaLLM(backend="auto")

    assert llm.active_backend == "transformers"
    assert remote.calls == 0, "on-device wins whenever it is available"


def test_explicitly_requesting_hf_api_is_honoured_even_while_remote_is_disabled(monkeypatch):
    """Naming the hosted backend *is* the explicit permission."""
    local, remote = patch_backends(monkeypatch, local_ok=True, remote_ok=True)

    llm = GemmaLLM(backend="hf_api")

    assert llm.active_backend == "hf_api"
    assert local.calls == 0, "an explicit request must not fall through the ladder"
    assert "external" in " ".join(llm.notes).lower()


def test_explicit_hf_api_still_degrades_to_heuristic_when_it_cannot_initialise(monkeypatch):
    patch_backends(monkeypatch, local_ok=True, remote_ok=False)
    assert GemmaLLM(backend="hf_api").active_backend == "heuristic"


def test_the_disabled_state_is_explained_in_the_notes(monkeypatch):
    patch_backends(monkeypatch, local_ok=False, remote_ok=True)
    notes = " ".join(GemmaLLM(backend="auto").notes)
    assert "ALLOW_REMOTE_INFERENCE" in notes


def test_hosted_client_is_never_constructed_while_remote_is_disabled(monkeypatch):
    """End-to-end version of the guarantee: no InferenceClient, no token read, no call."""
    monkeypatch.setattr(GemmaLLM, "_try_init_transformers", Spy(False))
    monkeypatch.setenv("HF_TOKEN", "hf_this_must_never_be_used")

    constructed: list = []

    class ExplodingClient:
        def __init__(self, *args, **kwargs):  # pragma: no cover - must never run
            constructed.append(kwargs)
            raise AssertionError("the hosted client was constructed with remote disabled")

    module = type(sys)("huggingface_hub")
    module.InferenceClient = ExplodingClient
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    llm = GemmaLLM(backend="auto")

    assert constructed == []
    assert llm.active_backend == "heuristic"
    assert llm._client is None


def test_remote_backends_are_labelled_remote_in_the_ui_string(monkeypatch):
    monkeypatch.setenv("ALLOW_REMOTE_INFERENCE", "true")
    patch_backends(monkeypatch, local_ok=False, remote_ok=True)

    label = GemmaLLM(backend="auto").backend_label
    assert "REMOTE" in label, "a viewer must be able to see that processing left the device"


def test_local_backends_are_not_labelled_remote(monkeypatch):
    patch_backends(monkeypatch, local_ok=True, remote_ok=False)
    assert "REMOTE" not in GemmaLLM(backend="transformers").backend_label
    assert "REMOTE" not in GemmaLLM(backend="heuristic").backend_label


# ---------------------------------------------------------------------------
# Chat-template shape probing
# ---------------------------------------------------------------------------


def test_chat_shapes_end_with_the_system_turn_folded_into_the_user_turn():
    shapes = GemmaLLM._chat_shapes("SYS", "USR")

    assert len(shapes) == 4
    assert [m["role"] for m in shapes[0]] == ["system", "user"]
    assert shapes[0][0]["content"] == [{"type": "text", "text": "SYS"}]
    assert [m["role"] for m in shapes[1]] == ["system", "user"]
    assert shapes[1][0]["content"] == "SYS"

    for fallback in shapes[2:]:
        assert [m["role"] for m in fallback] == ["user"], "no system role in the fallbacks"
    assert shapes[3][0]["content"] == "SYS\n\nUSR", "both halves survive the fold"
    assert "USR" in str(shapes[2][0]["content"])


class FakeInputs(dict):
    """Minimal stand-in for a tokenised batch: dict-like, with ``.to()``."""

    def __init__(self, n_tokens: int = 4):
        super().__init__(input_ids=FakeIds(n_tokens))
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeIds:
    def __init__(self, n: int):
        self.shape = (1, n)


class FakeProcessor:
    """A processor whose chat template only accepts certain message shapes.

    ``accepts`` is a predicate over the message list, mirroring how a real template
    raises when handed a role or a content type it does not know about.
    """

    def __init__(self, accepts=None, n_tokens: int = 4):
        self.accepts = accepts or (lambda messages: True)
        self.n_tokens = n_tokens
        self.template_calls: list = []
        self.direct_calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(messages)
        if not self.accepts(messages):
            raise ValueError("this template does not support that message shape")
        return FakeInputs(self.n_tokens)

    def __call__(self, text, **kwargs):
        self.direct_calls += 1
        return FakeInputs(self.n_tokens)

    def decode(self, tokens, skip_special_tokens=True):
        return "  decoded output  "


def _rejects_system_role(messages):
    return all(m["role"] != "system" for m in messages)


def _rejects_block_content(messages):
    return all(isinstance(m["content"], str) for m in messages)


def make_llm(monkeypatch, processor):
    """A heuristic-backed instance with a fake processor grafted on.

    Building via ``backend="heuristic"`` keeps the constructor away from torch and the
    network; the shape probe under test does not care which tier is nominally active.
    """
    llm = GemmaLLM(backend="heuristic")
    llm._processor = processor
    return llm


def test_the_richest_shape_is_used_when_the_template_accepts_it(monkeypatch):
    processor = FakeProcessor()
    llm = make_llm(monkeypatch, processor)

    llm._build_chat_inputs("SYS", "USR")

    assert llm._chat_shape_index == 0
    assert len(processor.template_calls) == 1
    assert not any("folded" in n for n in llm.notes)


def test_a_template_without_a_system_role_falls_through_to_the_folded_shape(monkeypatch):
    processor = FakeProcessor(accepts=_rejects_system_role)
    llm = make_llm(monkeypatch, processor)

    llm._build_chat_inputs("SYS", "USR")

    assert llm._chat_shape_index == 2, "first shape with no system role"
    used = processor.template_calls[-1]
    assert [m["role"] for m in used] == ["user"]
    assert "SYS" in str(used[0]["content"]) and "USR" in str(used[0]["content"])
    assert any("folded" in note for note in llm.notes), "the fallback must be visible"


def test_a_template_that_wants_plain_strings_falls_through_to_a_plain_string_shape(monkeypatch):
    processor = FakeProcessor(accepts=_rejects_block_content)
    llm = make_llm(monkeypatch, processor)

    llm._build_chat_inputs("SYS", "USR")

    assert llm._chat_shape_index == 1
    assert processor.template_calls[-1][0]["content"] == "SYS"


def test_the_working_shape_is_cached_and_reused(monkeypatch):
    processor = FakeProcessor(accepts=_rejects_system_role)
    llm = make_llm(monkeypatch, processor)

    llm._build_chat_inputs("SYS", "USR")
    first_probe = len(processor.template_calls)
    llm._build_chat_inputs("SYS", "USR 2")

    assert len(processor.template_calls) == first_probe + 1, "no re-probing after the first hit"
    assert llm._chat_shape_index == 2


def test_a_cached_shape_that_stops_working_is_re_probed(monkeypatch):
    processor = FakeProcessor()
    llm = make_llm(monkeypatch, processor)
    llm._build_chat_inputs("SYS", "USR")
    assert llm._chat_shape_index == 0

    processor.accepts = _rejects_system_role
    llm._build_chat_inputs("SYS", "USR")

    assert llm._chat_shape_index == 2, "the probe restarts rather than giving up"


def test_a_processor_with_no_usable_template_falls_back_to_plain_tokenisation(monkeypatch):
    processor = FakeProcessor(accepts=lambda messages: False)
    llm = make_llm(monkeypatch, processor)

    llm._build_chat_inputs("SYS", "USR")

    assert processor.direct_calls == 1
    assert llm._chat_shape_index is None
    assert len(processor.template_calls) == 4, "every shape was tried first"


# ---------------------------------------------------------------------------
# Local generation
# ---------------------------------------------------------------------------


class FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeModel:
    device = "cpu"

    def __init__(self, prompt_len: int = 4):
        self.prompt_len = prompt_len
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        # Prompt tokens followed by the "generated" ones the decoder will see.
        return [list(range(self.prompt_len)) + [101, 102, 103]]


def test_generate_transformers_decodes_only_the_new_tokens(monkeypatch):
    torch = type(sys)("torch")
    torch.inference_mode = FakeInferenceMode
    monkeypatch.setitem(sys.modules, "torch", torch)

    processor = FakeProcessor(n_tokens=4)
    llm = make_llm(monkeypatch, processor)
    model = FakeModel(prompt_len=4)
    llm._model = model

    result = llm._generate_transformers("SYS", "USR")

    assert result == "decoded output", "the decoded text is stripped"
    assert model.kwargs["max_new_tokens"] == llm.max_new_tokens
    assert model.kwargs["do_sample"] is False, "triage must be reproducible"
    assert "input_ids" in model.kwargs


def test_generate_transformers_moves_inputs_to_the_model_device(monkeypatch):
    torch = type(sys)("torch")
    torch.inference_mode = FakeInferenceMode
    monkeypatch.setitem(sys.modules, "torch", torch)

    captured = {}

    class RecordingProcessor(FakeProcessor):
        def apply_chat_template(self, messages, **kwargs):
            captured["inputs"] = super().apply_chat_template(messages, **kwargs)
            return captured["inputs"]

    llm = make_llm(monkeypatch, RecordingProcessor())
    llm._model = FakeModel()

    llm._generate_transformers("SYS", "USR")

    assert captured["inputs"].moved_to == "cpu"


# ---------------------------------------------------------------------------
# Hosted generation (fake client -- no network)
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, content):
        message = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": message})()]


class FakeClient:
    """Hosted client that can reject a system role, chat entirely, or both."""

    def __init__(self, *, allow_system=True, allow_chat=True):
        self.allow_system = allow_system
        self.allow_chat = allow_chat
        self.chat_calls: list = []
        self.text_calls: list = []

    def chat_completion(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        if not self.allow_chat:
            raise RuntimeError("chat_completion is unavailable")
        if not self.allow_system and any(m["role"] == "system" for m in messages):
            raise ValueError("this provider does not accept a system role")
        return FakeResponse("hosted output")

    def text_generation(self, prompt, **kwargs):
        self.text_calls.append((prompt, kwargs))
        return "text-generation output"


def _hosted_llm(monkeypatch, client):
    llm = GemmaLLM(backend="heuristic")
    llm._client = client
    return llm


def test_hosted_generation_is_deterministic(monkeypatch):
    client = FakeClient()
    llm = _hosted_llm(monkeypatch, client)

    assert llm._generate_hf_api("SYS", "USR") == "hosted output"

    _, kwargs = client.chat_calls[0]
    assert kwargs["temperature"] == 0.0, "triage must not be sampled"


def test_hosted_generation_retries_with_the_system_turn_folded_in(monkeypatch):
    client = FakeClient(allow_system=False)
    llm = _hosted_llm(monkeypatch, client)

    assert llm._generate_hf_api("SYS", "USR") == "hosted output"

    assert len(client.chat_calls) == 2
    retry_messages = client.chat_calls[1][0]
    assert [m["role"] for m in retry_messages] == ["user"]
    assert retry_messages[0]["content"] == "SYS\n\nUSR"
    assert client.text_calls == []


def test_hosted_generation_falls_back_to_text_generation(monkeypatch):
    client = FakeClient(allow_chat=False)
    llm = _hosted_llm(monkeypatch, client)

    assert llm._generate_hf_api("SYS", "USR") == "text-generation output"

    assert len(client.chat_calls) == 2, "both chat shapes are tried first"
    assert len(client.text_calls) == 1
    assert client.text_calls[0][0].startswith("SYS\n\nUSR")


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
