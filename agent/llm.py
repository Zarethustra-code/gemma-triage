"""Gemma 4 backend abstraction.

One class, :class:`GemmaLLM`, with one method, :meth:`GemmaLLM.generate`. Three
interchangeable backends sit behind it, selected by the ``GEMMA_BACKEND`` env var:

    1. ``transformers``  -- Gemma 4 running locally, on-device. Email content stays
                            on this machine. The real product.
    2. ``hf_api``        -- Gemma 4 hosted via the Hugging Face Inference API. Email
                            content is sent to an external provider.
    3. ``heuristic``     -- a dependency-free keyword engine honouring the same
                            JSON contract, so the demo can never hard-fail. Local.

``auto`` (the default) tries ``transformers`` and then ``heuristic``. It reaches the
hosted tier **only** when ``ALLOW_REMOTE_INFERENCE=true``; otherwise nothing ever
leaves the device without the operator having asked for it in so many words. Naming
``hf_api`` explicitly is itself that request, and is always honoured.

The heuristic tier is deliberately not a mock: it returns schema-valid output for
both prompt types, which means the UI, the tool executor and the test suite all
exercise the same code paths whether or not model weights are present.
:attr:`active_backend` is surfaced in the UI so a viewer always knows which engine
produced what they are looking at, and whether it ran here or elsewhere.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

# Default checkpoint. E4B is the effective-4B on-device variant; E2B is the lighter
# sibling for constrained hardware. Override with GEMMA_MODEL_ID.
DEFAULT_MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-E4B-it")

#: Model classes probed in order when loading locally. Older `transformers` builds
#: lack the newer names, so we walk down until one both exists and loads.
_MODEL_CLASS_CANDIDATES = (
    "Gemma4ForConditionalGeneration",
    "AutoModelForImageTextToText",
    "AutoModelForCausalLM",
)

VALID_BACKENDS = ("auto", "transformers", "hf_api", "heuristic")

#: After this many consecutive generation failures, drop to the heuristic tier
#: permanently rather than paying for a failing round trip on every email.
MAX_CONSECUTIVE_FAILURES = 2


def remote_inference_allowed() -> bool:
    """True when hosted (off-device) inference is explicitly permitted.

    Read at construction time, not import time, so a test or a Space can set
    ``ALLOW_REMOTE_INFERENCE`` and rebuild the engine. Defaults to ``false``:
    ``auto`` must never reach for a remote provider on its own.
    """
    return os.getenv("ALLOW_REMOTE_INFERENCE", "false").strip().lower() == "true"


class GemmaLLM:
    """Text-in / text-out wrapper around Gemma 4.

    Parameters
    ----------
    backend:
        One of ``auto``, ``transformers``, ``hf_api``, ``heuristic``. Defaults to
        the ``GEMMA_BACKEND`` env var, then ``auto``.
    model_id:
        Hugging Face repo id. Defaults to ``GEMMA_MODEL_ID`` then ``google/gemma-4-E4B-it``.
    max_new_tokens:
        Generation budget per call.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model_id: Optional[str] = None,
        max_new_tokens: int = 640,
    ) -> None:
        requested = (backend or os.environ.get("GEMMA_BACKEND") or "auto").strip().lower()
        if requested not in VALID_BACKENDS:
            requested = "auto"

        self.requested_backend: str = requested
        self.model_id: str = model_id or DEFAULT_MODEL_ID
        self.max_new_tokens: int = max_new_tokens

        #: Human-readable trail of what was tried and why it was rejected.
        self.notes: List[str] = []

        self._model: Any = None
        self._processor: Any = None
        self._client: Any = None
        self._degraded: bool = False
        self._consecutive_failures: int = 0

        #: Off-device inference is opt-in. `auto` never reaches for it on its own.
        self.allow_remote: bool = remote_inference_allowed()

        #: Chat-template message shape known to work (cached after first success).
        self._chat_shape_index: Optional[int] = None

        self._active_backend: str = self._resolve_backend(requested)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @property
    def active_backend(self) -> str:
        """The backend actually serving requests right now."""
        return self._active_backend

    @property
    def is_local(self) -> bool:
        """True when inference runs on this device, so email content stays here."""
        return self._active_backend in ("transformers", "heuristic")

    @property
    def is_remote(self) -> bool:
        """True when email content is sent to an external inference provider."""
        return self._active_backend == "hf_api"

    @property
    def backend_label(self) -> str:
        """Short description for the UI status bar."""
        labels = {
            "transformers": f"Gemma 4 on-device ({self.model_id}, transformers)",
            "hf_api": f"REMOTE — Gemma 4 via the Hugging Face Inference API ({self.model_id})",
            "heuristic": "Heuristic fallback engine, on-device (no Gemma weights loaded)",
        }
        label = labels.get(self._active_backend, self._active_backend)
        if self._degraded:
            label += " — degraded, serving from the heuristic fallback"
        return label

    def _resolve_backend(self, requested: str) -> str:
        """Pick a backend, honouring an explicit request and falling back sanely.

        Hosted inference is never chosen implicitly. ``auto`` walks
        ``transformers -> heuristic`` unless ``ALLOW_REMOTE_INFERENCE=true``, in which
        case ``hf_api`` is inserted between them. Asking for ``hf_api`` by name is an
        explicit, informed choice and is always honoured.
        """
        if requested == "heuristic":
            self.notes.append("Heuristic backend requested explicitly.")
            return "heuristic"

        if requested in ("transformers", "hf_api"):
            order = [requested]
            if requested == "hf_api":
                self.notes.append(
                    "Hosted backend requested explicitly — email content will be sent to "
                    "the configured external inference provider."
                )
        else:
            order = ["transformers"]
            if self.allow_remote:
                order.append("hf_api")
                self.notes.append(
                    "ALLOW_REMOTE_INFERENCE=true — `auto` may fall back to hosted inference."
                )
            else:
                self.notes.append(
                    "Remote inference is disabled (ALLOW_REMOTE_INFERENCE is not 'true'), "
                    "so `auto` resolves on-device only: transformers, then heuristic."
                )

        for name in order:
            ok = self._try_init_transformers() if name == "transformers" else self._try_init_hf_api()
            if ok:
                return name

        self.notes.append("Falling back to the heuristic engine; output stays schema-valid.")
        return "heuristic"

    def _try_init_transformers(self) -> bool:
        """Load Gemma 4 locally. Returns False (with a note) on any failure."""
        try:
            import torch  # noqa: F401
            import transformers
        except ImportError:
            self.notes.append(
                "transformers/torch not installed — see requirements-local.txt for the on-device stack."
            )
            return False

        try:
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError:  # pragma: no cover - defensive
            self.notes.append("transformers is installed but its API is unexpected.")
            return False

        # Processor first: Gemma 4 is multimodal, so AutoProcessor is the right entry
        # point. Fall back to a plain tokenizer for text-only checkpoints.
        processor = None
        for loader in (AutoProcessor, AutoTokenizer):
            try:
                processor = loader.from_pretrained(self.model_id)
                break
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                self.notes.append(f"{loader.__name__} failed for {self.model_id}: {exc}")
        if processor is None:
            return False

        # `dtype` replaced `torch_dtype` in recent transformers, and old builds reject
        # the new name while new builds warn on the old one. Try both before giving up
        # on half precision, then device_map alone, then bare defaults.
        device_map = os.environ.get("GEMMA_DEVICE", "auto")
        load_attempts: List[Dict[str, Any]] = []
        try:
            import torch

            for dtype_kw in ("dtype", "torch_dtype"):
                load_attempts.append({"device_map": device_map, dtype_kw: torch.bfloat16})
        except Exception:  # noqa: BLE001
            pass
        load_attempts.append({"device_map": device_map})
        load_attempts.append({})

        for class_name in _MODEL_CLASS_CANDIDATES:
            klass = getattr(transformers, class_name, None)
            if klass is None:
                continue
            for kwargs in load_attempts:
                try:
                    self._model = klass.from_pretrained(self.model_id, **kwargs)
                    self._processor = processor
                    self.notes.append(f"Loaded {self.model_id} with {class_name}.")
                    return True
                except TypeError:
                    continue
                except Exception as exc:  # noqa: BLE001
                    self.notes.append(f"{class_name} could not load {self.model_id}: {exc}")
                    break
        return False

    def _try_init_hf_api(self) -> bool:
        """Attach to the Hugging Face Inference API. Requires HF_TOKEN."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            self.notes.append("HF_TOKEN not set — the hosted Gemma 4 backend is unavailable.")
            return False
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            self.notes.append("huggingface_hub not installed — the hosted backend is unavailable.")
            return False

        try:
            self._client = InferenceClient(model=self.model_id, token=token)
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"InferenceClient init failed: {exc}")
            return False

        self.notes.append(f"Using the Hugging Face Inference API for {self.model_id}.")
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, system: str, user: str) -> str:
        """Run one system+user turn through the active backend and return raw text.

        A backend that throws at call time degrades to the heuristic engine for that
        call rather than propagating, so a flaky network never breaks a triage run.
        """
        backend = self._active_backend
        try:
            if backend == "transformers":
                result = self._generate_transformers(system, user)
            elif backend == "hf_api":
                result = self._generate_hf_api(system, user)
            else:
                return _heuristic_generate(system, user)
        except Exception as exc:  # noqa: BLE001 - the demo must never crash
            self._degraded = True
            self._consecutive_failures += 1
            self.notes.append(f"{backend} generation failed ({exc}); used the heuristic fallback.")
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                # A dead endpoint or a bad token would otherwise make every email in the
                # inbox pay for a failing round trip. Switch tiers for good instead.
                self._active_backend = "heuristic"
                self.notes.append(
                    f"Disabled the {backend} backend after "
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive failures."
                )
            return _heuristic_generate(system, user)

        self._consecutive_failures = 0
        return result

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _generate_transformers(self, system: str, user: str) -> str:
        import torch

        inputs = self._build_chat_inputs(system, user)

        device = getattr(self._model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        prompt_len = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic: triage should be reproducible
            )

        new_tokens = output[0][prompt_len:]
        decoder = getattr(self._processor, "decode", None) or self._processor.tokenizer.decode
        return decoder(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _chat_shapes(system: str, user: str) -> List[List[Dict[str, Any]]]:
        """Message shapes to probe, richest first.

        Chat templates disagree about two things: whether ``system`` is a role at all,
        and whether ``content`` may be a list of typed blocks. The last shape folds the
        system turn into the user turn, which every Gemma template accepts — so the
        local backend degrades to a different prompt shape, never to the heuristic.
        """
        folded = f"{system}\n\n{user}"
        return [
            [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": user}]},
            ],
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            [{"role": "user", "content": [{"type": "text", "text": folded}]}],
            [{"role": "user", "content": folded}],
        ]

    def _build_chat_inputs(self, system: str, user: str):
        """Tokenise one turn, caching the first message shape the template accepts."""
        shapes = self._chat_shapes(system, user)
        order = (
            [self._chat_shape_index]
            if self._chat_shape_index is not None
            else list(range(len(shapes)))
        )
        for i in order:
            try:
                inputs = self._processor.apply_chat_template(
                    shapes[i],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if self._chat_shape_index is None:
                    self._chat_shape_index = i
                    if i > 0:
                        self.notes.append(
                            f"Chat template needed shape #{i} (system folded into the user turn)."
                        )
                return inputs
            except Exception:  # noqa: BLE001
                continue
        if self._chat_shape_index is not None:
            # The cached shape stopped working; forget it and re-probe from scratch.
            self._chat_shape_index = None
            return self._build_chat_inputs(system, user)
        return self._processor(f"{system}\n\n{user}", return_tensors="pt")

    def _generate_hf_api(self, system: str, user: str) -> str:
        try:
            response = self._client.chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_new_tokens,
                temperature=0.0,  # deterministic: triage should be reproducible
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            pass
        # Some providers reject a separate system role outright. Fold and retry.
        try:
            response = self._client.chat_completion(
                messages=[{"role": "user", "content": f"{system}\n\n{user}"}],
                max_tokens=self.max_new_tokens,
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            prompt = f"{system}\n\n{user}\n"
            return str(
                self._client.text_generation(prompt, max_new_tokens=self.max_new_tokens)
            ).strip()


# =====================================================================
# Heuristic backend
# =====================================================================
#
# A dependency-free classifier that speaks the same JSON contract as Gemma 4.
# It reads the structured email block that `TriageAgent` puts in the user turn,
# so it needs no model, no token, and no network.

_SPAM_SIGNALS = (
    "unsubscribe", "click here", "you have won", "you've won", "winner", "lottery",
    "risk-free", "risk free", "act now", "limited time", "limited-time", "100% free",
    "no obligation", "claim your prize", "claim now", "exclusive offer", "special promotion",
    "make money", "work from home", "crypto", "bitcoin", "investment opportunity",
    "wire transfer", "verify your account", "confirm your password", "suspended account",
    "dear customer", "dear user", "congratulations", "viagra", "miracle", "guaranteed income",
    "double your", "one weird trick", "as seen on",
)

_URGENT_SIGNALS = (
    "urgent", "asap", "as soon as possible", "immediately", "emergency", "critical",
    "outage", "is down", "are down", "went down", "downtime", "production down",
    "p0", "p1", "sev1", "sev-1", "severity 1", "security breach", "breach", "incident",
    "escalat", "blocker", "blocking", "overdue", "final notice", "last warning",
    "expires today", "due today", "by end of day", "by eod", "eod today", "right away",
    "time-sensitive", "time sensitive", "cannot wait", "can't wait", "before close of business",
)

_ACTION_SIGNALS = (
    "please review", "please confirm", "please approve", "please send", "please sign",
    "please complete", "please respond", "please reply", "please let me know",
    "can you", "could you", "would you", "can we", "could we", "shall we",
    "let me know", "get back to me", "your input", "your feedback", "your approval",
    "your comments", "sign off", "sign the", "take a look", "have a look", "review it",
    "needs your", "waiting on you", "waiting for your", "action required", "action needed",
    "rsvp", "respond by", "reply by", "deadline", "submit", "approve", "confirm",
    "send me", "share the", "schedule", "book a", "are you available", "do you have time",
    "does that work", "works for you", "work for you", "what do you think", "any update",
)

_FYI_SIGNALS = (
    "newsletter", "digest", "no action", "no action is needed", "no reply needed",
    "do not reply", "for your information", "fyi", "notification", "release notes",
    "weekly update", "monthly update", "announcement", "recap", "minutes", "receipt",
    "your order", "has shipped", "status page", "changelog", "you are subscribed",
)

_MEETING_SIGNALS = (
    "meeting", "call", "sync", "interview", "demo", "standup", "stand-up", "appointment",
    "invite", "invitation", "zoom", "google meet", "microsoft teams", "1:1", "one-on-one",
    "catch up", "catch-up", "kickoff", "kick-off", "workshop", "review session", "onboarding session",
)

#: Word-boundary matcher, so "recall" or "demographics" never read as a meeting.
_MEETING_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _MEETING_SIGNALS) + r")\b", re.I
)

_NOREPLY_SENDERS = ("noreply", "no-reply", "donotreply", "newsletter", "marketing", "notifications", "mailer")

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_MONTH_DAY_RE = re.compile(
    r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2})\b", re.I
)
_TIME_12H_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", re.I)
_TIME_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TONE_RE = re.compile(r"in a ([a-z]+) tone", re.I)


def _parse_email_block(user: str) -> Dict[str, str]:
    """Read the ``KEY: value`` header block + ``BODY:`` section built by TriageAgent."""
    fields: Dict[str, str] = {"today": "", "from": "", "to": "", "subject": "", "date": "", "body": ""}
    if not user:
        return fields

    lines = user.splitlines()
    body_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("BODY:"):
            body_start = i
            remainder = stripped[5:].strip()
            if remainder:
                fields["body"] = remainder
            break
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip().lower()
            if key in fields:
                fields[key] = value.strip()

    if body_start is not None:
        tail = "\n".join(lines[body_start + 1 :]).strip()
        fields["body"] = (fields["body"] + "\n" + tail).strip() if fields["body"] else tail
    elif not fields["body"]:
        fields["body"] = user

    return fields


def _count_signals(text: str, signals) -> tuple[int, List[str]]:
    hits = [s for s in signals if s in text]
    return len(hits), hits[:4]


def _first_name(sender: str) -> str:
    """Pull a plausible first name out of a ``Name <addr>`` or bare-address sender."""
    sender = (sender or "").strip()
    if not sender:
        return "there"
    display = sender.split("<")[0].strip().strip('"')
    if display and "@" not in display:
        first = re.split(r"[\s,]+", display)[0]
        if first.lower() in {"the", "team", "support", "no-reply", "noreply"}:
            return display
        return first
    local = sender.split("<")[-1].strip(">").split("@")[0]
    local = re.split(r"[._\-+]", local)[0]
    return local.capitalize() if local else "there"


def _pick_summary_sentence(body: str, subject: str) -> str:
    """Prefer the sentence that carries the ask; fall back to the opening sentence."""
    clean = " ".join(body.split())
    if not clean:
        return subject or "(empty email)"
    sentences = [s.strip() for s in _SENTENCE_RE.split(clean) if len(s.strip()) > 15]
    if not sentences:
        sentences = [clean]
    lowered = [s.lower() for s in sentences]
    for i, low in enumerate(lowered):
        if any(sig in low for sig in _ACTION_SIGNALS + _URGENT_SIGNALS):
            return sentences[i]
    return sentences[0]


def _shorten(text: str, max_words: int = 24) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.rstrip(" ,;")
    return " ".join(words[:max_words]).rstrip(" ,;") + "…"


def _find_date(text: str) -> Optional[str]:
    """Return the earliest-mentioned date phrase in *text*, or None.

    Position decides, not pattern order. "Half the team is out on Wednesday, could
    we move it to Thursday?" proposes Thursday -- and because callers pass the
    subject line ahead of the body, the headline date wins over anything mentioned
    in passing further down.
    """
    candidates: List[tuple[int, str]] = []

    iso = _ISO_DATE_RE.search(text)
    if iso:
        candidates.append((iso.start(), iso.group(1)))

    for word in ("tomorrow", "today", "next week"):
        position = text.find(word)
        if position >= 0:
            candidates.append((position, word))

    month_day = _MONTH_DAY_RE.search(text)
    if month_day:
        candidates.append((month_day.start(), month_day.group(1)))

    for day in _WEEKDAYS:
        position = text.find(day)
        if position >= 0:
            prefixed = text[max(0, position - 5) : position] == "next "
            candidates.append((position, f"next {day}" if prefixed else day))

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _find_time(text: str) -> Optional[str]:
    """Return the earliest-mentioned time of day as 24-hour ``HH:MM``, or None."""
    candidates: List[tuple[int, str]] = []

    twelve = _TIME_12H_RE.search(text)
    if twelve:
        hour, minute = int(twelve.group(1)), int(twelve.group(2) or 0)
        meridiem = twelve.group(3).replace(".", "").lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23:
            candidates.append((twelve.start(), f"{hour:02d}:{minute:02d}"))

    twenty_four = _TIME_24H_RE.search(text)
    if twenty_four:
        candidates.append(
            (twenty_four.start(), f"{int(twenty_four.group(1)):02d}:{twenty_four.group(2)}")
        )

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _heuristic_triage(user: str) -> str:
    """Classify an email with keyword scoring and emit contract-valid JSON."""
    import json

    fields = _parse_email_block(user)
    subject = fields["subject"] or "(no subject)"
    body = fields["body"]
    sender = fields["from"]
    haystack = f"{subject}\n{body}".lower()
    sender_lc = sender.lower()

    spam_score, spam_hits = _count_signals(haystack, _SPAM_SIGNALS)
    urgent_score, urgent_hits = _count_signals(haystack, _URGENT_SIGNALS)
    action_score, action_hits = _count_signals(haystack, _ACTION_SIGNALS)
    fyi_score, fyi_hits = _count_signals(haystack, _FYI_SIGNALS)

    automated_sender = any(tag in sender_lc for tag in _NOREPLY_SENDERS)
    if automated_sender:
        fyi_score += 1
    if subject.strip().endswith("?"):
        # A question in the subject line is a request aimed at the recipient.
        action_score += 1

    # Decide. Spam is checked first so an "urgent!!!" scam does not reach the top.
    if spam_score >= 2 or (spam_score >= 1 and automated_sender and action_score == 0):
        category, evidence = "SPAM", spam_hits
    elif urgent_score >= 1 and not (automated_sender and urgent_score < 2):
        category, evidence = "URGENT", urgent_hits
    elif action_score >= 1:
        category, evidence = "ACTION_NEEDED", action_hits
    elif fyi_score >= 1 or automated_sender:
        category, evidence = "FYI", fyi_hits
    else:
        category, evidence = "FYI", []

    deadline_soon = any(w in haystack for w in ("today", "tomorrow", "by eod", "end of day", "deadline", "this week"))
    priority = {
        "URGENT": 5 if urgent_score >= 2 else 4,
        "ACTION_NEEDED": 4 if deadline_soon else 3,
        "FYI": 2,
        "SPAM": 1,
    }[category]

    summary = _shorten(_pick_summary_sentence(body, subject))
    evidence_str = ", ".join(f'"{e}"' for e in evidence) if evidence else "no strong keyword signals"
    reasoning = (
        f"Keyword engine matched {evidence_str} in the subject/body, "
        f"so this was filed as {category} at priority {priority}."
    )

    # ---- tool selection -------------------------------------------------
    actions: List[Dict[str, Any]] = []
    suggested_reply: Optional[str] = None
    date_phrase = _find_date(haystack)
    time_phrase = _find_time(haystack)
    # Only schedule something when the evidence is strong: a meeting word in the
    # subject line, or one in the body backed by an explicit time. Without this gate,
    # "thanks for the call last week" would silently book a meeting.
    wants_meeting = bool(_MEETING_RE.search(subject)) or (
        bool(_MEETING_RE.search(body)) and time_phrase is not None
    )

    if category == "URGENT":
        actions.append({"tool": "flag_urgent", "args": {"reason": _shorten(summary, 16)}})
        actions.append({"tool": "draft_reply", "args": {"tone": "urgent"}})
        actions.append(
            {"tool": "create_task", "args": {"title": f"Resolve: {subject}", "due": date_phrase or "today"}}
        )
    elif category == "ACTION_NEEDED":
        actions.append({"tool": "draft_reply", "args": {"tone": "professional"}})
        actions.append(
            {"tool": "create_task", "args": {"title": f"Follow up: {subject}", "due": date_phrase}}
        )

    if category in ("URGENT", "ACTION_NEEDED", "FYI") and wants_meeting and date_phrase:
        actions.append(
            {
                "tool": "create_calendar_event",
                "args": {"title": subject, "date": date_phrase, "time": time_phrase},
            }
        )

    if category in ("URGENT", "ACTION_NEEDED"):
        suggested_reply = _compose_reply(
            sender, subject, "urgent" if category == "URGENT" else "professional"
        )

    return json.dumps(
        {
            "category": category,
            "priority": priority,
            "summary": summary,
            "reasoning": reasoning,
            "suggested_reply": suggested_reply,
            "actions": actions,
        },
        ensure_ascii=False,
        indent=2,
    )


_TONE_LINES = {
    "urgent": (
        "I'm treating this as urgent and am looking into it right now. "
        "I'll send you an update as soon as I have something concrete."
    ),
    "professional": (
        "Thanks for flagging this. I've picked it up and will review it and come back to you "
        "with a clear answer shortly."
    ),
    "friendly": (
        "Thanks for the nudge! I've got this on my list and I'll get back to you soon."
    ),
    "brief": "Acknowledged — I'll follow up shortly.",
    "firm": (
        "I've noted the request. I'll confirm what is realistically achievable and respond "
        "before the date you mentioned."
    ),
}


def _compose_reply(sender: str, subject: str, tone: str) -> str:
    """Template-based reply used when no Gemma weights are available."""
    tone = tone if tone in _TONE_LINES else "professional"
    name = _first_name(sender)
    opener = f"Hi {name}," if name else "Hello,"
    subject_clause = f'about "{subject}"' if subject and subject != "(no subject)" else ""
    lead = f"Thanks for your email {subject_clause}.".replace("  ", " ").strip()
    return f"{opener}\n\n{lead}\n\n{_TONE_LINES[tone]}\n\nBest regards,\n[Your name]"


def _heuristic_reply(system: str, user: str) -> str:
    fields = _parse_email_block(user)
    tone_match = _TONE_RE.search(system or "")
    tone = tone_match.group(1).lower() if tone_match else "professional"
    return _compose_reply(fields["from"], fields["subject"], tone)


def _heuristic_generate(system: str, user: str) -> str:
    """Route to the right heuristic based on which prompt template is in play.

    The triage system prompt is the only one that names the category vocabulary,
    which makes it a reliable, prompt-file-agnostic discriminator.
    """
    system = system or ""
    if "ACTION_NEEDED" in system or '"category"' in system:
        return _heuristic_triage(user)
    return _heuristic_reply(system, user)
