"""Gemma-Triage — Gradio dashboard.

Entry point for both local runs and the Hugging Face Space. The app opens straight
into a working demo inbox: no login, no token, no model download required. If Gemma 4
weights are present the app uses them and says so in the status bar; if not, it falls
back to the heuristic engine and says that instead. Hosted inference is only ever used
when ``ALLOW_REMOTE_INFERENCE=true`` or the ``hf_api`` backend is named outright, and
the status bar says plainly when processing is happening off this device.

Run:  python app.py
"""

from __future__ import annotations

import html
import inspect
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

#: Gradio 6 moved `css`/`theme` from the Blocks constructor to launch(). Feature-detect
#: rather than version-parse so the app runs unchanged on Gradio 4, 5 and 6.
_LAUNCH_PARAMS = set(inspect.signature(gr.Blocks.launch).parameters)
_STYLE_ON_LAUNCH = {"css", "theme"} <= _LAUNCH_PARAMS

from agent import GemmaLLM, TriageAgent, tools
from agent.tools import STATUS_APPROVED, STATUS_CREATED, STATUS_FAILED, STATUS_PREPARED
from agent.triage import ProcessedEmail, inbox_stats, sort_results

APP_DIR = Path(__file__).resolve().parent
SAMPLE_INBOX = APP_DIR / "data" / "sample_inbox.json"

#: Size of the pre-allocated result row pool. Gradio needs a fixed component graph,
#: so rows are created once and shown/hidden per run.
MAX_ROWS = 14

#: Components per result row, in the order ``build_ui`` creates them and ``render_row``
#: / ``blank_row`` fill them:
#:
#:   group, header, accordion, detail, reply, tone, regenerate button,
#:   regenerate status, draft button, draft status, row state
#:
#: Gradio needs a fixed-width output tuple, so this number is asserted in the tests: a
#: row that gains or loses a component and forgets one of the two fillers fails loudly.
ROW_WIDGETS = 11

#: Default tone offered by every row's regenerate control. The vocabulary itself lives
#: in ``agent.tools.VALID_TONES`` — there is no second list.
DEFAULT_TONE = "professional"

#: Per-category presentation. Solid badges stay legible in both light and dark themes.
CATEGORY_STYLE: Dict[str, Dict[str, str]] = {
    "URGENT": {"color": "#dc2626", "icon": "🔴", "label": "URGENT"},
    "ACTION_NEEDED": {"color": "#d97706", "icon": "🟠", "label": "ACTION NEEDED"},
    "FYI": {"color": "#2563eb", "icon": "🔵", "label": "FYI"},
    "SPAM": {"color": "#6b7280", "icon": "⚪", "label": "SPAM"},
}

CUSTOM_CSS = """
.gt-header { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
.gt-badge {
  display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px;
  font-weight:700; letter-spacing:.06em; color:#fff; white-space:nowrap;
}
.gt-prio {
  display:inline-block; padding:3px 9px; border-radius:6px; font-size:11px;
  font-weight:700; border:1px solid currentColor; white-space:nowrap;
}
.gt-subject { font-weight:600; font-size:15px; }
.gt-summary { font-size:14px; opacity:.85; margin-top:6px; line-height:1.5; }
.gt-meta { font-size:12px; opacity:.65; margin-top:4px; }
.gt-card {
  border-left:3px solid var(--gt-accent,#2563eb); border-radius:6px; padding:10px 12px;
  margin:8px 0; background:rgba(127,127,127,.09);
}
.gt-card-title { font-weight:600; font-size:13px; }
.gt-card-detail { font-size:12.5px; opacity:.8; margin-top:3px; }
.gt-status {
  display:inline-block; padding:1px 8px; border-radius:999px; font-size:10.5px;
  font-weight:700; letter-spacing:.04em; text-transform:uppercase;
  border:1px solid currentColor; opacity:.75; white-space:nowrap; margin-left:4px;
}
.gt-section { font-size:11px; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; opacity:.6; margin:16px 0 4px; }
.gt-quote {
  white-space:pre-wrap; font-size:12.5px; line-height:1.55; opacity:.75;
  border-left:2px solid rgba(127,127,127,.35); padding-left:12px; margin:6px 0;
}
.gt-stats { display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 2px; }
.gt-stat {
  border:1px solid rgba(127,127,127,.3); border-radius:8px; padding:8px 14px;
  min-width:82px; text-align:center;
}
.gt-stat-n { font-size:20px; font-weight:700; line-height:1.1; }
.gt-stat-l { font-size:10.5px; letter-spacing:.06em; text-transform:uppercase; opacity:.65; }
.gt-bar {
  border-radius:8px; padding:10px 14px; font-size:13px;
  background:rgba(37,99,235,.10); border:1px solid rgba(37,99,235,.28);
}
.gt-err { border-left:3px solid #dc2626; padding-left:10px; font-size:12.5px; margin-top:8px; }
footer { display:none !important; }
"""


# =====================================================================
# Engine
# =====================================================================


class Engine:
    """Holds the live agent so the UI can rebuild it without restarting the app."""

    def __init__(self) -> None:
        self.agent: Optional[TriageAgent] = None
        self.error: Optional[str] = None
        self.build()

    def build(self, backend: Optional[str] = None, model_id: Optional[str] = None) -> None:
        try:
            llm = GemmaLLM(backend=backend or None, model_id=model_id or None)
            self.agent = TriageAgent(llm=llm)
            self.error = None
        except Exception as exc:  # noqa: BLE001 - a broken engine must not kill the UI
            self.agent = None
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def llm(self) -> Optional[GemmaLLM]:
        return self.agent.llm if self.agent else None


ENGINE = Engine()


def esc(text: Any) -> str:
    return html.escape(str(text or ""))


def status_bar_html() -> str:
    """Top-of-page banner: which engine is live, and the privacy posture."""
    llm = ENGINE.llm
    if llm is None:
        return (
            '<div class="gt-bar" style="background:rgba(220,38,38,.12);'
            'border-color:rgba(220,38,38,.35);">⚠️ <b>Engine unavailable</b> — '
            f"{esc(ENGINE.error)}</div>"
        )

    if llm.active_backend == "transformers":
        icon, lock, note = (
            "🟢",
            "🔒",
            "LOCAL processing — Gemma 4 runs on this device, so email content stays here.",
        )
    elif llm.active_backend == "ollama":
        icon, lock, note = (
            "🟢",
            "🔒",
            "LOCAL processing — Gemma 4 runs on this device through a local Ollama "
            "server, so email content stays here.",
        )
    elif llm.active_backend == "hf_api":
        icon, lock, note = (
            "🟡",
            "🌐",
            "REMOTE processing — email content is sent to the configured external "
            "inference provider for this session. Use the Transformers backend to keep "
            "email content on this device.",
        )
    else:
        icon, lock, note = (
            "⚪",
            "🔒",
            "LOCAL processing — no Gemma weights were loaded, so an on-device keyword "
            "engine is standing in. The JSON contract, tool calls and UI are identical; "
            "only the reasoning quality differs.",
        )

    # Remote inference is off by default and the banner has to be able to say so —
    # a privacy claim nobody can check in the UI is not a privacy claim.
    if llm.active_backend != "hf_api":
        note += (
            "  Remote inference is enabled for this session."
            if llm.allow_remote
            else "  Remote inference is disabled (ALLOW_REMOTE_INFERENCE is not 'true')."
        )

    accent = (
        'style="background:rgba(217,119,6,.12);border-color:rgba(217,119,6,.4);"'
        if llm.active_backend == "hf_api"
        else ""
    )
    return (
        f'<div class="gt-bar" {accent}>{icon} <b>Engine:</b> {esc(llm.backend_label)}'
        f'<br><span style="opacity:.75;font-size:12.5px;">{lock} {esc(note)}</span></div>'
    )


def engine_notes_md() -> str:
    llm = ENGINE.llm
    if llm is None:
        return f"Engine failed to build: `{ENGINE.error}`"
    lines = [
        f"- **Requested backend:** `{llm.requested_backend}`",
        f"- **Active backend:** `{llm.active_backend}`",
        f"- **Model id:** `{llm.model_id}`",
        "- **Processing location:** "
        + ("on this device" if llm.is_local else "external inference provider"),
        f"- **Remote inference permitted:** `{str(llm.allow_remote).lower()}` "
        "(`ALLOW_REMOTE_INFERENCE`)",
    ]
    lines += [f"- {n}" for n in llm.notes] or ["- (no fallback events)"]
    return "\n".join(lines)


# =====================================================================
# Inbox loading
# =====================================================================


def load_sample_inbox() -> List[Dict[str, Any]]:
    """Read the bundled demo inbox. Returns an empty list rather than raising."""
    try:
        data = json.loads(SAMPLE_INBOX.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    emails = data.get("emails", data) if isinstance(data, dict) else data
    return emails if isinstance(emails, list) else []


def connect_gmail(count: int, query: str) -> Tuple[List[Dict[str, Any]], str]:
    """Run the Gmail OAuth flow and fetch recent mail. Never raises into Gradio."""
    try:
        from gmail_integration import GOOGLE_LIBS_MISSING, GmailError, fetch_recent, gmail_available
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return [], "❌ The Gmail connector could not be loaded. See the server log for details."

    if not gmail_available():
        return [], f"❌ {GOOGLE_LIBS_MISSING}"

    try:
        emails = fetch_recent(int(count), query=query or "in:inbox")
    except GmailError as exc:
        # GmailError messages are written to be shown to a user.
        return [], f"❌ {exc}"
    except Exception:  # noqa: BLE001 - an arbitrary exception may carry paths or tokens
        traceback.print_exc()
        return [], "❌ Gmail could not be reached. See the server log for details."

    if not emails:
        return [], "⚠️ Gmail returned no messages for that query."
    return emails, f"✅ Fetched **{len(emails)}** message(s) from Gmail. Press **Run Triage**."


def create_gmail_draft(emails: Any, index: int, reply_text: str) -> str:
    """Create one Gmail draft, for one email, because a human pressed the button.

    Called from a click handler and nowhere else. There is no path from triage to
    here: classifying an inbox never touches Gmail, and this never sends anything.
    """
    if not isinstance(emails, list) or not 0 <= index < len(emails):
        return "❌ This row is not part of the current run. Re-run triage and try again."

    email = emails[index]
    body = (reply_text or "").strip()
    if not body:
        return "❌ The reply is empty. Write something first — no draft was created."

    try:
        from gmail_integration import (
            GOOGLE_LIBS_MISSING,
            GmailError,
            create_reply_draft,
            gmail_available,
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return "❌ The Gmail connector could not be loaded. See the server log for details."

    if not gmail_available():
        return f"❌ {GOOGLE_LIBS_MISSING}"

    try:
        created = create_reply_draft(email, body)
    except GmailError as exc:
        # Covers a missing/malformed recipient, a missing OAuth client, and an API
        # rejection. Every GmailError message is written to be shown to a user.
        return f"❌ **{STATUS_FAILED}** — {exc}"
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return (
            f"❌ **{STATUS_FAILED}** — the draft could not be created. "
            "See the server log for details."
        )

    threaded = " It is threaded onto the original conversation." if created.get("threaded") else ""
    return (
        f"✅ **{STATUS_APPROVED} → {STATUS_CREATED}** — draft to `{created['to']}`, subject "
        f"*{created['subject']}*.{threaded} It is sitting in your Gmail **Drafts** "
        "folder and has **not** been sent."
    )


def regenerate_handler(
    state: Any, tone: str, current_reply: str
) -> Tuple[str, str]:
    """Rewrite ONE row's reply in a new tone, using ONLY the reply-writing Gemma call.

    The whole point is what it does not do. It never re-triages the inbox — no
    ``process_inbox``, no ``process_email``, no reclassification of anything — and it
    never touches Gmail. It runs the second step of the agent loop again, for one
    email, and hands the text back to the same editable textbox the user was already
    looking at. Every other row is untouched because a row's context arrives in its own
    ``gr.State`` and the only output is its own textbox.

    Returns ``(reply_text, status_line)``. On any failure the current reply is returned
    unchanged, so a bad generation can never wipe text the user has written.
    """
    if not state or not isinstance(state, dict) or not state.get("email"):
        return current_reply, "Nothing to regenerate yet."

    if ENGINE.agent is None:
        return current_reply, f"Engine unavailable: {ENGINE.error}"

    tone = tools.normalize_tone(tone)
    try:
        text = tools.regenerate_reply(
            ENGINE.llm, state["email"], tone, state.get("decision_summary")
        )
    except Exception:  # noqa: BLE001 - a failed rewrite must not clear the textbox
        traceback.print_exc()
        return current_reply, "❌ The reply could not be regenerated. See the server log."

    return (
        text,
        f"Reply regenerated in a {tone} tone on `{ENGINE.llm.active_backend}`. "
        "Edit it freely — nothing was drafted or sent.",
    )


def load_demo(_: Any = None) -> Tuple[List[Dict[str, Any]], str]:
    emails = load_sample_inbox()
    if not emails:
        return [], f"❌ Could not read the demo inbox at `{SAMPLE_INBOX}`."
    return emails, f"📥 Demo inbox loaded — **{len(emails)}** synthetic emails. Press **Run Triage**."


# =====================================================================
# Rendering
# =====================================================================


def overview_html(processed: List[ProcessedEmail], total: int, done: int) -> str:
    stats = inbox_stats(processed)
    tiles = [
        ("Triaged", f"{done}/{total}", "inherit"),
        ("Urgent", stats["URGENT"], CATEGORY_STYLE["URGENT"]["color"]),
        ("Action", stats["ACTION_NEEDED"], CATEGORY_STYLE["ACTION_NEEDED"]["color"]),
        ("FYI", stats["FYI"], CATEGORY_STYLE["FYI"]["color"]),
        ("Spam", stats["SPAM"], CATEGORY_STYLE["SPAM"]["color"]),
        ("Drafts", stats["replies"], "#0891b2"),
        ("Tasks", stats["tasks"], "#7c3aed"),
        ("Events", stats["events"], "#059669"),
    ]
    cells = "".join(
        f'<div class="gt-stat"><div class="gt-stat-n" style="color:{color};">{esc(value)}</div>'
        f'<div class="gt-stat-l">{esc(label)}</div></div>'
        for label, value, color in tiles
    )
    return f'<div class="gt-stats">{cells}</div>'


def _tool_card(accent: str, icon: str, title: str, detail: str, status: str = "") -> str:
    chip = (
        f'<span class="gt-status">{esc(status)}</span>' if status else ""
    )
    return (
        f'<div class="gt-card" style="--gt-accent:{accent};">'
        f'<div class="gt-card-title">{icon} {esc(title)} {chip}</div>'
        f'<div class="gt-card-detail">{esc(detail)}</div></div>'
    )


def header_html(item: ProcessedEmail, position: int) -> str:
    style = CATEGORY_STYLE.get(item.decision.category, CATEGORY_STYLE["FYI"])
    subject = item.email.get("subject") or "(no subject)"
    sender = item.email.get("from") or "unknown sender"
    priority = item.decision.priority
    dots = "●" * priority + "○" * (5 - priority)

    flag = ""
    if item.flags:
        flag = ' <span class="gt-badge" style="background:#b91c1c;">FLAGGED</span>'
    degraded = ""
    if item.decision.parse_failed:
        degraded = ' <span class="gt-badge" style="background:#78716c;">FALLBACK PARSE</span>'

    return (
        f'<div class="gt-header">'
        f'<span style="opacity:.45;font-size:12px;font-weight:700;">#{position}</span>'
        f'<span class="gt-badge" style="background:{style["color"]};">{style["label"]}</span>'
        f'<span class="gt-prio" style="color:{style["color"]};">P{priority} {dots}</span>'
        f"{flag}{degraded}"
        f'<span class="gt-subject">{esc(subject)}</span>'
        f"</div>"
        f'<div class="gt-summary">{esc(item.decision.summary)}</div>'
        f'<div class="gt-meta">From {esc(sender)} · {esc(item.email.get("date", ""))} '
        f"· {item.elapsed_s:.1f}s</div>"
    )


def detail_html(item: ProcessedEmail) -> str:
    parts: List[str] = [
        '<div class="gt-section">Why Gemma filed it this way</div>',
        f'<div style="font-size:13px;line-height:1.6;">{esc(item.decision.reasoning)}</div>',
    ]

    actions: List[str] = []
    for flag in item.flags:
        actions.append(_tool_card("#dc2626", "🚩", flag.title, flag.detail))
    for task in item.tasks:
        actions.append(
            _tool_card("#7c3aed", "✅", task.title, task.detail, task.payload.get("status", ""))
        )
    for event in item.events:
        actions.append(
            _tool_card("#059669", "📅", event.title, event.detail, event.payload.get("status", ""))
        )

    parts.append('<div class="gt-section">Prepared actions</div>')
    if actions:
        parts.extend(actions)
        parts.append(
            '<div style="font-size:12px;opacity:.6;margin-top:6px;">'
            f"{esc(STATUS_PREPARED)} means the Google Tasks / Calendar request has been "
            "built and validated here. Nothing has been sent to Google — that would "
            f"show as “{esc(STATUS_CREATED)}”.</div>"
        )
    else:
        parts.append(
            '<div style="font-size:13px;opacity:.6;">No follow-up actions — '
            "the agent decided this email needs nothing.</div>"
        )

    if item.errors:
        parts.append('<div class="gt-section">Tool errors</div>')
        for error in item.errors:
            parts.append(f'<div class="gt-err">{esc(error)}</div>')

    body = str(item.email.get("body") or "")
    preview = body[:900] + ("…" if len(body) > 900 else "")
    parts.append('<div class="gt-section">Original email</div>')
    parts.append(f'<div class="gt-quote">{esc(preview)}</div>')

    return "".join(parts)


def blank_row() -> List[Any]:
    return [
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(label="Details"),
        gr.update(value=""),
        gr.update(value="", visible=False),
        gr.update(value=DEFAULT_TONE, visible=False),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        # A blank row owns no email, so its regenerate button has nothing to act on.
        None,
    ]


def render_row(item: ProcessedEmail, position: int) -> List[Any]:
    n_actions = len(item.tasks) + len(item.events) + len(item.flags)
    subject = (item.email.get("subject") or "(no subject)")[:70]
    label = f"▸ {subject} — reasoning, reply & {n_actions} action(s)"

    reply = item.reply_text
    has_reply = bool(reply)
    reply_update = (
        gr.update(
            value=reply,
            visible=True,
            label=f"✍️ Suggested reply (editable) — to {item.email.get('from', '')[:60]}",
        )
        if has_reply
        else gr.update(value="", visible=False)
    )

    tone = DEFAULT_TONE
    payload = item.reply_payload or {}
    if payload.get("tone") in tools.VALID_TONES:
        # Start the dropdown on the tone Gemma actually chose for this email.
        tone = payload["tone"]

    return [
        gr.update(visible=True),
        gr.update(value=header_html(item, position)),
        gr.update(label=label),
        gr.update(value=detail_html(item)),
        reply_update,
        # Rewriting a reply only makes sense where there is a reply.
        gr.update(value=tone, visible=has_reply),
        gr.update(visible=has_reply),
        gr.update(value="", visible=has_reply),
        # The draft button only exists where there is a reply to draft.
        gr.update(visible=has_reply),
        gr.update(value="", visible=False),
        # This row's own context, read by nothing but this row's regenerate button.
        {"email": item.email, "decision_summary": item.decision.summary},
    ]


def build_outputs(
    processed: List[ProcessedEmail], total: int, status: str
) -> List[Any]:
    """Assemble the full flat output tuple Gradio expects.

    The second element is the displayed-order email list, kept in a ``gr.State`` so a
    per-row draft button can resolve its row back to the right email even after the
    inbox has been re-sorted mid-run.
    """
    done = len(processed)
    rows: List[Any] = []
    for i in range(MAX_ROWS):
        rows.extend(render_row(processed[i], i + 1) if i < done else blank_row())

    trace = json.dumps(
        [p.to_dict(include_raw=True) for p in processed], indent=2, ensure_ascii=False
    )
    order = [p.email for p in processed]
    return [status, order, overview_html(processed, total, done), trace, *rows]


# =====================================================================
# Main action
# =====================================================================


def run_triage(emails: Optional[List[Dict[str, Any]]]):
    """Generator: streams partial results so judges see per-email progress."""
    emails = emails or []

    if not emails:
        yield build_outputs([], 0, "⚠️ No inbox loaded. Load the demo inbox or connect Gmail first.")
        return

    if ENGINE.agent is None:
        yield build_outputs([], 0, f"❌ Engine unavailable: `{ENGINE.error}`")
        return

    emails = emails[:MAX_ROWS]
    total = len(emails)
    processed: List[ProcessedEmail] = []

    yield build_outputs([], total, f"⏳ Starting triage of {total} emails…")

    for index, email in enumerate(emails, start=1):
        subject = str(email.get("subject", "(no subject)"))[:60]
        try:
            processed.append(ENGINE.agent.process_email(email))
        except Exception:  # noqa: BLE001 - one bad email must not end the run
            traceback.print_exc()
            continue
        yield build_outputs(
            sort_results(processed),
            total,
            f"⏳ Triaged **{index}/{total}** — last: _{subject}_",
        )

    final = sort_results(processed)
    stats = inbox_stats(final)
    backend = ENGINE.llm.active_backend if ENGINE.llm else "unknown"
    yield build_outputs(
        final,
        total,
        f"✅ Done — **{stats['total']}** emails triaged on `{backend}`: "
        f"{stats['URGENT']} urgent, {stats['ACTION_NEEDED']} needing action, "
        f"{stats['replies']} replies drafted, and {stats['tasks']} task / "
        f"{stats['events']} calendar payloads prepared for review. "
        "Nothing has been sent, filed or scheduled.",
    )


def reload_engine(backend: str, model_id: str) -> Tuple[str, str, str]:
    backend = (backend or "auto").strip()
    ENGINE.build(backend=backend, model_id=(model_id or "").strip() or None)
    ok = ENGINE.agent is not None
    message = (
        f"🔄 Engine rebuilt — now running `{ENGINE.llm.active_backend}`."
        if ok
        else f"❌ Engine rebuild failed: `{ENGINE.error}`"
    )
    return status_bar_html(), engine_notes_md(), message


# =====================================================================
# UI
# =====================================================================


def textbox(**kwargs: Any) -> gr.Textbox:
    """Textbox with a copy button, across Gradio 4/5/6 API differences."""
    for extra in ({"buttons": ["copy"]}, {"show_copy_button": True}, {}):
        try:
            return gr.Textbox(**extra, **kwargs)
        except Exception:  # noqa: BLE001 - try the next compatibility shim
            continue
    return gr.Textbox(**kwargs)


def build_ui() -> gr.Blocks:
    initial_emails = load_sample_inbox()

    blocks_kwargs: Dict[str, Any] = {"title": "Gemma-Triage"}
    if not _STYLE_ON_LAUNCH:  # Gradio <= 5 styles the Blocks constructor
        blocks_kwargs.update(css=CUSTOM_CSS, theme=gr.themes.Soft())

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            "# 📥 Gemma-Triage\n"
            "### Smart email agent & workflow automator, powered by Gemma 4\n"
            "Every email is classified, prioritised, summarised, answered with a draft reply, "
            "and mined for tasks and calendar events — by a single model that calls its own tools. "
            "With the Transformers backend that model runs on this device and email content "
            "stays here; the status bar below always names the engine actually in use."
        )

        status_bar = gr.HTML(status_bar_html())

        inbox_state = gr.State(initial_emails)
        #: Emails in the order they are displayed, so a row's draft button can find its own.
        display_state = gr.State([])

        with gr.Row():
            mode = gr.Radio(
                choices=["Demo Inbox (no login)", "Connect Gmail"],
                value="Demo Inbox (no login)",
                label="Inbox source",
                scale=3,
            )
            run_btn = gr.Button("⚡ Run Triage", variant="primary", scale=1)

        with gr.Group(visible=False) as gmail_panel:
            gr.Markdown(
                "**Gmail (optional).** Opens a Google consent screen in your browser and "
                "requests read-only + draft-compose access. Requires a local "
                "`credentials.json`. No send scope is ever requested, and a draft is only "
                "ever created when you press **📧 Create Gmail Draft** on a specific reply. "
                "Not available on the hosted Space — run locally for this path."
            )
            with gr.Row():
                gmail_count = gr.Slider(1, MAX_ROWS, value=10, step=1, label="Messages to fetch")
                gmail_query = gr.Textbox(value="in:inbox", label="Gmail search query", scale=2)
                gmail_btn = gr.Button("🔗 Connect & fetch", scale=1)

        status = gr.Markdown(
            f"📥 Demo inbox loaded — **{len(initial_emails)}** synthetic emails. Press **Run Triage**."
        )
        overview = gr.HTML()

        gr.Markdown("---")

        # --- pre-allocated result rows ---------------------------------
        row_components: List[Any] = []
        draft_bindings: List[Tuple[int, Any, Any, Any]] = []
        regen_bindings: List[Tuple[Any, Any, Any, Any, Any]] = []
        for i in range(MAX_ROWS):
            with gr.Group(visible=False) as group:
                head = gr.HTML()
                with gr.Accordion("Details", open=False) as accordion:
                    detail = gr.HTML()
                    reply = textbox(
                        label="✍️ Suggested reply (editable)",
                        lines=9,
                        visible=False,
                        interactive=True,
                        info=(
                            "Edit freely. Nothing is sent automatically — 🔁 rewrites this "
                            "text in the tone you pick, and 📧 creates a Gmail *draft* from "
                            "exactly what is in the box."
                        ),
                    )
                    with gr.Row():
                        tone_choice = gr.Dropdown(
                            choices=list(tools.VALID_TONES),
                            value=DEFAULT_TONE,
                            label="Reply tone",
                            visible=False,
                            interactive=True,
                            scale=2,
                        )
                        regen_btn = gr.Button(
                            "🔁 Regenerate reply", size="sm", visible=False, scale=1
                        )
                    regen_status = gr.Markdown(visible=False)
                    draft_btn = gr.Button(
                        "📧 Create Gmail Draft", size="sm", visible=False, variant="secondary"
                    )
                    draft_status = gr.Markdown(visible=False)
            #: This row's email + triage summary. Scoped to the row, so the regenerate
            #: button never has to look anything up in a shared list.
            row_state = gr.State(None)
            row_components.extend(
                [
                    group, head, accordion, detail, reply,
                    tone_choice, regen_btn, regen_status,
                    draft_btn, draft_status, row_state,
                ]
            )
            draft_bindings.append((i, draft_btn, reply, draft_status))
            regen_bindings.append((row_state, tone_choice, regen_btn, reply, regen_status))

        with gr.Accordion("🧠 Raw agent trace (the JSON Gemma actually produced)", open=False):
            trace = gr.Code(
                language="json", label="Structured output + the tool calls it produced"
            )

        with gr.Accordion("⚙️ Engine settings & backend transparency", open=False):
            gr.Markdown(
                "`GEMMA_BACKEND=auto` tries **transformers** (on-device) → **ollama** "
                "(on-device, via a local Ollama server on `OLLAMA_BASE_URL`) → "
                "**heuristic** (on-device). It only considers **hf_api** — which sends "
                "email content to an external inference provider — when "
                "`ALLOW_REMOTE_INFERENCE=true`. Selecting `hf_api` here is an explicit "
                "request for remote processing and is always honoured. Change the tier, "
                "then rebuild."
            )
            with gr.Row():
                backend_choice = gr.Dropdown(
                    choices=["auto", "transformers", "ollama", "hf_api", "heuristic"],
                    value=os.environ.get("GEMMA_BACKEND", "auto"),
                    label="Backend",
                )
                model_input = gr.Textbox(
                    value=os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-E4B-it"),
                    label="Model id",
                    scale=2,
                )
                reload_btn = gr.Button("🔄 Rebuild engine")
            reload_status = gr.Markdown()
            engine_notes = gr.Markdown(engine_notes_md())

        gr.Markdown(
            "<div style='opacity:.55;font-size:12px;margin-top:18px;'>"
            "Gemma-Triage · Apache-2.0 code · Gemma model weights are governed by the "
            "<a href='https://ai.google.dev/gemma/terms'>Gemma Terms of Use</a>.<br>"
            "Action lifecycle: <b>Suggested action → Prepared payload → Approved → "
            "Created (external)</b>. Tasks and calendar entries stop at "
            "<b>Prepared payload</b>; a Gmail draft reaches <b>Created (external)</b> only "
            "after you press the button, and is never sent."
            "</div>"
        )

        # --- wiring ----------------------------------------------------
        triage_outputs = [status, display_state, overview, trace, *row_components]

        run_btn.click(fn=run_triage, inputs=[inbox_state], outputs=triage_outputs)

        # One regenerate handler per row, wired to that row and nothing else. The inputs
        # name only this row's state, tone and textbox; the outputs name only this row's
        # textbox and status line. `triage_outputs` appears in neither, so pressing the
        # button reruns the reply-writing step for one email — the inbox is not
        # reclassified, no other row moves, and nothing is drafted or sent.
        for row_state, tone_choice, regen_btn, reply_box, regen_status in regen_bindings:
            regen_btn.click(
                fn=regenerate_handler,
                inputs=[row_state, tone_choice, reply_box],
                outputs=[reply_box, regen_status],
            )

        # One click handler per row. Each closes over its own row index and reads the
        # *edited* textbox, so the draft is whatever the user actually approved.
        for index, button, reply_box, reply_status in draft_bindings:
            button.click(
                fn=(lambda i: lambda emails, text: gr.update(
                    value=create_gmail_draft(emails, i, text), visible=True
                ))(index),
                inputs=[display_state, reply_box],
                outputs=[reply_status],
            )

        mode.change(
            fn=lambda choice: gr.update(visible=choice == "Connect Gmail"),
            inputs=[mode],
            outputs=[gmail_panel],
        ).then(
            fn=lambda choice: load_demo() if choice.startswith("Demo") else ([], "Connect Gmail below."),
            inputs=[mode],
            outputs=[inbox_state, status],
        )

        gmail_btn.click(
            fn=connect_gmail,
            inputs=[gmail_count, gmail_query],
            outputs=[inbox_state, status],
        )

        reload_btn.click(
            fn=reload_engine,
            inputs=[backend_choice, model_input],
            outputs=[status_bar, engine_notes, reload_status],
        )

    return demo


#: Module-level handle. Hugging Face Spaces picks this up, and `python app.py` launches it.
demo = build_ui()


def launch() -> None:
    kwargs: Dict[str, Any] = {
        "server_name": os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        "show_api": False,
    }
    if _STYLE_ON_LAUNCH:  # Gradio >= 6 styles at launch time
        kwargs.update(css=CUSTOM_CSS, theme=gr.themes.Soft())
    demo.queue().launch(**{k: v for k, v in kwargs.items() if k in _LAUNCH_PARAMS})


if __name__ == "__main__":
    launch()
