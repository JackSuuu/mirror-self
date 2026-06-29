"""
Textual TUI — immersive self-reflection interface for mirror-self.

Layout:
  ┌─ header (mode tabs) ──────────────────────────────────┐
  │  chat history (left)   │  memory panel (right)         │
  │  loading / streaming   │  retrieved journal entries    │
  ├─ input area ──────────────────────────────────────────┤
  └─ footer ──────────────────────────────────────────────┘
"""
from __future__ import annotations

from typing import Optional

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Header,
    Markdown,
    RichLog,
    Static,
    TextArea,
)

from mirrorself import config as cfg
from mirrorself.core import llm, prompts, retriever
from mirrorself.core.retriever import RetrievedEntry


# ── CSS ────────────────────────────────────────────────────────────────────────

APP_CSS = """
Screen {
    background: #0f1117;
}

/* ── Mode bar ── */
#mode-bar {
    height: 3;
    background: #1a1d2e;
    border-bottom: solid #2d3561;
    padding: 0 1;
    layout: horizontal;
    align: left middle;
}

.mode-btn {
    background: transparent;
    color: #606880;
    border: none;
    padding: 0 2;
    height: 3;
    content-align: center middle;
}

.mode-btn:hover {
    color: #c0c8f0;
    background: #252840;
}

.mode-btn.active {
    color: #f0b429;
    text-style: bold;
    background: #252840;
    border-bottom: solid #f0b429;
}

/* ── Main area ── */
#main {
    height: 1fr;
}

/* ── Chat column ── */
#chat-col {
    width: 2fr;
    border-right: solid #2d3561;
}

#chat-log {
    height: 1fr;
    background: #0f1117;
    padding: 1 2;
    scrollbar-color: #2d3561;
    scrollbar-background: #0f1117;
}

#streaming-box {
    height: auto;
    max-height: 45%;
    background: #0f1117;
    padding: 0 2 1 2;
    border-top: solid #1e2240;
}

/* ── Loading widget ── */
LoadingWidget {
    height: 1;
    padding: 0 0;
    display: none;
    color: #808898;
    background: #0f1117;
}

LoadingWidget.visible {
    display: block;
}

/* ── Memory panel ── */
#memory-panel {
    width: 1fr;
    min-width: 28;
    max-width: 45;
    background: #11131f;
    padding: 1;
}

#memory-title {
    color: #606880;
    text-style: italic;
    padding: 0 1;
    height: 1;
}

#memory-scroll {
    height: 1fr;
    scrollbar-color: #2d3561;
    scrollbar-background: #11131f;
}

.memory-entry {
    background: #1a1d2e;
    border: solid #2d3561;
    margin-bottom: 1;
    padding: 1;
    color: #9098c0;
}

/* ── Input area ── */
#input-area {
    height: 5;
    background: #1a1d2e;
    border-top: solid #2d3561;
}

#mode-hint {
    height: 1;
    padding: 0 2;
    color: #3a3f60;
    background: #1a1d2e;
}

ChatInput {
    background: #1e2140;
    color: #e0e4f8;
    border: none;
    height: 4;
    padding: 0 2;
    scrollbar-color: transparent;
    scrollbar-background: transparent;
}

ChatInput:focus {
    border-top: solid #f0b429;
}

ChatInput > .text-area--cursor {
    background: #f0b429;
    color: #0f1117;
}

ChatInput > .text-area--selection {
    background: #3d4170;
}

ChatInput > .text-area--cursor-line {
    background: #1e2140;
}
"""


# ── Loading widget (animated braille spinner) ──────────────────────────────────

_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class LoadingWidget(Static):
    """Animated braille-spinner loading indicator."""

    def __init__(self, **kwargs) -> None:
        super().__init__("", markup=True, **kwargs)
        self._frame: int = 0
        self._msg: str = ""
        self._timer = None

    def start(self, msg: str) -> None:
        self._msg = msg
        self._frame = 0
        self.add_class("visible")
        if self._timer:
            self._timer.stop()
        self._timer = self.set_interval(0.08, self._tick)

    def stop(self) -> None:
        if self._timer:
            self._timer.stop()
            self._timer = None
        self._msg = ""
        self.remove_class("visible")
        self.update("")

    def _tick(self) -> None:
        frame = _FRAMES[self._frame % len(_FRAMES)]
        self.update(f"[bold #f0b429]{frame}[/]  [dim]{self._msg}[/]")
        self._frame += 1


# ── IME-safe chat input (TextArea-based) ───────────────────────────────────────

class ChatInput(TextArea):
    """
    Chinese/Japanese/Korean IME-compatible input widget.
    TextArea handles IME composition (sent as paste events) correctly.
    Enter submits; Shift+Enter inserts a newline.
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, placeholder: str = "", **kwargs) -> None:
        super().__init__(
            "",
            show_line_numbers=False,
            tab_behavior="focus",
            soft_wrap=True,
            placeholder=placeholder,
            **kwargs,
        )

    def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.load_text("")
        elif event.key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.insert("\n")


# ── Memory entry widget ────────────────────────────────────────────────────────

class MemoryEntry(Static):
    def __init__(self, entry: RetrievedEntry) -> None:
        preview = entry.text[:200].replace("\n", " ")
        if len(entry.text) > 200:
            preview += "…"
        super().__init__(
            f"[bold #f0b429]{entry.date_label}[/]\n[#606880]{preview}[/]",
            markup=True,
            classes="memory-entry",
        )


# ── Mode bar ──────────────────────────────────────────────────────────────────

MODES = [
    ("chat",    "Chat  F1"),
    ("reflect", "Reflect  F2"),
    ("compare", "Timeline  F3"),
    ("pattern", "Patterns  F4"),
]

MODE_HINTS = {
    "chat":    "Ask anything — Enter to send, Shift+Enter for newline",
    "reflect": "Reflection questions generated automatically",
    "compare": "Type a topic to compare across all years",
    "pattern": "Describe your current state — find historical parallels",
}


class ModeBar(Widget):
    def compose(self) -> ComposeResult:
        for mode_id, label in MODES:
            yield Static(label, id=f"mode-{mode_id}", classes="mode-btn", markup=False)

    def set_active(self, mode: str) -> None:
        for mode_id, _ in MODES:
            btn = self.query_one(f"#mode-{mode_id}", Static)
            if mode_id == mode:
                btn.add_class("active")
            else:
                btn.remove_class("active")


# ── Main app ──────────────────────────────────────────────────────────────────

class MirrorSelfApp(App):
    CSS = APP_CSS
    TITLE = "mirror-self"
    SUB_TITLE = "the observer"

    BINDINGS = [
        Binding("f1", "set_mode('chat')",    "Chat",     show=True),
        Binding("f2", "set_mode('reflect')", "Reflect",  show=True),
        Binding("f3", "set_mode('compare')", "Timeline", show=True),
        Binding("f4", "set_mode('pattern')", "Patterns", show=True),
        Binding("ctrl+q", "quit",       "Quit",  show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=False),
    ]

    mode: reactive[str] = reactive("chat")
    _streaming: bool = False

    def __init__(self, conf: dict, initial_mode: str = "chat") -> None:
        super().__init__()
        self.conf = conf
        self._initial_mode = initial_mode
        self._history: list[dict] = []

    # ── Compose ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ModeBar(id="mode-bar")
        with Horizontal(id="main"):
            with Vertical(id="chat-col"):
                yield RichLog(id="chat-log", markup=True, wrap=True, highlight=False)
                with ScrollableContainer(id="streaming-box"):
                    yield LoadingWidget(id="loading-widget")
                    yield Markdown("", id="streaming-md")
            with Vertical(id="memory-panel"):
                yield Static("📎 Memories", id="memory-title", markup=True)
                yield ScrollableContainer(id="memory-scroll")
        with Vertical(id="input-area"):
            yield Static(MODE_HINTS["chat"], id="mode-hint", markup=False)
            yield ChatInput(placeholder=MODE_HINTS["chat"], id="user-input")
        yield Footer()

    def on_mount(self) -> None:
        self.mode = self._initial_mode
        self.query_one(ModeBar).set_active(self.mode)
        self.query_one("#user-input", ChatInput).focus()
        self._print_welcome()
        if self.mode == "reflect":
            self.run_reflect_mode()

    def _print_welcome(self) -> None:
        name = self.conf.get("user_name", "User")
        log = self.query_one("#chat-log", RichLog)
        log.write(
            f"[bold #f0b429]mirror-self[/] [#606880]— the observer is ready[/]\n"
            f"[#404860]Everything {name} has written — read, indexed, ready.[/]\n"
        )

    # ── Mode switching ─────────────────────────────────────────────────────────

    def action_set_mode(self, mode: str) -> None:
        if self._streaming:
            return
        self.mode = mode
        self.query_one(ModeBar).set_active(mode)
        hint = MODE_HINTS.get(mode, "")
        self.query_one("#mode-hint", Static).update(hint)
        inp = self.query_one("#user-input", ChatInput)
        inp.placeholder = hint
        log = self.query_one("#chat-log", RichLog)
        labels = {
            "chat":    "Chat",
            "reflect": "Reflect",
            "compare": "Timeline",
            "pattern": "Patterns",
        }
        log.write(f"\n[#2d3561]── [/][#f0b429]{labels.get(mode, mode)}[/][#2d3561] ──[/]\n")
        inp.focus()
        if mode == "reflect":
            self.run_reflect_mode()

    # ── Input handling ─────────────────────────────────────────────────────────

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        if not text or self._streaming:
            return
        self._handle_send(text)

    def _handle_send(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        display = text if len(text) < 200 else text[:197] + "..."
        log.write(f"\n[bold #a0c0ff]You[/]  [#e0e4f8]{display}[/]\n")

        if self.mode == "compare":
            self._run_compare(text)
        elif self.mode == "pattern":
            self._run_pattern(text)
        else:
            self._run_chat(text)

    # ── Loading helpers ────────────────────────────────────────────────────────

    def _loader_start(self, msg: str) -> None:
        self.query_one("#loading-widget", LoadingWidget).start(msg)

    def _loader_stop(self) -> None:
        self.query_one("#loading-widget", LoadingWidget).stop()

    def _clear_streaming(self) -> None:
        self.query_one("#streaming-md", Markdown).update("")
        self._loader_stop()

    # ── Chat mode ─────────────────────────────────────────────────────────────

    @work(thread=True)
    def _run_chat(self, text: str) -> None:
        self._streaming = True
        self.call_from_thread(self._clear_streaming)
        self.call_from_thread(self._loader_start, "Retrieving memories…")

        entries = retriever.search(text, self.conf)
        self.call_from_thread(self._update_memory_panel, entries)
        self.call_from_thread(self._loader_start, "Observer thinking…")

        messages = prompts.chat_messages(self._history, text, entries, self.conf)
        full = self._stream_to_ui(messages)

        self._history.append({"role": "user",      "content": text})
        self._history.append({"role": "assistant", "content": full})
        if len(self._history) > 12:
            self._history = self._history[-12:]

        self.call_from_thread(self._finalize_response, full)
        self._streaming = False

    # ── Reflect mode ──────────────────────────────────────────────────────────

    @work(thread=True)
    def run_reflect_mode(self) -> None:
        self._streaming = True
        self.call_from_thread(self._clear_streaming)
        self.call_from_thread(self._loader_start, "Reading recent entries…")

        recent = retriever.recent_months(self.conf, n_months=3)
        self.call_from_thread(self._update_memory_panel, recent[:6])
        self.call_from_thread(self._loader_start, "Generating reflection questions…")

        messages = prompts.reflect_prompt(recent, self.conf)
        full = self._stream_to_ui(messages)
        self.call_from_thread(self._finalize_response, full)
        self._streaming = False

    # ── Compare mode ──────────────────────────────────────────────────────────

    @work(thread=True)
    def _run_compare(self, topic: str) -> None:
        self._streaming = True
        self.call_from_thread(self._clear_streaming)
        self.call_from_thread(self._loader_start, f"Cross-year search: {topic[:20]}…")

        stats = self.conf.get("_stats", {})
        years = stats.get("years", list(range(2020, 2027)))
        entries = retriever.search_by_year(topic, self.conf, years, per_year=2)
        self.call_from_thread(self._update_memory_panel, entries)
        self.call_from_thread(self._loader_start, "Analysing timeline…")

        messages = prompts.compare_messages(topic, entries, self.conf)
        full = self._stream_to_ui(messages)
        self.call_from_thread(self._finalize_response, full)
        self._streaming = False

    # ── Pattern mode ──────────────────────────────────────────────────────────

    @work(thread=True)
    def _run_pattern(self, current_text: str) -> None:
        self._streaming = True
        self.call_from_thread(self._clear_streaming)
        self.call_from_thread(self._loader_start, "Finding similar past moments…")

        similar = retriever.search_emotional_low(current_text, self.conf, top_k=4)
        self.call_from_thread(self._update_memory_panel, similar)
        self.call_from_thread(self._loader_start, "Identifying pattern…")

        messages = prompts.pattern_messages(current_text, similar, self.conf)
        full = self._stream_to_ui(messages)
        self.call_from_thread(self._finalize_response, full)
        self._streaming = False

    # ── Streaming ─────────────────────────────────────────────────────────────

    def _stream_to_ui(self, messages: list[dict]) -> str:
        """Stream LLM tokens to the Markdown widget. Thread-safe via call_from_thread."""
        full = ""
        streaming_md = self.query_one("#streaming-md", Markdown)
        first_token = True

        for token in llm.stream_chat(messages, self.conf, max_tokens=1200):
            if first_token:
                first_token = False
                self.call_from_thread(self._loader_stop)
            full += token
            if len(full) % 5 == 0 or token in ("。", "！", "？", ".", "!", "?", "\n"):
                captured = full
                self.call_from_thread(streaming_md.update, captured)

        self.call_from_thread(streaming_md.update, full)
        return full

    def _finalize_response(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("\n[bold #f0b429]Observer[/]")
        log.write(text + "\n")
        log.scroll_end(animate=False)
        self.query_one("#streaming-md", Markdown).update("")
        self._loader_stop()

    # ── Memory panel ──────────────────────────────────────────────────────────

    def _update_memory_panel(self, entries: list[RetrievedEntry]) -> None:
        scroll = self.query_one("#memory-scroll", ScrollableContainer)
        scroll.remove_children()
        if not entries:
            scroll.mount(Static("[#404860]No memories found[/]", markup=True, classes="memory-entry"))
            return
        for entry in entries:
            scroll.mount(MemoryEntry(entry))

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        if self._streaming:
            return
        self.query_one("#chat-log", RichLog).clear()
        self._history = []
        self._print_welcome()

    def action_quit(self) -> None:
        self.exit()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(mode: str = "chat") -> None:
    conf = cfg.load()
    from mirrorself.core.indexer import collection_stats
    conf["_stats"] = collection_stats(conf)
    MirrorSelfApp(conf=conf, initial_mode=mode).run()
