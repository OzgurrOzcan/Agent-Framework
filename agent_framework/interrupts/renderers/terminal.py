"""TerminalRenderer — interactive prompts using rich + the stdlib `input()`.

Permission ASKs render with the tool name, arguments, and category in a
color-coded panel. Clarification questions with options render as numbered
choices; free-text input is accepted when `allow_free_text=True`.
"""

from __future__ import annotations

import asyncio
import json

from agent_framework.interrupts.models import (
    InterruptClass,
    InterruptEvent,
    InterruptResponse,
)
from agent_framework.interrupts.renderers.base import UIRenderer


class TerminalRenderer(UIRenderer):
    def __init__(self, *, no_color: bool = False) -> None:
        self._no_color = no_color
        try:
            from rich.console import Console
            self._console = Console(no_color=no_color)
            self._rich = True
        except Exception:
            self._console = None
            self._rich = False

    async def render(self, event: InterruptEvent) -> InterruptResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._render_blocking, event)

    # ----- private blocking renderer (runs in executor) -----
    def _render_blocking(self, event: InterruptEvent) -> InterruptResponse:
        if event.interrupt_class == InterruptClass.PERMISSION:
            return self._render_permission(event)
        return self._render_clarification(event)

    def _render_permission(self, event: InterruptEvent) -> InterruptResponse:
        cat = event.permission_category.value if event.permission_category else "?"
        title = f"Permission requested: {event.tool_name}  [category={cat}]"
        body = json.dumps(event.tool_args, indent=2, default=str)
        if self._rich and self._console is not None:
            from rich.panel import Panel
            style = {
                "info": "cyan",
                "warning": "yellow",
                "critical": "red bold",
            }.get(event.urgency, "cyan")
            self._console.print(Panel.fit(body, title=title, border_style=style))
        else:
            print(f"\n--- {title} ---\n{body}\n")
        while True:
            raw = input("Approve? [y]es / [n]o / [a]lways for these args: ").strip().lower()
            if raw in ("y", "yes"):
                return InterruptResponse(approved=True)
            if raw in ("n", "no"):
                return InterruptResponse(
                    approved=False,
                    feedback="User denied this action. Try a different approach.",
                )
            if raw in ("a", "always"):
                # The resolver caches `ASK_ONCE` approvals automatically; for
                # `ASK` we still only approve this one call. Treat "always"
                # like "yes" here — the cache lookup is upstream.
                return InterruptResponse(approved=True)

    def _render_clarification(self, event: InterruptEvent) -> InterruptResponse:
        title = event.question or "(no question)"
        if self._rich and self._console is not None:
            from rich.panel import Panel
            self._console.print(Panel.fit(title, title="Agent needs input", border_style="cyan"))
        else:
            print(f"\n[?] {title}\n")
        if event.options:
            for i, opt in enumerate(event.options, 1):
                print(f"  {i}) {opt}")
            if event.allow_free_text:
                print("  (or type a custom answer)")
            while True:
                raw = input("> ").strip()
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(event.options):
                        return InterruptResponse(
                            approved=True,
                            answer=event.options[idx],
                            selected_option=event.options[idx],
                        )
                if event.allow_free_text and raw:
                    return InterruptResponse(approved=True, answer=raw)
        else:
            raw = input("> ")
            return InterruptResponse(approved=True, answer=raw)
