"""Debate engine — orchestrates the multi-round Conclave of Experts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import re

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from cafm.config_manager import ConfigManager

# One distinct color per agent slot (cycles if more than 6 agents)
AGENT_COLORS = ["cyan", "yellow", "magenta", "blue", "green", "red"]
AGENT_BOLD   = ["bold cyan", "bold yellow", "bold magenta", "bold blue", "bold green", "bold red"]

_SPANISH_RE = re.compile(
    r'[\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1\u00bf\u00a1]|'
    r'\b(que|est\u00e1|es|en|de|la|el|los|las|con|por|para|una|uno|pero|como|m\u00e1s|'
    r'tambi\u00e9n|sobre|muy|tiene|hacer|qu\u00e9|c\u00f3mo|cu\u00e1l|esto|eso|son|hay|si)\b',
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Return 'Spanish' if the text appears to be Spanish, otherwise 'English'."""
    return "Spanish" if len(_SPANISH_RE.findall(text)) >= 2 else "English"
from cafm.context_manager import prepare_messages
from cafm.ollama_client import chat_stream, chat_sync

console = Console()


class DebateEntry:
    """One turn in the debate."""

    __slots__ = ("model", "round_num", "content")

    def __init__(self, model: str, round_num: int, content: str) -> None:
        self.model = model
        self.round_num = round_num
        self.content = content

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "round": self.round_num, "content": self.content}


class DebateEngine:
    """Runs a full multi-agent debate session."""

    def __init__(self, config: ConfigManager) -> None:
        self.config = config
        self.transcript: list[DebateEntry] = []
        self.user_query: str = ""

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, query: str) -> str:
        """Execute the full debate and return the final synthesised answer."""
        self.user_query = query
        self.transcript.clear()

        models = self.config.models
        n_rounds = self.config.rounds
        n_instances = self.config.instances
        prompts = self.config.system_prompts
        ctx_limit = self.config.context_limit
        strategy = self.config.context_strategy
        stream = self.config.data.get("stream_output", True)
        skeptic = self.config.skeptic_agent and n_instances > 1
        skeptic_idx = n_instances - 1  # last agent is the skeptic

        language = detect_language(query)
        lang_note = (
            f"\n\nIMPORTANT: You MUST respond ENTIRELY in {language}. "
            "Do not switch languages under any circumstances."
        )

        # ── Header ────────────────────────────────────────────────────
        console.print()
        console.rule("[bold cyan]Conclave of Experts — Debate Begins[/bold cyan]")
        console.print(f"  Query    : {query}")
        console.print(f"  Language : [bold]{language}[/bold]")
        for i, m in enumerate(models[:n_instances]):
            bold = AGENT_BOLD[i % len(AGENT_BOLD)]
            is_sk = skeptic and i == skeptic_idx
            tag = " [bold red]⚡ SKEPTIC[/bold red]" if is_sk else ""
            console.print(f"  [{bold}]Agent {i+1}[/{bold}]{tag}  : {m}")
        console.print(f"  Rounds   : {n_rounds}  |  Skeptic: [bold]{'ON' if skeptic else 'OFF'}[/bold]")
        console.rule()

        # ── Debate rounds — ALL agents speak in EVERY round ───────────
        for rnd in range(1, n_rounds + 1):
            is_first = rnd == 1
            console.print()
            console.rule(f"[bold yellow]Round {rnd} / {n_rounds}[/bold yellow]")

            for idx in range(n_instances):
                model = models[idx]
                is_this_skeptic = skeptic and idx == skeptic_idx

                if is_this_skeptic and is_first:
                    sp = prompts.get("skeptic_initial_round", prompts.get("skeptic_round", ""))
                elif is_this_skeptic:
                    sp = prompts.get("skeptic_round", prompts.get("debate_round", ""))
                elif is_first:
                    sp = prompts.get("initial_round", "")
                else:
                    sp = prompts.get("debate_round", "")

                messages = prepare_messages(
                    system_prompt=sp + lang_note,
                    user_query=query,
                    transcript_entries=[e.as_dict() for e in self.transcript],
                    context_limit=ctx_limit,
                    strategy=strategy,
                    summary_func=self._make_summary_func(),
                )
                content = self._generate(
                    model, messages, rnd, stream,
                    agent_idx=idx,
                    is_skeptic=is_this_skeptic,
                )
                self.transcript.append(DebateEntry(model, rnd, content))

        # ── Final synthesis — separate step after all rounds ──────────
        console.print()
        console.rule("[bold green]Final Synthesis[/bold green]")
        synth_prompt = prompts.get("final_synthesis", "") + lang_note
        final_model = models[0]
        messages = prepare_messages(
            system_prompt=synth_prompt,
            user_query=query,
            transcript_entries=[e.as_dict() for e in self.transcript],
            context_limit=ctx_limit,
            strategy=strategy,
            summary_func=self._make_summary_func(),
        )
        final_content = self._generate(
            final_model, messages, n_rounds + 1, stream,
            agent_idx=0, is_synthesis=True,
        )
        self.transcript.append(DebateEntry(final_model, n_rounds + 1, final_content))

        # Save log if configured
        if self.config.data.get("save_logs", False):
            self._save_log()

        return final_content

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        model: str,
        messages: list[dict],
        rnd: int,
        stream: bool,
        agent_idx: int = 0,
        is_synthesis: bool = False,
        is_skeptic: bool = False,
    ) -> str:
        """Generate a single model response (streaming or sync)."""
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        bold  = AGENT_BOLD[agent_idx % len(AGENT_BOLD)]

        if is_synthesis:
            border = "green"
            title  = f"[bold green] ✦ Agent {agent_idx+1} — {model} — Final Synthesis [/bold green]"
        elif is_skeptic:
            border = "red"
            title  = (
                f"[bold red] ⚡ SKEPTIC — Agent {agent_idx+1} [{color}]({model})[/{color}]"
                f" — Round {rnd} [/bold red]"
            )
        else:
            border = color
            title  = (
                f"[{bold}] Agent {agent_idx+1} [{color}]— {model}[/{color}] — Round {rnd} [{bold}][/]"
            )

        console.print()
        console.print(Panel("", title=title, border_style=border, padding=(0, 1)))

        text_color = "red" if is_skeptic else ("green" if is_synthesis else color)
        if stream:
            return self._stream_generate(model, messages, text_color)
        else:
            content = chat_sync(model, messages, self.config.context_limit)
            console.print(content, style=text_color)
            return content

    def _stream_generate(self, model: str, messages: list[dict], color: str = "white") -> str:
        """Stream tokens to console and accumulate result."""
        collected: list[str] = []
        buffer = Text(style=color)

        with Live(buffer, console=console, refresh_per_second=12, vertical_overflow="visible") as live:
            for token in chat_stream(model, messages, self.config.context_limit):
                collected.append(token)
                buffer.append(token, style=color)
                live.update(buffer)

        return "".join(collected)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_summary_func(self) -> Callable | None:
        """Return a summary callable for the context manager, or None."""
        summary_model = self.config.data.get("summary_model") or (
            self.config.models[0] if self.config.models else None
        )
        if summary_model is None:
            return None

        def _summarise(messages: list[dict[str, str]]) -> str:
            return chat_sync(summary_model, messages, self.config.context_limit)

        return _summarise

    def _save_log(self) -> None:
        """Persist the full debate transcript as a JSON file."""
        log_dir = Path(self.config.data.get("log_directory", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"debate_{ts}.json"
        payload = {
            "timestamp": ts,
            "query": self.user_query,
            "config": {
                "models": self.config.models,
                "instances": self.config.instances,
                "rounds": self.config.rounds,
                "context_limit": self.config.context_limit,
                "context_strategy": self.config.context_strategy,
            },
            "transcript": [e.as_dict() for e in self.transcript],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        console.print(f"\n[dim]Session log saved to {path}[/dim]")
