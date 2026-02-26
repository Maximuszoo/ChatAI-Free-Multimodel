"""CLI interface — settings menu, query prompt, and main loop."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm

from cafm.config_manager import ConfigManager
from cafm.ollama_client import list_local_models, validate_models, pull_model
from cafm.debate_engine import DebateEngine

console = Console()

BANNER = r"""
  ██████╗ █████╗ ███████╗███╗   ███╗
 ██╔════╝██╔══██╗██╔════╝████╗ ████║
 ██║     ███████║█████╗  ██╔████╔██║
 ██║     ██╔══██║██╔══╝  ██║╚██╔╝██║
 ╚██████╗██║  ██║██║     ██║ ╚═╝ ██║
  ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝
  ChatAI-Free-Multimodel  v1.0
"""


def show_banner() -> None:
    console.print(Panel(BANNER, style="bold cyan", expand=False))


def show_status(cfg: ConfigManager) -> None:
    """Display current configuration summary."""
    table = Table(title="Current Configuration", show_header=False, border_style="cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Instances", str(cfg.instances))
    table.add_row("Rounds", str(cfg.rounds))
    skeptic_on = cfg.skeptic_agent and cfg.instances > 1
    for i, m in enumerate(cfg.models[: cfg.instances], 1):
        is_sk = skeptic_on and i == cfg.instances
        label = f"  Agent {i} Model"
        value = f"{m}  [bold red]⚡ SKEPTIC[/bold red]" if is_sk else m
        table.add_row(label, value)
    table.add_row("Skeptic Agent", "[bold green]ON[/bold green]" if skeptic_on else "[dim]OFF[/dim]")
    table.add_row("Context Limit", f"{cfg.context_limit} tokens")
    table.add_row("Context Strategy", cfg.context_strategy)
    table.add_row("Stream Output", str(cfg.data.get("stream_output", True)))
    table.add_row("Save Logs", str(cfg.data.get("save_logs", True)))
    console.print(table)


def validate_and_fix_models(cfg: ConfigManager) -> bool:
    """Ensure all configured models are available; offer to pull missing ones.

    Returns True if all models are ready, False if unrecoverable.
    """
    models_needed = cfg.models[: cfg.instances]
    available, missing = validate_models(models_needed)

    if not missing:
        console.print("[green]✓ All models available.[/green]")
        return True

    console.print(f"[yellow]⚠ Missing models: {', '.join(missing)}[/yellow]\n")

    local = list_local_models()
    if local:
        console.print("Locally available models:")
        for m in local:
            console.print(f"  • {m}")
        console.print()

    for model in missing:
        choice = Prompt.ask(
            f"Model [bold]{model}[/bold] not found. [P]ull / [R]eplace / [S]kip?",
            choices=["p", "r", "s"],
            default="p",
        )
        if choice == "p":
            if not pull_model(model):
                console.print(f"[red]Could not pull {model}. Please resolve manually.[/red]")
                return False
        elif choice == "r":
            if not local:
                console.print("[red]No local models available to replace with.[/red]")
                return False
            replacement = _pick_model_by_number(local, f"Replace {model} with")
            if replacement is None:
                return False
            idx = cfg.models.index(model)
            cfg.set_model_at(idx, replacement)
            console.print(f"[green]Replaced {model} → {replacement}[/green]")
        else:
            console.print(f"[dim]Skipping {model}.[/dim]")
            return False

    return True


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pick_model_by_number(model_list: list[str], prompt_label: str = "Select model") -> str | None:
    """Show a numbered list of models and return the user's choice."""
    if not model_list:
        console.print("[red]No models available.[/red]")
        return None
    console.print()
    for idx, name in enumerate(model_list, 1):
        console.print(f"  [bold cyan]{idx:>3}[/bold cyan]) {name}")
    console.print()
    valid = [str(i) for i in range(1, len(model_list) + 1)]
    pick = Prompt.ask(prompt_label, choices=valid)
    return model_list[int(pick) - 1]


# ------------------------------------------------------------------
# Settings Menu
# ------------------------------------------------------------------

def settings_menu(cfg: ConfigManager) -> None:
    """Interactive settings editor."""
    while True:
        console.print()
        console.rule("[bold]Settings Menu[/bold]")
        console.print("  [1] Set number of instances")
        console.print("  [2] Set number of rounds")
        console.print("  [3] Assign models to instances")
        console.print("  [4] Set context limit")
        console.print("  [5] Toggle context strategy (sliding_window / summary)")
        console.print("  [6] Toggle stream output")
        console.print("  [7] Toggle save logs")
        sk_state = "[bold green]ON[/bold green]" if cfg.skeptic_agent else "[dim]OFF[/dim]"
        console.print(f"  [8] Toggle skeptic agent (last agent refutes others) — currently {sk_state}")
        console.print("  [0] Back to main menu")
        console.print()

        choice = Prompt.ask("Select option", choices=["0","1","2","3","4","5","6","7","8"], default="0")

        if choice == "0":
            break
        elif choice == "1":
            n = IntPrompt.ask("Number of instances (agents)", default=cfg.instances)
            cfg.set("instances", max(1, n))
            cfg.ensure_models_match_instances()
        elif choice == "2":
            r = IntPrompt.ask("Number of rounds", default=cfg.rounds)
            cfg.set("rounds", max(1, r))
        elif choice == "3":
            local = list_local_models()
            if not local:
                console.print("[red]No models found in Ollama. Pull some first.[/red]")
                continue
            for i in range(cfg.instances):
                current = cfg.models[i] if i < len(cfg.models) else "—"
                console.print(f"\n  Agent {i+1} (current: [bold]{current}[/bold])")
                picked = _pick_model_by_number(local, f"  Agent {i+1} model")
                if picked:
                    cfg.set_model_at(i, picked)
                    console.print(f"  [green]Agent {i+1} → {picked}[/green]")
        elif choice == "4":
            lim = IntPrompt.ask("Context token limit", default=cfg.context_limit)
            cfg.set("context_limit", max(512, lim))
        elif choice == "5":
            cur = cfg.context_strategy
            new = "summary" if cur == "sliding_window" else "sliding_window"
            cfg.set("context_strategy", new)
            console.print(f"  Strategy changed to: [bold]{new}[/bold]")
        elif choice == "6":
            cur = cfg.data.get("stream_output", True)
            cfg.set("stream_output", not cur)
            console.print(f"  Stream output: [bold]{not cur}[/bold]")
        elif choice == "7":
            cur = cfg.data.get("save_logs", True)
            cfg.set("save_logs", not cur)
            console.print(f"  Save logs: [bold]{not cur}[/bold]")
        elif choice == "8":
            cur = cfg.skeptic_agent
            cfg.set("skeptic_agent", not cur)
            state = "[bold green]ON[/bold green]" if not cur else "[dim]OFF[/dim]"
            console.print(f"  Skeptic agent: {state}")
            if not cur and cfg.instances < 2:
                console.print("  [yellow]Note: needs at least 2 agents to have a skeptic.[/yellow]")

    show_status(cfg)


# ------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------

def main_loop() -> None:
    """Primary CLI loop."""
    cfg = ConfigManager()
    show_banner()
    show_status(cfg)
    console.print()

    # Validate models
    if not validate_and_fix_models(cfg):
        console.print("[red]Cannot start debate — model issues unresolved.[/red]")
        console.print("Run again after pulling the required models with 'ollama pull <model>'.\n")
        sys.exit(1)

    cfg.ensure_models_match_instances()
    console.print("\n[bold green]System Ready.[/bold green]\n")

    engine = DebateEngine(cfg)

    while True:
        console.print("[bold cyan]Commands:[/bold cyan] type a query to start a debate, "
                       "[bold]/settings[/bold] to configure, [bold]/quit[/bold] to exit.\n")
        user_input = Prompt.ask("[bold]Enter your query[/bold]").strip()

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if user_input.lower() in ("/settings", "/config", "/s"):
            settings_menu(cfg)
            # Re-validate after settings change
            validate_and_fix_models(cfg)
            cfg.ensure_models_match_instances()
            engine = DebateEngine(cfg)
            continue

        engine.run(user_input)
        console.print()
