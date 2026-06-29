"""
CLI entry point for mirror-self.

Commands:
  mirror-self init     — index your journal into ChromaDB
  mirror-self chat     — open TUI in free-chat mode
  mirror-self reflect  — open TUI with today's reflection questions
  mirror-self compare  — open TUI in timeline-comparison mode
  mirror-self pattern  — open TUI in pattern-warning mode
  mirror-self status   — show index stats and config
  mirror-self config   — get/set configuration values
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from mirrorself import config as cfg

app = typer.Typer(
    name="mirror-self",
    help="Turn your personal journal into an AI reflection partner.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_index_exists() -> bool:
    from mirrorself.core.indexer import get_collection
    try:
        col = get_collection(cfg.chroma_path())
        return col.count() > 0
    except Exception:
        return False


def _require_index() -> None:
    if not _check_index_exists():
        console.print(
            "[red]No journal index found.[/] "
            "Run [bold]mirror-self init --journal /path/to/journal[/] first."
        )
        raise typer.Exit(1)


def _check_ollama(conf: dict) -> None:
    from mirrorself.core.llm import check_ollama
    ok, msg = check_ollama(conf)
    if not ok:
        console.print(f"[red]Ollama check failed:[/] {msg}")
        raise typer.Exit(1)


# ── init ──────────────────────────────────────────────────────────────────────

@app.command()
def init(
    journal: Optional[Path] = typer.Option(
        None, "--journal", "-j",
        help="Path to your journal directory (saved for future runs)",
        exists=True, file_okay=False, dir_okay=True, resolve_path=True,
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n",
        help="Your name — used to personalise the AI observer (default: User)",
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l",
        help='Language hint for responses: "auto" | "Chinese" | "English" | "mixed" | any description',
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d",
        help='Short description of your journal, e.g. "4 years of daily notes, Chinese/English"',
    ),
    embed_model: Optional[str] = typer.Option(
        None, "--embed-model",
        help="Ollama embedding model (default: nomic-embed-text)",
    ),
    llm_model: Optional[str] = typer.Option(
        None, "--llm-model",
        help="Ollama LLM model (default: qwen2.5:latest)",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-index all entries (ignores existing index)",
    ),
) -> None:
    """Index your journal and configure the AI observer."""
    conf = cfg.load()

    # Persist any supplied values
    if journal:
        conf["journal_path"] = str(journal)
    if name:
        conf["user_name"] = name
    if language:
        conf["language_hint"] = language
    if description:
        conf["journal_description"] = description
    if embed_model:
        conf["embed_model"] = embed_model
    if llm_model:
        conf["llm_model"] = llm_model
    cfg.save(conf)

    # Require journal path
    if not conf.get("journal_path"):
        console.print("[red]Please provide --journal /path/to/journal[/]")
        raise typer.Exit(1)

    journal_path = Path(conf["journal_path"])
    console.print(f"\n[bold]mirror-self init[/]")
    console.print(f"  Journal:     [cyan]{journal_path}[/]")
    console.print(f"  Name:        [cyan]{conf['user_name']}[/]")
    console.print(f"  Language:    [cyan]{conf['language_hint']}[/]")
    console.print(f"  Embed model: [cyan]{conf['embed_model']}[/]")
    console.print(f"  LLM model:   [cyan]{conf['llm_model']}[/]")
    console.print(f"  ChromaDB:    [cyan]{cfg.chroma_path()}[/]\n")

    from mirrorself.core.llm import check_ollama
    ok, msg = check_ollama(conf)
    if not ok:
        console.print(f"[red]{msg}[/]")
        raise typer.Exit(1)

    from mirrorself.core.indexer import parse_all
    console.print("[dim]Parsing journal files...[/]")
    entries = parse_all(journal_path)
    if not entries:
        console.print("[red]No entries found. Check your journal path.[/]")
        raise typer.Exit(1)
    years = sorted(set(e.year for e in entries))
    console.print(
        f"  Found [bold]{len(entries)}[/] entries across "
        f"[bold]{len(years)}[/] years: {', '.join(str(y) for y in years)}\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding + indexing", total=len(entries))

        def on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done)

        from mirrorself.core.indexer import index_journal
        new_count, total_count = index_journal(
            journal_path, conf, on_progress=on_progress, force=force
        )
        progress.update(task, completed=total_count)

    if new_count == 0:
        console.print("[green]Index already up to date.[/] Use [bold]--force[/] to re-index.")
    else:
        console.print(
            f"\n[bold green]Done![/] Indexed [bold]{new_count}[/] new entries "
            f"([bold]{total_count}[/] total)."
        )
    console.print("\nRun [bold]mirror-self chat[/] to start reflecting.\n")


# ── TUI commands ──────────────────────────────────────────────────────────────

@app.command()
def chat() -> None:
    """Open the TUI in free-chat mode."""
    _require_index()
    _check_ollama(cfg.load())
    from mirrorself.tui.app import run
    run(mode="chat")


@app.command()
def reflect() -> None:
    """Open the TUI with auto-generated today's reflection questions."""
    _require_index()
    _check_ollama(cfg.load())
    from mirrorself.tui.app import run
    run(mode="reflect")


@app.command()
def compare(
    topic: Optional[str] = typer.Argument(None, help="Topic to compare across years"),
) -> None:
    """Open the TUI in timeline-comparison mode."""
    _require_index()
    _check_ollama(cfg.load())
    from mirrorself.tui.app import run
    run(mode="compare")


@app.command()
def pattern() -> None:
    """Open the TUI in pattern-warning mode — describe your current state."""
    _require_index()
    _check_ollama(cfg.load())
    from mirrorself.tui.app import run
    run(mode="pattern")


# ── status ────────────────────────────────────────────────────────────────────

@app.command()
def status() -> None:
    """Show current configuration and index statistics."""
    conf = cfg.load()

    table = Table(title="mirror-self status", show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="dim")
    table.add_column("value", style="cyan")

    table.add_row("Journal path",     conf.get("journal_path") or "[red]not set[/]")
    table.add_row("Name",             conf.get("user_name", "User"))
    table.add_row("Language hint",    conf.get("language_hint", "auto"))
    table.add_row("Journal desc.",    conf.get("journal_description") or "—")
    table.add_row("LLM model",        conf.get("llm_model", ""))
    table.add_row("Embed model",      conf.get("embed_model", ""))
    table.add_row("Ollama URL",       conf.get("ollama_base_url", ""))
    table.add_row("ChromaDB path",    str(cfg.chroma_path()))

    console.print()
    console.print(table)

    from mirrorself.core.indexer import collection_stats
    stats = collection_stats(conf)
    if stats["count"] > 0:
        console.print(
            f"\n  [green]{stats['count']}[/] entries indexed across years: "
            f"[bold]{', '.join(str(y) for y in stats['years'])}[/]"
        )
    else:
        console.print("\n  [yellow]No entries indexed yet. Run [bold]mirror-self init[/].[/]")

    from mirrorself.core.llm import check_ollama
    ok, msg = check_ollama(conf)
    icon = "[green]✓[/]" if ok else "[red]✗[/]"
    console.print(f"  Ollama: {icon} {msg}\n")


# ── config ────────────────────────────────────────────────────────────────────

@app.command(name="config")
def config_cmd(
    journal:     Optional[str] = typer.Option(None, "--journal"),
    name:        Optional[str] = typer.Option(None, "--name"),
    language:    Optional[str] = typer.Option(None, "--language"),
    description: Optional[str] = typer.Option(None, "--description"),
    llm_model:   Optional[str] = typer.Option(None, "--llm-model"),
    embed_model: Optional[str] = typer.Option(None, "--embed-model"),
    ollama_url:  Optional[str] = typer.Option(None, "--ollama-url"),
    show:        bool          = typer.Option(False, "--show"),
) -> None:
    """Get or set configuration values."""
    conf = cfg.load()
    changed = False

    for key, val in [
        ("journal_path",        journal),
        ("user_name",           name),
        ("language_hint",       language),
        ("journal_description", description),
        ("llm_model",           llm_model),
        ("embed_model",         embed_model),
        ("ollama_base_url",     ollama_url),
    ]:
        if val is not None:
            conf[key] = val
            changed = True

    if changed:
        cfg.save(conf)
        console.print("[green]Config saved.[/]")

    if show or not changed:
        import json
        safe = {k: v for k, v in conf.items() if not k.startswith("_")}
        console.print_json(json.dumps(safe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
