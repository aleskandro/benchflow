from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for bootstrap environments
    Console = None
    RichHandler = None
    Panel = None
    Table = None
    RICH_AVAILABLE = False


_console = (
    Console(stderr=False, soft_wrap=True, highlight=False) if RICH_AVAILABLE else None
)
_error_console = (
    Console(stderr=True, soft_wrap=True, highlight=False) if RICH_AVAILABLE else None
)
_message_prefix: ContextVar[str] = ContextVar("benchflow_ui_message_prefix", default="")


def _prefix_message(message: str) -> str:
    prefix = _message_prefix.get().strip()
    if not prefix:
        return message
    if not message:
        return prefix
    return f"{prefix} {message}"


def emit(message: str = "", *, stderr: bool = False, end: str = "\n") -> None:
    if RICH_AVAILABLE:
        console = _error_console if stderr else _console
        console.print(message, end=end, markup=False, highlight=False)
        return
    stream = sys.stderr if stderr else sys.stdout
    print(message, file=stream, end=end)


def rule(title: str) -> None:
    title = _prefix_message(title)
    if RICH_AVAILABLE:
        _console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        emit(f"== {title} ==")


def panel(title: str, rows: Iterable[tuple[str, str]]) -> None:
    title = _prefix_message(title)
    items = [(label, value) for label, value in rows if value]
    if not items:
        return
    if RICH_AVAILABLE:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold white")
        table.add_column(style="white")
        for label, value in items:
            table.add_row(label, value)
        _console.print(Panel.fit(table, title=title, border_style="cyan"))
        return
    emit(title)
    for label, value in items:
        emit(f"  {label}: {value}")


def step(message: str) -> None:
    message = _prefix_message(message)
    if RICH_AVAILABLE:
        _console.print(f"[bold cyan]›[/bold cyan] {message}")
    else:
        emit(f"> {message}")


def detail(message: str) -> None:
    message = _prefix_message(message)
    if RICH_AVAILABLE:
        _console.print(f"  [dim]{message}[/dim]")
    else:
        emit(f"  {message}")


def success(message: str) -> None:
    message = _prefix_message(message)
    if RICH_AVAILABLE:
        _console.print(f"[bold green]✓[/bold green] {message}")
    else:
        emit(f"OK {message}")


def warning(message: str) -> None:
    message = _prefix_message(message)
    if RICH_AVAILABLE:
        _error_console.print(f"[bold yellow]![/bold yellow] {message}")
    else:
        emit(f"WARNING: {message}", stderr=True)


def error(message: str) -> None:
    message = _prefix_message(message)
    if RICH_AVAILABLE:
        _error_console.print(f"[bold red]error[/bold red] {message}")
    else:
        emit(f"ERROR: {message}", stderr=True)


@contextmanager
def ui_scope(prefix: str):
    normalized = str(prefix).strip()
    previous = _message_prefix.get().strip()
    combined = normalized if not previous else f"{previous} {normalized}"
    token = _message_prefix.set(combined)
    try:
        yield
    finally:
        _message_prefix.reset(token)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_benchflow_configured", False):
        root.setLevel(level.upper())
        return

    root.handlers.clear()
    if RICH_AVAILABLE:
        handler = RichHandler(
            show_time=False,
            show_level=True,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )
        logging.basicConfig(
            level=level.upper(), format="%(message)s", handlers=[handler]
        )
    else:
        logging.basicConfig(level=level.upper(), format="%(levelname)s: %(message)s")
    root._benchflow_configured = True  # type: ignore[attr-defined]
