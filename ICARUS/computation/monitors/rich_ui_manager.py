from __future__ import annotations

from types import TracebackType
from typing import Self

from rich.console import Console
from rich.console import ConsoleRenderable
from rich.console import RenderableType
from rich.console import RichCast
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

try:
    from ICARUS import ICARUS_CONSOLE

    _console = ICARUS_CONSOLE
except ImportError:
    _console = Console()


class RichUIManager:
    """
    Singleton class to manage the rich Live context and dynamic table of rows.
    Allows adding, updating, and removing arbitrary rows (jobs, info, logs, etc.).
    """

    def __init__(
        self,
        refresh_per_second: float = 1.0,
        console: Console | None = None,
    ) -> None:
        self.console = console or _console
        self.refresh_per_second = refresh_per_second
        self._rows: dict[str, RenderableType] = {}
        self._table = Table.grid(expand=True, padding=(0, 1))
        self._live: Live | None = None
        self._entered = False
        self._update_table()

    def _update_table(self) -> None:
        self._table = Table.grid(expand=True, padding=(0, 1))
        for key, renderable in self._rows.items():
            # Each row is a Panel with the key as the title
            self._table.add_row(Panel(renderable, title=key, expand=True))

    def add_row(self, key: str, content: RenderableType) -> None:
        """Add a new row or update an existing one by key."""
        self._rows[key] = content
        self._update_table()
        self.refresh()

    def update_row(self, key: str, content: RenderableType) -> None:
        """Update the content of an existing row."""
        if key in self._rows:
            self._rows[key] = content
            self._update_table()
            self.refresh()
        else:
            raise KeyError(f"Row '{key}' does not exist.")

    def remove_row(self, key: str) -> None:
        """Remove a row by key."""
        if key in self._rows:
            del self._rows[key]
            self._update_table()
            self.refresh()

    def refresh(self) -> None:
        if self._live:
            self._live.update(self._table, refresh=True)

    def __enter__(self) -> Self:
        if not self._entered:
            self._live = Live(
                self._table,
                console=self.console,
                refresh_per_second=self.refresh_per_second,
                screen=False,  # Not fullscreen
                auto_refresh=True,
                transient=True,
                redirect_stderr=False,
                redirect_stdout=False,
            )
            self._live.__enter__()
            self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._rows = {}
        if self._entered and self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._entered = False
            self._live = None

    @property
    def table(self) -> Table:
        return self._table

    @property
    def rows(self) -> dict[str, ConsoleRenderable | RichCast | str]:
        return self._rows

    @classmethod
    def get_instance(cls) -> RichUIManager:
        return cls()
