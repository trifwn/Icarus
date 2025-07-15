from __future__ import annotations

import builtins
import logging
import multiprocessing as mp
import sys
from logging.handlers import QueueHandler
from logging.handlers import QueueListener
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install

if TYPE_CHECKING:
    from ICARUS.computation.core import QueueLike

# Detect if running in a Jupyter Notebook environment
try:
    from IPython.core.getipython import get_ipython

    if get_ipython() is not None:
        IN_JUPYTER = True
    else:
        IN_JUPYTER = False
except ImportError:
    IN_JUPYTER = False


ICARUS_THEME = Theme(
    {
        # Logging levels
        "logging.level.debug": "dim cyan",
        "logging.level.info": "bold cyan",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        # Custom ICARUS styles
        "title": "bold blue underline",
        "section": "bold white on rgb(30,30,30)",
        "dimmed": "dim",
        "value": "bold green",
        "gradient": "italic magenta",
        "airfoil": "bold cyan",
        "warning_dim": "yellow dim",
        "opt": "bold magenta",
        "success": "bold green",
        "failure": "bold red",
        "highlight": "bold yellow underline",
        "unit": "italic white",
    },
)

# If running in Jupyter, set the console to use Rich's Jupyter console
if IN_JUPYTER:
    ICARUS_CONSOLE = Console(
        file=sys.stdout,
        force_jupyter=True,
        theme=ICARUS_THEME,
        # soft_wrap=True,
    )

else:
    # Rich Console for logging and output
    ICARUS_CONSOLE = Console(
        file=sys.stdout,
        force_terminal=True,
        theme=ICARUS_THEME,
        # soft_wrap=True,
        # highlight=False,  # Disable syntax highlighting for better performance in large outputs
    )

# Make console the default stream for print and logging
__print__ = builtins.print


def setup_logging() -> None:
    """Setup logging configuration to default."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(message)s",
        handlers=[
            RichHandler(
                console=ICARUS_CONSOLE,
                rich_tracebacks=True,
                show_level=True,
                show_time=True,
                show_path=True,
            ),
        ],
        force=True,
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    # builtins.print = ICARUS_CONSOLE.print


# Create a custom print function that sends to the logging queue
def queue_print(
    *args: object,
    sep: str = " ",
    end: str = "\n",
    file: Any | None = None,
    flush: Literal[False] = False,
) -> None:
    # Only intercept stdout prints (file=None or file=sys.stdout)
    import sys

    if file is not None and file is not sys.stdout:
        # Use original print for non-stdout outputs
        return __print__(*args, sep=sep, end=end, file=file, flush=flush)

    # Convert arguments to string using the same logic as print
    if args:
        output = sep.join(str(arg) for arg in args)
    else:
        output = ""

    # Add the end character
    output += end

    # Send to logging queue instead of stdout
    if output.strip():  # Only log non-empty output
        logger = logging.getLogger("print")
        # Remove trailing newlines since logger.info adds its own
        logger.info(output.rstrip("\n"))
    elif output == "\n":
        # Handle empty print() calls that just print a newline
        logger = logging.getLogger("print")
        logger.info("")


def setup_mp_logging(log_queue: QueueLike) -> QueueListener | None:
    """Set up global logging for the multiprocessing engine."""
    # Only configure rich logging in the main process
    if mp.current_process().name == "MainProcess":
        if not log_queue:
            raise ValueError("Log queue not initialized")

        # Rich handler (used only in the main process)
        rich_handler = RichHandler(
            console=ICARUS_CONSOLE,
            rich_tracebacks=True,
            show_path=False,
        )

        # Start QueueListener with the RichHandler
        listener = QueueListener(log_queue, rich_handler)

        # Configure root logger to send logs into the queue
        logging.basicConfig(
            level=logging.INFO,
            handlers=[QueueHandler(log_queue)],
            force=True,  # Overwrite any previous logging config
        )
    else:
        listener = None
        # Configure worker process logging to use the queue
        logging.basicConfig(
            level=logging.INFO,
            format="%(name)s - %(processName)s - %(message)s",
            handlers=[QueueHandler(log_queue)],
            force=True,
        )
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        builtins.print = queue_print
    return listener


setup_logging()
# Nice Traceback
# install(show_locals=True)
install(console=ICARUS_CONSOLE)
