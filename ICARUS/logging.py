from __future__ import annotations

import builtins
import logging
import multiprocessing as mp
import sys
from logging.handlers import QueueHandler
from logging.handlers import QueueListener
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler
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

ICARUS_THEME = "solarized-dark"

# If running in Jupyter, set the console to use Rich's Jupyter console
if IN_JUPYTER:
    ICARUS_CONSOLE = Console(
        file=sys.stdout,
        force_jupyter=True,
        # theme ="solarized-dark",
        # soft_wrap=True,
    )

else:
    # Rich Console for logging and output
    ICARUS_CONSOLE = Console(
        file=sys.stdout,
        force_terminal=True,
        # theme = "solarized-dark",
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
    builtins.print = ICARUS_CONSOLE.print


# Create a custom print function that sends to the logging queue
def queue_print(*args, sep=" ", end="\n", file=None, flush=False) -> None:
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
        # rich_handler.setFormatter(
        #     logging.Formatter(
        #         "%(name)s - %(message)s",
        #     )
        # )

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
install()
