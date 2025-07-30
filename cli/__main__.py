"""Main Entry Point for ICARUS CLI

This module provides the main entry point for the ICARUS CLI application,
using the streamlined implementation for better performance and maintainability.
"""

import sys
from pathlib import Path

# Add CLI directory to path
cli_dir = Path(__file__).parent
sys.path.insert(0, str(cli_dir.parent))
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False) -> None:
    """Set up logging with appropriate level."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ICARUS Aerodynamics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive TUI
  python -m cli.streamlined_main

  # Run with specific configuration
  python -m cli.streamlined_main --config path/to/config.json
        """,
    )

    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--workspace", help="Workspace to use")
    parser.add_argument("--theme", help="Theme to apply")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    return parser.parse_args()


def show_version() -> None:
    """Show version information and exit."""
    try:
        from ICARUS import __version__ as icarus_version
    except ImportError:
        icarus_version = "unknown"

    print("ICARUS CLI v2.0.0")
    print(f"ICARUS Core: v{icarus_version}")
    print("Copyright © 2025 ICARUS Team")


async def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    from cli.core.unified_config import get_config_manager

    config_manager = get_config_manager()
    await config_manager.load_config()

    if config_path:
        config_manager.import_config(config_path)

    return {}


def run_app() -> int:
    """Run the application."""
    args = parse_arguments()

    if args.version:
        show_version()
        return 0

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Measure startup time
    start_time = time.time()

    try:
        # Import only what's needed to run the app
        from cli.app.streamlined_app import IcarusApp

        # Create and run the app
        app = IcarusApp()

        # Apply command line overrides
        if args.config:
            asyncio.run(load_config(args.config))

        if args.workspace:
            from cli.core.unified_config import get_config_manager

            get_config_manager().switch_workspace(args.workspace)

        if args.theme:
            from cli.core.ui import ThemeManager

            ThemeManager().apply_theme(args.theme)

        # Log startup time
        startup_time = time.time() - start_time
        logger.info(f"Application startup time: {startup_time:.2f} seconds")

        # Run the app
        app.run()
        return 0

    except ImportError as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install textual rich matplotlib numpy")
        return 1
    except Exception as e:
        logger.error(f"Error running ICARUS CLI: {e}", exc_info=True)
        print(f"Error running ICARUS CLI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_app())
