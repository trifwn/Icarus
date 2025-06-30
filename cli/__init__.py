"""ICARUS CLI Module

This module provides a modern command-line interface for ICARUS Aerodynamics
using Rich and Typer for enhanced user experience.
"""

# Import new CLI functions
from .main import app, interactive


__all__ = [
    # New CLI functions
    "app",
    "interactive",
]
