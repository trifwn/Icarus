"""
This module provides various execution engines for running tasks in different modes.
It includes support for asynchronous, multiprocessing, sequential, threading, and adaptive execution strategies.

isort: skip-file
"""

from __future__ import annotations

from .async_engine import AsyncEngine
from .base_engine import AbstractEngine
from .engine_creator import create_execution_engine
from .multiprocessing_engine import MultiprocessingEngine
from .sequential_engine import SequentialExecutionEngine
from .threading_engine import ThreadingEngine

__all__ = [
    "AbstractEngine",
    "AsyncEngine",
    "MultiprocessingEngine",
    "SequentialExecutionEngine",
    "ThreadingEngine",
    "create_execution_engine",
]
