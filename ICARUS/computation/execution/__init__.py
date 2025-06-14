"""
This module provides various execution engines for running tasks in different modes.
It includes support for asynchronous, multiprocessing, sequential, threading, and adaptive execution strategies.

isort: skip-file
"""

from __future__ import annotations

from .adaptive_engine import AdaptiveExecutionEngine
from .async_engine import AsyncExecutionEngine
from .base_engine import BaseExecutionEngine
from .engine_creator import create_execution_engine
from .multiprocessing_engine import MultiprocessingExecutionEngine
from .sequential_engine import SequentialExecutionEngine
from .threading_engine import ThreadingExecutionEngine

__all__ = [
    "BaseExecutionEngine",
    "AsyncExecutionEngine",
    "MultiprocessingExecutionEngine",
    "SequentialExecutionEngine",
    "ThreadingExecutionEngine",
    "AdaptiveExecutionEngine",
    "create_execution_engine",
]
