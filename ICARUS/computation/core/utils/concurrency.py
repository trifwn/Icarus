"""
Concurrency utilities for the simulation framework.

This module provides helper functions and data structures to abstract away
the differences between Python's various concurrency models (threading,
asyncio, multiprocessing), allowing the framework to use the correct
synchronization primitives based on the selected ExecutionMode.
"""

import asyncio
import multiprocessing
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from typing import Coroutine
from typing import Protocol


class LockLike(Protocol):
    """
    A protocol that defines the basic interface for lock-like objects.

    This is used to ensure that any object passed as a lock adheres to the
    context manager protocol (i.e., supports __enter__ and __exit__ methods).
    """

    def acquire(self, *args, **kwargs) -> bool | Coroutine[None, None, bool]: ...
    def release(self, *args, **kwargs) -> None: ...

    # def __enter__(self): ...
    # def __exit__(self, exc_type, exc_val, exc_tb): ...


class EventLike(Protocol):
    """
    A protocol that defines the basic interface for event-like objects.

    This is used to ensure that any object passed as an event adheres to the
    expected methods (e.g., is_set, set, clear).
    """

    def is_set(self) -> bool: ...
    def set(self) -> None: ...


class DummyLock:
    def acquire(self, *args, **kwargs) -> bool:
        return True

    def release(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyEvent:
    def __init__(self):
        self._is_set = False

    def is_set(self):
        return self._is_set

    def set(self):
        self._is_set = True


class ConcurrencyType(Enum):
    """
    Defines the execution strategy for the simulation framework.
    """

    SEQUENTIAL = "sequential"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    ASYNC = "async"


class UnsupportedConcurrencyTypeError(Exception):
    """
    Exception raised when an unsupported concurrency type is encountered.
    """

    def __init__(self, type: ConcurrencyType):
        super().__init__(f"Unsupported concurrency type: {type.value}")
        self.type = type


@dataclass(frozen=True)
class ConcurrencyPrimitives:
    """
    A container for concurrency synchronization primitives.

    This structure holds factory functions for creating locks and events,
    ensuring that the correct type of primitive is used for the current
    execution mode.

    Attributes:
        lock: A callable that returns a new lock instance (e.g., threading.Lock).
        event: A callable that returns a new event instance (e.g., asyncio.Event).
    """

    lock: Callable[[], LockLike]
    event: Callable[[], EventLike]

    @classmethod
    def get_concurrency_primitives(cls, type: ConcurrencyType) -> "ConcurrencyPrimitives":
        """
        Get the appropriate synchronization primitives for a given execution mode.

        Args:
            mode: The execution mode (e.g., THREADING, ASYNC).

        Returns:
            A ConcurrencyPrimitives object containing factories for locks and events.

        Raises:
            ConfigurationError: If the execution mode is unsupported.
        """
        if type == ConcurrencyType.THREADING:
            return cls(lock=threading.Lock, event=threading.Event)
        elif type == ConcurrencyType.ASYNC:
            return cls(lock=asyncio.Lock, event=asyncio.Event)
        elif type == ConcurrencyType.MULTIPROCESSING:
            # Note: These primitives are created via a SyncManager in a real
            # multiprocessing setup, but we provide the base classes here.
            return cls(lock=multiprocessing.Lock, event=multiprocessing.Event)
        elif type == ConcurrencyType.SEQUENTIAL:
            # For sequential execution, real locks are not needed.
            # We can use dummy objects that do nothing.
            return cls(lock=DummyLock, event=DummyEvent)
        raise UnsupportedConcurrencyTypeError(type)
