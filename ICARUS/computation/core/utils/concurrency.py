"""
Concurrency utilities for the simulation framework.

This module provides helper functions and data structures to abstract away
the differences between Python's various concurrency models (threading,
asyncio, multiprocessing), allowing the framework to use the correct
synchronization primitives based on the selected ExecutionMode.
"""

import asyncio
import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass
from enum import Enum
from multiprocessing.managers import SyncManager
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import Protocol
from typing import Union
from typing import runtime_checkable


@runtime_checkable
class LockLike(Protocol):
    """
    A protocol that defines the basic interface for lock-like objects.

    This is used to ensure that any object passed as a lock adheres to the
    context manager protocol (i.e., supports __enter__ and __exit__ methods).
    """

    def acquire(self, *args, **kwargs) -> bool | Coroutine[None, None, bool]: ...
    def release(self, *args, **kwargs) -> None: ...

    # def __enter__(self, *args, **kwargs): ...
    # def __exit__(self, *args, **kwargs): ...


@runtime_checkable
class EventLike(Protocol):
    """
    A protocol that defines the basic interface for event-like objects.

    This is used to ensure that any object passed as an event adheres to the
    expected methods (e.g., is_set, set, clear).
    """

    def is_set(self) -> bool: ...
    def set(self) -> None: ...


@runtime_checkable
class QueueLike(Protocol):
    """
    A protocol that defines the basic interface for queue-like objects.

    This is used to ensure that any object passed as a queue adheres to the
    expected methods (e.g., put, get, empty).
    """

    def put(self, *args, **kwargs) -> None: ...
    def get(self, *args, **kwargs) -> Any: ...
    def empty(self) -> bool: ...
    def get_nowait(self, *args, **kwargs) -> Any: ...


class DummyLock:
    def acquire(self, *args, **kwargs) -> bool:
        return True

    def release(self, *args, **kwargs) -> None:
        pass

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class DummyEvent:
    def __init__(self) -> None:
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
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


ConcurrentVariable = Union[
    LockLike,
    EventLike,
    QueueLike,
]


class ConcurrencyFeature(Enum):
    """
    Enumeration of concurrent variable types used in the simulation framework.

    This enum is used to identify the type of concurrent variable being
    requested or set in the context of task execution.
    """

    LOCK = "lock"
    EVENT = "event"
    QUEUE = "queue"


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
    queue: Callable[[], QueueLike]

    @classmethod
    def from_multiprocessing_manager(
        cls,
        manager: SyncManager,
    ) -> "ConcurrencyPrimitives":
        """
        Create ConcurrencyPrimitives using a multiprocessing manager.

        Args:
            manager: A Manager instance to create locks and events.

        Returns:
            A ConcurrencyPrimitives object with locks and events created by the manager.
        """
        return cls(
            lock=manager.Lock,
            event=manager.Event,
            queue=manager.Queue,
        )

    @classmethod
    def from_type(cls, type: ConcurrencyType) -> "ConcurrencyPrimitives":
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
            return cls(lock=threading.Lock, event=threading.Event, queue=queue.Queue)
        elif type == ConcurrencyType.ASYNC:
            return cls(lock=asyncio.Lock, event=asyncio.Event, queue=queue.Queue)
        elif type == ConcurrencyType.MULTIPROCESSING:
            # Note: These primitives are created via a Manager in a real
            # multiprocessing setup, but we provide the base classes here.
            return cls(lock=mp.Lock, event=mp.Event, queue=mp.Queue)
        elif type == ConcurrencyType.SEQUENTIAL:
            # For sequential execution, real locks are not needed.
            # We can use dummy objects that do nothing.
            return cls(lock=DummyLock, event=DummyEvent, queue=queue.Queue)
        raise UnsupportedConcurrencyTypeError(type)

    def get_concurrent_variable(
        self,
        feature: ConcurrencyFeature,
    ) -> ConcurrentVariable:
        """
        Get a new instance of the requested concurrent variable type.

        Args:
            feature: The type of concurrent variable to create (LOCK, EVENT, QUEUE).

        Returns:
            A new instance of the requested concurrent variable type.
        """
        if feature == ConcurrencyFeature.LOCK:
            return self.lock()
        elif feature == ConcurrencyFeature.EVENT:
            return self.event()
        elif feature == ConcurrencyFeature.QUEUE:
            return self.queue()
        raise ValueError(f"Unsupported concurrency feature: {feature.value}")

    def get_concurrent_variables(
        self,
        features: dict[str, ConcurrencyFeature],
    ) -> dict[str, ConcurrentVariable]:
        """
        Get a dictionary of new instances for the requested concurrent variable types.

        Args:
            features: A list of ConcurrencyFeature types to create.

        Returns:
            A dictionary mapping feature names to their corresponding concurrent variable instances.
        """

        return {
            name: self.get_concurrent_variable(feature)
            for name, feature in features.items()
        }
