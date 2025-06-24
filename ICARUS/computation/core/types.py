"""
Core type definitions for the simulation framework.

This module contains the fundamental enums, type variables, and basic types
used throughout the simulation framework.
"""

import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import timedelta
from enum import Enum
from enum import auto
from typing import Any
from typing import List
from typing import Optional
from typing import TypeVar

from .utils.concurrency import ConcurrencyPrimitives
from .utils.concurrency import ConcurrencyType
from .utils.concurrency import EventLike
from .utils.concurrency import LockLike

# ===== TYPE VARIABLES =====

T = TypeVar("T")
R = TypeVar("R")
TaskInput = TypeVar("TaskInput", contravariant=True)
TaskOutput = TypeVar("TaskOutput", covariant=True)


# ===== ENUMS =====


class TaskState(Enum):
    """
    Defines all possible states a task can be in during its lifecycle.
    """

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRYING = auto()


class Priority(Enum):
    """
    Defines execution priority for tasks. Higher values are higher priority.
    """

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ExecutionMode(Enum):
    """
    Defines the execution strategy for the simulation framework.
    """

    SEQUENTIAL = "sequential"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    ASYNC = "async"
    ADAPTIVE = "adaptive"

    def __new__(cls, value):
        """
        Custom __new__ method to initialize concurrency primitives.
        """
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value):
        """
        Initialize the enum member with concurrency primitives.
        """
        self.concurrency_type: ConcurrencyType | None = None

    def get_primitives(self) -> ConcurrencyPrimitives:
        """
        Get the concurrency primitives for this execution mode.

        Returns:
            A ConcurrencyPrimitives object containing lock and event factories.
        """
        if self.concurrency_type is None:
            self.concurrency_type = ConcurrencyType(self.value)

        primitives = ConcurrencyPrimitives.get_concurrency_primitives(self.concurrency_type)
        return primitives

    def create_lock(self) -> LockLike:
        """
        Create a lock appropriate for the current execution mode.

        Returns:
            A lock object (e.g., threading.Lock, asyncio.Lock, multiprocessing.Lock, DummyLock).
        """
        return self.get_primitives().lock()

    def create_event(self) -> EventLike:
        """
        Create an event appropriate for the current execution mode.

        Returns:
            An event object (e.g., threading.Event, asyncio.Event, multiprocessing.Event, DummyEvent).
        """
        return self.get_primitives().event()

    @classmethod
    def validate(cls, value: str) -> "ExecutionMode":
        """
        Validate and return an ExecutionMode for the given value.

        Args:
            value: The string value to validate

        Returns:
            The corresponding ExecutionMode enum member

        Raises:
            ValueError: If the value is not a valid execution mode
        """
        try:
            return cls(value)
        except ValueError:
            valid_modes = [mode.value for mode in cls]
            raise ValueError(f"Invalid execution mode: {value}. Valid modes are: {valid_modes}")


# ===== CORE DATA TYPES =====
@dataclass(frozen=True)
class TaskId:
    """
    A unique, immutable identifier for a task.

    Wraps a UUID string to provide strong typing for task identification.
    """

    value: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __str__(self) -> str:
        """Return the string representation of the UUID."""
        return self.value

    def __hash__(self):
        return hash(self.value)


@dataclass
class TaskConfiguration:
    """
    Contains all configuration options for a single task's execution.

    Attributes:
        max_retries: Max retry attempts for the task.
        timeout: Optional execution timeout.
        priority: Task execution priority.
        resources: Dictionary of required resources.
        dependencies: List of TaskIds this task depends on.
        tags: List of string tags for categorization.
        checkpoint_interval: Optional interval for checkpointing.
    """

    max_retries: int = 3
    timeout: Optional[timedelta] = None
    priority: Priority = Priority.NORMAL
    resources: dict[str, Any] = field(default_factory=dict)
    dependencies: List[TaskId] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    checkpoint_interval: Optional[int] = None

    def merge(self, other: "TaskConfiguration") -> "TaskConfiguration":
        """
        Merge this configuration with another, with 'other' taking precedence.

        Args:
            other: The configuration to merge with.

        Returns:
            A new TaskConfiguration instance with the merged settings.
        """
        return TaskConfiguration(
            max_retries=other.max_retries if other.max_retries != 3 else self.max_retries,
            timeout=other.timeout or self.timeout,
            priority=other.priority if other.priority != Priority.NORMAL else self.priority,
            resources={**self.resources, **other.resources},
            dependencies=list(set(self.dependencies + other.dependencies)),
            tags=list(set(self.tags + other.tags)),
            checkpoint_interval=other.checkpoint_interval or self.checkpoint_interval,
        )
