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
from multiprocessing.managers import SyncManager
from typing import Any
from typing import List
from typing import Optional
from typing import Self
from typing import TypeVar

from .utils.concurrency import ConcurrencyPrimitives
from .utils.concurrency import ConcurrencyType

# ===== TYPE VARIABLES =====
T = TypeVar("T")
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

    def can_transition_to(self, new_state: "TaskState") -> bool:
        """Check if transition to new_state is valid."""
        valid_transitions = {
            TaskState.PENDING: {TaskState.QUEUED, TaskState.CANCELLED},
            TaskState.QUEUED: {TaskState.RUNNING, TaskState.CANCELLED},
            TaskState.RUNNING: {
                TaskState.PAUSED,
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.CANCELLED,
            },
            TaskState.PAUSED: {TaskState.RUNNING, TaskState.CANCELLED},
            TaskState.FAILED: {TaskState.RETRYING, TaskState.CANCELLED},
            TaskState.RETRYING: {
                TaskState.RUNNING,
                TaskState.FAILED,
                TaskState.CANCELLED,
            },
            # Terminal states
            TaskState.COMPLETED: set(),
            TaskState.CANCELLED: set(),
        }
        return new_state in valid_transitions.get(self, set())


class Priority(Enum):
    """
    Defines execution priority for tasks. Higher values are higher priority.
    """

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


# ===== DATA CLASSES =====
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

    def __hash__(self) -> int:
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
            max_retries=other.max_retries
            if other.max_retries != 3
            else self.max_retries,
            timeout=other.timeout or self.timeout,
            priority=other.priority
            if other.priority != Priority.NORMAL
            else self.priority,
            resources={**self.resources, **other.resources},
            dependencies=list(set(self.dependencies + other.dependencies)),
            tags=list(set(self.tags + other.tags)),
            checkpoint_interval=other.checkpoint_interval or self.checkpoint_interval,
        )


# ===== EXECUTION MODE =====
class ExecutionMode(Enum):
    """
    Defines the execution strategy for the simulation framework.
    """

    SEQUENTIAL = "sequential"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    ASYNC = "async"
    ADAPTIVE = "adaptive"

    def __new__(cls, value) -> Self:
        """
        Custom __new__ method to initialize concurrency primitives.
        """
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value) -> None:
        """
        Initialize the enum member with concurrency primitives.
        """
        self.concurrency_type: ConcurrencyType | None = None
        self._primitives: ConcurrencyPrimitives | None = None
        self._mp_manager: Optional[SyncManager] = None

    def set_multiprocessing_manager(self, manager: SyncManager) -> None:
        """
        Set the multiprocessing manager for this execution mode.

        Args:
            manager: A Manager instance to use for multiprocessing primitives.
        """
        if self.concurrency_type is not ConcurrencyType.MULTIPROCESSING:
            raise ValueError(
                f"Cannot set manager for non-multiprocessing mode: {self.value}",
            )
        self._mp_manager = manager
        self._primitives = ConcurrencyPrimitives.from_multiprocessing_manager(manager)

    def clear_multiprocessing_manager(self) -> None:
        """
        Unset the multiprocessing manager for this execution mode.
        """
        if self.concurrency_type is not ConcurrencyType.MULTIPROCESSING:
            raise ValueError(
                f"Cannot unset manager for non-multiprocessing mode: {self.value}",
            )
        self._mp_manager = None
        self._primitives = None

    @property
    def primitives(self) -> ConcurrencyPrimitives:
        """Get the concurrency primitives for this execution mode.
        If not already set, initialize them based on the concurrency type.
        Returns:
            A ConcurrencyPrimitives object containing lock and event factories.
        """
        if self._primitives is None:
            if self.concurrency_type is None:
                self.concurrency_type = ConcurrencyType(self.value)

            if (
                self.concurrency_type is ConcurrencyType.MULTIPROCESSING
                and self._mp_manager is not None
            ):
                self._primitives = ConcurrencyPrimitives.from_multiprocessing_manager(
                    self._mp_manager,
                )
            else:
                self._primitives = ConcurrencyPrimitives.from_type(
                    self.concurrency_type,
                )
        return self._primitives

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Remove non-serializable items
        state.pop("_primitives", None)
        state.pop("_mp_manager", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._primitives = None
        self._mp_manager = None
        self.concurrency_type = (
            ConcurrencyType(self.value)
            if "concurrency_type" not in state
            else state.get("concurrency_type")
        )
