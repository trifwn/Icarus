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
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar

# ===== TYPE VARIABLES =====

T = TypeVar("T")
R = TypeVar("R")
TaskInput = TypeVar("TaskInput")
TaskOutput = TypeVar("TaskOutput")


# ===== ENUMS =====


class TaskState(Enum):
    """
    Enhanced task state management.

    Defines all possible states a task can be in during its lifecycle,
    from initial creation through completion or failure.
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
    Task priority levels.

    Defines the execution priority for tasks, where higher values
    indicate higher priority.
    """

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ExecutionMode(Enum):
    """
    Execution strategies for the simulation framework.

    Defines different ways tasks can be executed, from sequential
    to various forms of parallel execution.
    """

    SEQUENTIAL = "sequential"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    ASYNC = "async"
    ADAPTIVE = "adaptive"


# ===== CORE DATA TYPES =====


@dataclass(frozen=True)
class TaskId:
    """
    Unique, immutable task identifier.

    Provides a strongly-typed wrapper around UUID strings to ensure
    task identification is type-safe and immutable.
    """

    value: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __str__(self) -> str:
        """Return the string representation of the task ID."""
        return self.value


@dataclass
class TaskConfiguration:
    """
    Comprehensive task configuration.

    Contains all configuration options for task execution including
    retry behavior, timeouts, resource requirements, and dependencies.

    Attributes:
        max_retries: Maximum number of retry attempts for failed tasks
        timeout: Optional timeout for task execution
        priority: Task execution priority level
        resources: Dictionary of required resources
        dependencies: List of task IDs that must complete before this task
        tags: List of string tags for task categorization
        checkpoint_interval: Optional interval for creating checkpoints
    """

    max_retries: int = 3
    timeout: Optional[timedelta] = None
    priority: Priority = Priority.NORMAL
    resources: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[TaskId] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    checkpoint_interval: Optional[int] = None

    def merge(self, other: "TaskConfiguration") -> "TaskConfiguration":
        """
        Merge configurations with other taking precedence.

        Args:
            other: Configuration to merge with this one

        Returns:
            New TaskConfiguration with merged settings
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
