"""
Core data structures for the simulation framework.

This module contains the primary data structures used throughout the framework
for progress tracking, task results, and other core data representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from enum import auto
from typing import Any
from typing import Generic

from .types import T
from .types import TaskId
from .types import TaskState


class ProgressEventType(Enum):
    STEP_COMPLETED = auto()
    STATUS_CHANGED = auto()
    ERROR_OCCURRED = auto()
    TASK_COMPLETED = auto()


@dataclass
class ProgressEvent:
    """
    Represents a progress update for a task.

    Attributes:
        task_id: Unique identifier of the task.
        name: Human-readable name of the task.
        current_step: Current step number.
        total_steps: Total number of steps.
        message: Optional progress message.
        percentage: Calculated completion percentage.
        timestamp: When this progress update was created.
        metadata: Additional metadata about the progress.
        completed: Whether the task has completed.
        error: Any error that occurred during execution.
    """

    task_id: TaskId
    name: str
    current_step: int
    total_steps: int
    message: str = ""
    percentage: float = field(init=False)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    error: Exception | BaseException | None = None

    def __post_init__(self) -> None:
        """Calculate derived fields and validate after initialization."""
        if self.current_step < 0 or self.total_steps < 0:
            raise ValueError("Steps cannot be negative")
        if self.current_step > self.total_steps:
            raise ValueError("Current step cannot exceed total steps")
        self.percentage = (
            (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        )

    @classmethod
    def step_completed(
        cls,
        task_id: TaskId,
        name: str,
        current_step: int,
        total_steps: int,
        message: str = "",
    ) -> ProgressEvent:
        """Create a step completion progress event."""
        return cls(
            task_id=task_id,
            name=name,
            current_step=current_step,
            total_steps=total_steps,
            message=message,
        )


@dataclass
class TaskResult(Generic[T]):
    """
    Represents the final result of a task's execution.

    Attributes:
        task_id: Unique identifier of the task.
        state: Final state of the task (e.g., COMPLETED, FAILED).
        output: The actual result data (if successful).
        error: Any exception that occurred during execution.
        execution_time: How long the task took to execute.
        retry_count: Number of retries attempted.
        metadata: Additional metadata about the execution.
        timestamp: When the result was created.
    """

    task_id: TaskId
    state: TaskState
    output: T | None = None
    error: Exception | BaseException | None = None
    execution_time: timedelta | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """Check if the task completed successfully."""
        return self.state == TaskState.COMPLETED and self.error is None
