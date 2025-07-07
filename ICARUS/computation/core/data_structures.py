"""
Core data structures for the simulation framework.

This module contains the primary data structures used throughout the framework
for progress tracking, task results, and other core data representations.
"""

import traceback
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from enum import auto
from typing import Any
from typing import Dict
from typing import Generic
from typing import Optional

from .protocols import SerializableMixin
from .types import T
from .types import TaskId
from .types import TaskState


class ProgressEventType(Enum):
    STEP_COMPLETED = auto()
    STATUS_CHANGED = auto()
    ERROR_OCCURRED = auto()
    TASK_COMPLETED = auto()


@dataclass
class ProgressEvent(SerializableMixin):
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    error: Optional[Exception] = None

    def __post_init__(self):
        """Calculate derived fields and validate after initialization."""
        if self.current_step < 0 or self.total_steps < 0:
            raise ValueError("Steps cannot be negative")
        if self.current_step > self.total_steps:
            raise ValueError("Current step cannot exceed total steps")
        self.percentage = (
            (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert progress update to a serializable dictionary."""
        error_info = None
        if self.error:
            error_info = {
                "type": type(self.error).__name__,
                "message": str(self.error),
                "traceback": traceback.format_exc(),
            }

        return {
            "task_id": str(self.task_id),
            "name": self.name,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "message": self.message,
            "percentage": self.percentage,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "completed": self.completed,
            "error": error_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgressEvent":
        """Create a ProgressUpdate instance from a dictionary."""
        error = None
        if data.get("error"):
            error = Exception(f"{data['error']['type']}: {data['error']['message']}")

        return cls(
            task_id=TaskId(data["task_id"]),
            name=data["name"],
            current_step=data["current_step"],
            total_steps=data["total_steps"],
            message=data.get("message", ""),
            metadata=data.get("metadata", {}),
            completed=data.get("completed", False),
            error=error,
        )

    @classmethod
    def step_completed(
        cls,
        task_id: TaskId,
        name: str,
        current_step: int,
        total_steps: int,
        message: str = "",
    ) -> "ProgressEvent":
        """Create a step completion progress event."""
        return cls(
            task_id=task_id,
            name=name,
            current_step=current_step,
            total_steps=total_steps,
            message=message,
        )


@dataclass
class TaskResult(SerializableMixin, Generic[T]):
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
    output: Optional[T] = None
    error: Optional[Exception] = None
    execution_time: Optional[timedelta] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """Check if the task completed successfully."""
        return self.state == TaskState.COMPLETED and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task result to a serializable dictionary."""
        serialized_output = None
        if self.output is not None:
            if isinstance(self.output, SerializableMixin):
                serialized_output = self.output.to_dict()
            else:
                # Fallback for non-serializable objects.
                # Consider logging a warning here.
                serialized_output = str(self.output)

        error_info = None
        if self.error:
            error_info = {
                "type": type(self.error).__name__,
                "message": str(self.error),
                "args": self.error.args,
                "traceback": traceback.format_exception(
                    type(self.error),
                    self.error,
                    self.error.__traceback__,
                ),
            }

        return {
            "task_id": str(self.task_id),
            "state": self.state.name,
            "output": serialized_output,
            "error": error_info,
            "execution_time": self.execution_time.total_seconds()
            if self.execution_time
            else None,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        """Create a TaskResult instance from a dictionary."""
        error = None
        if data.get("error"):
            error = Exception(f"<{data['error']['type']}> {data['error']['message']}")

        execution_time = None
        if data.get("execution_time") is not None:
            execution_time = timedelta(seconds=data["execution_time"])

        return cls(
            task_id=TaskId(data["task_id"]),
            state=TaskState[data["state"]],
            output=data.get("output"),
            error=error,
            execution_time=execution_time,
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {}),
        )
