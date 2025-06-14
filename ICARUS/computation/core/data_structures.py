"""
Core data structures for the simulation framework.

This module contains the primary data structures used throughout the framework
for progress tracking, task results, and other core data representations.
"""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import Generic
from typing import Optional

from .protocols import SerializableMixin
from .types import T
from .types import TaskId
from .types import TaskState


@dataclass
class ProgressUpdate(SerializableMixin):
    """
    Enhanced progress tracking with tqdm compatibility.

    Represents a progress update for a task, including current status,
    completion percentage, and any associated metadata or error information.

    Attributes:
        task_id: Unique identifier of the task
        name: Human-readable name of the task
        current_step: Current step number (0-based)
        total_steps: Total number of steps
        message: Optional progress message
        percentage: Calculated completion percentage
        timestamp: When this progress update was created
        metadata: Additional metadata about the progress
        completed: Whether the task has completed
        error: Any error that occurred during execution
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

    # Legacy compatibility fields for existing code
    current_iteration: int = field(init=False)
    max_iterations: int = field(init=False)
    progress_percentage: Optional[float] = field(init=False)

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        object.__setattr__(
            self,
            "percentage",
            (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0,
        )
        # Legacy compatibility
        object.__setattr__(self, "current_iteration", self.current_step)
        object.__setattr__(self, "max_iterations", self.total_steps)
        object.__setattr__(self, "progress_percentage", self.percentage)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert progress update to dictionary representation.

        Returns:
            Dictionary containing all progress update data
        """
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
            "error": str(self.error) if self.error else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgressUpdate":
        """
        Create progress update from dictionary representation.

        Args:
            data: Dictionary containing progress update data

        Returns:
            New ProgressUpdate instance
        """
        return cls(
            task_id=TaskId(data["task_id"]),
            name=data["name"],
            current_step=data["current_step"],
            total_steps=data["total_steps"],
            message=data.get("message", ""),
            metadata=data.get("metadata", {}),
            completed=data.get("completed", False),
            error=Exception(data["error"]) if data.get("error") else None,
        )


@dataclass
class TaskResult(SerializableMixin, Generic[T]):
    """
    Enhanced task results with type safety.

    Represents the final result of a task execution, including success/failure
    status, timing information, and any error details.

    Attributes:
        task_id: Unique identifier of the task
        state: Final state of the task
        result: The actual result data (if successful)
        error: Any error that occurred during execution
        execution_time: How long the task took to execute
        retry_count: Number of retries attempted
        metadata: Additional metadata about the execution
        timestamp: When the result was created
    """

    task_id: TaskId
    state: TaskState
    result: Optional[T] = None
    error: Optional[Exception] = None
    execution_time: Optional[timedelta] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """
        Check if the task completed successfully.

        Returns:
            True if the task completed without errors
        """
        return self.state == TaskState.COMPLETED and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task result to dictionary representation.

        Returns:
            Dictionary containing all task result data
        """
        return {
            "task_id": str(self.task_id),
            "state": self.state.name,
            "result": str(self.result) if self.result is not None else None,
            "error": str(self.error) if self.error else None,
            "execution_time": self.execution_time.total_seconds() if self.execution_time else None,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        """
        Create task result from dictionary representation.

        Args:
            data: Dictionary containing task result data

        Returns:
            New TaskResult instance
        """
        return cls(
            task_id=TaskId(data["task_id"]),
            state=TaskState[data["state"]],
            result=data.get("result"),
            error=Exception(data["error"]) if data.get("error") else None,
            execution_time=timedelta(seconds=data["execution_time"]) if data.get("execution_time") else None,
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {}),
        )
