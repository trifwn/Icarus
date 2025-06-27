"""
Core Task implementation.

This module contains the main Task class that represents a unit of work
in the simulation framework.
"""

from datetime import datetime

# from threading import Lock
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple

from .data_structures import ProgressEvent
from .protocols import TaskExecutor
from .types import TaskConfiguration
from .types import TaskId
from .types import TaskInput
from .types import TaskOutput
from .types import TaskState


class Task(Generic[TaskInput, TaskOutput]):
    """
    Represents a single, executable unit of work.

    A Task is primarily a data container holding its identity, configuration,
    input data, and the executor responsible for its logic. The lifecycle state
    (e.g., PENDING, RUNNING) is managed by the execution framework, not by the
    task itself, to ensure proper synchronization.

    Attributes:
        id: Unique identifier for this task.
        name: Human-readable name for the task.
        executor: The executor responsible for running this task.
        input: Input data for the task.
        config: Configuration for task execution.
        created_at: Timestamp of when the task was created.
        state: Current execution state of the task.
        id_num: A sequential numeric ID, mainly for display purposes.
    """

    _id_counter = 0
    # _id_lock = Lock()  # Lock to make the counter thread-safe

    def __init__(
        self,
        name: str,
        executor: TaskExecutor[TaskInput, TaskOutput],
        task_input: TaskInput,
        config: Optional[TaskConfiguration] = None,
        task_id: Optional[TaskId] = None,
    ):
        """
        Initialize a new task.

        Args:
            name: Human-readable name for the task.
            executor: The executor that will run this task.
            task_input: Input data for the task.
            config: Optional configuration (uses defaults if not provided).
            task_id: Optional task ID (generates one if not provided).
        """
        self.id = task_id or TaskId()
        self.name = name
        self.executor = executor
        self.input = task_input
        self.config = config or TaskConfiguration()
        self.created_at = datetime.now()

        # State is managed externally by the runner/scheduler
        self._state = TaskState.PENDING
        self._state_history: List[Tuple[TaskState, datetime]] = [(TaskState.PENDING, self.created_at)]

        # Assign a thread-safe numeric ID
        # with Task._id_lock:
        Task._id_counter += 1
        self.id_num = Task._id_counter

        # Progress tracking attributes
        self._progress = 0
        self._total_progress: int | None = None
        self._progress_message = ""
        self._last_progress_update = self.created_at

    @property
    def state(self) -> TaskState:
        """Returns the current state of the task."""
        return self._state

    @state.setter
    def state(self, new_state: TaskState) -> None:
        """
        Set the task's state.

        This setter is provided for convenience, but it is recommended to use
        `set_state` for thread-safe state transitions.

        Note: This method is not internally synchronized. The caller (e.g.,
        a task scheduler or worker) is responsible for ensuring thread-safe
        state transitions.

        Args:
            new_state: The new state to set for the task.
        """
        if new_state != self._state:
            self._state = new_state
            self._state_history.append((new_state, datetime.now()))

    def progress_probe(self) -> ProgressEvent:
        """
        Get a snapshot of the current progress.

        This method is useful for monitoring systems to poll the task's
        latest progress information.

        Returns:
            A ProgressUpdate data object for this task.
        """
        is_finished = self.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]
        error = None
        if self.state == TaskState.FAILED:
            # Note: The actual exception object is in the TaskResult.
            # This is just a placeholder for the probe.
            error = Exception("Task failed")

        return ProgressEvent(
            task_id=self.id,
            name=self.name,
            current_step=self._progress,
            total_steps=self._total_progress if self._total_progress else 100,
            message=self._progress_message,
            completed=is_finished,
            error=error,
        )

    def update_progress(self, progress_event: ProgressEvent) -> None:
        """
        Update the task's internal progress information.

        Args:
            progress_event: A ProgressEvent containing the current progress and any relevant message.
        """
        if progress_event.task_id != self.id:
            raise ValueError("ProgressEvent task_id does not match this task's ID")
        self._progress = progress_event.current_step
        self._total_progress = progress_event.total_steps
        self._progress_message = progress_event.message
        self._last_progress_update = datetime.now()

        # Update the state
        if progress_event.completed:
            self.state = TaskState.COMPLETED
        elif progress_event.error:
            self.state = TaskState.FAILED

    def get_state_history(self) -> List[Tuple[TaskState, datetime]]:
        """
        Get the complete state transition history.

        Returns:
            A list of (state, timestamp) tuples.
        """
        return self._state_history.copy()

    def get_progress(self) -> int:
        """Get current progress as a percentage."""
        if self._total_progress:
            return int((self._progress / self._total_progress) * 100)
        return self._progress

    def get_progress_message(self) -> str:
        """Get the current progress message."""
        return self._progress_message

    def __repr__(self) -> str:
        """String representation of the task."""
        return f"Task(id={self.id}, name='{self.name}', state={self.state.name})"

    def __eq__(self, other) -> bool:
        """Check for equality with another task based on ID."""
        return isinstance(other, Task) and self.id == other.id

    def __hash__(self) -> int:
        """Hash the task based on its unique ID."""
        return hash(self.id)
