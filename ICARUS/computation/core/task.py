"""
Core Task implementation.

This module contains the main Task class that represents a unit of work
in the simulation framework.
"""

from datetime import datetime

# from threading import Lock
from threading import Lock
from typing import Any
from typing import Callable
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple

from .data_structures import ProgressEvent
from .protocols import TaskExecutorProtocol
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
    _id_lock = Lock()  # Lock to make the counter thread-safe

    def __init__(
        self,
        name: str,
        executor: TaskExecutorProtocol[TaskInput, TaskOutput],
        task_input: TaskInput,
        config: Optional[TaskConfiguration] = None,
        task_id: Optional[TaskId] = None,
        progress_probe: Optional[Callable[[], ProgressEvent]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize a new task.

        Args:
            name: Human-readable name for the task.
            executor: The executor that will run this task.
            task_input: Input data for the task.
            config: Optional configuration (uses defaults if not provided).
            task_id: Optional task ID (generates one if not provided).
            progress_probe: Optional callable to probe task progress.
        """
        self.id = task_id or TaskId()
        self.name = name
        self.executor = executor
        self.input = task_input
        self.config = config or TaskConfiguration()
        self.created_at = datetime.now()
        self.metadata = metadata or {}

        # State is managed externally by the runner/scheduler
        self._state_history: List[Tuple[TaskState, datetime]] = [(TaskState.PENDING, self.created_at)]

        # Assign a thread-safe numeric ID
        # with Task._id_lock:
        Task._id_counter += 1
        self.id_num = Task._id_counter

        # Progress tracking attributes
        self.current_step: int = 0
        self.total_steps: int = 1
        self.progress_message = ""

        # Optional progress probe callable
        self.progress_probe = progress_probe

    @property
    def state(self) -> TaskState:
        """Returns the current state of the task."""
        return self._state_history[-1][0]

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
        currenct_state = self.state
        if new_state != currenct_state:
            self._state_history.append((new_state, datetime.now()))

    @property
    def state_history(self) -> List[Tuple[TaskState, datetime]]:
        """
        Get the complete state transition history.

        Returns:
            A list of (state, timestamp) tuples.
        """
        return self._state_history.copy()

    def register_progress(self, progress_event: ProgressEvent) -> None:
        """
        Update the task's internal progress information.

        Args:
            progress_event: A ProgressEvent containing the current progress and any relevant message.
        """
        if progress_event.task_id != self.id:
            raise ValueError("ProgressEvent task_id does not match this task's ID")

        self.current_step = progress_event.current_step
        self.total_steps = progress_event.total_steps
        self.progress_message = progress_event.message

        # Update the state
        if progress_event.completed:
            self.state = TaskState.COMPLETED
        elif progress_event.error:
            self.state = TaskState.FAILED

    def __repr__(self) -> str:
        """String representation of the task."""
        return f"Task(id={self.id}, name='{self.name}', state={self.state.name})"

    def __eq__(self, other) -> bool:
        """Check for equality with another task based on ID."""
        return isinstance(other, Task) and self.id == other.id

    def __hash__(self) -> int:
        """Hash the task based on its unique ID."""
        return hash(self.id)

    def __getstate__(self) -> dict[str, Any]:
        """Prepare the task for serialization."""
        state = self.__dict__.copy()
        # Remove non-serializable attributes
        state.pop("_id_lock", None)
        return state
