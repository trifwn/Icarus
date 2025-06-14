"""
Core Task implementation.

This module contains the main Task class that represents a unit of work
in the simulation framework, with full OOP design and progress tracking.
"""

from datetime import datetime
from threading import Lock
from threading import RLock
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple

from .data_structures import ProgressUpdate
from .protocols import TaskExecutor
from .types import TaskConfiguration
from .types import TaskId
from .types import TaskInput
from .types import TaskOutput
from .types import TaskState


class Task(Generic[TaskInput, TaskOutput]):
    """
    Enhanced task with full OOP design and progress bar support.

    Represents a single unit of work that can be executed by the simulation
    framework. Each task has a unique ID, configuration, executor, and
    maintains its own state and progress information.

    Attributes:
        id: Unique identifier for this task
        name: Human-readable name for the task
        executor: The executor responsible for running this task
        input: Input data for the task
        config: Configuration for task execution
        created_at: When the task was created
        state: Current execution state
        id_num: Numeric ID for progress bar compatibility
    """

    _id_counter = 0
    _id_lock = Lock()

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
            name: Human-readable name for the task
            executor: The executor that will run this task
            task_input: Input data for the task
            config: Optional configuration (uses defaults if not provided)
            task_id: Optional task ID (generates one if not provided)
        """
        self.id = task_id or TaskId()
        self.name = name
        self.executor = executor
        self.input = task_input
        self.config = config or TaskConfiguration()
        self.created_at = datetime.now()
        self._state_lock = RLock()
        self.state = TaskState.PENDING
        self._state_history: List[Tuple[TaskState, datetime]] = [(TaskState.PENDING, self.created_at)]

        # Add numeric ID for progress bar compatibility
        with Task._id_lock:
            Task._id_counter += 1
            self.id_num = Task._id_counter

        # Progress tracking
        self._current_progress = 0
        self._total_progress = 100
        self._progress_message = ""
        self._last_progress_update = datetime.now()

    @property
    def state(self) -> TaskState:
        """
        Thread-safe state access.

        Returns:
            Current task state
        """
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new_state: TaskState) -> None:
        """
        Thread-safe state modification with history tracking.

        Args:
            new_state: New state to set
        """
        with self._state_lock:
            if not hasattr(self, "_state"):
                self._state = TaskState.PENDING
                self._state_history = [(TaskState.PENDING, self.created_at)]

            if new_state != self._state:
                self._state = new_state
                self._state_history.append((new_state, datetime.now()))

    def progress_probe(self, task=None) -> ProgressUpdate:
        """
        Progress probe method for compatibility with tqdm monitoring.

        This method is called by the progress monitor to get current
        progress information from the task.

        Args:
            task: Optional task parameter (for compatibility)

        Returns:
            Current progress update for this task
        """
        return ProgressUpdate(
            task_id=self.id,
            name=self.name,
            current_step=self._current_progress,
            total_steps=self._total_progress,
            message=self._progress_message,
            completed=self.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED],
            error=None if self.state != TaskState.FAILED else Exception("Task failed"),
        )

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """
        Update task progress information.

        Args:
            current: Current step number
            total: Total number of steps
            message: Optional progress message
        """
        self._current_progress = current
        self._total_progress = total
        self._progress_message = message
        self._last_progress_update = datetime.now()

    def get_state_history(self) -> List[Tuple[TaskState, datetime]]:
        """
        Get complete state transition history.

        Returns:
            List of (state, timestamp) tuples showing state changes
        """
        with self._state_lock:
            return self._state_history.copy()

    def get_progress(self) -> int:
        """
        Get current progress percentage.

        Returns:
            Current progress as percentage (0-100)
        """
        if self._total_progress == 0:
            return 0
        return int((self._current_progress / self._total_progress) * 100)

    def get_progress_message(self) -> str:
        """
        Get current progress message.

        Returns:
            Current progress message string
        """
        return self._progress_message

    def __repr__(self) -> str:
        """
        String representation of the task.

        Returns:
            String containing task ID, name, and current state
        """
        return f"Task(id={self.id}, name='{self.name}', state={self.state.name})"

    def __eq__(self, other) -> bool:
        """
        Check equality with another task.

        Args:
            other: Other object to compare with

        Returns:
            True if both are tasks with the same ID
        """
        return isinstance(other, Task) and self.id == other.id

    def __hash__(self) -> int:
        """
        Hash function for task objects.

        Returns:
            Hash of the task ID
        """
        return hash(self.id.value)
