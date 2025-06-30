import logging

from .core import ConcurrencyFeature
from .core import ConcurrentVariable
from .core import ProgressEvent
from .core import QueueLike
from .core import Task
from .core import TaskResult
from .core.protocols import ProgressObserver
from .core.protocols import ProgressReporter


class Reporter(ProgressReporter):
    """Simple console progress reporter"""

    def __init__(self, tasks: list[Task] = []):
        self._observers: set[ProgressObserver] = set()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tasks = tasks
        self.event_queue: QueueLike | None = None

    def request_concurrent_vars(self) -> dict[str, ConcurrencyFeature]:
        """Request concurrent variables for this reporter"""
        reporter_vars = {
            "event_queue": ConcurrencyFeature.QUEUE,
        }
        for observer in self._observers:
            reporter_vars.update(observer.request_concurrent_vars())
        return reporter_vars

    def set_concurrent_vars(self, vars: dict[str, ConcurrentVariable]) -> None:
        """Set concurrent variables for this reporter"""
        event_queue = vars.get("event_queue")
        if not isinstance(event_queue, QueueLike):
            raise TypeError("Expected 'event_queue' to be a QueueLike instance")
        self.event_queue = event_queue

        for observer in self._observers:
            observer.set_concurrent_vars(vars)

    def add_observer(self, observer: ProgressObserver) -> None:
        """Add progress observer"""
        self._observers.add(observer)

    def report_progress(self, event: ProgressEvent) -> None:
        # Notify observers
        task = next((t for t in self.tasks if t.id == event.task_id), None)
        if task:
            task.register_progress(event)

        # Put event in the queue for processing
        if self.event_queue:
            try:
                self.event_queue.put(event)
            except Exception as e:
                self.logger.error(f"Error putting event in queue: {e}")

        for observer in self._observers:
            try:
                observer.on_progress_update(event)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")

    def report_completion(self, result: TaskResult) -> None:
        # task = next((t for t in self.tasks if t.id == result.task_id), None)

        # Notify observers
        for observer in self._observers:
            try:
                observer.on_task_completion(result)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")

    def __str__(self):
        return f"<Reporter with observers={(self._observers)}>"
