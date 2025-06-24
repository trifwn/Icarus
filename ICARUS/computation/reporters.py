import logging
from weakref import WeakSet
from ICARUS.computation.core.data_structures import ProgressEvent, TaskResult
from ICARUS.computation.core.protocols import ProgressObserver, ProgressReporter
from ICARUS.computation.core.task import Task


class Reporter(ProgressReporter):
    """Simple console progress reporter"""

    def __init__(self, tasks: list[Task] = []):
        self._observers: WeakSet[ProgressObserver] = WeakSet()
        self.logger = logging.getLogger(__name__)
        self.tasks = tasks

    def add_observer(self, observer: ProgressObserver) -> None:
        """Add progress observer"""
        self._observers.add(observer)

    async def report_progress(self, event: ProgressEvent) -> None:
        # Notify observers
        task = next((t for t in self.tasks if t.id == event.task_id), None)

        if task:
            task.update_progress(event)

        for observer in self._observers:
            try:
                await observer.on_progress_update(event)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")

    async def report_completion(self, result: TaskResult) -> None:
        # Notify observers
        for observer in self._observers:
            try:
                await observer.on_task_completion(result)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")

    def __str__(self):
        return f"<Reporter with observers={(self._observers)}>"
