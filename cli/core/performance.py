"""
Performance Optimization and Scalability Module

This module provides comprehensive performance optimization features including
asynchronous operation handling, intelligent caching, resource monitoring,
and background execution for long-running analyses.
"""

import asyncio
import gc
import hashlib
import json
import logging
import pickle
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import psutil


@dataclass
class ResourceUsage:
    """System resource usage information."""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    data: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)


@dataclass
class BackgroundTask:
    """Background task information."""

    task_id: str
    name: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    priority: int = 0  # Higher number = higher priority


class AsyncOperationManager:
    """Manages asynchronous operations for responsive UI."""

    def __init__(self, max_concurrent_operations: int = 10):
        self._logger = logging.getLogger(__name__)
        self._max_concurrent = max_concurrent_operations
        self._semaphore = asyncio.Semaphore(max_concurrent_operations)
        self._active_operations: Dict[str, asyncio.Task] = {}
        self._operation_callbacks: Dict[str, List[Callable]] = {}

    async def execute_async(
        self,
        operation_id: str,
        coro_func: Callable,
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Execute an operation asynchronously with resource management."""

        async with self._semaphore:
            try:
                # Register progress callback
                if progress_callback:
                    self._operation_callbacks[operation_id] = [progress_callback]

                # Create and start task
                if asyncio.iscoroutinefunction(coro_func):
                    coro = coro_func(*args, **kwargs)
                else:
                    # Wrap sync function in async
                    loop = asyncio.get_event_loop()
                    coro = loop.run_in_executor(None, coro_func, *args, **kwargs)

                if timeout:
                    coro = asyncio.wait_for(coro, timeout=timeout)

                task = asyncio.create_task(coro)
                self._active_operations[operation_id] = task

                # Execute with progress tracking
                result = await self._execute_with_progress(operation_id, task)

                self._logger.info(
                    f"Async operation {operation_id} completed successfully",
                )
                return result

            except asyncio.TimeoutError:
                self._logger.error(f"Operation {operation_id} timed out")
                raise
            except Exception as e:
                self._logger.error(f"Operation {operation_id} failed: {e}")
                raise
            finally:
                # Cleanup
                if operation_id in self._active_operations:
                    del self._active_operations[operation_id]
                if operation_id in self._operation_callbacks:
                    del self._operation_callbacks[operation_id]

    async def _execute_with_progress(
        self,
        operation_id: str,
        task: asyncio.Task,
    ) -> Any:
        """Execute task with progress monitoring."""
        start_time = time.time()

        while not task.done():
            # Notify progress callbacks
            elapsed = time.time() - start_time
            callbacks = self._operation_callbacks.get(operation_id, [])
            for callback in callbacks:
                try:
                    await callback(operation_id, elapsed, "running")
                except Exception as e:
                    self._logger.error(f"Progress callback error: {e}")

            await asyncio.sleep(0.1)  # Check every 100ms

        return await task

    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel an active operation."""
        if operation_id in self._active_operations:
            task = self._active_operations[operation_id]
            task.cancel()
            self._logger.info(f"Operation {operation_id} cancelled")
            return True
        return False

    def get_active_operations(self) -> List[str]:
        """Get list of active operation IDs."""
        return list(self._active_operations.keys())

    def get_operation_status(self, operation_id: str) -> Optional[str]:
        """Get status of an operation."""
        if operation_id in self._active_operations:
            task = self._active_operations[operation_id]
            if task.done():
                return "completed" if not task.cancelled() else "cancelled"
            return "running"
        return None


class IntelligentCache:
    """Intelligent caching system with configurable limits and LRU eviction."""

    def __init__(
        self,
        max_size_mb: int = 500,
        max_entries: int = 1000,
        default_ttl_seconds: Optional[int] = 3600,
    ):
        self._logger = logging.getLogger(__name__)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_entries = max_entries
        self._default_ttl = default_ttl_seconds

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size_bytes = 0
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        self._logger.info(
            f"Cache initialized: max_size={max_size_mb}MB, max_entries={max_entries}",
        )

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired:
                self._remove_entry(key)
                self._misses += 1
                return None

            # Update access info and move to end (most recently used)
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._cache.move_to_end(key)

            self._hits += 1
            return entry.data

    def put(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put item in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(data))
            except Exception:
                # If can't pickle, estimate size
                size_bytes = len(str(data)) * 2  # Rough estimate

            # Check if item is too large
            if size_bytes > self._max_size_bytes:
                self._logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False

            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Make space if needed
            while (
                len(self._cache) >= self._max_entries
                or self._current_size_bytes + size_bytes > self._max_size_bytes
            ):
                if not self._evict_lru():
                    break

            # Add new entry
            entry = CacheEntry(
                key=key,
                data=data,
                size_bytes=size_bytes,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                ttl_seconds=ttl_seconds or self._default_ttl,
            )

            self._cache[key] = entry
            self._current_size_bytes += size_bytes

            self._logger.debug(f"Cached item: {key} ({size_bytes} bytes)")
            return True

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_size_bytes -= entry.size_bytes
            del self._cache[key]

    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self._cache:
            return False

        # Get least recently used item (first in OrderedDict)
        lru_key = next(iter(self._cache))
        self._remove_entry(lru_key)
        self._evictions += 1

        self._logger.debug(f"Evicted LRU item: {lru_key}")
        return True

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            self._logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                self._logger.info(
                    f"Cleaned up {len(expired_keys)} expired cache entries",
                )

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "entries": len(self._cache),
                "size_mb": self._current_size_bytes / (1024 * 1024),
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": hit_rate,
                "evictions": self._evictions,
                "utilization_percent": (len(self._cache) / self._max_entries * 100),
            }

    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()


class ResourceMonitor:
    """Monitors system resources and provides optimization suggestions."""

    def __init__(self, monitoring_interval: float = 5.0):
        self._logger = logging.getLogger(__name__)
        self._monitoring_interval = monitoring_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Resource history
        self._resource_history: List[ResourceUsage] = []
        self._max_history_size = 1000

        # Thresholds for warnings
        self._cpu_warning_threshold = 80.0
        self._memory_warning_threshold = 85.0
        self._disk_warning_threshold = 90.0

        # Callbacks for resource events
        self._warning_callbacks: List[Callable] = []

    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self._logger.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self._logger.info("Resource monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                usage = self._get_current_usage()
                self._resource_history.append(usage)

                # Trim history if too large
                if len(self._resource_history) > self._max_history_size:
                    self._resource_history = self._resource_history[
                        -self._max_history_size :
                    ]

                # Check for warnings
                self._check_resource_warnings(usage)

            except Exception as e:
                self._logger.error(f"Error in resource monitoring: {e}")

            time.sleep(self._monitoring_interval)

    def _get_current_usage(self) -> ResourceUsage:
        """Get current system resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 * 1024 * 1024)

        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
        )

    def _check_resource_warnings(self, usage: ResourceUsage) -> None:
        """Check for resource warnings and notify callbacks."""
        warnings = []

        if usage.cpu_percent > self._cpu_warning_threshold:
            warnings.append(f"High CPU usage: {usage.cpu_percent:.1f}%")

        if usage.memory_percent > self._memory_warning_threshold:
            warnings.append(f"High memory usage: {usage.memory_percent:.1f}%")

        if usage.disk_usage_percent > self._disk_warning_threshold:
            warnings.append(f"High disk usage: {usage.disk_usage_percent:.1f}%")

        if warnings:
            for callback in self._warning_callbacks:
                try:
                    callback(warnings, usage)
                except Exception as e:
                    self._logger.error(f"Error in warning callback: {e}")

    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        return self._get_current_usage()

    def get_usage_history(self, minutes: int = 60) -> List[ResourceUsage]:
        """Get resource usage history for specified minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            usage for usage in self._resource_history if usage.timestamp >= cutoff_time
        ]

    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on current system state."""
        suggestions = []
        current_usage = self.get_current_usage()

        # CPU suggestions
        if current_usage.cpu_percent > 70:
            suggestions.append("Consider reducing concurrent analysis operations")
            suggestions.append("Enable background execution for long-running analyses")

        # Memory suggestions
        if current_usage.memory_percent > 75:
            suggestions.append("Clear analysis result cache to free memory")
            suggestions.append("Reduce cache size limits")
            suggestions.append("Close unused analysis sessions")

        if current_usage.memory_percent > 90:
            suggestions.append(
                "CRITICAL: Very high memory usage - consider restarting application",
            )

        # Disk suggestions
        if current_usage.disk_usage_percent > 85:
            suggestions.append("Clean up old analysis results and temporary files")
            suggestions.append("Archive or export old data")

        # Performance suggestions based on history
        if len(self._resource_history) > 10:
            recent_cpu = [u.cpu_percent for u in self._resource_history[-10:]]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)

            if avg_cpu > 50:
                suggestions.append("Consider enabling analysis result caching")
                suggestions.append(
                    "Use lower fidelity solvers for preliminary analyses",
                )

        return suggestions

    def add_warning_callback(self, callback: Callable) -> None:
        """Add callback for resource warnings."""
        self._warning_callbacks.append(callback)

    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        before_objects = len(gc.get_objects())
        collected = gc.collect()
        after_objects = len(gc.get_objects())

        stats = {
            "objects_before": before_objects,
            "objects_after": after_objects,
            "objects_collected": collected,
            "objects_freed": before_objects - after_objects,
        }

        self._logger.info(f"Garbage collection: {stats}")
        return stats


class BackgroundExecutor:
    """Manages background execution of long-running analyses."""

    def __init__(self, max_workers: int = 4):
        self._logger = logging.getLogger(__name__)
        self._max_workers = max_workers

        # Use both thread and process pools
        self._thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self._process_executor = ProcessPoolExecutor(max_workers=max_workers // 2)

        # Task management
        self._tasks: Dict[str, BackgroundTask] = {}
        self._task_futures: Dict[str, Union[asyncio.Future, Any]] = {}
        self._task_callbacks: Dict[str, List[Callable]] = {}

        # Task queue with priority
        self._task_queue: List[BackgroundTask] = []
        self._queue_lock = threading.Lock()

    def submit_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        *args,
        priority: int = 0,
        use_process: bool = False,
        progress_callback: Optional[Callable] = None,
        completion_callback: Optional[Callable] = None,
        **kwargs,
    ) -> str:
        """Submit a task for background execution."""

        task = BackgroundTask(
            task_id=task_id,
            name=name,
            status="pending",
            progress=0.0,
            created_at=datetime.now(),
            priority=priority,
        )

        self._tasks[task_id] = task

        # Set up callbacks
        callbacks = []
        if progress_callback:
            callbacks.append(progress_callback)
        if completion_callback:
            callbacks.append(completion_callback)
        if callbacks:
            self._task_callbacks[task_id] = callbacks

        # Submit to appropriate executor
        executor = self._process_executor if use_process else self._thread_executor

        try:
            future = executor.submit(self._execute_task, task_id, func, *args, **kwargs)
            self._task_futures[task_id] = future

            # Set up completion callback
            future.add_done_callback(lambda f: self._on_task_completed(task_id, f))

            self._logger.info(f"Background task submitted: {task_id} ({name})")
            return task_id

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self._logger.error(f"Failed to submit background task {task_id}: {e}")
            raise

    def _execute_task(self, task_id: str, func: Callable, *args, **kwargs) -> Any:
        """Execute a background task."""
        task = self._tasks[task_id]
        task.status = "running"
        task.started_at = datetime.now()

        try:
            # Notify progress callbacks
            self._notify_callbacks(task_id, "started", 0.0)

            # Execute function
            if asyncio.iscoroutinefunction(func):
                # Handle async functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(func(*args, **kwargs))
                finally:
                    loop.close()
            else:
                result = func(*args, **kwargs)

            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            task.progress = 100.0

            self._logger.info(f"Background task completed: {task_id}")
            return result

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()
            self._logger.error(f"Background task failed: {task_id} - {e}")
            raise

    def _on_task_completed(self, task_id: str, future) -> None:
        """Handle task completion."""
        try:
            if not future.cancelled():
                result = future.result()
                self._notify_callbacks(task_id, "completed", 100.0, result)
            else:
                self._tasks[task_id].status = "cancelled"
                self._notify_callbacks(task_id, "cancelled", 0.0)
        except Exception as e:
            self._tasks[task_id].status = "failed"
            self._tasks[task_id].error = str(e)
            self._notify_callbacks(task_id, "failed", 0.0, error=str(e))

    def _notify_callbacks(
        self,
        task_id: str,
        status: str,
        progress: float,
        result: Any = None,
        error: str = None,
    ) -> None:
        """Notify task callbacks."""
        callbacks = self._task_callbacks.get(task_id, [])
        for callback in callbacks:
            try:
                callback(task_id, status, progress, result, error)
            except Exception as e:
                self._logger.error(f"Error in task callback: {e}")

    def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get status of a background task."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[BackgroundTask]:
        """Get all background tasks."""
        return list(self._tasks.values())

    def get_running_tasks(self) -> List[BackgroundTask]:
        """Get currently running tasks."""
        return [task for task in self._tasks.values() if task.status == "running"]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a background task."""
        if task_id in self._task_futures:
            future = self._task_futures[task_id]
            cancelled = future.cancel()
            if cancelled or future.done():
                if task_id in self._tasks:
                    self._tasks[task_id].status = "cancelled"
                self._logger.info(f"Background task cancelled: {task_id}")
                return True
        return False

    def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up completed tasks older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        tasks_to_remove = [
            task_id
            for task_id, task in self._tasks.items()
            if task.status in ["completed", "failed", "cancelled"]
            and (task.completed_at or task.created_at) < cutoff_time
        ]

        for task_id in tasks_to_remove:
            del self._tasks[task_id]
            if task_id in self._task_futures:
                del self._task_futures[task_id]
            if task_id in self._task_callbacks:
                del self._task_callbacks[task_id]

        if tasks_to_remove:
            self._logger.info(f"Cleaned up {len(tasks_to_remove)} old background tasks")

        return len(tasks_to_remove)

    def shutdown(self) -> None:
        """Shutdown the background executor."""
        self._logger.info("Shutting down background executor")

        # Cancel all running tasks
        for task_id in list(self._task_futures.keys()):
            self.cancel_task(task_id)

        # Shutdown executors
        self._thread_executor.shutdown(wait=True)
        self._process_executor.shutdown(wait=True)

        self._logger.info("Background executor shutdown complete")


class PerformanceManager:
    """Main performance management coordinator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._logger = logging.getLogger(__name__)

        # Default configuration
        default_config = {
            "cache_max_size_mb": 500,
            "cache_max_entries": 1000,
            "cache_default_ttl_seconds": 3600,
            "max_concurrent_operations": 10,
            "max_background_workers": 4,
            "resource_monitoring_interval": 5.0,
            "auto_cleanup_interval_minutes": 30,
        }

        self._config = {**default_config, **(config or {})}

        # Initialize components
        self.async_manager = AsyncOperationManager(
            max_concurrent_operations=self._config["max_concurrent_operations"],
        )

        self.cache = IntelligentCache(
            max_size_mb=self._config["cache_max_size_mb"],
            max_entries=self._config["cache_max_entries"],
            default_ttl_seconds=self._config["cache_default_ttl_seconds"],
        )

        self.resource_monitor = ResourceMonitor(
            monitoring_interval=self._config["resource_monitoring_interval"],
        )

        self.background_executor = BackgroundExecutor(
            max_workers=self._config["max_background_workers"],
        )

        # Auto-cleanup timer
        self._cleanup_timer: Optional[threading.Timer] = None
        self._start_auto_cleanup()

        # Set up resource warning callback
        self.resource_monitor.add_warning_callback(self._on_resource_warning)

        self._logger.info("PerformanceManager initialized")

    def start(self) -> None:
        """Start performance monitoring and management."""
        self.resource_monitor.start_monitoring()
        self._logger.info("Performance management started")

    def stop(self) -> None:
        """Stop performance monitoring and management."""
        self.resource_monitor.stop_monitoring()
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        self._logger.info("Performance management stopped")

    def _start_auto_cleanup(self) -> None:
        """Start automatic cleanup timer."""
        interval_seconds = self._config["auto_cleanup_interval_minutes"] * 60

        def cleanup_task():
            try:
                self.perform_maintenance()
            except Exception as e:
                self._logger.error(f"Error in auto cleanup: {e}")
            finally:
                # Schedule next cleanup
                self._cleanup_timer = threading.Timer(interval_seconds, cleanup_task)
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()

        self._cleanup_timer = threading.Timer(interval_seconds, cleanup_task)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _on_resource_warning(self, warnings: List[str], usage: ResourceUsage) -> None:
        """Handle resource warnings."""
        self._logger.warning(f"Resource warnings: {warnings}")

        # Auto-optimize if memory is critically high
        if usage.memory_percent > 90:
            self._logger.info("Performing emergency memory cleanup")
            self.emergency_cleanup()

    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform routine maintenance tasks."""
        self._logger.info("Performing maintenance tasks")

        maintenance_stats = {}

        # Clean up expired cache entries
        expired_cleaned = self.cache.cleanup_expired()
        maintenance_stats["cache_expired_cleaned"] = expired_cleaned

        # Clean up old background tasks
        tasks_cleaned = self.background_executor.cleanup_completed_tasks()
        maintenance_stats["background_tasks_cleaned"] = tasks_cleaned

        # Force garbage collection
        gc_stats = self.resource_monitor.force_garbage_collection()
        maintenance_stats["garbage_collection"] = gc_stats

        self._logger.info(f"Maintenance completed: {maintenance_stats}")
        return maintenance_stats

    def emergency_cleanup(self) -> Dict[str, Any]:
        """Perform emergency cleanup when resources are critically low."""
        self._logger.warning("Performing emergency cleanup")

        cleanup_stats = {}

        # Clear cache
        self.cache.clear()
        cleanup_stats["cache_cleared"] = True

        # Force garbage collection multiple times
        for i in range(3):
            gc_stats = self.resource_monitor.force_garbage_collection()
            cleanup_stats[f"gc_round_{i+1}"] = gc_stats

        # Cancel non-critical background tasks
        running_tasks = self.background_executor.get_running_tasks()
        cancelled_count = 0
        for task in running_tasks:
            if task.priority < 5:  # Cancel low priority tasks
                if self.background_executor.cancel_task(task.task_id):
                    cancelled_count += 1

        cleanup_stats["background_tasks_cancelled"] = cancelled_count

        self._logger.warning(f"Emergency cleanup completed: {cleanup_stats}")
        return cleanup_stats

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "resource_usage": self.resource_monitor.get_current_usage(),
            "cache_stats": self.cache.get_stats(),
            "active_operations": len(self.async_manager.get_active_operations()),
            "background_tasks": {
                "total": len(self.background_executor.get_all_tasks()),
                "running": len(self.background_executor.get_running_tasks()),
            },
            "optimization_suggestions": self.resource_monitor.get_optimization_suggestions(),
            "config": self._config,
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update performance configuration."""
        self._config.update(new_config)
        self._logger.info(f"Performance configuration updated: {new_config}")

    def shutdown(self) -> None:
        """Shutdown performance manager."""
        self._logger.info("Shutting down performance manager")

        self.stop()
        self.background_executor.shutdown()

        self._logger.info("Performance manager shutdown complete")


# Global performance manager instance
performance_manager: Optional[PerformanceManager] = None


def get_performance_manager() -> PerformanceManager:
    """Get global performance manager instance."""
    global performance_manager
    if performance_manager is None:
        performance_manager = PerformanceManager()
    return performance_manager


def initialize_performance_system(
    config: Optional[Dict[str, Any]] = None,
) -> PerformanceManager:
    """Initialize the performance system with configuration."""
    global performance_manager
    performance_manager = PerformanceManager(config)
    performance_manager.start()
    return performance_manager
