"""Real-time Updater - Handles real-time plot updates during analysis

This module provides real-time plot updating capabilities for long-running
analyses, allowing users to see progress and results as they develop.
"""

import threading
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from rich.console import Console


class RealTimeUpdater:
    """Manages real-time plot updates."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the real-time updater.

        Args:
            console: Rich console for output (optional)
        """
        self.console = console or Console()

        # Track active updates
        self.active_updates: Dict[str, Dict[str, Any]] = {}
        self.update_counter = 0

        # Animation objects
        self.animations: Dict[str, FuncAnimation] = {}

        # Data buffers for streaming data
        self.data_buffers: Dict[str, Dict[str, List]] = {}

    def start_updates(
        self,
        plot: Figure,
        data_source: Any,
        update_interval: float = 1.0,
        max_points: Optional[int] = None,
        update_callback: Optional[Callable] = None,
    ) -> str:
        """Start real-time updates for a plot.

        Args:
            plot: Figure to update
            data_source: Source of real-time data
            update_interval: Update interval in seconds
            max_points: Maximum number of points to keep (for streaming)
            update_callback: Custom update callback function

        Returns:
            Update ID for tracking
        """
        try:
            update_id = f"update_{self.update_counter}"
            self.update_counter += 1

            # Initialize data buffer
            self.data_buffers[update_id] = {"x": [], "y": [], "timestamps": []}

            # Store update configuration
            self.active_updates[update_id] = {
                "plot": plot,
                "data_source": data_source,
                "update_interval": update_interval,
                "max_points": max_points,
                "update_callback": update_callback,
                "start_time": time.time(),
                "update_count": 0,
                "is_running": True,
            }

            # Start the update mechanism
            if hasattr(data_source, "get_data"):
                # Data source with get_data method
                self._start_polling_updates(update_id)
            elif callable(data_source):
                # Data source is a function
                self._start_callback_updates(update_id)
            elif hasattr(data_source, "__iter__"):
                # Data source is iterable (streaming)
                self._start_streaming_updates(update_id)
            else:
                raise ValueError("Unsupported data source type")

            self.console.print(
                f"[green]✓[/green] Started real-time updates: {update_id}",
            )
            return update_id

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to start real-time updates: {e}")
            raise

    def _start_polling_updates(self, update_id: str) -> None:
        """Start polling-based updates."""
        update_info = self.active_updates[update_id]
        plot = update_info["plot"]

        def update_function(frame):
            if not update_info["is_running"]:
                return

            try:
                # Get new data
                new_data = update_info["data_source"].get_data()

                # Update plot
                self._update_plot_data(update_id, new_data)

                # Increment counter
                update_info["update_count"] += 1

            except Exception as e:
                self.console.print(f"[red]✗[/red] Update error: {e}")

        # Create animation
        interval_ms = int(update_info["update_interval"] * 1000)
        animation = FuncAnimation(
            plot,
            update_function,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

        self.animations[update_id] = animation

    def _start_callback_updates(self, update_id: str) -> None:
        """Start callback-based updates."""
        update_info = self.active_updates[update_id]

        def update_thread():
            while update_info["is_running"]:
                try:
                    # Get new data from callback
                    new_data = update_info["data_source"]()

                    # Update plot
                    self._update_plot_data(update_id, new_data)

                    # Increment counter
                    update_info["update_count"] += 1

                    # Wait for next update
                    time.sleep(update_info["update_interval"])

                except Exception as e:
                    self.console.print(f"[red]✗[/red] Update error: {e}")
                    break

        # Start update thread
        thread = threading.Thread(target=update_thread, daemon=True)
        thread.start()
        update_info["thread"] = thread

    def _start_streaming_updates(self, update_id: str) -> None:
        """Start streaming-based updates."""
        update_info = self.active_updates[update_id]

        def stream_thread():
            try:
                for data_point in update_info["data_source"]:
                    if not update_info["is_running"]:
                        break

                    # Update plot with new data point
                    self._update_plot_data(update_id, data_point)

                    # Increment counter
                    update_info["update_count"] += 1

                    # Small delay to prevent overwhelming
                    time.sleep(0.01)

            except Exception as e:
                self.console.print(f"[red]✗[/red] Streaming error: {e}")

        # Start streaming thread
        thread = threading.Thread(target=stream_thread, daemon=True)
        thread.start()
        update_info["thread"] = thread

    def _update_plot_data(self, update_id: str, new_data: Dict[str, Any]) -> None:
        """Update plot with new data."""
        try:
            update_info = self.active_updates[update_id]
            plot = update_info["plot"]
            buffer = self.data_buffers[update_id]

            # Add new data to buffer
            if isinstance(new_data, dict):
                for key, value in new_data.items():
                    if key not in buffer:
                        buffer[key] = []

                    if isinstance(value, (list, tuple)):
                        buffer[key].extend(value)
                    else:
                        buffer[key].append(value)

            # Apply max points limit
            max_points = update_info.get("max_points")
            if max_points:
                for key in buffer:
                    if len(buffer[key]) > max_points:
                        buffer[key] = buffer[key][-max_points:]

            # Update plot
            if update_info.get("update_callback"):
                # Use custom callback
                update_info["update_callback"](plot, buffer)
            else:
                # Default update behavior
                self._default_plot_update(plot, buffer)

            # Redraw
            plot.canvas.draw()
            plot.canvas.flush_events()

        except Exception as e:
            self.console.print(f"[red]✗[/red] Plot update error: {e}")

    def _default_plot_update(self, plot: Figure, buffer: Dict[str, List]) -> None:
        """Default plot update behavior."""
        # Update first axis (simplified)
        if plot.axes and buffer:
            ax = plot.axes[0]

            # Update line data
            if ax.lines and "x" in buffer and "y" in buffer:
                line = ax.lines[0]
                line.set_xdata(buffer["x"])
                line.set_ydata(buffer["y"])

                # Rescale axes
                ax.relim()
                ax.autoscale_view()

    def stop_updates(self, update_id: str) -> bool:
        """Stop real-time updates.

        Args:
            update_id: ID of the update to stop

        Returns:
            True if successful, False otherwise
        """
        try:
            if update_id not in self.active_updates:
                self.console.print(f"[red]✗[/red] Update {update_id} not found")
                return False

            update_info = self.active_updates[update_id]
            update_info["is_running"] = False

            # Stop animation if exists
            if update_id in self.animations:
                self.animations[update_id].event_source.stop()
                del self.animations[update_id]

            # Wait for thread to finish
            if "thread" in update_info:
                update_info["thread"].join(timeout=1.0)

            # Clean up
            del self.active_updates[update_id]
            if update_id in self.data_buffers:
                del self.data_buffers[update_id]

            self.console.print(
                f"[green]✓[/green] Stopped real-time updates: {update_id}",
            )
            return True

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to stop updates: {e}")
            return False

    def pause_updates(self, update_id: str) -> bool:
        """Pause real-time updates.

        Args:
            update_id: ID of the update to pause

        Returns:
            True if successful, False otherwise
        """
        try:
            if update_id not in self.active_updates:
                return False

            update_info = self.active_updates[update_id]
            update_info["is_paused"] = True

            if update_id in self.animations:
                self.animations[update_id].pause()

            self.console.print(f"[yellow]⏸[/yellow] Paused updates: {update_id}")
            return True

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to pause updates: {e}")
            return False

    def resume_updates(self, update_id: str) -> bool:
        """Resume paused real-time updates.

        Args:
            update_id: ID of the update to resume

        Returns:
            True if successful, False otherwise
        """
        try:
            if update_id not in self.active_updates:
                return False

            update_info = self.active_updates[update_id]
            update_info["is_paused"] = False

            if update_id in self.animations:
                self.animations[update_id].resume()

            self.console.print(f"[green]▶[/green] Resumed updates: {update_id}")
            return True

        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to resume updates: {e}")
            return False

    def get_update_status(self, update_id: str) -> Optional[Dict[str, Any]]:
        """Get status of real-time updates.

        Args:
            update_id: ID of the update

        Returns:
            Status dictionary or None if not found
        """
        if update_id not in self.active_updates:
            return None

        update_info = self.active_updates[update_id]
        current_time = time.time()

        return {
            "update_id": update_id,
            "is_running": update_info["is_running"],
            "is_paused": update_info.get("is_paused", False),
            "update_count": update_info["update_count"],
            "elapsed_time": current_time - update_info["start_time"],
            "update_interval": update_info["update_interval"],
            "data_points": len(self.data_buffers.get(update_id, {}).get("x", [])),
        }

    def list_active_updates(self) -> List[Dict[str, Any]]:
        """Get list of all active updates.

        Returns:
            List of update status dictionaries
        """
        return [
            self.get_update_status(update_id)
            for update_id in self.active_updates.keys()
        ]

    def stop_all_updates(self) -> int:
        """Stop all active real-time updates.

        Returns:
            Number of updates stopped
        """
        update_ids = list(self.active_updates.keys())
        stopped_count = 0

        for update_id in update_ids:
            if self.stop_updates(update_id):
                stopped_count += 1

        self.console.print(
            f"[green]✓[/green] Stopped {stopped_count} real-time updates",
        )
        return stopped_count

    def create_data_source(
        self,
        data_function: Callable,
        source_type: str = "polling",
    ) -> Any:
        """Create a data source wrapper.

        Args:
            data_function: Function that returns data
            source_type: Type of data source (polling, streaming, callback)

        Returns:
            Data source object
        """
        if source_type == "polling":
            return DataSourcePolling(data_function)
        elif source_type == "streaming":
            return DataSourceStreaming(data_function)
        elif source_type == "callback":
            return data_function
        else:
            raise ValueError(f"Unsupported source type: {source_type}")


class DataSourcePolling:
    """Polling-based data source."""

    def __init__(self, data_function: Callable):
        self.data_function = data_function

    def get_data(self) -> Dict[str, Any]:
        return self.data_function()


class DataSourceStreaming:
    """Streaming-based data source."""

    def __init__(self, data_generator: Callable):
        self.data_generator = data_generator

    def __iter__(self):
        return self.data_generator()


# Example usage and data source implementations
class AnalysisDataSource:
    """Example data source for analysis progress."""

    def __init__(self, analysis_runner):
        self.analysis_runner = analysis_runner
        self.iteration = 0

    def get_data(self) -> Dict[str, Any]:
        """Get current analysis data."""
        if hasattr(self.analysis_runner, "get_current_state"):
            state = self.analysis_runner.get_current_state()
            return {
                "iteration": [self.iteration],
                "residual": [state.get("residual", 0)],
                "cl": [state.get("cl", 0)],
                "cd": [state.get("cd", 0)],
            }
        else:
            # Simulate data for demo
            self.iteration += 1
            return {
                "iteration": [self.iteration],
                "residual": [1.0 / self.iteration],
                "cl": [0.5 + 0.1 * (self.iteration % 10)],
                "cd": [0.01 + 0.001 * (self.iteration % 5)],
            }
