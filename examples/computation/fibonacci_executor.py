"""
Fibonacci calculation executor for simulation framework demo.

This module provides an executor that calculates Fibonacci numbers
with progress reporting, showcasing the simulation framework capabilities.
"""

import asyncio
from typing import Tuple

from ICARUS.computation.core.context import ExecutionContext
from ICARUS.computation.core.protocols import TaskExecutor


class FibonacciExecutor(TaskExecutor[int, Tuple[int, int]]):
    """
    Executor that calculates Fibonacci numbers with progress reporting.

    This executor demonstrates how to implement a TaskExecutor that:
    - Performs actual computation work
    - Reports progress during execution
    - Handles cancellation gracefully
    - Validates input parameters
    """

    def __init__(self, delay_per_step: float = 0.01, max_number: int = 10000):
        """
        Initialize the Fibonacci executor.

        Args:
            delay_per_step: Artificial delay per calculation step for demo purposes
            max_number: Maximum Fibonacci number to calculate for safety
        """
        self.delay_per_step = delay_per_step
        self.max_number = max_number

    async def execute(self, n: int, context: ExecutionContext) -> Tuple[int, int]:
        """
        Calculate the nth Fibonacci number with progress reporting.

        Args:
            n: Position in Fibonacci sequence to calculate
            context: Execution context for progress reporting and cancellation

        Returns:
            Tuple of (position, fibonacci_value)

        Raises:
            asyncio.CancelledError: If task is cancelled during execution
            ValueError: If n is invalid
        """
        if not await self.validate_input(n):
            raise ValueError(f"Invalid Fibonacci input: {n}")

        # Handle edge cases
        if n <= 0:
            context.report_progress(1, 1, "Completed: F(0) = 0")
            return (n, 0)
        elif n == 1:
            context.report_progress(1, 1, "Completed: F(1) = 1")
            return (n, 1)

        # Initialize for iterative calculation
        a, b = 0, 1
        total_steps = n

        # Report initial progress
        context.report_progress(1, total_steps, f"Starting F({n}) calculation...")

        # Use time.sleep instead of asyncio.sleep for multiprocessing compatibility
        if self.delay_per_step > 0:
            import time

            time.sleep(self.delay_per_step)

        # Iterative Fibonacci calculation with progress reporting
        for i in range(2, n + 1):
            # Calculate next Fibonacci number
            a, b = b, a + b

            # Report progress every few steps or at key milestones
            if i % max(1, n // 20) == 0 or i == n:
                percentage = (i / n) * 100
                context.report_progress(i, n, f"F({i}) = {b:,} ({percentage:.1f}%)")

            # Add artificial delay for demo visualization
            if self.delay_per_step > 0:
                import time

                time.sleep(self.delay_per_step)
                await asyncio.sleep(0)  # Yield control to event loop

        # Final progress report
        context.report_progress(n, n, f"Completed: F({n}) = {b:,}")

        return (n, b)

    async def validate_input(self, n: int) -> bool:
        """
        Validate that the input is a valid Fibonacci sequence position.

        Args:
            n: Position to validate

        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(n, int):
            return False

        if n < 0:
            return False

        if n > self.max_number:
            return False

        return True

    async def cancel(self) -> None:
        """
        Handle cancellation of the Fibonacci calculation.
        This method can be used to clean up resources or state if needed.
        """
        pass

    async def cleanup(self) -> None:
        pass
