from __future__ import annotations

import asyncio

from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import TaskExecutor


class SummationExecutor(TaskExecutor):
    """Example executor for summation tasks"""

    async def execute(self, n: int, context: ExecutionContext) -> int:
        """Calculate sum with progress reporting"""
        total = 0
        for i in range(n):
            if context.is_cancelled:
                raise asyncio.CancelledError("Task was cancelled")

            total += i

            # Report progress every 10 iterations
            if i % 10 == 0:
                await context.report_progress(i, n, f"Processing {i}/{n}")

            # Simulate work
            await asyncio.sleep(0.001)

        await context.report_progress(n, n, "Completed")
        return total

    async def validate_input(self, n: int) -> bool:
        """Validate that n is a positive integer"""
        return isinstance(n, int) and n > 0

    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass
