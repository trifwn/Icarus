from asyncio.locks import Lock as asyncio_lock
from collections import defaultdict
from multiprocessing.synchronize import Lock as mp_lock
from pathlib import Path
from threading import _RLock as threading_lock
from typing import Any

from ICARUS.computation.core import ConcurrencyPrimitives
from ICARUS.computation.core import LockLike
from ICARUS.computation.core.protocols import ResourceManager


class SimpleResourceManager(ResourceManager):
    """Basic resource manager implementation"""

    def __init__(self, primitives: ConcurrencyPrimitives) -> None:
        self._resources: dict[str, Any] = {}
        self._locks: dict[str, LockLike] = defaultdict(primitives.lock)

    @property
    def locks(self) -> dict[str, LockLike]:
        """Return a dictionary of locks for resource management"""
        return self._locks

    async def acquire_resources(self, requirements: dict[str, Any]) -> dict[str, Any]:
        """Acquire resources based on requirements"""
        acquired = {}
        for name, requirement in requirements.items():
            lock = self.locks.get(name)

            if isinstance(lock, mp_lock):
                # For multiprocessing locks, we need to use a context manager
                with lock:
                    if name not in self._resources:
                        self._resources[name] = self._create_resource(name, requirement)
                    acquired[name] = self._resources[name]

            elif isinstance(lock, asyncio_lock):
                # For asyncio locks, we need to use an async context manager
                async with lock:
                    if name not in self._resources:
                        self._resources[name] = self._create_resource(name, requirement)
                    acquired[name] = self._resources[name]

            elif isinstance(lock, threading_lock):
                # For threading locks, we can use a context manager
                with lock:
                    if name not in self._resources:
                        self._resources[name] = self._create_resource(name, requirement)
                    acquired[name] = self._resources[name]

            else:
                if name not in self._resources:
                    self._resources[name] = self._create_resource(name, requirement)
                acquired[name] = self._resources[name]

        return acquired

    async def release_resources(self, resources: dict[str, Any]) -> None:
        """Release resources (no-op for simple manager)"""
        pass

    def _create_resource(self, name: str, requirement: Any) -> Any:
        """Create a resource based on requirement"""
        if isinstance(requirement, dict) and requirement.get("type") == "file":
            return Path(requirement["path"])
        return requirement
