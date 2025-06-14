from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any
from typing import Dict


class SimpleResourceManager:
    """Basic resource manager implementation"""

    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self._locks: Dict[str, Lock] = defaultdict(Lock)

    async def acquire_resources(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Acquire resources based on requirements"""
        acquired = {}
        for name, requirement in requirements.items():
            with self._locks[name]:
                if name not in self._resources:
                    self._resources[name] = self._create_resource(name, requirement)
                acquired[name] = self._resources[name]
        return acquired

    async def release_resources(self, resources: Dict[str, Any]) -> None:
        """Release resources (no-op for simple manager)"""
        pass

    def _create_resource(self, name: str, requirement: Any) -> Any:
        """Create a resource based on requirement"""
        if isinstance(requirement, dict) and requirement.get("type") == "file":
            return Path(requirement["path"])
        return requirement
