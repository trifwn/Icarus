"""API adaptation layer for external tool updates."""

import asyncio
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import aiohttp
import semver

from .models import AuthenticationConfig
from .models import ExternalToolConfig


@dataclass
class APIVersion:
    """API version information."""

    version: str
    release_date: str
    deprecated: bool = False
    sunset_date: Optional[str] = None
    breaking_changes: List[str] = field(default_factory=list)
    migration_guide: Optional[str] = None


@dataclass
class APIChange:
    """API change notification."""

    endpoint: str
    change_type: str  # 'added', 'modified', 'deprecated', 'removed'
    description: str
    version: str
    impact_level: str  # 'low', 'medium', 'high', 'breaking'
    migration_required: bool = False
    migration_steps: List[str] = field(default_factory=list)


class APIAdapter:
    """Handles API adaptation and version management for external tools."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self._tools: Dict[str, ExternalToolConfig] = {}
        self._adapters: Dict[str, Dict[str, Callable]] = {}
        self._version_cache: Dict[str, APIVersion] = {}
        self._change_handlers: Dict[str, List[Callable]] = {}

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def register_tool(self, tool_config: ExternalToolConfig) -> None:
        """Register an external tool configuration."""
        self._tools[tool_config.name] = tool_config
        self._adapters[tool_config.name] = {}
        self._change_handlers[tool_config.name] = []
        self.logger.info(f"Registered external tool: {tool_config.name}")

    def register_adapter(
        self,
        tool_name: str,
        version: str,
        adapter_func: Callable,
    ) -> None:
        """Register a version-specific adapter function."""
        if tool_name not in self._adapters:
            self._adapters[tool_name] = {}

        self._adapters[tool_name][version] = adapter_func
        self.logger.info(f"Registered adapter for {tool_name} v{version}")

    def register_change_handler(
        self,
        tool_name: str,
        handler: Callable[[APIChange], None],
    ) -> None:
        """Register a handler for API changes."""
        if tool_name not in self._change_handlers:
            self._change_handlers[tool_name] = []

        self._change_handlers[tool_name].append(handler)
        self.logger.info(f"Registered change handler for {tool_name}")

    async def check_api_version(self, tool_name: str) -> Optional[APIVersion]:
        """Check the current API version for a tool."""
        tool_config = self._tools.get(tool_name)
        if not tool_config:
            raise ValueError(f"Tool not registered: {tool_name}")

        # Find version endpoint
        version_endpoint = None
        for endpoint in tool_config.api_endpoints:
            if "version" in endpoint.name.lower():
                version_endpoint = endpoint
                break

        if not version_endpoint:
            self.logger.warning(f"No version endpoint found for {tool_name}")
            return None

        try:
            headers = {}
            if version_endpoint.auth_config:
                headers = await self._get_auth_headers(version_endpoint.auth_config)

            async with self.session.get(
                version_endpoint.url,
                headers=headers,
                timeout=version_endpoint.timeout,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    version_info = self._parse_version_response(data)

                    if version_info:
                        self._version_cache[tool_name] = version_info
                        return version_info
                else:
                    self.logger.error(
                        f"Version check failed for {tool_name}: {response.status}",
                    )

        except Exception as e:
            self.logger.error(f"Error checking version for {tool_name}: {e}")

        return None

    def _parse_version_response(self, data: Dict[str, Any]) -> Optional[APIVersion]:
        """Parse version information from API response."""
        try:
            # Common version response formats
            version = None
            if "version" in data:
                version = data["version"]
            elif "api_version" in data:
                version = data["api_version"]
            elif "v" in data:
                version = data["v"]

            if not version:
                return None

            return APIVersion(
                version=version,
                release_date=data.get("release_date", ""),
                deprecated=data.get("deprecated", False),
                sunset_date=data.get("sunset_date"),
                breaking_changes=data.get("breaking_changes", []),
                migration_guide=data.get("migration_guide"),
            )
        except Exception as e:
            self.logger.error(f"Error parsing version response: {e}")
            return None

    async def _get_auth_headers(
        self,
        auth_config: AuthenticationConfig,
    ) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        headers = {}

        if auth_config.type.value == "api_key":
            api_key = auth_config.get_credential("api_key")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        elif auth_config.type.value == "basic_auth":
            username = auth_config.get_credential("username")
            password = auth_config.get_credential("password")
            if username and password:
                import base64

                credentials = base64.b64encode(
                    f"{username}:{password}".encode(),
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"

        return headers

    async def make_api_call(
        self,
        tool_name: str,
        endpoint_name: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make an API call with automatic version adaptation."""
        tool_config = self._tools.get(tool_name)
        if not tool_config:
            raise ValueError(f"Tool not registered: {tool_name}")

        # Find endpoint
        endpoint = None
        for ep in tool_config.api_endpoints:
            if ep.name == endpoint_name:
                endpoint = ep
                break

        if not endpoint:
            raise ValueError(f"Endpoint not found: {endpoint_name}")

        # Check if we need to adapt the request
        current_version = await self.check_api_version(tool_name)
        if current_version and tool_config.version:
            adapted_data = await self._adapt_request(
                tool_name,
                endpoint_name,
                data,
                tool_config.version,
                current_version.version,
            )
            if adapted_data is not None:
                data = adapted_data

        # Make the API call
        try:
            headers = endpoint.headers.copy()
            if endpoint.auth_config:
                auth_headers = await self._get_auth_headers(endpoint.auth_config)
                headers.update(auth_headers)

            kwargs = {"headers": headers, "timeout": endpoint.timeout, "params": params}

            if data:
                kwargs["json"] = data

            async with self.session.request(
                endpoint.method,
                endpoint.url,
                **kwargs,
            ) as response:
                if response.status < 400:
                    response_data = await response.json()

                    # Adapt response if needed
                    if current_version and tool_config.version:
                        adapted_response = await self._adapt_response(
                            tool_name,
                            endpoint_name,
                            response_data,
                            current_version.version,
                            tool_config.version,
                        )
                        if adapted_response is not None:
                            response_data = adapted_response

                    return response_data
                else:
                    self.logger.error(
                        f"API call failed: {response.status} - {await response.text()}",
                    )
                    return None

        except Exception as e:
            self.logger.error(f"API call error for {tool_name}.{endpoint_name}: {e}")
            return None

    async def _adapt_request(
        self,
        tool_name: str,
        endpoint_name: str,
        data: Optional[Dict[str, Any]],
        from_version: str,
        to_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Adapt request data for version compatibility."""
        if not data or from_version == to_version:
            return data

        # Check if we have a specific adapter
        adapters = self._adapters.get(tool_name, {})
        adapter_key = f"{from_version}->{to_version}"

        if adapter_key in adapters:
            try:
                return adapters[adapter_key](data, "request")
            except Exception as e:
                self.logger.error(f"Request adaptation failed: {e}")

        # Try generic version-based adaptation
        return await self._generic_request_adaptation(data, from_version, to_version)

    async def _adapt_response(
        self,
        tool_name: str,
        endpoint_name: str,
        data: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Adapt response data for version compatibility."""
        if from_version == to_version:
            return data

        # Check if we have a specific adapter
        adapters = self._adapters.get(tool_name, {})
        adapter_key = f"{from_version}->{to_version}"

        if adapter_key in adapters:
            try:
                return adapters[adapter_key](data, "response")
            except Exception as e:
                self.logger.error(f"Response adaptation failed: {e}")

        # Try generic version-based adaptation
        return await self._generic_response_adaptation(data, from_version, to_version)

    async def _generic_request_adaptation(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """Generic request adaptation based on version differences."""
        # This is a simplified implementation
        # In practice, you'd have more sophisticated version-specific logic

        try:
            from_ver = semver.VersionInfo.parse(from_version)
            to_ver = semver.VersionInfo.parse(to_version)

            # If upgrading to a newer version
            if to_ver > from_ver:
                # Add new required fields with defaults
                if to_ver.major > from_ver.major:
                    # Major version change - might need significant adaptation
                    data = await self._adapt_major_version_request(
                        data,
                        from_ver,
                        to_ver,
                    )
                elif to_ver.minor > from_ver.minor:
                    # Minor version change - add new optional fields
                    data = await self._adapt_minor_version_request(
                        data,
                        from_ver,
                        to_ver,
                    )

            # If downgrading to an older version
            elif to_ver < from_ver:
                # Remove fields that don't exist in older version
                data = await self._adapt_downgrade_request(data, from_ver, to_ver)

        except Exception as e:
            self.logger.warning(f"Generic request adaptation failed: {e}")

        return data

    async def _generic_response_adaptation(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """Generic response adaptation based on version differences."""
        try:
            from_ver = semver.VersionInfo.parse(from_version)
            to_ver = semver.VersionInfo.parse(to_version)

            # If response is from newer version, adapt to older format
            if from_ver > to_ver:
                data = await self._adapt_newer_response(data, from_ver, to_ver)
            # If response is from older version, add missing fields
            elif from_ver < to_ver:
                data = await self._adapt_older_response(data, from_ver, to_ver)

        except Exception as e:
            self.logger.warning(f"Generic response adaptation failed: {e}")

        return data

    async def _adapt_major_version_request(
        self,
        data: Dict[str, Any],
        from_ver: semver.VersionInfo,
        to_ver: semver.VersionInfo,
    ) -> Dict[str, Any]:
        """Adapt request for major version changes."""
        # Example adaptations for major version changes
        adapted_data = data.copy()

        # Common patterns for major version changes
        if "id" in adapted_data and to_ver.major >= 2:
            # v2+ might require UUID format
            if isinstance(adapted_data["id"], int):
                adapted_data["id"] = str(adapted_data["id"])

        return adapted_data

    async def _adapt_minor_version_request(
        self,
        data: Dict[str, Any],
        from_ver: semver.VersionInfo,
        to_ver: semver.VersionInfo,
    ) -> Dict[str, Any]:
        """Adapt request for minor version changes."""
        adapted_data = data.copy()

        # Add default values for new optional fields
        # This would be customized based on actual API changes

        return adapted_data

    async def _adapt_downgrade_request(
        self,
        data: Dict[str, Any],
        from_ver: semver.VersionInfo,
        to_ver: semver.VersionInfo,
    ) -> Dict[str, Any]:
        """Adapt request for downgrade to older version."""
        adapted_data = data.copy()

        # Remove fields that don't exist in older versions
        # This would be based on actual API documentation

        return adapted_data

    async def _adapt_newer_response(
        self,
        data: Dict[str, Any],
        from_ver: semver.VersionInfo,
        to_ver: semver.VersionInfo,
    ) -> Dict[str, Any]:
        """Adapt response from newer API to older format."""
        adapted_data = data.copy()

        # Remove or transform fields that didn't exist in older versions

        return adapted_data

    async def _adapt_older_response(
        self,
        data: Dict[str, Any],
        from_ver: semver.VersionInfo,
        to_ver: semver.VersionInfo,
    ) -> Dict[str, Any]:
        """Adapt response from older API to newer format."""
        adapted_data = data.copy()

        # Add missing fields with default values

        return adapted_data

    async def check_for_updates(self, tool_name: str) -> List[APIChange]:
        """Check for API updates and changes."""
        tool_config = self._tools.get(tool_name)
        if not tool_config:
            raise ValueError(f"Tool not registered: {tool_name}")

        current_version = await self.check_api_version(tool_name)
        if not current_version:
            return []

        # Compare with cached version
        cached_version = self._version_cache.get(tool_name)
        if not cached_version or cached_version.version == current_version.version:
            return []

        # Detect changes (simplified implementation)
        changes = []

        try:
            cached_ver = semver.VersionInfo.parse(cached_version.version)
            current_ver = semver.VersionInfo.parse(current_version.version)

            if current_ver > cached_ver:
                if current_ver.major > cached_ver.major:
                    changes.append(
                        APIChange(
                            endpoint="*",
                            change_type="modified",
                            description=f"Major version update: {cached_ver} -> {current_ver}",
                            version=current_version.version,
                            impact_level="breaking",
                            migration_required=True,
                            migration_steps=current_version.breaking_changes,
                        ),
                    )
                elif current_ver.minor > cached_ver.minor:
                    changes.append(
                        APIChange(
                            endpoint="*",
                            change_type="added",
                            description=f"Minor version update: {cached_ver} -> {current_ver}",
                            version=current_version.version,
                            impact_level="medium",
                        ),
                    )
                else:
                    changes.append(
                        APIChange(
                            endpoint="*",
                            change_type="modified",
                            description=f"Patch version update: {cached_ver} -> {current_ver}",
                            version=current_version.version,
                            impact_level="low",
                        ),
                    )

        except Exception as e:
            self.logger.error(f"Error detecting changes for {tool_name}: {e}")

        # Notify change handlers
        for change in changes:
            await self._notify_change_handlers(tool_name, change)

        return changes

    async def _notify_change_handlers(self, tool_name: str, change: APIChange) -> None:
        """Notify registered change handlers."""
        handlers = self._change_handlers.get(tool_name, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(change)
                else:
                    handler(change)
            except Exception as e:
                self.logger.error(f"Change handler error: {e}")

    async def update_tool_config(self, tool_name: str, new_version: str) -> bool:
        """Update tool configuration to new version."""
        tool_config = self._tools.get(tool_name)
        if not tool_config:
            return False

        try:
            # Update version
            tool_config.version = new_version
            tool_config.last_updated = tool_config.last_updated.__class__.now()

            # Check if tool is still available with new version
            version_info = await self.check_api_version(tool_name)
            tool_config.is_available = version_info is not None

            self.logger.info(f"Updated {tool_name} to version {new_version}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update {tool_name}: {e}")
            return False

    def get_tool_status(self, tool_name: str) -> Dict[str, Any]:
        """Get current status of a tool."""
        tool_config = self._tools.get(tool_name)
        if not tool_config:
            return {"error": "Tool not found"}

        cached_version = self._version_cache.get(tool_name)

        return {
            "name": tool_config.name,
            "configured_version": tool_config.version,
            "current_version": cached_version.version if cached_version else None,
            "is_available": tool_config.is_available,
            "last_updated": tool_config.last_updated.isoformat(),
            "endpoints": len(tool_config.api_endpoints),
            "has_adapters": len(self._adapters.get(tool_name, {})) > 0,
        }

    def list_registered_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    async def test_tool_connectivity(self, tool_name: str) -> Dict[str, Any]:
        """Test connectivity to all endpoints of a tool."""
        tool_config = self._tools.get(tool_name)
        if not tool_config:
            return {"error": "Tool not found"}

        results = {}

        for endpoint in tool_config.api_endpoints:
            try:
                headers = endpoint.headers.copy()
                if endpoint.auth_config:
                    auth_headers = await self._get_auth_headers(endpoint.auth_config)
                    headers.update(auth_headers)

                async with self.session.get(
                    endpoint.url,
                    headers=headers,
                    timeout=endpoint.timeout,
                ) as response:
                    results[endpoint.name] = {
                        "status": response.status,
                        "available": response.status < 400,
                        "response_time": response.headers.get(
                            "X-Response-Time",
                            "unknown",
                        ),
                    }
            except Exception as e:
                results[endpoint.name] = {
                    "status": "error",
                    "available": False,
                    "error": str(e),
                }

        return results
