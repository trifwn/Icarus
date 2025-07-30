"""
Plugin marketplace integration for ICARUS CLI plugins.

This module provides tools for publishing and discovering plugins
through various marketplace platforms.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import requests

from ..models import PluginType


class MarketplaceConfig:
    """Configuration for marketplace integration."""

    def __init__(self, config_data: Dict[str, Any]):
        self.name = config_data.get("name", "default")
        self.base_url = config_data.get(
            "base_url",
            "https://plugins.icarus.example.com",
        )
        self.api_key = config_data.get("api_key")
        self.username = config_data.get("username")
        self.timeout = config_data.get("timeout", 30)
        self.verify_ssl = config_data.get("verify_ssl", True)


class PluginMarketplace:
    """
    Plugin marketplace integration that handles plugin publishing,
    discovery, and installation from various marketplace platforms.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config_file = Path.home() / ".icarus" / "marketplace.json"
        self.marketplaces: Dict[str, MarketplaceConfig] = {}

        # Load marketplace configurations
        self._load_marketplace_config()

    def _load_marketplace_config(self) -> None:
        """Load marketplace configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    config_data = json.load(f)

                for name, marketplace_config in config_data.get(
                    "marketplaces",
                    {},
                ).items():
                    self.marketplaces[name] = MarketplaceConfig(marketplace_config)

                self.logger.info(
                    f"Loaded {len(self.marketplaces)} marketplace configurations",
                )
            else:
                # Create default configuration
                self._create_default_config()

        except Exception as e:
            self.logger.error(f"Failed to load marketplace config: {e}")
            self._create_default_config()

    def _create_default_config(self) -> None:
        """Create default marketplace configuration."""
        default_config = {
            "marketplaces": {
                "official": {
                    "name": "ICARUS Official Plugin Repository",
                    "base_url": "https://plugins.icarus.example.com",
                    "verify_ssl": True,
                    "timeout": 30,
                },
                "community": {
                    "name": "ICARUS Community Plugins",
                    "base_url": "https://community-plugins.icarus.example.com",
                    "verify_ssl": True,
                    "timeout": 30,
                },
            },
        }

        # Create config directory
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # Save default config
        with open(self.config_file, "w") as f:
            json.dump(default_config, f, indent=2)

        # Load the default config
        for name, marketplace_config in default_config["marketplaces"].items():
            self.marketplaces[name] = MarketplaceConfig(marketplace_config)

    def add_marketplace(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Add a new marketplace configuration.

        Args:
            name: Marketplace name
            config: Marketplace configuration

        Returns:
            True if added successfully, False otherwise
        """
        try:
            self.marketplaces[name] = MarketplaceConfig(config)
            self._save_marketplace_config()
            self.logger.info(f"Added marketplace: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add marketplace {name}: {e}")
            return False

    def remove_marketplace(self, name: str) -> bool:
        """
        Remove a marketplace configuration.

        Args:
            name: Marketplace name

        Returns:
            True if removed successfully, False otherwise
        """
        try:
            if name in self.marketplaces:
                del self.marketplaces[name]
                self._save_marketplace_config()
                self.logger.info(f"Removed marketplace: {name}")
                return True
            else:
                self.logger.warning(f"Marketplace not found: {name}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to remove marketplace {name}: {e}")
            return False

    def _save_marketplace_config(self) -> None:
        """Save marketplace configuration to file."""
        try:
            config_data = {
                "marketplaces": {
                    name: {
                        "name": marketplace.name,
                        "base_url": marketplace.base_url,
                        "api_key": marketplace.api_key,
                        "username": marketplace.username,
                        "timeout": marketplace.timeout,
                        "verify_ssl": marketplace.verify_ssl,
                    }
                    for name, marketplace in self.marketplaces.items()
                },
            }

            with open(self.config_file, "w") as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save marketplace config: {e}")

    def search_plugins(
        self,
        query: str,
        marketplace: str = None,
        plugin_type: PluginType = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search for plugins in marketplaces.

        Args:
            query: Search query
            marketplace: Specific marketplace to search (None for all)
            plugin_type: Filter by plugin type
            limit: Maximum number of results

        Returns:
            List of plugin search results
        """
        results = []

        marketplaces_to_search = (
            [marketplace] if marketplace else list(self.marketplaces.keys())
        )

        for marketplace_name in marketplaces_to_search:
            if marketplace_name not in self.marketplaces:
                self.logger.warning(f"Unknown marketplace: {marketplace_name}")
                continue

            try:
                marketplace_results = self._search_marketplace(
                    marketplace_name,
                    query,
                    plugin_type,
                    limit,
                )

                # Add marketplace info to results
                for result in marketplace_results:
                    result["marketplace"] = marketplace_name

                results.extend(marketplace_results)

            except Exception as e:
                self.logger.error(
                    f"Search failed for marketplace {marketplace_name}: {e}",
                )

        # Sort results by relevance/popularity
        results.sort(key=lambda x: x.get("downloads", 0), reverse=True)

        return results[:limit]

    def _search_marketplace(
        self,
        marketplace_name: str,
        query: str,
        plugin_type: PluginType = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search a specific marketplace."""
        marketplace = self.marketplaces[marketplace_name]

        # Build search parameters
        params = {"q": query, "limit": limit}

        if plugin_type:
            params["type"] = plugin_type.value

        # Make API request
        response = requests.get(
            f"{marketplace.base_url}/api/search",
            params=params,
            headers=self._get_headers(marketplace),
            timeout=marketplace.timeout,
            verify=marketplace.verify_ssl,
        )

        response.raise_for_status()

        return response.json().get("plugins", [])

    def get_plugin_details(
        self,
        plugin_id: str,
        marketplace: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a plugin.

        Args:
            plugin_id: Plugin identifier
            marketplace: Marketplace name

        Returns:
            Plugin details or None if not found
        """
        try:
            if marketplace not in self.marketplaces:
                self.logger.error(f"Unknown marketplace: {marketplace}")
                return None

            marketplace_config = self.marketplaces[marketplace]

            response = requests.get(
                f"{marketplace_config.base_url}/api/plugins/{plugin_id}",
                headers=self._get_headers(marketplace_config),
                timeout=marketplace_config.timeout,
                verify=marketplace_config.verify_ssl,
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            self.logger.error(f"Failed to get plugin details: {e}")
            return None

    def download_plugin(
        self,
        plugin_id: str,
        marketplace: str,
        output_path: str,
        version: str = None,
    ) -> bool:
        """
        Download a plugin from marketplace.

        Args:
            plugin_id: Plugin identifier
            marketplace: Marketplace name
            output_path: Path to save plugin
            version: Specific version to download

        Returns:
            True if download successful, False otherwise
        """
        try:
            if marketplace not in self.marketplaces:
                self.logger.error(f"Unknown marketplace: {marketplace}")
                return False

            marketplace_config = self.marketplaces[marketplace]

            # Build download URL
            url = f"{marketplace_config.base_url}/api/plugins/{plugin_id}/download"
            if version:
                url += f"?version={version}"

            self.logger.info(f"Downloading plugin {plugin_id} from {marketplace}")

            # Download plugin
            response = requests.get(
                url,
                headers=self._get_headers(marketplace_config),
                timeout=marketplace_config.timeout,
                verify=marketplace_config.verify_ssl,
                stream=True,
            )

            response.raise_for_status()

            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.logger.info(f"Plugin downloaded to: {output_path}")

            # Verify download if checksum provided
            checksum = response.headers.get("X-Plugin-Checksum")
            if checksum:
                if not self._verify_download_checksum(output_path, checksum):
                    self.logger.error("Download checksum verification failed")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Plugin download failed: {e}")
            return False

    def publish_plugin(
        self,
        plugin_package_path: str,
        marketplace: str,
        publish_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish a plugin to marketplace.

        Args:
            plugin_package_path: Path to plugin package
            marketplace: Marketplace name
            publish_config: Publishing configuration

        Returns:
            True if published successfully, False otherwise
        """
        try:
            if marketplace not in self.marketplaces:
                self.logger.error(f"Unknown marketplace: {marketplace}")
                return False

            marketplace_config = self.marketplaces[marketplace]

            if not marketplace_config.api_key:
                self.logger.error(f"API key required for publishing to {marketplace}")
                return False

            package_path = Path(plugin_package_path)
            if not package_path.exists():
                self.logger.error(f"Plugin package not found: {package_path}")
                return False

            self.logger.info(f"Publishing plugin to {marketplace}")

            # Prepare upload data
            files = {"package": open(package_path, "rb")}
            data = publish_config or {}

            # Add package metadata
            data["checksum"] = self._calculate_file_checksum(package_path)
            data["size"] = package_path.stat().st_size

            try:
                response = requests.post(
                    f"{marketplace_config.base_url}/api/plugins/publish",
                    files=files,
                    data=data,
                    headers=self._get_headers(marketplace_config),
                    timeout=marketplace_config.timeout
                    * 3,  # Longer timeout for uploads
                    verify=marketplace_config.verify_ssl,
                )

                response.raise_for_status()

                result = response.json()
                plugin_id = result.get("plugin_id")

                self.logger.info(f"Plugin published successfully. ID: {plugin_id}")
                return True

            finally:
                files["package"].close()

        except Exception as e:
            self.logger.error(f"Plugin publishing failed: {e}")
            return False

    def update_plugin(
        self,
        plugin_id: str,
        plugin_package_path: str,
        marketplace: str,
        update_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing plugin in marketplace.

        Args:
            plugin_id: Plugin identifier
            plugin_package_path: Path to updated plugin package
            marketplace: Marketplace name
            update_config: Update configuration

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if marketplace not in self.marketplaces:
                self.logger.error(f"Unknown marketplace: {marketplace}")
                return False

            marketplace_config = self.marketplaces[marketplace]

            if not marketplace_config.api_key:
                self.logger.error(
                    f"API key required for updating plugin in {marketplace}",
                )
                return False

            package_path = Path(plugin_package_path)
            if not package_path.exists():
                self.logger.error(f"Plugin package not found: {package_path}")
                return False

            self.logger.info(f"Updating plugin {plugin_id} in {marketplace}")

            # Prepare upload data
            files = {"package": open(package_path, "rb")}
            data = update_config or {}

            # Add package metadata
            data["checksum"] = self._calculate_file_checksum(package_path)
            data["size"] = package_path.stat().st_size

            try:
                response = requests.put(
                    f"{marketplace_config.base_url}/api/plugins/{plugin_id}",
                    files=files,
                    data=data,
                    headers=self._get_headers(marketplace_config),
                    timeout=marketplace_config.timeout * 3,
                    verify=marketplace_config.verify_ssl,
                )

                response.raise_for_status()

                self.logger.info(f"Plugin {plugin_id} updated successfully")
                return True

            finally:
                files["package"].close()

        except Exception as e:
            self.logger.error(f"Plugin update failed: {e}")
            return False

    def unpublish_plugin(self, plugin_id: str, marketplace: str) -> bool:
        """
        Unpublish a plugin from marketplace.

        Args:
            plugin_id: Plugin identifier
            marketplace: Marketplace name

        Returns:
            True if unpublished successfully, False otherwise
        """
        try:
            if marketplace not in self.marketplaces:
                self.logger.error(f"Unknown marketplace: {marketplace}")
                return False

            marketplace_config = self.marketplaces[marketplace]

            if not marketplace_config.api_key:
                self.logger.error(
                    f"API key required for unpublishing from {marketplace}",
                )
                return False

            self.logger.info(f"Unpublishing plugin {plugin_id} from {marketplace}")

            response = requests.delete(
                f"{marketplace_config.base_url}/api/plugins/{plugin_id}",
                headers=self._get_headers(marketplace_config),
                timeout=marketplace_config.timeout,
                verify=marketplace_config.verify_ssl,
            )

            response.raise_for_status()

            self.logger.info(f"Plugin {plugin_id} unpublished successfully")
            return True

        except Exception as e:
            self.logger.error(f"Plugin unpublishing failed: {e}")
            return False

    def get_my_plugins(self, marketplace: str) -> List[Dict[str, Any]]:
        """
        Get list of plugins published by current user.

        Args:
            marketplace: Marketplace name

        Returns:
            List of user's plugins
        """
        try:
            if marketplace not in self.marketplaces:
                self.logger.error(f"Unknown marketplace: {marketplace}")
                return []

            marketplace_config = self.marketplaces[marketplace]

            if not marketplace_config.api_key:
                self.logger.error(
                    f"API key required for accessing user plugins in {marketplace}",
                )
                return []

            response = requests.get(
                f"{marketplace_config.base_url}/api/user/plugins",
                headers=self._get_headers(marketplace_config),
                timeout=marketplace_config.timeout,
                verify=marketplace_config.verify_ssl,
            )

            response.raise_for_status()

            return response.json().get("plugins", [])

        except Exception as e:
            self.logger.error(f"Failed to get user plugins: {e}")
            return []

    def get_plugin_stats(
        self,
        plugin_id: str,
        marketplace: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get plugin statistics.

        Args:
            plugin_id: Plugin identifier
            marketplace: Marketplace name

        Returns:
            Plugin statistics or None if not found
        """
        try:
            if marketplace not in self.marketplaces:
                self.logger.error(f"Unknown marketplace: {marketplace}")
                return None

            marketplace_config = self.marketplaces[marketplace]

            response = requests.get(
                f"{marketplace_config.base_url}/api/plugins/{plugin_id}/stats",
                headers=self._get_headers(marketplace_config),
                timeout=marketplace_config.timeout,
                verify=marketplace_config.verify_ssl,
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            self.logger.error(f"Failed to get plugin stats: {e}")
            return None

    def _get_headers(self, marketplace: MarketplaceConfig) -> Dict[str, str]:
        """Get HTTP headers for marketplace requests."""
        headers = {
            "User-Agent": "ICARUS-CLI-Plugin-SDK/1.0.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if marketplace.api_key:
            headers["Authorization"] = f"Bearer {marketplace.api_key}"

        return headers

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _verify_download_checksum(
        self,
        file_path: Path,
        expected_checksum: str,
    ) -> bool:
        """Verify downloaded file checksum."""
        actual_checksum = self._calculate_file_checksum(file_path)
        return actual_checksum == expected_checksum

    def list_marketplaces(self) -> List[Dict[str, Any]]:
        """
        List configured marketplaces.

        Returns:
            List of marketplace information
        """
        return [
            {
                "name": name,
                "display_name": marketplace.name,
                "base_url": marketplace.base_url,
                "has_api_key": bool(marketplace.api_key),
                "username": marketplace.username,
            }
            for name, marketplace in self.marketplaces.items()
        ]

    def test_marketplace_connection(self, marketplace: str) -> Dict[str, Any]:
        """
        Test connection to a marketplace.

        Args:
            marketplace: Marketplace name

        Returns:
            Connection test results
        """
        result = {
            "marketplace": marketplace,
            "connected": False,
            "authenticated": False,
            "error": None,
            "response_time": None,
        }

        try:
            if marketplace not in self.marketplaces:
                result["error"] = f"Unknown marketplace: {marketplace}"
                return result

            marketplace_config = self.marketplaces[marketplace]

            # Test basic connection
            start_time = datetime.now()

            response = requests.get(
                f"{marketplace_config.base_url}/api/health",
                headers=self._get_headers(marketplace_config),
                timeout=marketplace_config.timeout,
                verify=marketplace_config.verify_ssl,
            )

            end_time = datetime.now()
            result["response_time"] = (end_time - start_time).total_seconds()

            if response.status_code == 200:
                result["connected"] = True

                # Test authentication if API key provided
                if marketplace_config.api_key:
                    auth_response = requests.get(
                        f"{marketplace_config.base_url}/api/user/profile",
                        headers=self._get_headers(marketplace_config),
                        timeout=marketplace_config.timeout,
                        verify=marketplace_config.verify_ssl,
                    )

                    result["authenticated"] = auth_response.status_code == 200

        except Exception as e:
            result["error"] = str(e)

        return result
