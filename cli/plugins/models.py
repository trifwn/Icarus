"""
Plugin system data models and structures.
"""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class PluginStatus(Enum):
    """Plugin status enumeration."""

    UNKNOWN = "unknown"
    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    UPDATING = "updating"
    INSTALLING = "installing"
    UNINSTALLING = "uninstalling"


class PluginType(Enum):
    """Plugin type enumeration."""

    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    EXPORT = "export"
    IMPORT = "import"
    WORKFLOW = "workflow"
    INTEGRATION = "integration"
    UTILITY = "utility"
    THEME = "theme"


class SecurityLevel(Enum):
    """Plugin security level enumeration."""

    SAFE = "safe"  # No system access, UI only
    RESTRICTED = "restricted"  # Limited system access
    ELEVATED = "elevated"  # Full system access
    DANGEROUS = "dangerous"  # Potentially harmful operations


@dataclass
class PluginDependency:
    """Plugin dependency specification."""

    name: str
    version: str
    optional: bool = False
    source: Optional[str] = None  # PyPI, git, local, etc.


@dataclass
class PluginPermission:
    """Plugin permission specification."""

    name: str
    description: str
    required: bool = True
    granted: bool = False


@dataclass
class PluginAuthor:
    """Plugin author information."""

    name: str
    email: Optional[str] = None
    url: Optional[str] = None


@dataclass
class PluginVersion:
    """Plugin version information."""

    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        return version

    @classmethod
    def from_string(cls, version_str: str) -> "PluginVersion":
        """Parse version from string."""
        parts = version_str.split("-")
        version_parts = parts[0].split(".")
        pre_release = parts[1] if len(parts) > 1 else None

        return cls(
            major=int(version_parts[0]),
            minor=int(version_parts[1]),
            patch=int(version_parts[2]) if len(version_parts) > 2 else 0,
            pre_release=pre_release,
        )


@dataclass
class PluginManifest:
    """Plugin manifest containing metadata and configuration."""

    name: str
    version: PluginVersion
    description: str
    author: PluginAuthor
    plugin_type: PluginType
    security_level: SecurityLevel

    # Entry points
    main_module: str
    main_class: str

    # Dependencies and requirements
    dependencies: List[PluginDependency] = field(default_factory=list)
    python_version: str = ">=3.8"
    icarus_version: str = ">=1.0.0"

    # Permissions and security
    permissions: List[PluginPermission] = field(default_factory=list)
    sandbox_enabled: bool = True

    # Metadata
    homepage: Optional[str] = None
    repository: Optional[str] = None
    documentation: Optional[str] = None
    license: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    # Installation and configuration
    install_requires: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        """Create manifest from dictionary."""
        return cls(
            name=data["name"],
            version=PluginVersion.from_string(data["version"]),
            description=data["description"],
            author=PluginAuthor(**data["author"]),
            plugin_type=PluginType(data["type"]),
            security_level=SecurityLevel(data.get("security_level", "restricted")),
            main_module=data["main_module"],
            main_class=data["main_class"],
            dependencies=[
                PluginDependency(**dep) for dep in data.get("dependencies", [])
            ],
            python_version=data.get("python_version", ">=3.8"),
            icarus_version=data.get("icarus_version", ">=1.0.0"),
            permissions=[
                PluginPermission(**perm) for perm in data.get("permissions", [])
            ],
            sandbox_enabled=data.get("sandbox_enabled", True),
            homepage=data.get("homepage"),
            repository=data.get("repository"),
            documentation=data.get("documentation"),
            license=data.get("license"),
            keywords=data.get("keywords", []),
            install_requires=data.get("install_requires", []),
            config_schema=data.get("config_schema"),
            default_config=data.get("default_config"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "name": self.name,
            "version": str(self.version),
            "description": self.description,
            "author": {
                "name": self.author.name,
                "email": self.author.email,
                "url": self.author.url,
            },
            "type": self.plugin_type.value,
            "security_level": self.security_level.value,
            "main_module": self.main_module,
            "main_class": self.main_class,
            "dependencies": [
                {
                    "name": dep.name,
                    "version": dep.version,
                    "optional": dep.optional,
                    "source": dep.source,
                }
                for dep in self.dependencies
            ],
            "python_version": self.python_version,
            "icarus_version": self.icarus_version,
            "permissions": [
                {
                    "name": perm.name,
                    "description": perm.description,
                    "required": perm.required,
                    "granted": perm.granted,
                }
                for perm in self.permissions
            ],
            "sandbox_enabled": self.sandbox_enabled,
            "homepage": self.homepage,
            "repository": self.repository,
            "documentation": self.documentation,
            "license": self.license,
            "keywords": self.keywords,
            "install_requires": self.install_requires,
            "config_schema": self.config_schema,
            "default_config": self.default_config,
        }


@dataclass
class PluginInfo:
    """Complete plugin information including runtime state."""

    manifest: PluginManifest
    status: PluginStatus
    path: str

    # Runtime information
    loaded_at: Optional[datetime] = None
    last_error: Optional[str] = None
    instance: Optional[Any] = None
    config: Optional[Dict[str, Any]] = None

    # Installation information
    installed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    install_source: Optional[str] = None

    @property
    def id(self) -> str:
        """Unique plugin identifier."""
        return f"{self.manifest.name}@{self.manifest.version}"

    @property
    def is_active(self) -> bool:
        """Check if plugin is active."""
        return self.status == PluginStatus.ACTIVE

    @property
    def is_loaded(self) -> bool:
        """Check if plugin is loaded."""
        return self.status in [PluginStatus.LOADED, PluginStatus.ACTIVE]

    @property
    def has_error(self) -> bool:
        """Check if plugin has an error."""
        return self.status == PluginStatus.ERROR or self.last_error is not None


@dataclass
class PluginConfig:
    """Plugin configuration settings."""

    plugin_name: str
    settings: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    auto_update: bool = False
    permissions_granted: List[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.settings[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plugin_name": self.plugin_name,
            "settings": self.settings,
            "enabled": self.enabled,
            "auto_update": self.auto_update,
            "permissions_granted": self.permissions_granted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginConfig":
        """Create from dictionary."""
        return cls(
            plugin_name=data["plugin_name"],
            settings=data.get("settings", {}),
            enabled=data.get("enabled", True),
            auto_update=data.get("auto_update", False),
            permissions_granted=data.get("permissions_granted", []),
        )


@dataclass
class PluginRegistry:
    """Plugin registry containing all plugin information."""

    plugins: Dict[str, PluginInfo] = field(default_factory=dict)
    configs: Dict[str, PluginConfig] = field(default_factory=dict)
    last_scan: Optional[datetime] = None

    def add_plugin(self, plugin_info: PluginInfo) -> None:
        """Add plugin to registry."""
        self.plugins[plugin_info.id] = plugin_info

        # Create default config if not exists
        if plugin_info.manifest.name not in self.configs:
            self.configs[plugin_info.manifest.name] = PluginConfig(
                plugin_name=plugin_info.manifest.name,
                settings=plugin_info.manifest.default_config or {},
            )

    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin by ID."""
        return self.plugins.get(plugin_id)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Get all plugins of a specific type."""
        return [
            plugin
            for plugin in self.plugins.values()
            if plugin.manifest.plugin_type == plugin_type
        ]

    def get_active_plugins(self) -> List[PluginInfo]:
        """Get all active plugins."""
        return [plugin for plugin in self.plugins.values() if plugin.is_active]

    def remove_plugin(self, plugin_id: str) -> bool:
        """Remove plugin from registry."""
        if plugin_id in self.plugins:
            plugin_name = self.plugins[plugin_id].manifest.name
            del self.plugins[plugin_id]
            if plugin_name in self.configs:
                del self.configs[plugin_name]
            return True
        return False
