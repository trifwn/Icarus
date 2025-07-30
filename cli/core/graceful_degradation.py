"""Graceful Degradation System

This module provides graceful degradation capabilities when dependencies
are missing or components fail, ensuring the system remains functional
with reduced capabilities rather than complete failure.
"""

import functools
import importlib
import logging
import subprocess
import warnings
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from .error_handler import ErrorContext
from .error_handler import ErrorHandler


class DegradationLevel(Enum):
    """Levels of system degradation."""

    NONE = "none"  # Full functionality
    MINIMAL = "minimal"  # Minor features disabled
    MODERATE = "moderate"  # Some major features disabled
    SEVERE = "severe"  # Only core features available
    CRITICAL = "critical"  # Minimal functionality only


class ComponentStatus(Enum):
    """Status of system components."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    FALLBACK = "fallback"


@dataclass
class DependencyInfo:
    """Information about a system dependency."""

    name: str
    type: str  # 'python_package', 'executable', 'service', 'file'
    required: bool = True
    fallbacks: List[str] = field(default_factory=list)
    check_function: Optional[Callable[[], bool]] = None
    install_instructions: str = ""
    description: str = ""
    affects_components: List[str] = field(default_factory=list)


@dataclass
class ComponentInfo:
    """Information about a system component."""

    name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    fallback_implementation: Optional[Callable] = None
    degraded_implementation: Optional[Callable] = None
    status: ComponentStatus = ComponentStatus.AVAILABLE
    error_message: Optional[str] = None


@dataclass
class DegradationReport:
    """Report on system degradation status."""

    overall_level: DegradationLevel
    available_components: List[str]
    degraded_components: List[str]
    unavailable_components: List[str]
    missing_dependencies: List[str]
    active_fallbacks: List[str]
    recommendations: List[str]
    user_impact: str


class GracefulDegradationManager:
    """Manages graceful degradation of system functionality."""

    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler
        self.logger = logging.getLogger("graceful_degradation")

        # System state
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.components: Dict[str, ComponentInfo] = {}
        self.degradation_level = DegradationLevel.NONE
        self.checked_dependencies: Set[str] = set()

        # Configuration
        self.auto_check_on_import = True
        self.warn_on_degradation = True
        self.fallback_enabled = True

        # Initialize built-in dependencies and components
        self._register_builtin_dependencies()
        self._register_builtin_components()

    def _register_builtin_dependencies(self):
        """Register built-in system dependencies."""

        # Python packages
        self.register_dependency(
            DependencyInfo(
                name="numpy",
                type="python_package",
                required=True,
                fallbacks=["python_math"],
                install_instructions="pip install numpy",
                description="Numerical computing library",
                affects_components=["analysis", "visualization", "data_processing"],
            ),
        )

        self.register_dependency(
            DependencyInfo(
                name="matplotlib",
                type="python_package",
                required=False,
                fallbacks=["rich_plotting", "ascii_plots"],
                install_instructions="pip install matplotlib",
                description="Plotting and visualization library",
                affects_components=["visualization", "plotting"],
            ),
        )

        self.register_dependency(
            DependencyInfo(
                name="pandas",
                type="python_package",
                required=False,
                fallbacks=["csv_processing"],
                install_instructions="pip install pandas",
                description="Data analysis and manipulation library",
                affects_components=["data_processing", "import_export"],
            ),
        )

        self.register_dependency(
            DependencyInfo(
                name="rich",
                type="python_package",
                required=True,
                fallbacks=["basic_console"],
                install_instructions="pip install rich",
                description="Rich text and beautiful formatting library",
                affects_components=["ui", "console_output"],
            ),
        )

        self.register_dependency(
            DependencyInfo(
                name="textual",
                type="python_package",
                required=True,
                fallbacks=["basic_cli"],
                install_instructions="pip install textual",
                description="Terminal User Interface framework",
                affects_components=["tui", "interface"],
            ),
        )

        # External executables
        self.register_dependency(
            DependencyInfo(
                name="xfoil",
                type="executable",
                required=False,
                fallbacks=["foil2wake", "openfoam"],
                install_instructions="Install XFoil from https://web.mit.edu/drela/Public/web/xfoil/",
                description="Airfoil analysis and design tool",
                affects_components=["airfoil_analysis"],
                check_function=self._check_xfoil,
            ),
        )

        self.register_dependency(
            DependencyInfo(
                name="avl",
                type="executable",
                required=False,
                fallbacks=["gnvp3", "lspt"],
                install_instructions="Install AVL from https://web.mit.edu/drela/Public/web/avl/",
                description="Vortex Lattice Method for aircraft analysis",
                affects_components=["aircraft_analysis"],
                check_function=self._check_avl,
            ),
        )

        # System services
        self.register_dependency(
            DependencyInfo(
                name="internet_connection",
                type="service",
                required=False,
                fallbacks=["offline_mode"],
                install_instructions="Check network connection",
                description="Internet connectivity for cloud features",
                affects_components=["collaboration", "cloud_sync", "updates"],
                check_function=self._check_internet,
            ),
        )

    def _register_builtin_components(self):
        """Register built-in system components."""

        self.register_component(
            ComponentInfo(
                name="airfoil_analysis",
                description="Airfoil analysis capabilities",
                dependencies=["xfoil"],
                fallback_implementation=self._airfoil_analysis_fallback,
                degraded_implementation=self._airfoil_analysis_degraded,
            ),
        )

        self.register_component(
            ComponentInfo(
                name="aircraft_analysis",
                description="Aircraft analysis capabilities",
                dependencies=["avl"],
                fallback_implementation=self._aircraft_analysis_fallback,
                degraded_implementation=self._aircraft_analysis_degraded,
            ),
        )

        self.register_component(
            ComponentInfo(
                name="visualization",
                description="Data visualization and plotting",
                dependencies=["matplotlib"],
                fallback_implementation=self._visualization_fallback,
                degraded_implementation=self._visualization_degraded,
            ),
        )

        self.register_component(
            ComponentInfo(
                name="data_processing",
                description="Advanced data processing",
                dependencies=["pandas", "numpy"],
                fallback_implementation=self._data_processing_fallback,
                degraded_implementation=self._data_processing_degraded,
            ),
        )

        self.register_component(
            ComponentInfo(
                name="collaboration",
                description="Real-time collaboration features",
                dependencies=["internet_connection"],
                fallback_implementation=self._collaboration_fallback,
                degraded_implementation=self._collaboration_degraded,
            ),
        )

        self.register_component(
            ComponentInfo(
                name="ui",
                description="User interface",
                dependencies=["textual", "rich"],
                fallback_implementation=self._ui_fallback,
                degraded_implementation=self._ui_degraded,
            ),
        )

    def register_dependency(self, dependency: DependencyInfo):
        """Register a system dependency."""
        self.dependencies[dependency.name] = dependency

    def register_component(self, component: ComponentInfo):
        """Register a system component."""
        self.components[component.name] = component

    def check_dependency(self, name: str) -> bool:
        """Check if a dependency is available."""
        if name in self.checked_dependencies:
            # Return cached result
            return name in [
                dep
                for dep, info in self.dependencies.items()
                if self._is_dependency_available(dep)
            ]

        if name not in self.dependencies:
            self.logger.warning(f"Unknown dependency: {name}")
            return False

        dependency = self.dependencies[name]
        available = self._is_dependency_available(name)

        self.checked_dependencies.add(name)

        if not available and dependency.required:
            self.logger.error(f"Required dependency '{name}' is not available")
            if self.error_handler:
                context = ErrorContext(
                    component="dependency_checker",
                    operation="check_dependency",
                    user_data={"dependency": name},
                )
                self.error_handler.handle_error(
                    ImportError(f"Required dependency '{name}' not found"),
                    context,
                )

        return available

    def _is_dependency_available(self, name: str) -> bool:
        """Check if a specific dependency is available."""
        if name not in self.dependencies:
            return False

        dependency = self.dependencies[name]

        # Use custom check function if provided
        if dependency.check_function:
            try:
                return dependency.check_function()
            except Exception as e:
                self.logger.debug(f"Custom check failed for {name}: {e}")
                return False

        # Default checks based on dependency type
        if dependency.type == "python_package":
            return self._check_python_package(name)
        elif dependency.type == "executable":
            return self._check_executable(name)
        elif dependency.type == "service":
            return self._check_service(name)
        elif dependency.type == "file":
            return self._check_file(name)
        else:
            self.logger.warning(f"Unknown dependency type: {dependency.type}")
            return False

    def _check_python_package(self, package_name: str) -> bool:
        """Check if a Python package is available."""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False

    def _check_executable(self, executable_name: str) -> bool:
        """Check if an executable is available."""
        try:
            result = subprocess.run(
                [executable_name, "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return False

    def _check_service(self, service_name: str) -> bool:
        """Check if a service is available."""
        # This would implement service-specific checks
        return True  # Placeholder

    def _check_file(self, file_path: str) -> bool:
        """Check if a file exists."""
        return Path(file_path).exists()

    # Specific dependency check functions
    def _check_xfoil(self) -> bool:
        """Check if XFoil is available."""
        try:
            result = subprocess.run(
                ["xfoil"],
                input="\n\nquit\n",
                text=True,
                capture_output=True,
                timeout=5,
            )
            return True  # XFoil doesn't have --version, so any response is good
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return False

    def _check_avl(self) -> bool:
        """Check if AVL is available."""
        try:
            result = subprocess.run(
                ["avl"],
                input="quit\n",
                text=True,
                capture_output=True,
                timeout=5,
            )
            return True  # AVL doesn't have --version, so any response is good
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return False

    def _check_internet(self) -> bool:
        """Check if internet connection is available."""
        try:
            import socket

            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def check_all_dependencies(self) -> Dict[str, bool]:
        """Check all registered dependencies."""
        results = {}
        for name in self.dependencies:
            results[name] = self.check_dependency(name)
        return results

    def update_component_status(self):
        """Update the status of all components based on dependency availability."""
        for component_name, component in self.components.items():
            # Check if all dependencies are available
            all_available = True
            some_available = False

            for dep_name in component.dependencies:
                if self.check_dependency(dep_name):
                    some_available = True
                else:
                    all_available = False

            # Update component status
            if all_available:
                component.status = ComponentStatus.AVAILABLE
                component.error_message = None
            elif some_available:
                component.status = ComponentStatus.DEGRADED
                component.error_message = "Some dependencies unavailable"
            else:
                # Check for fallbacks
                fallback_available = self._check_fallbacks(component.dependencies)
                if fallback_available:
                    component.status = ComponentStatus.FALLBACK
                    component.error_message = "Using fallback implementation"
                else:
                    component.status = ComponentStatus.UNAVAILABLE
                    component.error_message = "All dependencies unavailable"

    def _check_fallbacks(self, dependencies: List[str]) -> bool:
        """Check if fallbacks are available for dependencies."""
        for dep_name in dependencies:
            if dep_name in self.dependencies:
                fallbacks = self.dependencies[dep_name].fallbacks
                for fallback in fallbacks:
                    if self.check_dependency(fallback):
                        return True
        return False

    def get_degradation_level(self) -> DegradationLevel:
        """Calculate overall system degradation level."""
        self.update_component_status()

        total_components = len(self.components)
        if total_components == 0:
            return DegradationLevel.NONE

        available_count = sum(
            1
            for comp in self.components.values()
            if comp.status == ComponentStatus.AVAILABLE
        )
        degraded_count = sum(
            1
            for comp in self.components.values()
            if comp.status == ComponentStatus.DEGRADED
        )
        fallback_count = sum(
            1
            for comp in self.components.values()
            if comp.status == ComponentStatus.FALLBACK
        )

        functional_ratio = (
            available_count + degraded_count * 0.7 + fallback_count * 0.5
        ) / total_components

        if functional_ratio >= 0.95:
            return DegradationLevel.NONE
        elif functional_ratio >= 0.8:
            return DegradationLevel.MINIMAL
        elif functional_ratio >= 0.6:
            return DegradationLevel.MODERATE
        elif functional_ratio >= 0.3:
            return DegradationLevel.SEVERE
        else:
            return DegradationLevel.CRITICAL

    def get_degradation_report(self) -> DegradationReport:
        """Get comprehensive degradation report."""
        self.update_component_status()
        degradation_level = self.get_degradation_level()

        available_components = [
            name
            for name, comp in self.components.items()
            if comp.status == ComponentStatus.AVAILABLE
        ]
        degraded_components = [
            name
            for name, comp in self.components.items()
            if comp.status == ComponentStatus.DEGRADED
        ]
        unavailable_components = [
            name
            for name, comp in self.components.items()
            if comp.status == ComponentStatus.UNAVAILABLE
        ]

        missing_dependencies = [
            name
            for name, dep in self.dependencies.items()
            if not self.check_dependency(name)
        ]

        active_fallbacks = [
            name
            for name, comp in self.components.items()
            if comp.status == ComponentStatus.FALLBACK
        ]

        recommendations = self._generate_recommendations(
            missing_dependencies,
            degraded_components,
            unavailable_components,
        )

        user_impact = self._describe_user_impact(
            degradation_level,
            unavailable_components,
        )

        return DegradationReport(
            overall_level=degradation_level,
            available_components=available_components,
            degraded_components=degraded_components,
            unavailable_components=unavailable_components,
            missing_dependencies=missing_dependencies,
            active_fallbacks=active_fallbacks,
            recommendations=recommendations,
            user_impact=user_impact,
        )

    def _generate_recommendations(
        self,
        missing_deps: List[str],
        degraded_comps: List[str],
        unavailable_comps: List[str],
    ) -> List[str]:
        """Generate recommendations for improving system functionality."""
        recommendations = []

        # Recommendations for missing dependencies
        for dep_name in missing_deps:
            if dep_name in self.dependencies:
                dep = self.dependencies[dep_name]
                if dep.install_instructions:
                    recommendations.append(
                        f"Install {dep_name}: {dep.install_instructions}",
                    )

        # Recommendations for degraded components
        if degraded_comps:
            recommendations.append(
                f"Components with reduced functionality: {', '.join(degraded_comps)}. "
                "Check dependency status for full functionality.",
            )

        # Recommendations for unavailable components
        if unavailable_comps:
            recommendations.append(
                f"Unavailable components: {', '.join(unavailable_comps)}. "
                "Install required dependencies to enable these features.",
            )

        return recommendations

    def _describe_user_impact(
        self,
        level: DegradationLevel,
        unavailable: List[str],
    ) -> str:
        """Describe the impact on user experience."""
        if level == DegradationLevel.NONE:
            return "All features are fully functional."
        elif level == DegradationLevel.MINIMAL:
            return (
                "Minor features may be unavailable, but core functionality is intact."
            )
        elif level == DegradationLevel.MODERATE:
            return "Some important features are unavailable or running with reduced capabilities."
        elif level == DegradationLevel.SEVERE:
            return (
                "Many features are unavailable. Only core functionality is accessible."
            )
        else:  # CRITICAL
            return "System is running with minimal functionality. Many features are unavailable."

    # Fallback implementations
    def _airfoil_analysis_fallback(self):
        """Fallback implementation for airfoil analysis."""
        return "Airfoil analysis unavailable - XFoil not found. Please install XFoil for full functionality."

    def _airfoil_analysis_degraded(self):
        """Degraded implementation for airfoil analysis."""
        return "Using simplified airfoil analysis - some features may be limited."

    def _aircraft_analysis_fallback(self):
        """Fallback implementation for aircraft analysis."""
        return "Aircraft analysis unavailable - AVL not found. Please install AVL for full functionality."

    def _aircraft_analysis_degraded(self):
        """Degraded implementation for aircraft analysis."""
        return "Using simplified aircraft analysis - some features may be limited."

    def _visualization_fallback(self):
        """Fallback implementation for visualization."""
        return "Advanced plotting unavailable - using text-based output."

    def _visualization_degraded(self):
        """Degraded implementation for visualization."""
        return "Using simplified plotting - some chart types may be unavailable."

    def _data_processing_fallback(self):
        """Fallback implementation for data processing."""
        return "Advanced data processing unavailable - using basic Python operations."

    def _data_processing_degraded(self):
        """Degraded implementation for data processing."""
        return "Using simplified data processing - some operations may be slower."

    def _collaboration_fallback(self):
        """Fallback implementation for collaboration."""
        return "Collaboration features unavailable - working in offline mode."

    def _collaboration_degraded(self):
        """Degraded implementation for collaboration."""
        return "Limited collaboration features - some real-time features unavailable."

    def _ui_fallback(self):
        """Fallback implementation for UI."""
        return "Using basic command-line interface - advanced UI features unavailable."

    def _ui_degraded(self):
        """Degraded implementation for UI."""
        return "Using simplified interface - some visual features may be limited."


def require_dependency(
    dependency_name: str,
    fallback_func: Optional[Callable] = None,
    error_message: Optional[str] = None,
):
    """Decorator to require a dependency for a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not degradation_manager.check_dependency(dependency_name):
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    msg = (
                        error_message
                        or f"Dependency '{dependency_name}' is required for this function"
                    )
                    raise ImportError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def graceful_import(module_name: str, fallback_value=None):
    """Import a module gracefully, returning fallback if not available."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if degradation_manager.warn_on_degradation:
            warnings.warn(f"Module '{module_name}' not available, using fallback")
        return fallback_value


# Global degradation manager instance
degradation_manager = GracefulDegradationManager()
