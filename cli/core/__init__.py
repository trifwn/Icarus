"""ICARUS CLI Core Module

This module provides the foundational components for the enhanced CLI architecture
including state management, UI framework, core services, and comprehensive error handling.
"""

from .config import ConfigManager as NewConfigManager
from .error_analytics import ErrorAnalytics
from .error_analytics import error_analytics

# Error handling system
from .error_handler import ErrorContext
from .error_handler import ErrorHandler
from .error_handler import ErrorSeverity
from .error_handler import handle_errors
from .error_integration import IntegratedErrorManager
from .error_integration import check_system_dependencies
from .error_integration import error_context
from .error_integration import get_error_summary
from .error_integration import handle_error
from .error_integration import integrated_error_handler
from .error_integration import integrated_error_manager
from .error_integration import run_health_check
from .error_integration import safe_operation
from .graceful_degradation import GracefulDegradationManager
from .graceful_degradation import degradation_manager
from .graceful_degradation import require_dependency
from .services import ExportService
from .services import ValidationService
from .state import ConfigManager
from .state import HistoryManager
from .state import SessionManager
from .ui import LayoutManager
from .ui import NotificationSystem
from .ui import ProgressManager
from .ui import ThemeManager
from .workflow import TemplateManager
from .workflow import WorkflowEngine

__all__ = [
    "SessionManager",
    "ConfigManager",
    "NewConfigManager",
    "HistoryManager",
    "ThemeManager",
    "LayoutManager",
    "ProgressManager",
    "NotificationSystem",
    "WorkflowEngine",
    "TemplateManager",
    "ValidationService",
    "ExportService",
    # Error handling system
    "ErrorHandler",
    "ErrorContext",
    "ErrorSeverity",
    "handle_errors",
    "ErrorAnalytics",
    "error_analytics",
    "GracefulDegradationManager",
    "degradation_manager",
    "require_dependency",
    "IntegratedErrorManager",
    "integrated_error_manager",
    "error_context",
    "safe_operation",
    "integrated_error_handler",
    "handle_error",
    "get_error_summary",
    "run_health_check",
    "check_system_dependencies",
]
