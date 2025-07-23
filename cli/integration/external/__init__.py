"""External Tool Integration System

This module provides integration capabilities with external tools including:
- CAD file import with geometry validation
- Cloud service integration with secure authentication
- External tool export with format conversion
- API adaptation layer for external tool updates
"""

# Core functionality (minimal dependencies)
from .cad_integration import CADIntegration

# Always available models
from .models import APIEndpoint
from .models import AuthenticationConfig
from .models import CADFile
from .models import CADFormat
from .models import CloudService
from .models import ExportFormat
from .models import ExportFormatType
from .models import ExportResult
from .models import GeometryValidationResult
from .models import ValidationStatus

# Optional components with external dependencies
try:
    from .cloud_integration import CloudIntegration

    CLOUD_AVAILABLE = True
except ImportError:
    CloudIntegration = None
    CLOUD_AVAILABLE = False

try:
    from .export_manager import ExportManager

    EXPORT_AVAILABLE = True
except ImportError:
    ExportManager = None
    EXPORT_AVAILABLE = False

try:
    from .api_adapter import APIAdapter

    API_ADAPTER_AVAILABLE = True
except ImportError:
    APIAdapter = None
    API_ADAPTER_AVAILABLE = False

__all__ = [
    "CADIntegration",
    "CloudIntegration",
    "ExportManager",
    "APIAdapter",
    "CADFile",
    "CloudService",
    "ExportFormat",
    "GeometryValidationResult",
    "AuthenticationConfig",
    "ExportResult",
    "APIEndpoint",
    "CADFormat",
    "ExportFormatType",
    "ValidationStatus",
    "CLOUD_AVAILABLE",
    "EXPORT_AVAILABLE",
    "API_ADAPTER_AVAILABLE",
]
