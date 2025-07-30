"""Data models for external tool integration."""

import datetime
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class CADFormat(Enum):
    """Supported CAD file formats."""

    STEP = "step"
    IGES = "iges"
    STL = "stl"
    OBJ = "obj"
    PLY = "ply"


class CloudServiceType(Enum):
    """Supported cloud service types."""

    AWS_S3 = "aws_s3"
    GOOGLE_CLOUD = "google_cloud"
    AZURE_BLOB = "azure_blob"
    DROPBOX = "dropbox"
    ONEDRIVE = "onedrive"


class ExportFormatType(Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    XML = "xml"
    HDF5 = "hdf5"
    MATLAB = "matlab"
    EXCEL = "excel"
    PARAVIEW = "paraview"
    TECPLOT = "tecplot"


class ValidationStatus(Enum):
    """Geometry validation status."""

    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


class AuthenticationType(Enum):
    """Authentication types for external services."""

    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    BASIC_AUTH = "basic_auth"
    CERTIFICATE = "certificate"


@dataclass
class GeometryInfo:
    """Information about geometry in CAD file."""

    vertices: int
    faces: int
    edges: int
    volume: Optional[float] = None
    surface_area: Optional[float] = None
    bounding_box: Optional[Dict[str, float]] = None
    units: Optional[str] = None


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    severity: ValidationStatus
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class GeometryValidationResult:
    """Result of geometry validation."""

    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    geometry_info: Optional[GeometryInfo] = None
    validation_time: datetime.datetime = field(default_factory=datetime.datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if geometry is valid (no errors)."""
        return self.status != ValidationStatus.ERROR

    @property
    def has_warnings(self) -> bool:
        """Check if geometry has warnings."""
        return any(issue.severity == ValidationStatus.WARNING for issue in self.issues)


@dataclass
class CADFile:
    """CAD file representation."""

    path: Path
    format: CADFormat
    name: str
    size: int
    created_at: datetime.datetime
    modified_at: datetime.datetime
    validation_result: Optional[GeometryValidationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if CAD file is valid."""
        return self.validation_result is not None and self.validation_result.is_valid


@dataclass
class AuthenticationConfig:
    """Authentication configuration for external services."""

    type: AuthenticationType
    credentials: Dict[str, str]
    endpoint: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3

    def get_credential(self, key: str) -> Optional[str]:
        """Get a specific credential value."""
        return self.credentials.get(key)


@dataclass
class CloudService:
    """Cloud service configuration."""

    name: str
    type: CloudServiceType
    auth_config: AuthenticationConfig
    base_url: str
    bucket_name: Optional[str] = None
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportFormat:
    """Export format configuration."""

    type: ExportFormatType
    extension: str
    mime_type: str
    supports_compression: bool = False
    max_file_size: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportResult:
    """Result of export operation."""

    success: bool
    output_path: Optional[Path] = None
    format: Optional[ExportFormat] = None
    file_size: Optional[int] = None
    export_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIEndpoint:
    """External API endpoint configuration."""

    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    auth_config: Optional[AuthenticationConfig] = None
    timeout: int = 30
    retry_count: int = 3
    rate_limit: Optional[int] = None
    version: str = "v1"

    def get_full_url(self, path: str = "") -> str:
        """Get full URL with path."""
        return f"{self.url.rstrip('/')}/{path.lstrip('/')}" if path else self.url


@dataclass
class ExternalToolConfig:
    """Configuration for external tool integration."""

    name: str
    executable_path: Optional[Path] = None
    api_endpoints: List[APIEndpoint] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    version: Optional[str] = None
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)
    is_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionJob:
    """File format conversion job."""

    id: str
    input_file: Path
    output_file: Path
    source_format: str
    target_format: str
    status: str = "pending"
    progress: float = 0.0
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    errors: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
