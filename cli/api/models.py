"""
Pydantic data models for ICARUS CLI API

These models provide JSON serializable data structures that are compatible
with both the TUI and future web interfaces.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class AnalysisType(str, Enum):
    """Types of analysis available in ICARUS"""

    AIRFOIL = "airfoil"
    AIRPLANE = "airplane"
    PROPULSION = "propulsion"
    MISSION = "mission"
    OPTIMIZATION = "optimization"


class SolverType(str, Enum):
    """Available solver types"""

    XFOIL = "xfoil"
    AVL = "avl"
    GENUVP = "genuvp"
    OPENFOAM = "openfoam"
    XFLR5 = "xflr5"
    FOIL2WAKE = "foil2wake"


class AnalysisStatus(str, Enum):
    """Status of an analysis execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BaseAPIModel(BaseModel):
    """Base model with common configuration"""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)},
    )


class AnalysisConfig(BaseAPIModel):
    """Configuration for an analysis run"""

    id: UUID = Field(default_factory=uuid4)
    analysis_type: AnalysisType
    target: str = Field(..., description="Target file or identifier for analysis")
    solver: SolverType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output_format: str = Field(default="json")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class AnalysisResult(BaseAPIModel):
    """Result of an analysis execution"""

    id: UUID
    config_id: UUID
    status: AnalysisStatus
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None  # seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowStep(BaseAPIModel):
    """Individual step in a workflow"""

    id: str
    name: str
    analysis_config: AnalysisConfig
    dependencies: List[str] = Field(default_factory=list)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    order: int = 0


class Workflow(BaseAPIModel):
    """Complete workflow definition"""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    steps: List[WorkflowStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class WorkflowExecution(BaseAPIModel):
    """Workflow execution state and results"""

    id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    status: AnalysisStatus
    current_step: Optional[str] = None
    step_results: Dict[str, AnalysisResult] = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class UserPreferences(BaseAPIModel):
    """User preferences and settings"""

    theme: str = "default"
    layout: str = "standard"
    auto_save: bool = True
    default_solver: Optional[SolverType] = None
    recent_files: List[str] = Field(default_factory=list, max_length=10)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseAPIModel):
    """Current application session state"""

    id: UUID = Field(default_factory=uuid4)
    user_id: str
    workspace: str
    active_analyses: List[UUID] = Field(default_factory=list)
    recent_results: List[UUID] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    collaboration_session: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)


class SolverInfo(BaseAPIModel):
    """Information about an available solver"""

    name: SolverType
    version: Optional[str] = None
    available: bool = True
    path: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    supported_formats: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseAPIModel):
    """Result of parameter or configuration validation"""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class ErrorResponse(BaseAPIModel):
    """Standardized error response"""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class ScreenData(BaseAPIModel):
    """Data structure for UI screen rendering"""

    screen_id: str
    title: str
    content: Dict[str, Any]
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InputEvent(BaseAPIModel):
    """User input event data"""

    event_type: str
    component_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class WebSocketMessage(BaseAPIModel):
    """WebSocket message structure"""

    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class CollaborationEvent(BaseAPIModel):
    """Collaboration event for real-time updates"""

    event_type: str
    user_id: str
    session_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# Request/Response models for API endpoints


class AnalysisRequest(BaseAPIModel):
    """Request to start an analysis"""

    config: AnalysisConfig


class AnalysisResponse(BaseAPIModel):
    """Response from analysis request"""

    analysis_id: UUID
    status: AnalysisStatus
    message: str = "Analysis started successfully"


class WorkflowRequest(BaseAPIModel):
    """Request to execute a workflow"""

    workflow: Workflow


class WorkflowResponse(BaseAPIModel):
    """Response from workflow execution request"""

    execution_id: UUID
    status: AnalysisStatus
    message: str = "Workflow execution started"


class SessionRequest(BaseAPIModel):
    """Request to create or update a session"""

    user_id: str
    workspace: str
    preferences: Optional[UserPreferences] = None


class SessionResponse(BaseAPIModel):
    """Response with session information"""

    session: SessionState
    message: str = "Session created successfully"
