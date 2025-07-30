"""
Data models for ICARUS integration layer.
"""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class AnalysisType(Enum):
    """Types of analyses supported by ICARUS."""

    AIRFOIL_POLAR = "airfoil_polar"
    AIRPLANE_POLAR = "airplane_polar"
    AIRPLANE_STABILITY = "airplane_stability"
    MISSION_ANALYSIS = "mission_analysis"
    OPTIMIZATION = "optimization"
    CONCEPTUAL_DESIGN = "conceptual_design"


class SolverType(Enum):
    """Types of solvers available in ICARUS."""

    XFOIL = "xfoil"
    AVL = "avl"
    GENUVP = "genuvp"
    XFLR5 = "xflr5"
    OPENFOAM = "openfoam"
    FOIL2WAKE = "foil2wake"
    ICARUS_LSPT = "icarus_lspt"


@dataclass
class SolverInfo:
    """Information about a solver."""

    name: str
    solver_type: SolverType
    version: Optional[str] = None
    executable_path: Optional[str] = None
    is_available: bool = False
    supported_analyses: List[AnalysisType] = field(default_factory=list)
    description: str = ""
    fidelity_level: int = 1  # 1=low, 2=medium, 3=high
    capabilities: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """Configuration for an analysis run."""

    analysis_type: AnalysisType
    solver_type: SolverType
    target: str  # Airfoil name, airplane file, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    solver_parameters: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "json"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "analysis_type": self.analysis_type.value,
            "solver_type": self.solver_type.value,
            "target": self.target,
            "parameters": self.parameters,
            "solver_parameters": self.solver_parameters,
            "output_format": self.output_format,
            "metadata": self.metadata,
        }


@dataclass
class ValidationError:
    """Represents a validation error."""

    field: str
    message: str
    error_type: str
    suggested_value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of parameter validation."""

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def add_error(
        self,
        field: str,
        message: str,
        error_type: str = "validation",
        suggested_value: Any = None,
    ) -> None:
        """Add a validation error."""
        self.errors.append(
            ValidationError(
                field=field,
                message=message,
                error_type=error_type,
                suggested_value=suggested_value,
            ),
        )
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def add_suggestion(self, message: str) -> None:
        """Add a suggestion."""
        self.suggestions.append(message)


@dataclass
class AnalysisResult:
    """Raw result from an analysis."""

    analysis_id: str
    config: AnalysisConfig
    status: str  # "success", "failed", "running", "queued"
    start_time: datetime
    end_time: Optional[datetime] = None
    raw_data: Any = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get analysis duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def is_successful(self) -> bool:
        """Check if analysis was successful."""
        return self.status == "success"


@dataclass
class ProcessedResult:
    """Processed and formatted analysis result."""

    analysis_result: AnalysisResult
    formatted_data: Dict[str, Any] = field(default_factory=dict)
    plots: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    export_formats: List[str] = field(default_factory=list)

    def get_plot_data(self, plot_type: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific plot type."""
        for plot in self.plots:
            if plot.get("type") == plot_type:
                return plot
        return None

    def get_table_data(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific table."""
        for table in self.tables:
            if table.get("name") == table_name:
                return table
        return None


@dataclass
class AnalysisProgress:
    """Progress information for running analysis."""

    analysis_id: str
    progress_percent: float
    current_step: str
    total_steps: int
    completed_steps: int
    estimated_time_remaining: Optional[float] = None
    status_message: str = ""
