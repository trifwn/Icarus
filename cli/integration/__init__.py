"""
ICARUS Module Integration Layer

This module provides a unified interface to all ICARUS analysis modules,
solver management, parameter validation, and result processing.
"""

from .analysis_service import AnalysisService
from .models import AnalysisConfig
from .models import AnalysisResult
from .models import AnalysisType
from .models import ProcessedResult
from .models import SolverInfo
from .models import SolverType
from .models import ValidationResult
from .parameter_validator import ParameterValidator
from .result_processor import ResultProcessor
from .solver_manager import SolverManager

__all__ = [
    "AnalysisService",
    "SolverManager",
    "ParameterValidator",
    "ResultProcessor",
    "AnalysisType",
    "SolverType",
    "AnalysisConfig",
    "AnalysisResult",
    "SolverInfo",
    "ValidationResult",
    "ProcessedResult",
]
