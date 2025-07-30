"""Error Explanation System

This module provides educational error explanations and solution suggestions
to help users understand and resolve issues they encounter.
"""

import json
import re
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class ErrorCategory(Enum):
    """Categories of errors."""

    USER_INPUT = "user_input"
    SOLVER_ERROR = "solver_error"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION = "configuration"
    DATA_ERROR = "data_error"
    NETWORK_ERROR = "network_error"


class SolutionType(Enum):
    """Types of error solutions."""

    QUICK_FIX = "quick_fix"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    CONFIGURATION_CHANGE = "configuration_change"
    ALTERNATIVE_APPROACH = "alternative_approach"
    EXTERNAL_RESOURCE = "external_resource"


@dataclass
class ErrorSolution:
    """A solution for an error."""

    id: str
    title: str
    description: str
    solution_type: SolutionType
    steps: List[str] = field(default_factory=list)
    code_example: Optional[str] = None
    related_help: List[str] = field(default_factory=list)
    difficulty: str = "easy"  # easy, medium, hard
    estimated_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "solution_type": self.solution_type.value,
            "steps": self.steps,
            "code_example": self.code_example,
            "related_help": self.related_help,
            "difficulty": self.difficulty,
            "estimated_time": self.estimated_time,
        }


@dataclass
class ErrorExplanation:
    """Detailed explanation of an error."""

    error_pattern: str  # Regex pattern to match error
    title: str
    explanation: str
    category: ErrorCategory
    common_causes: List[str] = field(default_factory=list)
    solutions: List[ErrorSolution] = field(default_factory=list)
    prevention_tips: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)
    learning_resources: List[str] = field(default_factory=list)

    def matches_error(self, error_message: str) -> bool:
        """Check if this explanation matches an error message."""
        return bool(re.search(self.error_pattern, error_message, re.IGNORECASE))


class ErrorExplanationSystem:
    """System for providing educational error explanations and solutions."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("cli/learning/data")
        self.error_explanations: List[ErrorExplanation] = []
        self.error_history: List[Dict[str, Any]] = []

        # Initialize built-in error explanations
        self._initialize_error_explanations()

    def _initialize_error_explanations(self) -> None:
        """Initialize built-in error explanations."""
        # XFoil convergence errors
        self._create_xfoil_errors()

        # File and data errors
        self._create_file_errors()

        # Configuration errors
        self._create_config_errors()

        # System errors
        self._create_system_errors()

    def _create_xfoil_errors(self) -> None:
        """Create XFoil-specific error explanations."""
        # XFoil convergence failure
        convergence_solutions = [
            ErrorSolution(
                id="reduce_angle_range",
                title="Reduce Angle of Attack Range",
                description="XFoil often fails to converge at high angles of attack near stall.",
                solution_type=SolutionType.PARAMETER_ADJUSTMENT,
                steps=[
                    "Reduce maximum angle of attack to 12-15 degrees",
                    "Use smaller angle increments (0.5° instead of 1°)",
                    "Start analysis from a converged point (usually 0°)",
                ],
                difficulty="easy",
                estimated_time="1 minute",
            ),
            ErrorSolution(
                id="adjust_reynolds",
                title="Adjust Reynolds Number",
                description="Very low or high Reynolds numbers can cause convergence issues.",
                solution_type=SolutionType.PARAMETER_ADJUSTMENT,
                steps=[
                    "Try Reynolds numbers between 1e5 and 1e7",
                    "Avoid extremely low Re < 1e4 or high Re > 1e8",
                    "Use typical flight Reynolds numbers for your application",
                ],
                related_help=["reynolds_number_guide"],
                difficulty="easy",
                estimated_time="1 minute",
            ),
            ErrorSolution(
                id="enable_viscous",
                title="Enable Viscous Analysis",
                description="Inviscid analysis may not converge for some airfoils.",
                solution_type=SolutionType.CONFIGURATION_CHANGE,
                steps=[
                    "Enable viscous analysis in XFoil settings",
                    "Allow boundary layer transition prediction",
                    "Check that Reynolds number is specified",
                ],
                difficulty="easy",
                estimated_time="30 seconds",
            ),
        ]

        xfoil_convergence = ErrorExplanation(
            error_pattern=r"xfoil.*convergence|convergence.*failed|not.*converged",
            title="XFoil Convergence Failure",
            explanation="""XFoil failed to converge to a solution. This is common when:

• Operating conditions are extreme (high angle of attack, very low/high Reynolds number)
• The airfoil geometry has issues (sharp corners, poor quality)
• Solver settings are inappropriate for the problem

Convergence failures don't necessarily mean your analysis is wrong - they often indicate you're pushing the limits of the flow physics or solver capabilities.""",
            category=ErrorCategory.SOLVER_ERROR,
            common_causes=[
                "High angle of attack near or beyond stall",
                "Very low Reynolds numbers (< 1e4)",
                "Poor airfoil geometry quality",
                "Inappropriate solver settings",
                "Extreme Mach numbers",
            ],
            solutions=convergence_solutions,
            prevention_tips=[
                "Start with moderate operating conditions",
                "Use well-validated airfoil geometries",
                "Enable viscous analysis for realistic results",
                "Check airfoil geometry for discontinuities",
            ],
            learning_resources=["xfoil_parameters", "convergence_issues"],
        )
        self.error_explanations.append(xfoil_convergence)

        # XFoil geometry errors
        geometry_solutions = [
            ErrorSolution(
                id="check_airfoil_format",
                title="Check Airfoil Data Format",
                description="Ensure airfoil coordinates are in the correct format.",
                solution_type=SolutionType.QUICK_FIX,
                steps=[
                    "Verify coordinates start from trailing edge (x=1)",
                    "Check that upper surface comes first, then lower surface",
                    "Ensure coordinates are normalized (0 ≤ x ≤ 1)",
                    "Remove any duplicate points",
                ],
                code_example="""# Correct airfoil format:
1.0000  0.0000  # Trailing edge
0.9500  0.0123  # Upper surface
...
0.0000  0.0000  # Leading edge
...
0.9500 -0.0087  # Lower surface
1.0000  0.0000  # Trailing edge""",
                difficulty="medium",
                estimated_time="5 minutes",
            ),
        ]

        xfoil_geometry = ErrorExplanation(
            error_pattern=r"geometry.*error|invalid.*airfoil|bad.*coordinates",
            title="Airfoil Geometry Error",
            explanation="""The airfoil geometry data is invalid or improperly formatted.

Common issues include:
• Incorrect coordinate format or ordering
• Duplicate or missing points
• Coordinates outside valid range
• Self-intersecting geometry""",
            category=ErrorCategory.DATA_ERROR,
            common_causes=[
                "Incorrect coordinate file format",
                "Coordinates not normalized to chord length",
                "Missing leading or trailing edge points",
                "Self-intersecting airfoil geometry",
            ],
            solutions=geometry_solutions,
            prevention_tips=[
                "Use validated airfoil databases (UIUC, etc.)",
                "Check geometry visually before analysis",
                "Ensure proper coordinate format and ordering",
            ],
        )
        self.error_explanations.append(xfoil_geometry)

    def _create_file_errors(self) -> None:
        """Create file and data error explanations."""
        file_solutions = [
            ErrorSolution(
                id="check_file_path",
                title="Verify File Path",
                description="Ensure the file path is correct and accessible.",
                solution_type=SolutionType.QUICK_FIX,
                steps=[
                    "Check that the file path is spelled correctly",
                    "Verify the file exists in the specified location",
                    "Ensure you have read permissions for the file",
                    "Try using an absolute path instead of relative",
                ],
                difficulty="easy",
                estimated_time="1 minute",
            ),
            ErrorSolution(
                id="check_file_format",
                title="Check File Format",
                description="Verify the file is in the expected format.",
                solution_type=SolutionType.PARAMETER_ADJUSTMENT,
                steps=[
                    "Check file extension matches expected format",
                    "Open file in text editor to verify contents",
                    "Compare with working example files",
                    "Try converting to supported format if needed",
                ],
                difficulty="medium",
                estimated_time="3 minutes",
            ),
        ]

        file_not_found = ErrorExplanation(
            error_pattern=r"file.*not.*found|no.*such.*file|cannot.*open.*file",
            title="File Not Found",
            explanation="""The specified file could not be found or opened.

This usually happens when:
• The file path is incorrect
• The file doesn't exist at the specified location
• You don't have permission to access the file
• The file is locked by another application""",
            category=ErrorCategory.USER_INPUT,
            common_causes=[
                "Incorrect file path or filename",
                "File moved or deleted",
                "Insufficient file permissions",
                "File locked by another application",
            ],
            solutions=file_solutions,
            prevention_tips=[
                "Use absolute paths when possible",
                "Verify file exists before referencing",
                "Check file permissions and ownership",
            ],
        )
        self.error_explanations.append(file_not_found)

    def _create_config_errors(self) -> None:
        """Create configuration error explanations."""
        config_solutions = [
            ErrorSolution(
                id="reset_config",
                title="Reset Configuration",
                description="Reset configuration to default values.",
                solution_type=SolutionType.CONFIGURATION_CHANGE,
                steps=[
                    "Go to Settings > Reset to Defaults",
                    "Or delete the configuration file to regenerate",
                    "Restart the application",
                    "Reconfigure your preferences",
                ],
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            ErrorSolution(
                id="check_config_syntax",
                title="Check Configuration Syntax",
                description="Verify configuration file syntax is correct.",
                solution_type=SolutionType.CONFIGURATION_CHANGE,
                steps=[
                    "Open configuration file in text editor",
                    "Check for missing commas, brackets, or quotes",
                    "Validate JSON syntax using online validator",
                    "Compare with default configuration file",
                ],
                difficulty="medium",
                estimated_time="5 minutes",
            ),
        ]

        config_error = ErrorExplanation(
            error_pattern=r"configuration.*error|config.*invalid|settings.*corrupt",
            title="Configuration Error",
            explanation="""There's an issue with the application configuration.

This can happen when:
• Configuration file is corrupted or has syntax errors
• Settings contain invalid values
• Configuration file permissions are incorrect
• Incompatible settings from different versions""",
            category=ErrorCategory.CONFIGURATION,
            common_causes=[
                "Corrupted configuration file",
                "Invalid configuration values",
                "Syntax errors in config file",
                "Version compatibility issues",
            ],
            solutions=config_solutions,
            prevention_tips=[
                "Backup configuration before making changes",
                "Use the settings interface instead of editing files directly",
                "Validate configuration after changes",
            ],
        )
        self.error_explanations.append(config_error)

    def _create_system_errors(self) -> None:
        """Create system-level error explanations."""
        memory_solutions = [
            ErrorSolution(
                id="reduce_problem_size",
                title="Reduce Problem Size",
                description="Reduce the computational requirements of your analysis.",
                solution_type=SolutionType.PARAMETER_ADJUSTMENT,
                steps=[
                    "Reduce the number of analysis points",
                    "Use coarser angle of attack increments",
                    "Analyze fewer configurations simultaneously",
                    "Close other applications to free memory",
                ],
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            ErrorSolution(
                id="increase_system_memory",
                title="Increase Available Memory",
                description="Free up system memory for the analysis.",
                solution_type=SolutionType.ALTERNATIVE_APPROACH,
                steps=[
                    "Close unnecessary applications",
                    "Restart the computer to clear memory",
                    "Consider upgrading system RAM",
                    "Use batch processing for large studies",
                ],
                difficulty="easy",
                estimated_time="5 minutes",
            ),
        ]

        memory_error = ErrorExplanation(
            error_pattern=r"memory.*error|out.*of.*memory|insufficient.*memory",
            title="Memory Error",
            explanation="""The system has run out of available memory.

This typically occurs with:
• Large parameter studies or optimization runs
• High-resolution analyses with many data points
• Multiple simultaneous analyses
• Insufficient system RAM for the problem size""",
            category=ErrorCategory.SYSTEM_ERROR,
            common_causes=[
                "Problem size too large for available memory",
                "Memory leak in long-running analyses",
                "Insufficient system RAM",
                "Other applications using too much memory",
            ],
            solutions=memory_solutions,
            prevention_tips=[
                "Monitor memory usage during large analyses",
                "Break large studies into smaller batches",
                "Close unnecessary applications before analysis",
                "Consider system memory upgrades for large problems",
            ],
        )
        self.error_explanations.append(memory_error)

    def explain_error(
        self,
        error_message: str,
        context: Dict[str, Any] = None,
    ) -> Optional[ErrorExplanation]:
        """Get explanation for an error message."""
        for explanation in self.error_explanations:
            if explanation.matches_error(error_message):
                # Log the error for learning
                self._log_error(error_message, explanation, context)
                return explanation
        return None

    def get_solutions(self, error_message: str) -> List[ErrorSolution]:
        """Get solutions for an error message."""
        explanation = self.explain_error(error_message)
        if explanation:
            return explanation.solutions
        return []

    def _log_error(
        self,
        error_message: str,
        explanation: ErrorExplanation,
        context: Dict[str, Any] = None,
    ) -> None:
        """Log error occurrence for analytics."""
        error_entry = {
            "timestamp": None,  # Would use datetime in real implementation
            "error_message": error_message,
            "explanation_id": explanation.title,
            "category": explanation.category.value,
            "context": context or {},
        }
        self.error_history.append(error_entry)

    def get_common_errors(
        self,
        category: ErrorCategory = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get most common errors from history."""
        filtered_errors = self.error_history
        if category:
            filtered_errors = [
                e for e in self.error_history if e["category"] == category.value
            ]

        # Count occurrences
        error_counts = {}
        for error in filtered_errors:
            key = error["explanation_id"]
            error_counts[key] = error_counts.get(key, 0) + 1

        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {"error": error, "count": count} for error, count in sorted_errors[:limit]
        ]

    def add_error_explanation(self, explanation: ErrorExplanation) -> None:
        """Add a new error explanation."""
        self.error_explanations.append(explanation)

    def get_error_categories(self) -> List[ErrorCategory]:
        """Get all error categories."""
        return list(ErrorCategory)

    def save_error_data(self, filepath: Path = None) -> None:
        """Save error explanations and history."""
        if filepath is None:
            filepath = self.data_dir / "error_data.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "error_explanations": [
                {
                    "error_pattern": exp.error_pattern,
                    "title": exp.title,
                    "explanation": exp.explanation,
                    "category": exp.category.value,
                    "common_causes": exp.common_causes,
                    "solutions": [sol.to_dict() for sol in exp.solutions],
                    "prevention_tips": exp.prevention_tips,
                    "related_errors": exp.related_errors,
                    "learning_resources": exp.learning_resources,
                }
                for exp in self.error_explanations
            ],
            "error_history": self.error_history,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_error_data(self, filepath: Path = None) -> None:
        """Load error explanations and history."""
        if filepath is None:
            filepath = self.data_dir / "error_data.json"

        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)

            # Load error explanations
            self.error_explanations = []
            for exp_data in data.get("error_explanations", []):
                solutions = []
                for sol_data in exp_data.get("solutions", []):
                    solution = ErrorSolution(
                        id=sol_data["id"],
                        title=sol_data["title"],
                        description=sol_data["description"],
                        solution_type=SolutionType(sol_data["solution_type"]),
                        steps=sol_data.get("steps", []),
                        code_example=sol_data.get("code_example"),
                        related_help=sol_data.get("related_help", []),
                        difficulty=sol_data.get("difficulty", "easy"),
                        estimated_time=sol_data.get("estimated_time"),
                    )
                    solutions.append(solution)

                explanation = ErrorExplanation(
                    error_pattern=exp_data["error_pattern"],
                    title=exp_data["title"],
                    explanation=exp_data["explanation"],
                    category=ErrorCategory(exp_data["category"]),
                    common_causes=exp_data.get("common_causes", []),
                    solutions=solutions,
                    prevention_tips=exp_data.get("prevention_tips", []),
                    related_errors=exp_data.get("related_errors", []),
                    learning_resources=exp_data.get("learning_resources", []),
                )
                self.error_explanations.append(explanation)

            # Load error history
            self.error_history = data.get("error_history", [])
