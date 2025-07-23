"""
Solver Management System with automatic discovery.

This module handles discovery, validation, and management of all available
ICARUS solvers and their capabilities.
"""

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .models import AnalysisType
from .models import SolverInfo
from .models import SolverType

# Add ICARUS to path if not already there
icarus_path = Path(__file__).parent.parent.parent / "ICARUS"
if str(icarus_path) not in sys.path:
    sys.path.insert(0, str(icarus_path))

try:
    import ICARUS
    from ICARUS.computation import Solver
    from ICARUS.computation.analyses import Analysis
    from ICARUS.computation.analyses import BaseAnalysisInput

    ICARUS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ICARUS not available: {e}")
    ICARUS_AVAILABLE = False


class SolverManager:
    """Manages ICARUS solvers with automatic discovery and validation."""

    def __init__(self):
        self._solvers: Dict[str, SolverInfo] = {}
        self._solver_instances: Dict[str, Any] = {}
        self._analysis_mapping: Dict[AnalysisType, List[SolverType]] = {}
        self._logger = logging.getLogger(__name__)

        # Initialize solver discovery
        self._discover_solvers()

    def _discover_solvers(self) -> None:
        """Automatically discover available ICARUS solvers."""
        if not ICARUS_AVAILABLE:
            self._logger.warning("ICARUS not available, cannot discover solvers")
            return

        # Define solver configurations
        solver_configs = {
            SolverType.XFOIL: {
                "module_path": "ICARUS.solvers.Xfoil",
                "analyses": [AnalysisType.AIRFOIL_POLAR],
                "description": "2D airfoil analysis using XFoil",
                "fidelity": 2,
                "executable": getattr(ICARUS, "Xfoil_exe", None),
            },
            SolverType.AVL: {
                "module_path": "ICARUS.solvers.AVL",
                "analyses": [
                    AnalysisType.AIRPLANE_POLAR,
                    AnalysisType.AIRPLANE_STABILITY,
                ],
                "description": "Vortex lattice method for aircraft analysis",
                "fidelity": 2,
                "executable": getattr(ICARUS, "AVL_exe", None),
            },
            SolverType.GENUVP: {
                "module_path": "ICARUS.solvers.GenuVP",
                "analyses": [
                    AnalysisType.AIRPLANE_POLAR,
                    AnalysisType.AIRPLANE_STABILITY,
                ],
                "description": "High-fidelity vortex particle method",
                "fidelity": 3,
                "executable": getattr(ICARUS, "GenuVP3_exe", None),
            },
            SolverType.XFLR5: {
                "module_path": "ICARUS.solvers.XFLR5",
                "analyses": [AnalysisType.AIRFOIL_POLAR, AnalysisType.AIRPLANE_POLAR],
                "description": "Combined 2D/3D analysis tool",
                "fidelity": 2,
                "executable": None,
            },
            SolverType.OPENFOAM: {
                "module_path": "ICARUS.solvers.OpenFoam",
                "analyses": [AnalysisType.AIRFOIL_POLAR, AnalysisType.AIRPLANE_POLAR],
                "description": "CFD analysis using OpenFOAM",
                "fidelity": 3,
                "executable": None,
            },
            SolverType.FOIL2WAKE: {
                "module_path": "ICARUS.solvers.Foil2Wake",
                "analyses": [AnalysisType.AIRFOIL_POLAR],
                "description": "Unsteady airfoil analysis",
                "fidelity": 3,
                "executable": getattr(ICARUS, "Foil_exe", None),
            },
            SolverType.ICARUS_LSPT: {
                "module_path": "ICARUS.solvers.Icarus_LSPT",
                "analyses": [AnalysisType.AIRPLANE_POLAR],
                "description": "ICARUS Lifting Surface Panel Theory",
                "fidelity": 2,
                "executable": None,
            },
        }

        # Discover each solver
        for solver_type, config in solver_configs.items():
            try:
                solver_info = self._discover_solver(solver_type, config)
                if solver_info:
                    self._solvers[solver_type.value] = solver_info
                    self._update_analysis_mapping(solver_type, config["analyses"])
            except Exception as e:
                self._logger.warning(
                    f"Failed to discover solver {solver_type.value}: {e}",
                )

    def _discover_solver(
        self,
        solver_type: SolverType,
        config: Dict[str, Any],
    ) -> Optional[SolverInfo]:
        """Discover a specific solver."""
        try:
            # Try to import the solver module
            module = importlib.import_module(config["module_path"])

            # Check if executable exists (if required)
            executable_path = config.get("executable")
            is_available = True

            if executable_path:
                is_available = os.path.exists(executable_path) and os.access(
                    executable_path,
                    os.X_OK,
                )

            # Get version if available
            version = getattr(module, "__version__", None) or getattr(
                ICARUS,
                "__version__",
                "unknown",
            )

            # Create solver info
            solver_info = SolverInfo(
                name=solver_type.value.title(),
                solver_type=solver_type,
                version=version,
                executable_path=executable_path,
                is_available=is_available,
                supported_analyses=config["analyses"],
                description=config["description"],
                fidelity_level=config["fidelity"],
                capabilities=self._get_solver_capabilities(module),
                requirements=self._get_solver_requirements(solver_type),
            )

            self._logger.info(
                f"Discovered solver: {solver_info.name} (available: {is_available})",
            )
            return solver_info

        except ImportError as e:
            self._logger.warning(f"Could not import solver {solver_type.value}: {e}")
            return None
        except Exception as e:
            self._logger.error(f"Error discovering solver {solver_type.value}: {e}")
            return None

    def _get_solver_capabilities(self, module: Any) -> Dict[str, Any]:
        """Extract solver capabilities from module."""
        capabilities = {}

        try:
            # Look for common capability indicators
            if hasattr(module, "analyses"):
                analyses_attr = module.analyses
                if hasattr(analyses_attr, "__iter__") and not isinstance(
                    analyses_attr,
                    str,
                ):
                    capabilities["analyses"] = [
                        str(analysis) for analysis in analyses_attr
                    ]
                else:
                    capabilities["analyses"] = [str(analyses_attr)]

            # Check for specific analysis classes
            analysis_classes = []
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, "__bases__"):
                        if any("Analysis" in base.__name__ for base in attr.__bases__):
                            analysis_classes.append(attr_name)
                except Exception:
                    continue

            if analysis_classes:
                capabilities["analysis_classes"] = analysis_classes

        except Exception as e:
            self._logger.debug(f"Error extracting capabilities from module: {e}")

        return capabilities

    def _get_solver_requirements(self, solver_type: SolverType) -> List[str]:
        """Get requirements for a specific solver."""
        requirements_map = {
            SolverType.XFOIL: ["xfoil executable"],
            SolverType.AVL: ["avl executable"],
            SolverType.GENUVP: ["genuvp executable"],
            SolverType.OPENFOAM: ["openfoam installation"],
            SolverType.FOIL2WAKE: ["foil2wake executable"],
            SolverType.XFLR5: ["xflr5 installation"],
            SolverType.ICARUS_LSPT: [],
        }
        return requirements_map.get(solver_type, [])

    def _update_analysis_mapping(
        self,
        solver_type: SolverType,
        analyses: List[AnalysisType],
    ) -> None:
        """Update the analysis to solver mapping."""
        for analysis_type in analyses:
            if analysis_type not in self._analysis_mapping:
                self._analysis_mapping[analysis_type] = []
            if solver_type not in self._analysis_mapping[analysis_type]:
                self._analysis_mapping[analysis_type].append(solver_type)

    def get_available_solvers(self) -> List[SolverInfo]:
        """Get all available solvers."""
        return [solver for solver in self._solvers.values() if solver.is_available]

    def get_all_solvers(self) -> List[SolverInfo]:
        """Get all discovered solvers (available and unavailable)."""
        return list(self._solvers.values())

    def get_solver_info(self, solver_name: str) -> Optional[SolverInfo]:
        """Get information about a specific solver."""
        return self._solvers.get(solver_name)

    def get_solvers_for_analysis(self, analysis_type: AnalysisType) -> List[SolverInfo]:
        """Get all solvers that support a specific analysis type."""
        solver_types = self._analysis_mapping.get(analysis_type, [])
        return [
            self._solvers[st.value] for st in solver_types if st.value in self._solvers
        ]

    def get_recommended_solver(
        self,
        analysis_type: AnalysisType,
        prefer_high_fidelity: bool = False,
    ) -> Optional[SolverInfo]:
        """Get recommended solver for an analysis type."""
        available_solvers = [
            s for s in self.get_solvers_for_analysis(analysis_type) if s.is_available
        ]

        if not available_solvers:
            return None

        # Sort by fidelity and availability
        if prefer_high_fidelity:
            available_solvers.sort(key=lambda s: s.fidelity_level, reverse=True)
        else:
            available_solvers.sort(key=lambda s: s.fidelity_level)

        return available_solvers[0]

    def is_solver_available(self, solver_name: str) -> bool:
        """Check if a solver is available."""
        solver_info = self.get_solver_info(solver_name)
        return solver_info is not None and solver_info.is_available

    def validate_solver_for_analysis(
        self,
        solver_name: str,
        analysis_type: AnalysisType,
    ) -> bool:
        """Validate if a solver supports a specific analysis type."""
        solver_info = self.get_solver_info(solver_name)
        if not solver_info:
            return False
        return analysis_type in solver_info.supported_analyses

    def get_solver_status_report(self) -> Dict[str, Any]:
        """Generate a comprehensive status report of all solvers."""
        report = {
            "total_solvers": len(self._solvers),
            "available_solvers": len(self.get_available_solvers()),
            "unavailable_solvers": len(self._solvers)
            - len(self.get_available_solvers()),
            "solvers": {},
            "analysis_coverage": {},
        }

        # Solver details
        for solver_name, solver_info in self._solvers.items():
            report["solvers"][solver_name] = {
                "available": solver_info.is_available,
                "version": solver_info.version,
                "fidelity": solver_info.fidelity_level,
                "supported_analyses": [a.value for a in solver_info.supported_analyses],
                "requirements_met": solver_info.is_available,
            }

        # Analysis coverage
        for analysis_type, solver_types in self._analysis_mapping.items():
            available_count = sum(
                1 for st in solver_types if self.is_solver_available(st.value)
            )
            report["analysis_coverage"][analysis_type.value] = {
                "total_solvers": len(solver_types),
                "available_solvers": available_count,
                "solver_names": [st.value for st in solver_types],
            }

        return report
