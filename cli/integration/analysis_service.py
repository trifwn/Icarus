"""
Unified Analysis Service for ICARUS modules.

This service provides a unified interface to all ICARUS analysis capabilities,
integrating solver management, parameter validation, and result processing.
"""

import asyncio
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from ..core.performance import get_performance_manager
from .models import AnalysisConfig
from .models import AnalysisProgress
from .models import AnalysisResult
from .models import AnalysisType
from .models import ProcessedResult
from .models import ValidationResult
from .parameter_validator import ParameterValidator
from .result_processor import ResultProcessor
from .solver_manager import SolverManager

# Add ICARUS to path if not already there
icarus_path = Path(__file__).parent.parent.parent / "ICARUS"
if str(icarus_path) not in sys.path:
    sys.path.insert(0, str(icarus_path))

try:
    import ICARUS
    from ICARUS.airfoils import Airfoil
    from ICARUS.computation import Solver
    from ICARUS.computation.analyses import Analysis
    from ICARUS.computation.analyses import BaseAnalysisInput
    from ICARUS.vehicle import Airplane

    ICARUS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ICARUS not available: {e}")
    ICARUS_AVAILABLE = False


class AnalysisService:
    """Unified service for ICARUS analysis operations."""

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self.solver_manager = SolverManager()
        self.parameter_validator = ParameterValidator()
        self.result_processor = ResultProcessor()

        # Performance management
        self.performance_manager = get_performance_manager()

        # Track running analyses
        self._running_analyses: Dict[str, AnalysisProgress] = {}
        self._progress_callbacks: Dict[str, List[Callable]] = {}

        # Thread pool for analysis execution
        self._executor = ThreadPoolExecutor(max_workers=4)

        self._logger.info("AnalysisService initialized with performance optimization")

    def get_available_analysis_types(self) -> List[AnalysisType]:
        """Get all available analysis types."""
        return list(AnalysisType)

    def get_available_solvers(self) -> List[Dict[str, Any]]:
        """Get information about all available solvers."""
        solvers = self.solver_manager.get_available_solvers()
        return [
            {
                "name": solver.name,
                "type": solver.solver_type.value,
                "version": solver.version,
                "description": solver.description,
                "fidelity": solver.fidelity_level,
                "supported_analyses": [a.value for a in solver.supported_analyses],
                "is_available": solver.is_available,
            }
            for solver in solvers
        ]

    def get_solvers_for_analysis(
        self,
        analysis_type: AnalysisType,
    ) -> List[Dict[str, Any]]:
        """Get solvers that support a specific analysis type."""
        solvers = self.solver_manager.get_solvers_for_analysis(analysis_type)
        return [
            {
                "name": solver.name,
                "type": solver.solver_type.value,
                "description": solver.description,
                "fidelity": solver.fidelity_level,
                "is_available": solver.is_available,
            }
            for solver in solvers
        ]

    def get_recommended_solver(
        self,
        analysis_type: AnalysisType,
        prefer_high_fidelity: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get recommended solver for an analysis type."""
        solver = self.solver_manager.get_recommended_solver(
            analysis_type,
            prefer_high_fidelity,
        )
        if not solver:
            return None

        return {
            "name": solver.name,
            "type": solver.solver_type.value,
            "description": solver.description,
            "fidelity": solver.fidelity_level,
            "reason": "Recommended based on availability and fidelity preference",
        }

    def validate_analysis_config(self, config: AnalysisConfig) -> ValidationResult:
        """Validate an analysis configuration."""
        # Basic validation
        result = self.parameter_validator.validate_analysis_config(config)

        # Check solver availability
        if not self.solver_manager.is_solver_available(config.solver_type.value):
            result.add_error(
                "solver_type",
                f"Solver '{config.solver_type.value}' is not available",
                "availability_error",
            )

            # Suggest alternative solvers
            alternatives = self.get_solvers_for_analysis(config.analysis_type)
            if alternatives:
                available_alternatives = [
                    s["name"] for s in alternatives if s["is_available"]
                ]
                if available_alternatives:
                    result.add_suggestion(
                        f"Available alternatives: {', '.join(available_alternatives)}",
                    )

        # Check solver compatibility
        if not self.solver_manager.validate_solver_for_analysis(
            config.solver_type.value,
            config.analysis_type,
        ):
            result.add_error(
                "solver_type",
                f"Solver '{config.solver_type.value}' does not support analysis type '{config.analysis_type.value}'",
                "compatibility_error",
            )

        return result

    def get_parameter_suggestions(self, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Get parameter suggestions for an analysis type."""
        return self.parameter_validator.get_parameter_suggestions(analysis_type)

    async def run_analysis(
        self,
        config: AnalysisConfig,
        progress_callback: Optional[Callable] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
    ) -> AnalysisResult:
        """Run an analysis with the given configuration."""
        # Generate cache key for this analysis
        cache_key = self._generate_analysis_cache_key(config)

        # Check cache first if enabled
        if use_cache:
            cached_result = self.performance_manager.cache.get(cache_key)
            if cached_result:
                self._logger.info(f"Analysis result found in cache: {cache_key}")
                return cached_result

        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())

        # Validate configuration
        validation_result = self.validate_analysis_config(config)
        if not validation_result.is_valid:
            return AnalysisResult(
                analysis_id=analysis_id,
                config=config,
                status="failed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=f"Configuration validation failed: {validation_result.errors[0].message}",
            )

        # Create analysis result
        result = AnalysisResult(
            analysis_id=analysis_id,
            config=config,
            status="running",
            start_time=datetime.now(),
        )

        # Set up progress tracking
        if progress_callback:
            self._progress_callbacks[analysis_id] = [progress_callback]

        progress = AnalysisProgress(
            analysis_id=analysis_id,
            progress_percent=0.0,
            current_step="Initializing analysis",
            total_steps=5,
            completed_steps=0,
        )
        self._running_analyses[analysis_id] = progress
        self._notify_progress(analysis_id, progress)

        try:
            # Execute analysis using async operation manager for better resource management
            raw_result = await self.performance_manager.async_manager.execute_async(
                operation_id=analysis_id,
                coro_func=self._execute_analysis_async,
                config=config,
                analysis_id=analysis_id,
                timeout=3600.0,  # 1 hour timeout
                progress_callback=lambda op_id,
                elapsed,
                status: self._async_progress_callback(op_id, elapsed, status),
            )

            # Update result
            result.raw_data = raw_result
            result.status = "success"
            result.end_time = datetime.now()

            # Cache the result if successful
            if use_cache and result.status == "success":
                self.performance_manager.cache.put(
                    cache_key,
                    result,
                    ttl_seconds=cache_ttl,
                )
                self._logger.info(f"Analysis result cached: {cache_key}")

            self._logger.info(f"Analysis {analysis_id} completed successfully")

        except Exception as e:
            result.status = "failed"
            result.end_time = datetime.now()
            result.error_message = str(e)
            self._logger.error(f"Analysis {analysis_id} failed: {e}")

        finally:
            # Clean up progress tracking
            if analysis_id in self._running_analyses:
                del self._running_analyses[analysis_id]
            if analysis_id in self._progress_callbacks:
                del self._progress_callbacks[analysis_id]

        return result

    def _execute_analysis_sync(self, config: AnalysisConfig, analysis_id: str) -> Any:
        """Execute analysis synchronously (runs in thread pool)."""
        if not ICARUS_AVAILABLE:
            raise RuntimeError("ICARUS is not available")

        # Update progress
        self._update_progress(analysis_id, 1, "Loading target")

        # Load target (airfoil, airplane, etc.)
        target_object = self._load_target(config)

        # Update progress
        self._update_progress(analysis_id, 2, "Configuring solver")

        # Get solver instance
        solver_instance = self._get_solver_instance(config)

        # Update progress
        self._update_progress(analysis_id, 3, "Running analysis")

        # Execute analysis based on type
        if config.analysis_type == AnalysisType.AIRFOIL_POLAR:
            result = self._run_airfoil_polar_analysis(
                config,
                target_object,
                solver_instance,
            )
        elif config.analysis_type == AnalysisType.AIRPLANE_POLAR:
            result = self._run_airplane_polar_analysis(
                config,
                target_object,
                solver_instance,
            )
        elif config.analysis_type == AnalysisType.AIRPLANE_STABILITY:
            result = self._run_stability_analysis(
                config,
                target_object,
                solver_instance,
            )
        else:
            raise ValueError(f"Analysis type {config.analysis_type} not implemented")

        # Update progress
        self._update_progress(analysis_id, 4, "Processing results")

        return result

    def _load_target(self, config: AnalysisConfig) -> Any:
        """Load the analysis target (airfoil, airplane, etc.)."""
        target = config.target

        if config.analysis_type == AnalysisType.AIRFOIL_POLAR:
            # Load airfoil
            if Path(target).exists():
                # Load from file
                return Airfoil.from_file(target)
            else:
                # Try to create from NACA name
                if target.upper().startswith("NACA"):
                    naca_digits = target.upper().replace("NACA", "").strip()
                    return Airfoil.naca(naca_digits)
                else:
                    # Try to load from web or return mock airfoil for testing
                    try:
                        return Airfoil.load_from_web(target)
                    except Exception:
                        # For testing, create a simple NACA 0012
                        return Airfoil.naca("0012")

        elif config.analysis_type in [
            AnalysisType.AIRPLANE_POLAR,
            AnalysisType.AIRPLANE_STABILITY,
        ]:
            # Load airplane - for now return a placeholder since we don't have airplane loading implemented
            if Path(target).exists():
                # In a real implementation, this would load the airplane file
                return f"Airplane from file: {target}"
            else:
                raise FileNotFoundError(f"Airplane file not found: {target}")

        else:
            raise ValueError(f"Unknown analysis type: {config.analysis_type}")

    def _get_solver_instance(self, config: AnalysisConfig) -> Any:
        """Get solver instance for the configuration."""
        # This would create the appropriate solver instance
        # For now, return a placeholder
        return f"Solver instance for {config.solver_type.value}"

    def _run_airfoil_polar_analysis(
        self,
        config: AnalysisConfig,
        airfoil: Any,
        solver: Any,
    ) -> Any:
        """Run airfoil polar analysis."""
        # Placeholder implementation
        # In real implementation, this would:
        # 1. Create appropriate analysis input object
        # 2. Configure solver parameters
        # 3. Execute analysis
        # 4. Return results

        import numpy as np

        # Generate mock data for demonstration
        alpha = np.linspace(
            config.parameters.get("min_aoa", -10),
            config.parameters.get("max_aoa", 15),
            int(
                (
                    config.parameters.get("max_aoa", 15)
                    - config.parameters.get("min_aoa", -10)
                )
                / config.parameters.get("aoa_step", 0.5),
            )
            + 1,
        )

        # Mock airfoil polar data
        cl = 2 * np.pi * np.sin(np.radians(alpha)) * np.cos(np.radians(alpha))
        cd = 0.01 + 0.02 * (np.radians(alpha)) ** 2
        cm = -0.1 * np.ones_like(alpha)

        return {
            "polars": {
                "alpha": alpha,
                "cl": cl,
                "cd": cd,
                "cm": cm,
            },
            "reynolds": config.parameters.get("reynolds"),
            "mach": config.parameters.get("mach", 0.0),
        }

    def _run_airplane_polar_analysis(
        self,
        config: AnalysisConfig,
        airplane: Any,
        solver: Any,
    ) -> Any:
        """Run airplane polar analysis."""
        # Placeholder implementation
        import numpy as np

        alpha = np.linspace(
            config.parameters.get("min_aoa", -5),
            config.parameters.get("max_aoa", 15),
            21,
        )

        # Mock airplane polar data
        CL = 0.1 * alpha + 0.5
        CD = 0.02 + 0.001 * alpha**2
        CM = -0.05 * alpha

        return {
            "alpha": alpha,
            "CL": CL,
            "CD": CD,
            "CM": CM,
            "velocity": config.parameters.get("velocity"),
            "altitude": config.parameters.get("altitude"),
        }

    def _run_stability_analysis(
        self,
        config: AnalysisConfig,
        airplane: Any,
        solver: Any,
    ) -> Any:
        """Run stability analysis."""
        # Placeholder implementation
        return {
            "stability_derivatives": {
                "CLa": 5.2,
                "CMa": -0.8,
                "CYb": -0.5,
                "Clb": -0.1,
                "Cnb": 0.15,
            },
            "trim_conditions": {
                "alpha_trim": 2.5,
                "elevator_trim": -1.2,
            },
        }

    def _update_progress(
        self,
        analysis_id: str,
        completed_steps: int,
        current_step: str,
    ) -> None:
        """Update analysis progress."""
        if analysis_id not in self._running_analyses:
            return

        progress = self._running_analyses[analysis_id]
        progress.completed_steps = completed_steps
        progress.current_step = current_step
        progress.progress_percent = (completed_steps / progress.total_steps) * 100

        self._notify_progress(analysis_id, progress)

    def _notify_progress(self, analysis_id: str, progress: AnalysisProgress) -> None:
        """Notify progress callbacks."""
        callbacks = self._progress_callbacks.get(analysis_id, [])
        for callback in callbacks:
            try:
                callback(progress)
            except Exception as e:
                self._logger.error(f"Error in progress callback: {e}")

    def process_result(self, analysis_result: AnalysisResult) -> ProcessedResult:
        """Process a raw analysis result."""
        return self.result_processor.process_result(analysis_result)

    def export_result(
        self,
        processed_result: ProcessedResult,
        format_type: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Export a processed result."""
        return self.result_processor.export_result(
            processed_result,
            format_type,
            output_path,
        )

    def get_analysis_progress(self, analysis_id: str) -> Optional[AnalysisProgress]:
        """Get progress for a running analysis."""
        return self._running_analyses.get(analysis_id)

    def cancel_analysis(self, analysis_id: str) -> bool:
        """Cancel a running analysis."""
        if analysis_id in self._running_analyses:
            # In a real implementation, this would signal the analysis to stop
            del self._running_analyses[analysis_id]
            if analysis_id in self._progress_callbacks:
                del self._progress_callbacks[analysis_id]
            self._logger.info(f"Analysis {analysis_id} cancelled")
            return True
        return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        solver_report = self.solver_manager.get_solver_status_report()

        return {
            "icarus_available": ICARUS_AVAILABLE,
            "running_analyses": len(self._running_analyses),
            "solver_status": solver_report,
            "supported_analyses": [a.value for a in AnalysisType],
            "service_status": "operational" if ICARUS_AVAILABLE else "limited",
        }

    def _generate_analysis_cache_key(self, config: AnalysisConfig) -> str:
        """Generate cache key for analysis configuration."""
        return self.performance_manager.cache.generate_key(
            analysis_type=config.analysis_type.value,
            solver_type=config.solver_type.value,
            target=config.target,
            parameters=config.parameters,
        )

    async def _execute_analysis_async(
        self,
        config: AnalysisConfig,
        analysis_id: str,
    ) -> Any:
        """Execute analysis asynchronously."""
        # Run the synchronous analysis in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._execute_analysis_sync,
            config,
            analysis_id,
        )

    async def _async_progress_callback(
        self,
        operation_id: str,
        elapsed: float,
        status: str,
    ) -> None:
        """Handle async operation progress updates."""
        if operation_id in self._running_analyses:
            progress = self._running_analyses[operation_id]
            progress.current_step = (
                f"Running analysis ({status}) - {elapsed:.1f}s elapsed"
            )
            self._notify_progress(operation_id, progress)

    async def run_analysis_background(
        self,
        config: AnalysisConfig,
        task_name: Optional[str] = None,
        priority: int = 0,
        progress_callback: Optional[Callable] = None,
        completion_callback: Optional[Callable] = None,
    ) -> str:
        """Submit analysis for background execution."""
        analysis_id = str(uuid.uuid4())
        task_name = task_name or f"Analysis: {config.analysis_type.value}"

        def analysis_wrapper():
            """Wrapper function for background execution."""
            return asyncio.run(self.run_analysis(config, progress_callback))

        # Submit to background executor
        task_id = self.performance_manager.background_executor.submit_task(
            task_id=analysis_id,
            name=task_name,
            func=analysis_wrapper,
            priority=priority,
            use_process=False,  # Keep in same process for now
            progress_callback=progress_callback,
            completion_callback=completion_callback,
        )

        self._logger.info(f"Analysis submitted for background execution: {task_id}")
        return task_id

    def get_background_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a background analysis task."""
        task = self.performance_manager.background_executor.get_task_status(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status,
            "progress": task.progress,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat()
            if task.completed_at
            else None,
            "error": task.error,
            "priority": task.priority,
        }

    def cancel_background_analysis(self, task_id: str) -> bool:
        """Cancel a background analysis task."""
        return self.performance_manager.background_executor.cancel_task(task_id)

    def get_all_background_tasks(self) -> List[Dict[str, Any]]:
        """Get all background analysis tasks."""
        tasks = self.performance_manager.background_executor.get_all_tasks()
        return [
            {
                "task_id": task.task_id,
                "name": task.name,
                "status": task.status,
                "progress": task.progress,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat()
                if task.completed_at
                else None,
                "error": task.error,
                "priority": task.priority,
            }
            for task in tasks
        ]

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get analysis cache statistics."""
        return self.performance_manager.cache.get_stats()

    def clear_analysis_cache(self) -> None:
        """Clear the analysis cache."""
        self.performance_manager.cache.clear()
        self._logger.info("Analysis cache cleared")

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report including analysis-specific metrics."""
        base_report = self.performance_manager.get_performance_report()

        # Add analysis-specific metrics
        base_report["analysis_service"] = {
            "running_analyses": len(self._running_analyses),
            "cache_stats": self.get_cache_statistics(),
            "background_tasks": {
                "total": len(self.get_all_background_tasks()),
                "running": len(
                    [
                        t
                        for t in self.get_all_background_tasks()
                        if t["status"] == "running"
                    ],
                ),
            },
        }

        return base_report

    def optimize_performance(self) -> Dict[str, Any]:
        """Perform performance optimization tasks."""
        optimization_results = {}

        # Clean up expired cache entries
        expired_cleaned = self.performance_manager.cache.cleanup_expired()
        optimization_results["cache_expired_cleaned"] = expired_cleaned

        # Clean up old background tasks
        tasks_cleaned = (
            self.performance_manager.background_executor.cleanup_completed_tasks()
        )
        optimization_results["background_tasks_cleaned"] = tasks_cleaned

        # Force garbage collection
        gc_stats = self.performance_manager.resource_monitor.force_garbage_collection()
        optimization_results["garbage_collection"] = gc_stats

        self._logger.info(f"Performance optimization completed: {optimization_results}")
        return optimization_results

    def shutdown(self) -> None:
        """Shutdown the analysis service."""
        self._logger.info("Shutting down AnalysisService")

        # Cancel all running analyses
        for analysis_id in list(self._running_analyses.keys()):
            self.cancel_analysis(analysis_id)

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        self._logger.info("AnalysisService shutdown complete")
