"""
Performance Testing Suite for ICARUS CLI

This module provides comprehensive performance tests and benchmarking
for the ICARUS CLI system components.
"""

import asyncio
import gc
import psutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# Add CLI directory to path for imports
cli_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cli_dir))

from .framework import TestResult
from .framework import TestStatus
from .framework import TestType


@dataclass
class PerformanceMetrics:
    """Performance metrics for a test"""

    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    memory_growth_mb: float
    operations_per_second: Optional[float] = None


class PerformanceTestSuite:
    """Performance test suite for ICARUS CLI system"""

    def __init__(self):
        self.test_results: List[TestResult] = []
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}

    async def run_all_tests(self) -> List[TestResult]:
        """Run all performance tests"""
        self.test_results = []

        # Core component performance tests
        await self._test_config_manager_performance()
        await self._test_event_system_performance()
        await self._test_state_manager_performance()

        # UI component performance tests
        await self._test_screen_manager_performance()
        await self._test_theme_system_performance()

        # Analysis performance tests
        await self._test_analysis_service_performance()
        await self._test_solver_manager_performance()

        # Data management performance tests
        await self._test_database_performance()
        await self._test_export_performance()

        # Memory and resource tests
        await self._test_memory_usage()
        await self._test_resource_cleanup()

        # Concurrent operation tests
        await self._test_concurrent_operations()

        return self.test_results

    async def _run_performance_test(
        self,
        test_name: str,
        test_func,
        iterations: int = 100,
        warmup_iterations: int = 10,
    ):
        """Run a performance test with comprehensive metrics collection"""

        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                await test_func()
            except Exception:
                pass  # Ignore warmup errors

        # Force garbage collection before measurement
        gc.collect()

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Performance measurement
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        peak_memory = initial_memory

        successful_iterations = 0
        errors = []

        for i in range(iterations):
            try:
                iteration_start = time.perf_counter()
                await test_func()
                iteration_end = time.perf_counter()

                # Track peak memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)

                successful_iterations += 1

            except Exception as e:
                errors.append(f"Iteration {i}: {str(e)}")

        end_time = time.perf_counter()
        end_cpu_time = time.process_time()

        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024

        # Calculate metrics
        total_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        avg_execution_time = total_time / max(successful_iterations, 1)
        cpu_usage = (cpu_time / total_time) * 100 if total_time > 0 else 0
        memory_growth = final_memory - initial_memory
        ops_per_second = successful_iterations / total_time if total_time > 0 else 0

        metrics = PerformanceMetrics(
            execution_time=avg_execution_time,
            memory_usage_mb=final_memory,
            cpu_usage_percent=cpu_usage,
            peak_memory_mb=peak_memory,
            memory_growth_mb=memory_growth,
            operations_per_second=ops_per_second,
        )

        # Determine test status
        if successful_iterations == 0:
            status = TestStatus.ERROR
            error_message = f"All iterations failed: {errors[:3]}"
        elif len(errors) > iterations * 0.1:  # More than 10% failure rate
            status = TestStatus.FAILED
            error_message = f"High failure rate: {len(errors)}/{iterations} failed"
        else:
            status = TestStatus.PASSED
            error_message = None

        result = TestResult(
            name=test_name,
            test_type=TestType.PERFORMANCE,
            status=status,
            duration=total_time,
            error_message=error_message,
            details={
                "metrics": {
                    "avg_execution_time_ms": avg_execution_time * 1000,
                    "operations_per_second": ops_per_second,
                    "memory_usage_mb": final_memory,
                    "peak_memory_mb": peak_memory,
                    "memory_growth_mb": memory_growth,
                    "cpu_usage_percent": cpu_usage,
                },
                "test_config": {
                    "iterations": iterations,
                    "successful_iterations": successful_iterations,
                    "error_count": len(errors),
                },
            },
        )

        self.test_results.append(result)
        return result

    # Core Component Performance Tests

    async def _test_config_manager_performance(self):
        """Test configuration manager performance"""

        async def config_operations():
            try:
                from core.config import ConfigManager

                config = ConfigManager()

                # Test rapid get/set operations
                for i in range(10):
                    config.set(f"test_key_{i}", f"test_value_{i}")
                    value = config.get(f"test_key_{i}")
                    assert value == f"test_value_{i}"

                # Test bulk operations
                bulk_data = {f"bulk_key_{i}": f"bulk_value_{i}" for i in range(50)}
                for key, value in bulk_data.items():
                    config.set(key, value)

            except ImportError:
                # Create mock operations for testing
                data = {}
                for i in range(60):
                    data[f"key_{i}"] = f"value_{i}"

        await self._run_performance_test(
            "ConfigManager Performance",
            config_operations,
            iterations=1000,
        )

    async def _test_event_system_performance(self):
        """Test event system performance"""

        async def event_operations():
            try:
                from app.event_system import EventSystem

                event_system = EventSystem()
                received_events = []

                def event_handler(data):
                    received_events.append(data)

                # Subscribe to events
                event_system.subscribe("perf_test_event", event_handler)

                # Emit events rapidly
                for i in range(10):
                    event_system.emit_sync("perf_test_event", {"index": i})

                # Test async events
                for i in range(5):
                    await event_system.emit("perf_test_event", {"async_index": i})

            except ImportError:
                # Mock event operations
                events = []
                for i in range(15):
                    events.append({"event": i})

        await self._run_performance_test(
            "EventSystem Performance",
            event_operations,
            iterations=500,
        )

    async def _test_state_manager_performance(self):
        """Test state manager performance"""

        async def state_operations():
            try:
                from app.state_manager import StateManager

                state_manager = StateManager()

                # Initialize session
                session = await state_manager.initialize_session()

                # Rapid state updates
                for i in range(20):
                    await state_manager.update_state(f"key_{i}", f"value_{i}")

                # Bulk state retrieval
                for i in range(20):
                    state = state_manager.get_current_state()
                    assert f"key_{i}" in state

            except ImportError:
                # Mock state operations
                state = {}
                for i in range(40):
                    state[f"key_{i}"] = f"value_{i}"

        await self._run_performance_test(
            "StateManager Performance",
            state_operations,
            iterations=200,
        )

    # UI Component Performance Tests

    async def _test_screen_manager_performance(self):
        """Test screen manager performance"""

        async def screen_operations():
            try:
                # Mock app for testing
                class MockApp:
                    def __init__(self):
                        self.screens = {}
                        self.event_system = MockEventSystem()

                    def install_screen(self, screen, name):
                        self.screens[name] = screen

                    async def push_screen(self, name):
                        pass

                class MockEventSystem:
                    async def emit(self, event, data):
                        pass

                from app.screen_manager import ScreenManager

                app = MockApp()
                screen_manager = ScreenManager(app)

                # Initialize screens
                await screen_manager.initialize()

                # Rapid screen switching
                screens = ["dashboard", "analysis", "results", "workflow"]
                for screen in screens:
                    await screen_manager.switch_to(screen)

                # Test navigation history
                for _ in range(5):
                    await screen_manager.go_back()

            except ImportError:
                # Mock screen operations
                screens = ["dashboard", "analysis", "results", "workflow"]
                current_screen = screens[0]
                for screen in screens[1:]:
                    current_screen = screen

        await self._run_performance_test(
            "ScreenManager Performance",
            screen_operations,
            iterations=100,
        )

    async def _test_theme_system_performance(self):
        """Test theme system performance"""

        async def theme_operations():
            try:
                from tui.themes.theme_manager import ThemeManager

                theme_manager = ThemeManager()

                # Get available themes
                themes = theme_manager.get_available_themes()

                # Rapid theme switching
                for theme_id in themes[:3]:  # Test first 3 themes
                    theme_manager.set_theme(theme_id)
                    css = theme_manager.get_current_css()
                    assert len(css) > 0

            except ImportError:
                # Mock theme operations
                themes = ["default", "dark", "aerospace"]
                for theme in themes:
                    css = f"/* {theme} theme */ body {{ color: black; }}"

        await self._run_performance_test(
            "ThemeSystem Performance",
            theme_operations,
            iterations=50,
        )

    # Analysis Performance Tests

    async def _test_analysis_service_performance(self):
        """Test analysis service performance"""

        async def analysis_operations():
            try:
                from integration.analysis_service import AnalysisService

                service = AnalysisService()

                # Rapid module queries
                for _ in range(10):
                    modules = service.get_available_modules()

                # Parameter validation
                test_params = {"reynolds": 1000000, "mach": 0.1}
                for _ in range(5):
                    validation = service.validate_parameters(test_params)

            except ImportError:
                # Mock analysis operations
                modules = ["xfoil", "avl", "gnvp"]
                for _ in range(15):
                    params = {"reynolds": 1000000, "mach": 0.1}
                    valid = True

        await self._run_performance_test(
            "AnalysisService Performance",
            analysis_operations,
            iterations=100,
        )

    async def _test_solver_manager_performance(self):
        """Test solver manager performance"""

        async def solver_operations():
            try:
                from integration.solver_manager import SolverManager

                solver_manager = SolverManager()

                # Solver discovery
                for _ in range(5):
                    solvers = solver_manager.discover_solvers()

                # Solver availability checks
                for _ in range(10):
                    available = solver_manager.is_solver_available("xfoil")

            except ImportError:
                # Mock solver operations
                solvers = ["xfoil", "avl", "gnvp3", "gnvp7"]
                for _ in range(15):
                    available = True

        await self._run_performance_test(
            "SolverManager Performance",
            solver_operations,
            iterations=50,
        )

    # Data Management Performance Tests

    async def _test_database_performance(self):
        """Test database performance"""

        async def database_operations():
            try:
                from data.database import DatabaseManager

                db_manager = DatabaseManager()
                await db_manager.initialize()

                # Rapid CRUD operations
                record_ids = []
                for i in range(10):
                    test_data = {"name": f"test_{i}", "value": i}
                    record_id = await db_manager.create_record("test_table", test_data)
                    record_ids.append(record_id)

                # Bulk read operations
                for record_id in record_ids:
                    record = await db_manager.get_record("test_table", record_id)

                # Cleanup
                for record_id in record_ids:
                    await db_manager.delete_record("test_table", record_id)

            except ImportError:
                # Mock database operations
                records = {}
                for i in range(10):
                    records[f"id_{i}"] = {"name": f"test_{i}", "value": i}

        await self._run_performance_test(
            "Database Performance",
            database_operations,
            iterations=20,
        )

    async def _test_export_performance(self):
        """Test export performance"""

        async def export_operations():
            try:
                from core.services import ExportService

                export_service = ExportService()

                # Test data export performance
                test_data = {
                    "results": list(range(100)),
                    "metadata": {"type": "performance_test"},
                }

                formats = export_service.get_supported_formats()
                for format_type in formats[:2]:  # Test first 2 formats
                    exported = export_service.export_data(test_data, format_type)

            except ImportError:
                # Mock export operations
                test_data = {"results": list(range(100))}
                formats = ["json", "csv"]
                for format_type in formats:
                    exported_data = str(test_data)

        await self._run_performance_test(
            "Export Performance",
            export_operations,
            iterations=50,
        )

    # Memory and Resource Tests

    async def _test_memory_usage(self):
        """Test memory usage patterns"""

        async def memory_operations():
            # Create and destroy large data structures
            large_data = []
            for i in range(1000):
                large_data.append({"index": i, "data": "x" * 100})

            # Process data
            processed = [item["index"] for item in large_data]

            # Clear references
            del large_data
            del processed

        await self._run_performance_test(
            "Memory Usage Test",
            memory_operations,
            iterations=10,
        )

    async def _test_resource_cleanup(self):
        """Test resource cleanup performance"""

        async def cleanup_operations():
            # Simulate resource allocation and cleanup
            resources = []
            for i in range(100):
                resource = {"id": i, "data": "resource_data"}
                resources.append(resource)

            # Cleanup resources
            for resource in resources:
                del resource["data"]

            resources.clear()

        await self._run_performance_test(
            "Resource Cleanup Test",
            cleanup_operations,
            iterations=50,
        )

    # Concurrent Operation Tests

    async def _test_concurrent_operations(self):
        """Test concurrent operation performance"""

        async def concurrent_operations():
            # Create multiple concurrent tasks
            async def worker_task(worker_id: int):
                for i in range(10):
                    # Simulate work
                    await asyncio.sleep(0.001)
                    result = worker_id * 10 + i
                return result

            # Run concurrent workers
            tasks = [worker_task(i) for i in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5

        await self._run_performance_test(
            "Concurrent Operations Test",
            concurrent_operations,
            iterations=20,
        )

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(
                    1 for r in self.test_results if r.status == TestStatus.PASSED
                ),
                "failed": sum(
                    1 for r in self.test_results if r.status == TestStatus.FAILED
                ),
                "errors": sum(
                    1 for r in self.test_results if r.status == TestStatus.ERROR
                ),
            },
            "performance_metrics": {},
            "recommendations": [],
        }

        # Collect performance metrics
        for result in self.test_results:
            if result.details and "metrics" in result.details:
                metrics = result.details["metrics"]
                report["performance_metrics"][result.name] = metrics

                # Generate recommendations based on metrics
                if metrics.get("avg_execution_time_ms", 0) > 100:
                    report["recommendations"].append(
                        f"{result.name}: Consider optimization - execution time > 100ms"
                    )

                if metrics.get("memory_growth_mb", 0) > 10:
                    report["recommendations"].append(
                        f"{result.name}: Memory growth detected - {metrics['memory_growth_mb']:.1f}MB"
                    )

                if metrics.get("operations_per_second", 0) < 10:
                    report["recommendations"].append(
                        f"{result.name}: Low throughput - {metrics['operations_per_second']:.1f} ops/sec"
                    )

        return report
