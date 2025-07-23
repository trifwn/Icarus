# This file has been moved to utils/
# The following code is a placeholder to indicate the file's previous location.
# Please refer to the new location for the actual implementation.

# Original code would be here if it were not deleted.

# The file has been successfully moved to the utils directory.

# If you are looking for the functionality that was in this file,
# please check the utils directory for the new implementation.

# Thank you for your understanding.

# End of file.
                "wing": {
                    "span": 10.0,
                    "chord_root": 1.5,
                    "chord_tip": 0.8,
                    "airfoil_root": "naca0012",
                    "airfoil_tip": "naca0012",
                    "sweep": 0.0,
                    "dihedral": 2.0,
                },
                "fuselage": {
                    "length": 8.0,
                    "diameter": 1.2,
                    "fineness_ratio": 6.67,
                },
                "empennage": {
                    "horizontal_tail": {
                        "span": 3.0,
                        "chord": 0.8,
                        "airfoil": "naca0009",
                    },
                    "vertical_tail": {
                        "span": 2.0,
                        "chord": 1.0,
                        "airfoil": "naca0009",
                    },
                },
            },
            "mass_properties": {
                "empty_weight": 800.0,
                "max_takeoff_weight": 1200.0,
                "cg_location": [4.0, 0.0, 0.0],
            },
        }

    @staticmethod
    def get_sample_analysis_results() -> Dict[str, Any]:
        """Get sample analysis results"""
        return {
            "analysis_type": "airfoil_polar",
            "airfoil": "naca0012",
            "conditions": {
                "reynolds": 1000000,
                "mach": 0.1,
                "temperature": 288.15,
                "pressure": 101325,
            },
            "results": {
                "alpha": list(range(-5, 11)),
                "cl": [i * 0.1 for i in range(-5, 11)],
                "cd": [0.007 + abs(i) * 0.0005 for i in range(-5, 11)],
                "cm": [-0.05 - i * 0.005 for i in range(-5, 11)],
            },
            "metadata": {
                "solver": "xfoil",
                "convergence": "good",
                "iterations": 50,
                "timestamp": "2025-01-01T12:00:00Z",
            },
        }

    @staticmethod
    def get_sample_workflow_config() -> Dict[str, Any]:
        """Get sample workflow configuration"""
        return {
            "name": "Airfoil Analysis Workflow",
            "description": "Complete airfoil analysis with visualization",
            "steps": [
                {
                    "id": "load_airfoil",
                    "name": "Load Airfoil",
                    "type": "data_load",
                    "config": {"file": "naca0012.dat"},
                },
                {
                    "id": "run_analysis",
                    "name": "Run XFoil Analysis",
                    "type": "analysis",
                    "config": {
                        "solver": "xfoil",
                        "reynolds": 1000000,
                        "alpha_range": [-5, 10, 1],
                    },
                    "dependencies": ["load_airfoil"],
                },
                {
                    "id": "generate_plots",
                    "name": "Generate Plots",
                    "type": "visualization",
                    "config": {"plot_types": ["polar", "pressure_distribution"]},
                    "dependencies": ["run_analysis"],
                },
                {
                    "id": "export_results",
                    "name": "Export Results",
                    "type": "export",
                    "config": {"formats": ["json", "csv"]},
                    "dependencies": ["run_analysis", "generate_plots"],
                },
            ],
        }


class MockComponents:
    """Mock implementations of ICARUS CLI components for testing"""

    class MockApp:
        """Mock application for testing"""

        def __init__(self):
            self.screens = {}
            self.event_system = MockComponents.MockEventSystem()
            self.log = MockComponents.MockLogger()
            self.state = {}

        def install_screen(self, screen, name):
            self.screens[name] = screen

        async def push_screen(self, name):
            pass

        async def pop_screen(self):
            pass

    class MockEventSystem:
        """Mock event system for testing"""

        def __init__(self):
            self.subscribers = {}
            self.events = []

        def subscribe(self, event_type, callback):
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)

        def emit_sync(self, event_type, data):
            self.events.append({"type": event_type, "data": data})
            if event_type in self.subscribers:
                for callback in self.subscribers[event_type]:
                    callback(data)

        async def emit(self, event_type, data):
            self.events.append({"type": event_type, "data": data})
            if event_type in self.subscribers:
                for callback in self.subscribers[event_type]:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)

    class MockLogger:
        """Mock logger for testing"""

        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(("INFO", message))

        def warning(self, message):
            self.messages.append(("WARNING", message))

        def error(self, message):
            self.messages.append(("ERROR", message))

        def debug(self, message):
            self.messages.append(("DEBUG", message))

    class MockDatabase:
        """Mock database for testing"""

        def __init__(self):
            self.data = {}
            self.next_id = 1

        async def initialize(self):
            pass

        async def create_record(self, table, data):
            record_id = str(self.next_id)
            self.next_id += 1

            if table not in self.data:
                self.data[table] = {}

            self.data[table][record_id] = data.copy()
            return record_id

        async def get_record(self, table, record_id):
            if table in self.data and record_id in self.data[table]:
                return self.data[table][record_id].copy()
            return None

        async def update_record(self, table, record_id, data):
            if table in self.data and record_id in self.data[table]:
                self.data[table][record_id].update(data)
                return True
            return False

        async def delete_record(self, table, record_id):
            if table in self.data and record_id in self.data[table]:
                del self.data[table][record_id]
                return True
            return False

        async def query_records(self, table, conditions=None):
            if table not in self.data:
                return []

            records = []
            for record_id, record_data in self.data[table].items():
                if conditions is None:
                    records.append({"id": record_id, **record_data})
                else:
                    # Simple condition matching
                    match = True
                    for key, value in conditions.items():
                        if key not in record_data or record_data[key] != value:
                            match = False
                            break
                    if match:
                        records.append({"id": record_id, **record_data})

            return records

    class MockSolver:
        """Mock solver for testing"""

        def __init__(self, name: str):
            self.name = name
            self.available = True

        async def run_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
            # Simulate analysis execution
            await asyncio.sleep(0.01)  # Simulate processing time

            # Generate mock results based on solver type
            if self.name == "xfoil":
                return {
                    "status": "completed",
                    "solver": self.name,
                    "results": TestFixtures.get_sample_analysis_results()["results"],
                    "convergence": "good",
                }
            elif self.name == "avl":
                return {
                    "status": "completed",
                    "solver": self.name,
                    "stability_derivatives": {
                        "cma": -0.15,
                        "cmq": -8.5,
                        "cla": 5.2,
                    },
                    "trim_conditions": {
                        "alpha": 3.2,
                        "elevator": -1.8,
                    },
                }
            else:
                return {
                    "status": "completed",
                    "solver": self.name,
                    "results": {"message": f"Mock results from {self.name}"},
                }

        def is_available(self) -> bool:
            return self.available

        def get_info(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "version": "1.0.0",
                "available": self.available,
                "capabilities": ["analysis", "optimization"],
            }


class TestEnvironment:
    """Test environment manager"""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir
        self.created_files = []
        self.created_dirs = []

    @asynccontextmanager
    async def temporary_workspace(self):
        """Create a temporary workspace for testing"""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="icarus_test_"))

        try:
            # Create standard directory structure
            directories = [
                "airfoils",
                "aircraft",
                "results",
                "workflows",
                "exports",
                "config",
            ]

            for directory in directories:
                dir_path = self.temp_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.created_dirs.append(dir_path)

            # Create sample files
            await self._create_sample_files()

            yield self.temp_dir

        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

    async def _create_sample_files(self):
        """Create sample files for testing"""
        # Create sample airfoil file
        airfoil_file = self.temp_dir / "airfoils" / "naca0012.dat"
        with open(airfoil_file, "w") as f:
            f.write(TestFixtures.get_sample_airfoil_data())
        self.created_files.append(airfoil_file)

        # Create sample aircraft configuration
        aircraft_file = self.temp_dir / "aircraft" / "test_aircraft.json"
        with open(aircraft_file, "w") as f:
            json.dump(TestFixtures.get_sample_aircraft_config(), f, indent=2)
        self.created_files.append(aircraft_file)

        # Create sample workflow
        workflow_file = self.temp_dir / "workflows" / "sample_workflow.json"
        with open(workflow_file, "w") as f:
            json.dump(TestFixtures.get_sample_workflow_config(), f, indent=2)
        self.created_files.append(workflow_file)


class TestAssertions:
    """Custom assertions for ICARUS CLI testing"""

    @staticmethod
    def assert_analysis_result_valid(result: Dict[str, Any]):
        """Assert that an analysis result is valid"""
        assert "status" in result, "Analysis result should have status"
        assert "results" in result, "Analysis result should have results"

        if result["status"] == "completed":
            assert (
                result["results"] is not None
            ), "Completed analysis should have results"

    @staticmethod
    def assert_airfoil_data_valid(data: Dict[str, Any]):
        """Assert that airfoil data is valid"""
        required_fields = ["alpha", "cl", "cd"]
        for field in required_fields:
            assert field in data, f"Airfoil data should contain {field}"
            assert isinstance(data[field], list), f"{field} should be a list"
            assert len(data[field]) > 0, f"{field} should not be empty"

    @staticmethod
    def assert_workflow_valid(workflow: Dict[str, Any]):
        """Assert that a workflow configuration is valid"""
        assert "name" in workflow, "Workflow should have a name"
        assert "steps" in workflow, "Workflow should have steps"
        assert isinstance(workflow["steps"], list), "Steps should be a list"
        assert len(workflow["steps"]) > 0, "Workflow should have at least one step"

        for step in workflow["steps"]:
            assert "id" in step, "Step should have an id"
            assert "name" in step, "Step should have a name"
            assert "type" in step, "Step should have a type"

    @staticmethod
    def assert_performance_acceptable(
        metrics: Dict[str, Any],
        thresholds: Dict[str, float],
    ):
        """Assert that performance metrics meet acceptable thresholds"""
        for metric, threshold in thresholds.items():
            if metric in metrics:
                actual_value = metrics[metric]
                assert (
                    actual_value <= threshold
                ), f"{metric} ({actual_value}) exceeds threshold ({threshold})"

    @staticmethod
    def assert_memory_usage_reasonable(
        initial_mb: float,
        final_mb: float,
        max_growth_mb: float = 50.0,
    ):
        """Assert that memory usage growth is reasonable"""
        growth = final_mb - initial_mb
        assert (
            growth <= max_growth_mb
        ), f"Memory growth ({growth:.1f}MB) exceeds limit ({max_growth_mb}MB)"


class TestDataGenerator:
    """Generate test data for various scenarios"""

    @staticmethod
    def generate_polar_data(alpha_range: List[float]) -> Dict[str, List[float]]:
        """Generate realistic polar data for given alpha range"""

        cl_data = []
        cd_data = []
        cm_data = []

        for alpha in alpha_range:
            # Simple linear lift curve with stall
            if abs(alpha) < 15:
                cl = alpha * 0.1
            else:
                cl = 15 * 0.1 * (1 - (abs(alpha) - 15) * 0.05)  # Stall region

            # Parabolic drag polar
            cd = 0.007 + (cl**2) * 0.05

            # Moment coefficient
            cm = -0.05 - alpha * 0.005

            cl_data.append(cl)
            cd_data.append(cd)
            cm_data.append(cm)

        return {
            "alpha": alpha_range,
            "cl": cl_data,
            "cd": cd_data,
            "cm": cm_data,
        }

    @staticmethod
    def generate_aircraft_derivatives() -> Dict[str, float]:
        """Generate realistic stability derivatives"""
        return {
            "cla": 5.2,  # Lift curve slope
            "cma": -0.15,  # Pitching moment derivative
            "cmq": -8.5,  # Pitch damping
            "cna": 0.8,  # Side force derivative
            "cnb": 0.12,  # Yaw stability
            "cnr": -0.25,  # Yaw damping
            "clb": -0.08,  # Dihedral effect
            "clp": -0.45,  # Roll damping
            "clr": 0.15,  # Roll due to yaw rate
        }

    @staticmethod
    def generate_test_matrix(parameters: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate test matrix from parameter ranges"""
        import itertools

        keys = list(parameters.keys())
        values = list(parameters.values())

        test_cases = []
        for combination in itertools.product(*values):
            test_case = dict(zip(keys, combination))
            test_cases.append(test_case)

        return test_cases


# Convenience functions for common test operations


async def run_with_timeout(coro, timeout_seconds: float = 30.0):
    """Run a coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise AssertionError(f"Operation timed out after {timeout_seconds} seconds")


def create_mock_analysis_service():
    """Create a mock analysis service for testing"""
    service = Mock()
    service.get_available_modules = Mock(return_value=["xfoil", "avl", "gnvp"])
    service.validate_parameters = Mock(return_value={"valid": True})
    service.run_analysis = AsyncMock(
        return_value=TestFixtures.get_sample_analysis_results(),
    )
    return service


def create_mock_solver_manager():
    """Create a mock solver manager for testing"""
    manager = Mock()
    manager.discover_solvers = Mock(return_value=["xfoil", "avl", "gnvp3", "gnvp7"])
    manager.is_solver_available = Mock(return_value=True)
    manager.get_solver_info = Mock(return_value={"name": "xfoil", "version": "6.99"})
    return manager
