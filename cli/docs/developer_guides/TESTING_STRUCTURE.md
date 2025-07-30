# ICARUS CLI Testing Structure

This document describes the testing structure for the ICARUS CLI project.

## Testing Directory Structure

```
tests/
├── cli/                  # CLI-specific tests
│   ├── analysis/         # Analysis system tests
│   ├── collaboration/    # Collaboration system tests
│   ├── framework/        # Core framework tests
│   ├── performance/      # Performance tests
│   ├── security/         # Security tests
│   └── workflow/         # Workflow system tests
├── config/               # Configuration tests
├── functional/           # Functional tests
├── TestData/             # Test data files
├── unit/                 # Unit tests
└── conftest.py           # Pytest configuration
```

## Test Types

### Unit Tests

Unit tests focus on testing individual components in isolation. These tests are located in the `tests/unit/` directory and are organized by module.

**Example:**
```python
def test_parameter_validator():
    validator = ParameterValidator()
    result = validator.validate({"alpha": 5.0})
    assert result.is_valid
    assert len(result.errors) == 0
```

### Integration Tests

Integration tests focus on testing the interaction between components. These tests are located in the `tests/cli/` directory and are organized by subsystem.

**Example:**
```python
def test_analysis_service_with_solver():
    service = AnalysisService()
    solver = MockSolver()
    service.register_solver(solver)
    result = service.run_analysis({"solver": "mock", "params": {"alpha": 5.0}})
    assert result.status == "success"
    assert "cl" in result.data
```

### Functional Tests

Functional tests focus on testing complete features from a user perspective. These tests are located in the `tests/functional/` directory.

**Example:**
```python
def test_airfoil_analysis_workflow():
    app = create_test_app()
    app.run_command("analyze airfoil NACA0012 --alpha 5.0")
    assert app.last_result.status == "success"
    assert "cl" in app.last_result.data
```

### Performance Tests

Performance tests focus on testing the performance of the system. These tests are located in the `tests/cli/performance/` directory.

**Example:**
```python
def test_analysis_performance():
    service = AnalysisService()
    start_time = time.time()
    service.run_analysis({"solver": "xfoil", "params": {"alpha": 5.0}})
    end_time = time.time()
    assert end_time - start_time < 2.0  # Analysis should complete in under 2 seconds
```

### Security Tests

Security tests focus on testing the security features of the system. These tests are located in the `tests/cli/security/` directory.

**Example:**
```python
def test_plugin_sandboxing():
    manager = PluginManager()
    plugin = create_malicious_test_plugin()
    result = manager.validate_plugin(plugin)
    assert not result.is_valid
    assert "security violation" in result.errors[0]
```

## Test Data

Test data files are located in the `tests/TestData/` directory. These files include:

- Sample airfoil data
- Sample airplane configurations
- Sample analysis results
- Mock solver implementations
- Test configuration files

## Running Tests

Tests can be run using pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/cli/framework/test_cli.py

# Run tests with specific marker
pytest -m "performance"

# Run tests with coverage report
pytest --cov=cli
```

## Test Fixtures

Common test fixtures are defined in `tests/conftest.py`. These fixtures include:

- `test_app`: Creates a test instance of the CLI application
- `mock_solver`: Creates a mock solver for testing
- `test_config`: Creates a test configuration
- `test_data_dir`: Path to test data directory

## Test Markers

The following pytest markers are used to categorize tests:

- `unit`: Unit tests
- `integration`: Integration tests
- `functional`: Functional tests
- `performance`: Performance tests
- `security`: Security tests
- `slow`: Tests that take a long time to run

## Continuous Integration

Tests are automatically run on CI/CD pipelines for:

- Pull requests
- Merges to main branch
- Release tags

The CI/CD pipeline runs:

1. Linting and code formatting checks
2. Unit tests
3. Integration tests
4. Functional tests
5. Performance tests (on scheduled runs)
6. Security tests
7. Coverage reporting
