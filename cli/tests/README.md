# ICARUS CLI Testing Guide

This guide describes how to run and contribute to the ICARUS CLI test suite using [pytest](https://docs.pytest.org/).

## Test Categories

- **Unit Tests**: Test individual functions or classes in isolation. Located in `cli/tests/unit/`.
- **Integration Tests**: Test interactions between components. Located in `cli/tests/integration/`.
- **Functional/End-to-End Tests**: Test complete workflows and user scenarios. Located in `cli/tests/functional/`.
- **Performance Tests**: Benchmark performance and identify bottlenecks. Located in `cli/tests/performance/`.

## Running Tests

### Run All Tests
```bash
pytest cli/tests
```

### Run a Specific Category
```bash
# Unit tests
pytest cli/tests/unit

# Integration tests
pytest cli/tests/integration
```

### Run a Specific Test File
```bash
pytest cli/tests/unit/test_config.py
```

### Additional Options
- `-v` for verbose output
- `-k <expression>` to run tests matching a substring

## Writing Tests
- Name test files as `test_*.py` and test functions as `test_*` for pytest discovery.
- Use fixtures for setup/teardown.
- Keep tests small and focused.
- Add docstrings to explain test purpose.

## Contributing
- Place new tests in the appropriate category folder.
- Ensure tests are easy to read and maintain.
- Remove or refactor legacy tests that do not follow these guidelines.
# Run in CI/CD mode with fail-fast and reporting
python -m cli.testing.test_runner --ci --fail-fast
```

## Test Types

### Unit Tests (`unit_tests.py`)

Test individual components in isolation using mocks and stubs.

**Coverage:**
- Configuration Manager
- Event System
- State Manager
- Screen Manager
- Theme System
- Analysis Service
- Data Management
- Plugin System
- Collaboration System

**Example:**
```python
# Test configuration manager
await test_suite._test_config_manager()
```

### Integration Tests (`integration_tests.py`)

Test component interactions and ICARUS module connections.

**Coverage:**
- XFoil Integration
- AVL Integration
- GNVP Integration
- Analysis Workflow Integration
- Data Visualization Integration
- API Layer Integration
- WebSocket Integration
- Plugin System Integration

**Example:**
```python
# Test XFoil solver integration
await test_suite._test_xfoil_integration()
```

### End-to-End Tests (`e2e_tests.py`)

Test complete workflows and user scenarios.

**Coverage:**
- Complete Airfoil Analysis Workflow
- Aircraft Design Workflow
- Optimization Workflow
- New User Onboarding
- Collaborative Analysis
- Multi-Solver Comparison

**Example:**
```python
# Test complete airfoil analysis workflow
await test_suite._test_airfoil_analysis_workflow()
```

### Performance Tests (`performance_tests.py`)

Benchmark performance and identify bottlenecks.

**Coverage:**
- Component Performance (Config, Events, State)
- UI Performance (Screens, Themes)
- Analysis Performance
- Database Performance
- Memory Usage Analysis
- Concurrent Operations

**Metrics Collected:**
- Execution time
- Memory usage
- CPU usage
- Operations per second
- Memory growth

## Test Configuration

### Configuration File (`test_config.py`)

The testing framework uses a comprehensive configuration system:

```python
from cli.testing import TEST_CONFIG

# Modify test settings
TEST_CONFIG.default_timeout = 60.0
TEST_CONFIG.performance_iterations = 200
```

### Environment Variables

```bash
# Enable CI mode
export CI=true

# Set test output directory
export TEST_OUTPUT_DIR=/path/to/reports

# Enable verbose logging
export TEST_VERBOSE=true
```

## Test Utilities

### Test Fixtures (`test_utils.py`)

Common test data and fixtures:

```python
from cli.testing import TestFixtures

# Get sample airfoil data
airfoil_data = TestFixtures.get_sample_airfoil_data()

# Get sample aircraft configuration
aircraft_config = TestFixtures.get_sample_aircraft_config()
```

### Mock Components

```python
from cli.testing import MockComponents

# Create mock application
app = MockComponents.MockApp()

# Create mock database
db = MockComponents.MockDatabase()
```

### Test Assertions

```python
from cli.testing import TestAssertions

# Assert analysis result is valid
TestAssertions.assert_analysis_result_valid(result)

# Assert performance is acceptable
TestAssertions.assert_performance_acceptable(metrics, thresholds)
```

## Test Reports

The framework generates comprehensive test reports:

### HTML Report
- Visual test results with charts
- Performance metrics visualization
- Error analysis and recommendations
- Located in `cli/testing/reports/`

### JSON Report
- Machine-readable test results
- Detailed metrics and timing data
- Integration with monitoring systems

### JUnit XML
- CI/CD compatible format
- Integration with Jenkins, GitHub Actions, etc.
- Test result visualization in CI systems

### Performance Report
- Detailed performance metrics
- Trend analysis and recommendations
- Memory usage and CPU profiling

## Writing Tests

### Unit Test Example

```python
async def _test_my_component(self):
    """Test my component functionality"""

    async def test_basic_operations():
        from my_module import MyComponent

        component = MyComponent()

        # Test initialization
        await component.initialize()
        assert component.is_initialized

        # Test operations
        result = await component.do_something()
        assert result is not None

        # Test cleanup
        await component.cleanup()

    await self._run_test("My Component Basic Operations", test_basic_operations)
```

### Integration Test Example

```python
async def _test_component_integration(self):
    """Test component integration"""

    async def test_integration():
        from component_a import ComponentA
        from component_b import ComponentB

        comp_a = ComponentA()
        comp_b = ComponentB()

        # Test interaction
        result = await comp_a.interact_with(comp_b)
        assert result.success

        # Verify state changes
        assert comp_a.state == "updated"
        assert comp_b.state == "processed"

    await self._run_test("Component Integration", test_integration)
```

### Performance Test Example

```python
async def _test_component_performance(self):
    """Test component performance"""

    async def performance_operations():
        from my_module import MyComponent

        component = MyComponent()

        # Perform operations to benchmark
        for i in range(100):
            await component.fast_operation()

    await self._run_performance_test(
        "Component Performance",
        performance_operations,
        iterations=1000
    )
```

## Best Practices

### Test Organization
- Group related tests in the same test method
- Use descriptive test names
- Include docstrings explaining test purpose
- Tag tests with appropriate categories

### Test Data
- Use fixtures for common test data
- Create realistic test scenarios
- Clean up test data after tests
- Use temporary directories for file operations

### Mocking
- Mock external dependencies
- Use consistent mock configurations
- Verify mock interactions
- Reset mocks between tests

### Performance Testing
- Use appropriate iteration counts
- Include warmup iterations
- Monitor memory usage
- Set realistic performance thresholds

### Error Handling
- Test both success and failure cases
- Verify error messages and types
- Test error recovery mechanisms
- Include edge cases and boundary conditions

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure CLI directory is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Test Timeouts**
```python
# Increase timeout for slow tests
TEST_CONFIG.default_timeout = 120.0
```

**Memory Issues**
```python
# Reduce test data size
TEST_CONFIG.sample_data_size = 50
```

**Mock Failures**
```python
# Reset mocks between tests
mock_component.reset_mock()
```

### Debug Mode

```bash
# Run with debug output
python -m cli.testing.test_runner --verbose --type unit

# Run specific test
python -c "
import asyncio
from cli.testing import UnitTestSuite
suite = UnitTestSuite()
asyncio.run(suite._test_config_manager())
"
```

### Performance Debugging

```bash
# Run performance tests with detailed metrics
python -m cli.testing.test_runner --type performance --verbose

# Generate performance report
python -c "
import asyncio
from cli.testing import PerformanceTestSuite
suite = PerformanceTestSuite()
asyncio.run(suite.run_all_tests())
report = suite.generate_performance_report()
print(report)
"
```

## Contributing

### Adding New Tests

1. Choose appropriate test type (unit/integration/e2e/performance)
2. Add test method to relevant test suite
3. Use existing patterns and utilities
4. Include proper assertions and error handling
5. Update documentation

### Test Guidelines

- Follow existing naming conventions
- Include comprehensive docstrings
- Use appropriate test categories and tags
- Ensure tests are deterministic
- Clean up resources after tests

### Performance Considerations

- Keep unit tests fast (< 100ms each)
- Use mocks to avoid external dependencies
- Monitor test execution time
- Optimize slow tests or move to integration suite

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: ICARUS CLI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r cli/requirements.txt
    - name: Run tests
      run: python -m cli.testing.test_runner --ci
    - name: Upload test reports
      uses: actions/upload-artifact@v2
      with:
        name: test-reports
        path: cli/testing/reports/
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh 'python -m cli.testing.test_runner --ci'
            }
            post {
                always {
                    publishTestResults testResultsPattern: 'cli/testing/reports/junit_*.xml'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'cli/testing/reports',
                        reportFiles: '*.html',
                        reportName: 'Test Report'
                    ])
                }
            }
        }
    }
}
```

## License

This testing framework is part of the ICARUS CLI project and follows the same license terms.
