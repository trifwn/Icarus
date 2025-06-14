#!/usr/bin/env python3
"""
Test runner script for the ICARUS simulation framework.

This script runs all simulation tests and generates reports.
Use this to validate the simulation framework functionality.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

this_file = Path(__file__).resolve()
basic_simultion_file = this_file.parent / "unit" / "computation" / "test_simulation_basic.py"
performance_simulation_file = this_file.parent / "unit" / "computation" / "test_simulation_performance.py"


def run_basic_tests():
    """Run basic functionality tests."""
    print("🧪 Running basic functionality tests...")
    cmd = ["python", "-m", "pytest", basic_simultion_file, "-v", "--tb=short"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Basic tests passed!")
    else:
        print("❌ Basic tests failed!")
        print(result.stdout)
        print(result.stderr)

    return result.returncode == 0


def run_performance_tests():
    """Run performance benchmark tests."""
    print("⚡ Running performance benchmarks...")
    cmd = [
        "python",
        "-m",
        "pytest",
        performance_simulation_file,
        "-v",
        "--tb=short",
        "-m",
        "performance",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Performance tests passed!")
    else:
        print("❌ Performance tests failed!")
        print(result.stdout)
        print(result.stderr)

    return result.returncode == 0


def run_stress_tests():
    """Run stress tests."""
    print("🔥 Running stress tests...")
    cmd = [
        "python",
        "-m",
        "pytest",
        performance_simulation_file,
        "-v",
        "--tb=short",
        "-m",
        "stress",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Stress tests passed!")
    else:
        print("❌ Stress tests failed!")
        print(result.stdout)
        print(result.stderr)

    return result.returncode == 0


def run_integration_tests():
    """Run integration tests."""
    print("🔗 Running integration tests...")
    cmd = ["python", "-m", "pytest", f"{basic_simultion_file}::TestIntegration", "-v", "--tb=short"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Integration tests passed!")
    else:
        print("❌ Integration tests failed!")
        print(result.stdout)
        print(result.stderr)

    return result.returncode == 0


def generate_coverage_report():
    """Generate test coverage report."""
    print("📊 Generating coverage report...")
    # Install coverage if not available
    try:
        import importlib.util

        spec = importlib.util.find_spec("coverage")
        if spec is None:
            print("Installing coverage...")
            subprocess.run([sys.executable, "-m", "pip", "install", "coverage"])
    except ImportError:
        print("Installing coverage...")
        subprocess.run([sys.executable, "-m", "pip", "install", "coverage"])

    # Run tests with coverage
    cmd = ["python", "-m", "coverage", "run", "--source=ICARUS.computation", "-m", "pytest", "test_simulation_basic.py"]
    subprocess.run(cmd, capture_output=True)

    # Generate report
    cmd = ["python", "-m", "coverage", "report", "-m"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("Coverage Report:")
    print(result.stdout)

    # Generate HTML report
    cmd = ["python", "-m", "coverage", "html"]
    subprocess.run(cmd, capture_output=True)
    print("📝 HTML coverage report generated in htmlcov/")


def validate_configuration():
    """Validate configuration loading."""
    print("⚙️  Validating configuration...")

    try:
        from ICARUS.computation.core import SimulationConfig

        # Test default config
        default_config = SimulationConfig()
        print(f"✅ Default config created: {default_config.execution_mode}")

        # Test config file loading (if file exists)
        config_file = Path("config/simulation_config.yaml")
        if config_file.exists():
            try:
                file_config = SimulationConfig.from_file(config_file)
                print(f"✅ Config loaded from file: {file_config.execution_mode}")
            except Exception as e:
                print(f"❌ Config file loading failed: {e}")
                return False

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="ICARUS Simulation Framework Test Runner")
    parser.add_argument("--basic", action="store_true", help="Run basic tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--stress", action="store_true", help="Run stress tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")

    args = parser.parse_args()

    # Default to running all tests if no specific test type is specified
    if not any([args.basic, args.performance, args.stress, args.integration, args.coverage]):
        args.all = True

    start_time = time.time()
    results = []

    print("🚀 ICARUS Simulation Framework Test Suite")
    print("=" * 50)

    # Configuration validation
    if not validate_configuration():
        print("❌ Configuration validation failed, aborting tests")
        sys.exit(1)

    # Run selected tests
    if args.all or args.basic:
        results.append(("Basic Tests", run_basic_tests()))

    if args.all or args.integration:
        results.append(("Integration Tests", run_integration_tests()))

    if args.all or args.performance:
        results.append(("Performance Tests", run_performance_tests()))

    if args.all or args.stress:
        results.append(("Stress Tests", run_stress_tests()))

    # Generate coverage report if requested
    if args.all or args.coverage:
        generate_coverage_report()

    # Summary
    execution_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("📋 Test Summary:")

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")

    if all_passed:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
