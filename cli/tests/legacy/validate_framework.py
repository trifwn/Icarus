#!/usr/bin/env python3
"""
Validation script for ICARUS CLI Testing Framework

This script validates that the testing framework is properly structured
and all components are accessible.
"""

import sys
from pathlib import Path


def validate_framework_structure():
    """Validate the testing framework structure"""
    print("🔍 Validating ICARUS CLI Testing Framework")
    print("=" * 50)

    # Check if testing directory exists
    testing_dir = Path("cli/testing")
    if not testing_dir.exists():
        print("❌ Testing directory not found")
        return False

    print("✅ Testing directory found")

    # Check required files
    required_files = [
        "framework.py",
        "unit_tests.py",
        "integration_tests.py",
        "e2e_tests.py",
        "performance_tests.py",
        "test_runner.py",
        "test_utils.py",
        "test_config.py",
        "__init__.py",
        "README.md",
    ]

    missing_files = []
    for file_name in required_files:
        file_path = testing_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - MISSING")
            missing_files.append(file_name)

    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False

    # Check file sizes (ensure they're not empty)
    print("\n📊 File sizes:")
    for file_name in required_files:
        if file_name.endswith(".md"):
            continue
        file_path = testing_dir / file_name
        size = file_path.stat().st_size
        print(f"   {file_name}: {size} bytes")
        if size < 100:  # Very small files might be incomplete
            print(f"   ⚠️  {file_name} seems very small")

    # Try to import key components
    print("\n🔧 Testing imports:")

    # Add CLI directory to path
    cli_dir = Path("cli")
    sys.path.insert(0, str(cli_dir))

    try:
        from tests import TestFramework

        print("✅ TestFramework import successful")
    except ImportError as e:
        print(f"❌ TestFramework import failed: {e}")
        return False

    try:
        from tests import UnitTestSuite

        print("✅ UnitTestSuite import successful")
    except ImportError as e:
        print(f"❌ UnitTestSuite import failed: {e}")
        return False

    try:
        from tests import TestFixtures

        print("✅ TestFixtures import successful")
    except ImportError as e:
        print(f"❌ TestFixtures import failed: {e}")
        return False

    try:
        from tests import TEST_CONFIG

        print("✅ TEST_CONFIG import successful")
    except ImportError as e:
        print(f"❌ TEST_CONFIG import failed: {e}")
        return False

    # Check class structure
    print("\n🏗️  Testing class structure:")

    try:
        framework = TestFramework()
        assert hasattr(framework, "test_suites")
        assert hasattr(framework, "run_all_tests")
        print("✅ TestFramework structure valid")
    except Exception as e:
        print(f"❌ TestFramework structure invalid: {e}")
        return False

    try:
        unit_suite = UnitTestSuite()
        assert hasattr(unit_suite, "test_results")
        assert hasattr(unit_suite, "run_all_tests")
        print("✅ UnitTestSuite structure valid")
    except Exception as e:
        print(f"❌ UnitTestSuite structure invalid: {e}")
        return False

    # Check test fixtures
    try:
        airfoil_data = TestFixtures.get_sample_airfoil_data()
        assert len(airfoil_data) > 100
        print("✅ TestFixtures working")
    except Exception as e:
        print(f"❌ TestFixtures not working: {e}")
        return False

    print("\n🎉 Framework validation completed successfully!")
    print("\n📋 Summary:")
    print("   ✅ All required files present")
    print("   ✅ All imports working")
    print("   ✅ Class structures valid")
    print("   ✅ Test fixtures functional")

    print("\n🚀 Ready to run tests!")
    print("   Usage: python -m cli.testing.test_runner")
    print("   Quick: python cli/testing/run_tests.py")

    return True


if __name__ == "__main__":
    success = validate_framework_structure()
    sys.exit(0 if success else 1)
