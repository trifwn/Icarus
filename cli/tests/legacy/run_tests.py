#!/usr/bin/env python3
"""
Simple test execution script for ICARUS CLI

This script provides a simple way to run tests without complex command-line arguments.
"""

import asyncio
import sys
from pathlib import Path

# Add CLI directory to path
cli_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cli_dir))

from tests.test_runner import IcarusTestRunner


async def main():
    """Run tests with simple configuration"""
    print("🚀 ICARUS CLI Test Execution")
    print("=" * 50)

    # Create test runner
    runner = IcarusTestRunner()

    # Validate environment
    print("🔍 Validating test environment...")
    if not await runner.validate_test_environment():
        print("❌ Environment validation failed")
        return 1

    print("✅ Environment validation passed")
    print()

    # Run tests
    try:
        print("🎯 Running test suites...")
        results = await runner.run_tests(
            test_types=[
                "unit",
                "integration",
            ],  # Skip E2E and performance for quick run
            verbose=True,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {results['total']}")
        print(f"✅ Passed: {results['passed']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"💥 Errors: {results['errors']}")
        print(f"⏭️  Skipped: {results['skipped']}")
        print(f"⏱️  Duration: {results['duration']:.2f}s")

        success_rate = (results["passed"] / max(results["total"], 1)) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")

        if results["failed"] == 0 and results["errors"] == 0:
            print("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print("\n⚠️  SOME TESTS FAILED")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted")
        return 130
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
