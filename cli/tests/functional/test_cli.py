#!/usr/bin/env python3
"""
Test script for ICARUS CLI Demo functionality

This script tests the essential CLI functionality without requiring the TUI.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_airfoil_analysis():
    """Test airfoil analysis from CLI."""
    print("Testing Airfoil Analysis CLI...")

    from argparse import Namespace

    from cli.icarus_demo import run_cli_analysis

    # Create mock args
    args = Namespace(
        cli="airfoil",
        target="NACA2412",
        reynolds=1000000,
        min_aoa=-5,
        max_aoa=10,
        aoa_step=1.0,
        output=None,
        verbose=True,
    )

    # Run analysis
    result = run_cli_analysis(args)

    if result == 0:
        print("✓ Airfoil analysis CLI test passed")
    else:
        print("✗ Airfoil analysis CLI test failed")

    return result


def test_airplane_analysis():
    """Test airplane analysis from CLI."""
    print("\nTesting Airplane Analysis CLI...")

    # For airplane analysis, we'll use the workflow directly since it has mock data
    try:
        import asyncio

        from cli.workflows.airplane_workflow import analyze_demo_airplane
        from cli.workflows.airplane_workflow import print_airplane_summary

        # Run analysis
        results = asyncio.run(
            analyze_demo_airplane(
                velocity=50.0,
                altitude=1000.0,
                angle_range=(-5, 15, 1.0),
            ),
        )

        if results.get("success"):
            print("Analysis completed successfully")
            print_airplane_summary(results)
            print("✓ Airplane analysis workflow test passed")
            return 0
        else:
            print(f"Analysis failed: {results.get('error')}")
            print("✗ Airplane analysis workflow test failed")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        print("✗ Airplane analysis workflow test failed")
        return 1


def main():
    """Run all tests."""
    print("ICARUS CLI Demo - CLI Functionality Test")
    print("=" * 50)

    airfoil_result = test_airfoil_analysis()
    airplane_result = test_airplane_analysis()

    print("\n" + "=" * 50)
    if airfoil_result == 0 and airplane_result == 0:
        print("✓ All CLI tests passed!")
        return 0
    else:
        print("✗ Some CLI tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
