#!/usr/bin/env python3
"""
Demonstration of ICARUS Integration Layer

This script demonstrates how to use the ICARUS integration layer
to perform aerodynamic analyses.
"""

import asyncio
import sys
from pathlib import Path

# Add CLI to path
cli_path = Path(__file__).parent.parent
if str(cli_path) not in sys.path:
    sys.path.insert(0, str(cli_path))

from integration import AnalysisConfig
from integration import AnalysisService
from integration import AnalysisType
from integration import SolverType


async def demo_airfoil_analysis():
    """Demonstrate airfoil polar analysis."""
    print("🛩️  ICARUS Integration Layer Demo")
    print("=" * 50)

    # Initialize the analysis service
    service = AnalysisService()

    # Show system status
    print("\n📊 System Status:")
    status = service.get_system_status()
    print(f"  ICARUS Available: {status['icarus_available']}")
    print(f"  Available Solvers: {status['solver_status']['available_solvers']}")
    print(f"  Service Status: {status['service_status']}")

    # Show available solvers
    print("\n🔧 Available Solvers:")
    solvers = service.get_available_solvers()
    for solver in solvers[:3]:  # Show first 3
        print(
            f"  - {solver['name']} ({solver['type']}) - Fidelity: {solver['fidelity']}",
        )
    if len(solvers) > 3:
        print(f"  ... and {len(solvers) - 3} more")

    # Get recommended solver for airfoil analysis
    recommended = service.get_recommended_solver(AnalysisType.AIRFOIL_POLAR)
    print(f"\n💡 Recommended solver for airfoil analysis: {recommended['name']}")

    # Create analysis configuration
    config = AnalysisConfig(
        analysis_type=AnalysisType.AIRFOIL_POLAR,
        solver_type=SolverType.XFOIL,
        target="NACA0012",
        parameters={
            "reynolds": 1e6,
            "mach": 0.0,
            "min_aoa": -5,
            "max_aoa": 15,
            "aoa_step": 1.0,
        },
    )

    # Validate configuration
    print("\n✅ Validating analysis configuration...")
    validation = service.validate_analysis_config(config)
    if validation.is_valid:
        print("  Configuration is valid!")
    else:
        print("  Configuration has errors:")
        for error in validation.errors:
            print(f"    - {error.field}: {error.message}")
        return

    # Run analysis
    print("\n🚀 Running airfoil polar analysis...")
    print("  Target: NACA 0012")
    print(f"  Solver: {config.solver_type.value}")
    print(f"  Reynolds: {config.parameters['reynolds']:,.0f}")
    print(
        f"  Angle range: {config.parameters['min_aoa']}° to {config.parameters['max_aoa']}°",
    )

    def progress_callback(progress):
        print(
            f"  Progress: {progress.progress_percent:5.1f}% - {progress.current_step}",
        )

    result = await service.run_analysis(config, progress_callback)

    if result.is_successful:
        print(f"✅ Analysis completed successfully in {result.duration:.2f} seconds")

        # Process results
        print("\n📈 Processing results...")
        processed = service.process_result(result)

        # Show summary
        summary = processed.summary
        print(f"  Analysis Type: {summary['analysis_type']}")
        print(f"  Data Points: {summary['data_points']}")

        # Show key performance metrics
        if "key_results" in summary and summary["key_results"]:
            perf = summary["key_results"]
            print("\n🎯 Key Performance Metrics:")
            if "max_cl" in perf:
                print(
                    f"  Maximum CL: {perf['max_cl']['value']:.3f} at α = {perf['max_cl']['alpha']:.1f}°",
                )
            if "max_ld" in perf:
                print(
                    f"  Maximum L/D: {perf['max_ld']['value']:.1f} at α = {perf['max_ld']['alpha']:.1f}°",
                )
            if "stall_angle" in perf:
                print(f"  Stall Angle: {perf['stall_angle']:.1f}°")

        # Show available plots and tables
        print(
            f"\n📊 Generated {len(processed.plots)} plots and {len(processed.tables)} tables",
        )
        for plot in processed.plots:
            print(f"  - {plot['title']}")

        print(f"\n💾 Available export formats: {', '.join(processed.export_formats)}")

        # Demonstrate export
        try:
            export_path = service.export_result(processed, "json")
            print(f"  Exported results to: {export_path}")
        except Exception as e:
            print(f"  Export demo skipped: {e}")

    else:
        print(f"❌ Analysis failed: {result.error_message}")

    print("\n🎉 Demo completed!")


def demo_parameter_suggestions():
    """Demonstrate parameter suggestions."""
    print("\n💡 Parameter Suggestions Demo")
    print("-" * 30)

    service = AnalysisService()

    # Get suggestions for airfoil polar analysis
    suggestions = service.get_parameter_suggestions(AnalysisType.AIRFOIL_POLAR)

    print("Suggested parameters for airfoil polar analysis:")
    for key, value in suggestions.items():
        if not key.endswith("_description"):
            description = suggestions.get(f"{key}_description", "")
            print(f"  {key}: {value}")
            if description:
                print(f"    ({description})")


def demo_solver_capabilities():
    """Demonstrate solver capability queries."""
    print("\n🔍 Solver Capabilities Demo")
    print("-" * 30)

    service = AnalysisService()

    # Show solvers for different analysis types
    analysis_types = [AnalysisType.AIRFOIL_POLAR, AnalysisType.AIRPLANE_POLAR]

    for analysis_type in analysis_types:
        solvers = service.get_solvers_for_analysis(analysis_type)
        print(f"\nSolvers for {analysis_type.value}:")
        for solver in solvers:
            status = "✅" if solver["is_available"] else "❌"
            print(f"  {status} {solver['name']} (Fidelity: {solver['fidelity']})")


async def main():
    """Run all demos."""
    try:
        await demo_airfoil_analysis()
        demo_parameter_suggestions()
        demo_solver_capabilities()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
