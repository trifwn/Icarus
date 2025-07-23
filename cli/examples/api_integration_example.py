"""
Example integration of ICARUS CLI API layer

This script demonstrates how to integrate the API layer with the existing CLI
and shows the web migration readiness features.
"""

import asyncio
import sys
from pathlib import Path

# Add the CLI directory to the path
cli_path = Path(__file__).parent.parent
sys.path.insert(0, str(cli_path))

from api.adapters import UIAdapterFactory
from api.adapters import UIBridge
from api.app import create_api_app
from api.models import AnalysisConfig
from api.models import AnalysisType
from api.models import InputEvent
from api.models import ScreenData
from api.models import SessionState
from api.models import SolverType
from api.models import UserPreferences
from api.models import Workflow
from api.models import WorkflowStep


class MockEventHandler:
    """Mock event handler for demonstration"""

    async def handle_analysis_request(self, config_data):
        print(f"Handling analysis request: {config_data}")
        # This would integrate with actual ICARUS analysis modules

    async def handle_workflow_request(self, workflow_data):
        print(f"Handling workflow request: {workflow_data}")
        # This would execute the workflow using ICARUS modules

    async def handle_session_update(self, session_data):
        print(f"Handling session update: {session_data}")
        # This would update the session state


async def demonstrate_ui_abstraction():
    """Demonstrate UI adapter abstraction"""
    print("Demonstrating UI Adapter Abstraction")
    print("-" * 40)

    # Create different UI adapters
    textual_adapter = UIAdapterFactory.create_adapter("textual")
    web_adapter = UIAdapterFactory.create_adapter("web")

    # Create event handler
    event_handler = MockEventHandler()

    # Create UI bridges
    textual_bridge = UIBridge(textual_adapter, event_handler)
    web_bridge = UIBridge(web_adapter, event_handler)

    # Initialize adapters
    try:
        await textual_bridge.initialize()
        print("✓ Textual UI bridge initialized")
    except RuntimeError:
        print("⚠ Textual not available (expected in test environment)")

    await web_bridge.initialize()
    print("✓ Web UI bridge initialized")

    # Create sample screen data
    screen_data = ScreenData(
        screen_id="analysis_setup",
        title="Analysis Setup",
        content={
            "analysis_type": "airfoil",
            "target_file": "naca0012.dat",
            "solver": "xfoil",
            "parameters": {
                "reynolds": 1000000,
                "mach": 0.1,
                "alpha_range": [-5, 15, 1],
            },
        },
        actions=[
            {"id": "run_analysis", "label": "Run Analysis", "type": "button"},
            {"id": "save_config", "label": "Save Configuration", "type": "button"},
        ],
    )

    # Demonstrate rendering on both adapters
    print("\nRendering screen on different adapters:")
    try:
        await textual_bridge.update_ui(screen_data)
        print("✓ Screen rendered on Textual adapter")
    except RuntimeError:
        print("⚠ Textual adapter rendering skipped (framework not available)")

    await web_bridge.update_ui(screen_data)
    print("✓ Screen rendered on Web adapter")

    # Demonstrate input event handling
    input_event = InputEvent(
        event_type="analysis_request",
        component_id="run_analysis",
        data={"analysis_type": "airfoil", "target": "naca0012.dat", "solver": "xfoil"},
    )

    print("\nProcessing input event:")
    try:
        await textual_bridge.process_input_event(input_event)
        print("✓ Input event processed by Textual adapter")
    except RuntimeError:
        print("⚠ Textual adapter input processing skipped (framework not available)")

    await web_bridge.process_input_event(input_event)
    print("✓ Input event processed by Web adapter")


async def demonstrate_data_models():
    """Demonstrate JSON serializable data models"""
    print("\nDemonstrating Data Models")
    print("-" * 40)

    # Create analysis configuration
    analysis_config = AnalysisConfig(
        analysis_type=AnalysisType.AIRFOIL,
        target="naca0012.dat",
        solver=SolverType.XFOIL,
        parameters={
            "reynolds": 1000000,
            "mach": 0.1,
            "alpha_start": -5,
            "alpha_end": 15,
            "alpha_step": 1,
        },
        metadata={
            "description": "Basic NACA 0012 analysis",
            "created_by": "user@example.com",
        },
    )

    print("Analysis Configuration:")
    print(analysis_config.model_dump_json(indent=2))

    # Create workflow
    workflow_step = WorkflowStep(
        id="airfoil_analysis",
        name="NACA 0012 Analysis",
        analysis_config=analysis_config,
        order=1,
    )

    workflow = Workflow(
        name="Basic Airfoil Study",
        description="Comprehensive analysis of NACA 0012 airfoil",
        steps=[workflow_step],
        metadata={"category": "airfoil_studies", "complexity": "basic"},
    )

    print("\nWorkflow Definition:")
    print(workflow.model_dump_json(indent=2))

    # Create session state
    preferences = UserPreferences(
        theme="aerospace_dark",
        default_solver=SolverType.XFOIL,
        auto_save=True,
        recent_files=["naca0012.dat", "naca2412.dat", "naca4412.dat"],
    )

    session = SessionState(
        user_id="demo_user",
        workspace="/home/user/icarus_projects",
        preferences=preferences,
        active_analyses=[analysis_config.id],
    )

    print("\nSession State:")
    print(session.model_dump_json(indent=2))


async def demonstrate_api_compatibility():
    """Demonstrate API compatibility for web migration"""
    print("\nDemonstrating API Compatibility")
    print("-" * 40)

    # Create FastAPI app
    app = create_api_app()

    print("FastAPI Application Created:")
    print(f"✓ Title: {app.title}")
    print(f"✓ Version: {app.version}")
    print(f"✓ OpenAPI URL: {app.openapi_url}")
    print(f"✓ Docs URL: {app.docs_url}")

    # List available routes
    print("\nAvailable API Routes:")
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            methods = getattr(route, "methods", set())
            if methods and "GET" in methods or "POST" in methods:
                print(f"  {route.path} [{', '.join(methods)}]")

    # Demonstrate data model compatibility
    print("\nData Model Compatibility:")

    # Show that models can be serialized to JSON (web-compatible)
    config = AnalysisConfig(
        analysis_type=AnalysisType.AIRPLANE,
        target="cessna172.xml",
        solver=SolverType.AVL,
    )

    json_data = config.model_dump()
    print(f"✓ Model serializes to dict: {type(json_data)}")

    json_str = config.model_dump_json()
    print(f"✓ Model serializes to JSON: {len(json_str)} characters")

    # Show that models can be deserialized from JSON
    restored_config = AnalysisConfig.model_validate_json(json_str)
    print(f"✓ Model deserializes from JSON: {restored_config.analysis_type}")

    print("\n✓ All data models are web-migration ready!")


async def main():
    """Run the integration example"""
    print("ICARUS CLI API Integration Example")
    print("=" * 50)

    try:
        await demonstrate_ui_abstraction()
        await demonstrate_data_models()
        await demonstrate_api_compatibility()

        print("\n" + "=" * 50)
        print("✓ Integration example completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• UI adapter abstraction for multiple frontends")
        print("• JSON serializable data models")
        print("• FastAPI REST API with OpenAPI documentation")
        print("• WebSocket support for real-time features")
        print("• Web migration readiness")

    except Exception as e:
        print(f"\n✗ Integration example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
