"""
Test script for ICARUS CLI API layer

This script demonstrates and tests the API layer functionality.
"""

import asyncio

from .adapters import UIAdapterFactory
from .app import create_api_app
from .models import AnalysisConfig
from .models import AnalysisType
from .models import SessionState
from .models import SolverType
from .models import UserPreferences
from .models import WebSocketMessage
from .models import Workflow
from .models import WorkflowStep
from .websocket import WebSocketManager


async def test_data_models():
    """Test Pydantic data models"""
    print("Testing Pydantic data models...")

    # Test AnalysisConfig
    config = AnalysisConfig(
        analysis_type=AnalysisType.AIRFOIL,
        target="naca0012.dat",
        solver=SolverType.XFOIL,
        parameters={"reynolds": 1000000, "mach": 0.1},
    )

    # Test JSON serialization
    config_json = config.model_dump_json()
    print(f"AnalysisConfig JSON: {config_json}")

    # Test deserialization
    config_restored = AnalysisConfig.model_validate_json(config_json)
    assert config.id == config_restored.id
    print("✓ AnalysisConfig serialization/deserialization works")

    # Test Workflow
    workflow_step = WorkflowStep(
        id="step1",
        name="Airfoil Analysis",
        analysis_config=config,
        dependencies=[],
        order=1,
    )

    workflow = Workflow(
        name="Basic Airfoil Study",
        description="Simple airfoil analysis workflow",
        steps=[workflow_step],
    )

    workflow_json = workflow.model_dump_json()
    print(f"Workflow JSON length: {len(workflow_json)} characters")
    print("✓ Workflow serialization works")

    # Test SessionState
    preferences = UserPreferences(
        theme="aerospace",
        default_solver=SolverType.XFOIL,
        recent_files=["naca0012.dat", "naca2412.dat"],
    )

    session = SessionState(
        user_id="test_user",
        workspace="/home/user/icarus_workspace",
        preferences=preferences,
    )

    session_json = session.model_dump_json()
    print(f"SessionState JSON: {session_json}")
    print("✓ SessionState serialization works")


async def test_ui_adapters():
    """Test UI adapter abstraction"""
    print("\nTesting UI adapters...")

    # Test adapter factory
    try:
        textual_adapter = UIAdapterFactory.create_adapter("textual")
        print("✓ Textual adapter created")

        web_adapter = UIAdapterFactory.create_adapter("web")
        print("✓ Web adapter created")

        # Test initialization (may fail if Textual not available, which is expected)
        try:
            await textual_adapter.initialize()
            print("✓ Textual adapter initialized")
        except RuntimeError as e:
            print(f"⚠ Textual adapter initialization failed (expected): {e}")

        await web_adapter.initialize()
        print("✓ Web adapter initialized")

    except Exception as e:
        print(f"✗ UI adapter test failed: {e}")


async def test_websocket_manager():
    """Test WebSocket manager functionality"""
    print("\nTesting WebSocket manager...")

    # Create a mock WebSocket connection
    class MockWebSocket:
        def __init__(self):
            self.messages = []
            self.closed = False

        async def send_text(self, text):
            self.messages.append(text)

        async def close(self):
            self.closed = True

    ws_manager = WebSocketManager()

    # Test connection management
    mock_ws = MockWebSocket()
    connection = await ws_manager.add_connection(mock_ws)
    print(f"✓ Connection added: {connection.session_id}")

    # Test authentication
    await ws_manager.authenticate_connection(connection.session_id, "test_user")
    print("✓ Connection authenticated")

    # Test room management
    await ws_manager.join_collaboration_room(connection.session_id, "room1")
    print("✓ Joined collaboration room")

    # Test message sending
    message = WebSocketMessage(type="test_message", payload={"data": "test"})

    sent = await ws_manager.send_to_session(connection.session_id, message)
    assert sent == True
    assert len(mock_ws.messages) == 1
    print("✓ Message sent to session")

    # Test broadcasting
    sent_count = await ws_manager.broadcast_to_room("room1", message)
    assert sent_count == 1
    print("✓ Message broadcast to room")

    # Test stats
    stats = await ws_manager.get_connection_stats()
    print(f"✓ Connection stats: {stats}")

    # Cleanup
    await ws_manager.remove_connection(connection.session_id)
    print("✓ Connection removed")


async def test_api_app():
    """Test FastAPI application creation"""
    print("\nTesting FastAPI application...")

    try:
        app = create_api_app()
        print("✓ FastAPI app created successfully")

        # Check that routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/health",
            "/analysis/start",
            "/workflow/execute",
            "/session",
            "/ws",
        ]

        for expected_route in expected_routes:
            if any(expected_route in route for route in routes):
                print(f"✓ Route {expected_route} found")
            else:
                print(f"✗ Route {expected_route} missing")

    except Exception as e:
        print(f"✗ FastAPI app creation failed: {e}")


async def main():
    """Run all tests"""
    print("ICARUS CLI API Layer Test Suite")
    print("=" * 40)

    await test_data_models()
    await test_ui_adapters()
    await test_websocket_manager()
    await test_api_app()

    print("\n" + "=" * 40)
    print("Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main())
