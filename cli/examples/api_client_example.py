"""
Example API client for testing ICARUS CLI API

This script demonstrates how to interact with the ICARUS CLI API
using HTTP requests and WebSocket connections.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the CLI directory to the path
cli_path = Path(__file__).parent.parent
sys.path.insert(0, str(cli_path))

try:
    import httpx
    import websockets

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print(
        "Note: httpx and websockets not available. Install with: pip install httpx websockets",
    )

from api.models import AnalysisConfig
from api.models import AnalysisType
from api.models import SolverType
from api.models import Workflow
from api.models import WorkflowStep


async def test_rest_api():
    """Test REST API endpoints"""
    if not DEPENDENCIES_AVAILABLE:
        print("Skipping REST API test - dependencies not available")
        return

    print("Testing REST API endpoints...")
    base_url = "http://127.0.0.1:8000"

    async with httpx.AsyncClient() as client:
        try:
            # Test health endpoint
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✓ Health check passed")
                print(f"  Response: {response.json()}")
            else:
                print(f"✗ Health check failed: {response.status_code}")

            # Test system info
            response = await client.get(f"{base_url}/system/info")
            if response.status_code == 200:
                print("✓ System info retrieved")
                info = response.json()
                print(f"  Version: {info.get('version')}")
                print(f"  Available solvers: {len(info.get('available_solvers', []))}")

            # Test analysis creation
            analysis_config = AnalysisConfig(
                analysis_type=AnalysisType.AIRFOIL,
                target="naca0012.dat",
                solver=SolverType.XFOIL,
                parameters={"reynolds": 1000000, "mach": 0.1},
            )

            response = await client.post(
                f"{base_url}/analysis/start",
                json={"config": analysis_config.model_dump()},
            )

            if response.status_code == 200:
                print("✓ Analysis started")
                result = response.json()
                analysis_id = result["analysis_id"]
                print(f"  Analysis ID: {analysis_id}")

                # Check analysis status
                await asyncio.sleep(1)  # Give it a moment
                response = await client.get(f"{base_url}/analysis/{analysis_id}")
                if response.status_code == 200:
                    status = response.json()
                    print(f"  Analysis status: {status['status']}")

            # Test session creation
            session_data = {"user_id": "test_user", "workspace": "/tmp/test_workspace"}

            response = await client.post(f"{base_url}/session", json=session_data)
            if response.status_code == 200:
                print("✓ Session created")
                session = response.json()["session"]
                print(f"  Session ID: {session['id']}")

        except httpx.ConnectError:
            print("✗ Could not connect to API server")
            print("  Make sure to start the server with: python -m cli.api.server")
        except Exception as e:
            print(f"✗ API test failed: {e}")


async def test_websocket():
    """Test WebSocket connection"""
    if not DEPENDENCIES_AVAILABLE:
        print("Skipping WebSocket test - dependencies not available")
        return

    print("\nTesting WebSocket connection...")

    try:
        uri = "ws://127.0.0.1:8000/ws"
        async with websockets.connect(uri) as websocket:
            print("✓ WebSocket connected")

            # Send authentication message
            auth_message = {"type": "authenticate", "payload": {"user_id": "test_user"}}
            await websocket.send(json.dumps(auth_message))

            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)

            if response_data.get("type") == "authenticated":
                print("✓ WebSocket authentication successful")
                session_id = response_data["payload"]["session_id"]
                print(f"  Session ID: {session_id}")

            # Send ping
            ping_message = {"type": "ping", "payload": {"timestamp": "test"}}
            await websocket.send(json.dumps(ping_message))

            # Wait for pong
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)

            if response_data.get("type") == "pong":
                print("✓ WebSocket ping/pong successful")

    except websockets.exceptions.ConnectionRefused:
        print("✗ Could not connect to WebSocket")
        print("  Make sure to start the server with: python -m cli.api.server")
    except asyncio.TimeoutError:
        print("✗ WebSocket timeout")
    except Exception as e:
        print(f"✗ WebSocket test failed: {e}")


async def demonstrate_workflow_api():
    """Demonstrate workflow API usage"""
    if not DEPENDENCIES_AVAILABLE:
        print("Skipping workflow API test - dependencies not available")
        return

    print("\nTesting Workflow API...")
    base_url = "http://127.0.0.1:8000"

    # Create a sample workflow
    analysis_config = AnalysisConfig(
        analysis_type=AnalysisType.AIRFOIL,
        target="naca2412.dat",
        solver=SolverType.XFOIL,
        parameters={"reynolds": 500000, "mach": 0.05},
    )

    workflow_step = WorkflowStep(
        id="step1",
        name="NACA 2412 Analysis",
        analysis_config=analysis_config,
        order=1,
    )

    workflow = Workflow(
        name="NACA 2412 Study",
        description="Analysis of NACA 2412 airfoil",
        steps=[workflow_step],
    )

    async with httpx.AsyncClient() as client:
        try:
            # Save workflow
            response = await client.post(
                f"{base_url}/workflow",
                json=workflow.model_dump(),
            )

            if response.status_code == 200:
                print("✓ Workflow saved")

                # Execute workflow
                response = await client.post(
                    f"{base_url}/workflow/execute",
                    json={"workflow": workflow.model_dump()},
                )

                if response.status_code == 200:
                    print("✓ Workflow execution started")
                    result = response.json()
                    execution_id = result["execution_id"]
                    print(f"  Execution ID: {execution_id}")

                    # Check execution status
                    await asyncio.sleep(2)  # Give it time to process
                    response = await client.get(
                        f"{base_url}/workflow/execution/{execution_id}",
                    )
                    if response.status_code == 200:
                        execution = response.json()
                        print(f"  Execution status: {execution['status']}")

        except httpx.ConnectError:
            print("✗ Could not connect to API server")
        except Exception as e:
            print(f"✗ Workflow API test failed: {e}")


def show_api_documentation():
    """Show information about API documentation"""
    print("\nAPI Documentation:")
    print("=" * 40)
    print("Once the server is running, you can access:")
    print("• Interactive API docs: http://127.0.0.1:8000/docs")
    print("• ReDoc documentation: http://127.0.0.1:8000/redoc")
    print("• OpenAPI JSON schema: http://127.0.0.1:8000/openapi.json")
    print("\nTo start the server:")
    print("  python -m cli.api.server")


async def main():
    """Run the API client example"""
    print("ICARUS CLI API Client Example")
    print("=" * 40)

    if not DEPENDENCIES_AVAILABLE:
        print("Installing missing dependencies...")
        print("Run: pip install httpx websockets")
        print()

    await test_rest_api()
    await test_websocket()
    await demonstrate_workflow_api()

    show_api_documentation()

    print("\n" + "=" * 40)
    print("API client example completed!")


if __name__ == "__main__":
    asyncio.run(main())
