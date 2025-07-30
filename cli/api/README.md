# ICARUS CLI API Layer

This directory contains the API layer foundation for the ICARUS CLI, designed to support web migration readiness and provide a unified interface for multiple frontend implementations.

## Overview

The API layer provides:

- **REST API** with FastAPI and OpenAPI documentation
- **WebSocket support** for real-time features and collaboration
- **UI adapter abstraction** for multiple frontend support (TUI, Web)
- **JSON serializable data models** with Pydantic
- **Web migration readiness** with clear separation of concerns

## Architecture

```
cli/api/
├── __init__.py          # Package initialization
├── models.py            # Pydantic data models
├── adapters.py          # UI adapter abstraction layer
├── websocket.py         # WebSocket manager and real-time features
├── app.py               # FastAPI application
├── server.py            # Development server
├── test_api.py          # Test suite
└── README.md            # This file
```

## Key Components

### 1. Data Models (`models.py`)

JSON-serializable Pydantic models that work across TUI and web interfaces:

- `AnalysisConfig` - Configuration for analysis runs
- `AnalysisResult` - Results from analysis execution
- `Workflow` / `WorkflowStep` - Workflow definitions and execution
- `SessionState` - User session management
- `WebSocketMessage` - Real-time communication
- `CollaborationEvent` - Multi-user collaboration events

### 2. UI Adapters (`adapters.py`)

Abstract interface allowing the same business logic to work with different UIs:

- `UIAdapter` - Abstract base class for UI implementations
- `TextualUIAdapter` - Textual TUI implementation
- `WebUIAdapter` - Future web interface implementation
- `UIBridge` - Bridge between adapters and business logic

### 3. WebSocket Manager (`websocket.py`)

Real-time communication support:

- Connection management and authentication
- Collaboration room management
- Message broadcasting and routing
- Progress updates and notifications

### 4. FastAPI Application (`app.py`)

REST API with comprehensive endpoints:

- Analysis management (`/analysis/*`)
- Workflow execution (`/workflow/*`)
- Session handling (`/session/*`)
- System information (`/system/*`)
- WebSocket endpoint (`/ws`)

## Usage Examples

### Starting the API Server

```bash
# Start development server
python -m cli.api.server

# Or with uvicorn directly
uvicorn cli.api.app:create_api_app --host 127.0.0.1 --port 8000
```

### Using Data Models

```python
from cli.api.models import AnalysisConfig, AnalysisType, SolverType

# Create analysis configuration
config = AnalysisConfig(
    analysis_type=AnalysisType.AIRFOIL,
    target="naca0012.dat",
    solver=SolverType.XFOIL,
    parameters={"reynolds": 1000000, "mach": 0.1}
)

# Serialize to JSON (web-compatible)
json_data = config.model_dump_json()

# Deserialize from JSON
restored_config = AnalysisConfig.model_validate_json(json_data)
```

### Using UI Adapters

```python
from cli.api.adapters import UIAdapterFactory, UIBridge
from cli.api.models import ScreenData

# Create adapter for current interface
adapter = UIAdapterFactory.create_adapter("textual")  # or "web"
await adapter.initialize()

# Create screen data
screen_data = ScreenData(
    screen_id="analysis_setup",
    title="Analysis Setup",
    content={"analysis_type": "airfoil", "target": "naca0012.dat"}
)

# Render screen
await adapter.render_screen(screen_data)
```

### WebSocket Communication

```python
import asyncio
import json
import websockets

async def websocket_client():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Authenticate
        auth_msg = {
            "type": "authenticate",
            "payload": {"user_id": "user123"}
        }
        await websocket.send(json.dumps(auth_msg))

        # Receive messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")
```

## API Endpoints

### Analysis Endpoints

- `POST /analysis/start` - Start new analysis
- `GET /analysis/{analysis_id}` - Get analysis result
- `GET /analysis` - List analyses
- `DELETE /analysis/{analysis_id}` - Cancel analysis

### Workflow Endpoints

- `POST /workflow/execute` - Execute workflow
- `GET /workflow/execution/{execution_id}` - Get execution status
- `GET /workflow` - List workflows
- `POST /workflow` - Save workflow

### Session Endpoints

- `POST /session` - Create session
- `GET /session/{session_id}` - Get session
- `PUT /session/{session_id}` - Update session
- `DELETE /session/{session_id}` - Delete session

### System Endpoints

- `GET /health` - Health check
- `GET /system/info` - System information

### WebSocket Endpoint

- `WS /ws` - Real-time communication

## Web Migration Readiness

The API layer is designed for easy migration to web applications:

### 1. Separation of Concerns

- **Business Logic**: UI-agnostic services
- **Data Models**: JSON-serializable with standard formats
- **API Layer**: RESTful endpoints with OpenAPI documentation
- **UI Layer**: Abstract adapters for different implementations

### 2. Technology Compatibility

| Component | TUI Technology | Web Technology | Shared Code |
|-----------|---------------|----------------|-------------|
| Business Logic | Python Classes | Python/FastAPI | 95% |
| Data Models | Pydantic | JSON Schema | 100% |
| Real-time Features | AsyncIO | WebSocket | 85% |
| Authentication | Local | JWT/OAuth2 | 80% |

### 3. Migration Benefits

- **Reduced Development Time**: Reuse 80%+ of business logic
- **Consistent Behavior**: Same functionality across interfaces
- **Easier Maintenance**: Single source of truth
- **Flexible Deployment**: Support multiple interfaces simultaneously

## Testing

Run the test suite:

```bash
# Run API tests
python -m cli.api.test_api

# Run integration example
python cli/examples/api_integration_example.py

# Test with API client (requires server running)
python cli/examples/api_client_example.py
```

## Dependencies

Required packages (already in `cli/requirements.txt`):

```
fastapi>=0.100.0     # REST API framework
websockets>=11.0     # WebSocket support
uvicorn>=0.23.0      # ASGI server
pydantic>=2.0.0      # Data validation and serialization
```

Optional for testing:

```
httpx                # HTTP client for testing
websockets           # WebSocket client for testing
```

## Development

### Adding New Endpoints

1. Define data models in `models.py`
2. Add endpoint to `app.py`
3. Update tests in `test_api.py`
4. Document in this README

### Adding New UI Adapters

1. Inherit from `UIAdapter` in `adapters.py`
2. Implement all abstract methods
3. Add to `UIAdapterFactory`
4. Test with integration examples

### WebSocket Message Types

Current message types:
- `authenticate` - User authentication
- `join_room` / `leave_room` - Collaboration rooms
- `collaboration_event` - Real-time collaboration
- `ping` / `pong` - Connection health
- `progress_update` - Analysis progress
- `analysis_result` - Analysis completion
- `error_notification` - Error messages

## Future Enhancements

- Database integration (SQLAlchemy)
- Authentication and authorization
- Rate limiting and security
- Plugin system integration
- Advanced collaboration features
- Performance monitoring
- Caching layer

## Documentation

When the server is running, access:

- **Interactive API Docs**: http://127.0.0.1:8000/docs
- **ReDoc Documentation**: http://127.0.0.1:8000/redoc
- **OpenAPI Schema**: http://127.0.0.1:8000/openapi.json
