"""
FastAPI application for ICARUS CLI API layer

This module creates the REST API application with OpenAPI documentation,
providing endpoints for analysis, workflow management, and session handling.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import AnalysisConfig
from .models import AnalysisRequest
from .models import AnalysisResponse
from .models import AnalysisResult
from .models import AnalysisStatus
from .models import ErrorResponse
from .models import SessionRequest
from .models import SessionResponse
from .models import SessionState
from .models import SolverInfo
from .models import SolverType
from .models import ValidationResult
from .models import Workflow
from .models import WorkflowExecution
from .models import WorkflowRequest
from .models import WorkflowResponse
from .websocket import handle_websocket_connection
from .websocket import websocket_manager

try:
    from cli.collaboration import get_collaboration_manager
    from cli.collaboration import initialize_collaboration
except ImportError:
    # Fallback for testing without full collaboration setup
    get_collaboration_manager = None
    initialize_collaboration = None

logger = logging.getLogger(__name__)

# In-memory storage for demonstration (would be replaced with proper database)
analyses: Dict[UUID, AnalysisResult] = {}
workflows: Dict[UUID, Workflow] = {}
workflow_executions: Dict[UUID, WorkflowExecution] = {}
sessions: Dict[UUID, SessionState] = {}

# Global collaboration manager
collaboration_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global collaboration_manager

    logger.info("Starting ICARUS CLI API server...")

    # Startup tasks
    if initialize_collaboration is not None:
        collaboration_manager = await initialize_collaboration()
    else:
        collaboration_manager = None

    yield

    # Shutdown tasks
    logger.info("Shutting down ICARUS CLI API server...")
    await websocket_manager.shutdown()
    if collaboration_manager:
        await collaboration_manager.stop()


def create_api_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    app = FastAPI(
        title="ICARUS CLI API",
        description="REST API for ICARUS aerodynamics software CLI interface",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware for web interface support
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handler for custom error responses
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.__class__.__name__,
                message=str(exc.detail),
            ).model_dump(),
        )

    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "ICARUS CLI API"}

    # System information endpoint
    @app.get("/system/info", tags=["System"])
    async def get_system_info():
        """Get system information and available solvers"""
        # This would query the actual ICARUS system for available solvers
        solvers = [
            SolverInfo(
                name=SolverType.XFOIL,
                available=True,
                capabilities=["airfoil_analysis"],
            ),
            SolverInfo(
                name=SolverType.AVL,
                available=True,
                capabilities=["airplane_analysis"],
            ),
            SolverInfo(
                name=SolverType.GENUVP,
                available=False,
                capabilities=["vortex_particle_method"],
            ),
        ]

        return {
            "version": "1.0.0",
            "available_solvers": [solver.model_dump() for solver in solvers],
            "websocket_stats": await websocket_manager.get_connection_stats(),
        }

    # Analysis endpoints
    @app.post("/analysis/start", response_model=AnalysisResponse, tags=["Analysis"])
    async def start_analysis(request: AnalysisRequest):
        """Start a new analysis"""
        try:
            # Validate the analysis configuration
            validation = validate_analysis_config(request.config)
            if not validation.valid:
                raise HTTPException(status_code=400, detail=validation.errors)

            # Create analysis result entry
            result = AnalysisResult(
                id=request.config.id,
                config_id=request.config.id,
                status=AnalysisStatus.PENDING,
            )
            analyses[result.id] = result

            # Start analysis asynchronously (would integrate with actual ICARUS modules)
            asyncio.create_task(execute_analysis(request.config, result.id))

            return AnalysisResponse(analysis_id=result.id, status=result.status)

        except Exception as e:
            logger.error(f"Failed to start analysis: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/analysis/{analysis_id}",
        response_model=AnalysisResult,
        tags=["Analysis"],
    )
    async def get_analysis_result(analysis_id: UUID):
        """Get analysis result by ID"""
        if analysis_id not in analyses:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return analyses[analysis_id]

    @app.get("/analysis", response_model=List[AnalysisResult], tags=["Analysis"])
    async def list_analyses(status: Optional[AnalysisStatus] = None, limit: int = 100):
        """List analyses with optional status filter"""
        results = list(analyses.values())

        if status:
            results = [r for r in results if r.status == status]

        return results[:limit]

    @app.delete("/analysis/{analysis_id}", tags=["Analysis"])
    async def cancel_analysis(analysis_id: UUID):
        """Cancel a running analysis"""
        if analysis_id not in analyses:
            raise HTTPException(status_code=404, detail="Analysis not found")

        result = analyses[analysis_id]
        if result.status == AnalysisStatus.RUNNING:
            result.status = AnalysisStatus.CANCELLED
            # Would also cancel the actual analysis process

        return {"message": "Analysis cancelled"}

    # Workflow endpoints
    @app.post("/workflow/execute", response_model=WorkflowResponse, tags=["Workflow"])
    async def execute_workflow(request: WorkflowRequest):
        """Execute a workflow"""
        try:
            # Create workflow execution entry
            execution = WorkflowExecution(
                workflow_id=request.workflow.id,
                status=AnalysisStatus.PENDING,
            )
            workflow_executions[execution.id] = execution

            # Store the workflow if it's new
            if request.workflow.id not in workflows:
                workflows[request.workflow.id] = request.workflow

            # Start workflow execution asynchronously
            asyncio.create_task(execute_workflow_async(request.workflow, execution.id))

            return WorkflowResponse(execution_id=execution.id, status=execution.status)

        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/workflow/execution/{execution_id}",
        response_model=WorkflowExecution,
        tags=["Workflow"],
    )
    async def get_workflow_execution(execution_id: UUID):
        """Get workflow execution status and results"""
        if execution_id not in workflow_executions:
            raise HTTPException(status_code=404, detail="Workflow execution not found")

        return workflow_executions[execution_id]

    @app.get("/workflow", response_model=List[Workflow], tags=["Workflow"])
    async def list_workflows():
        """List all workflows"""
        return list(workflows.values())

    @app.post("/workflow", response_model=Workflow, tags=["Workflow"])
    async def save_workflow(workflow: Workflow):
        """Save a workflow"""
        workflows[workflow.id] = workflow
        return workflow

    # Session endpoints
    @app.post("/session", response_model=SessionResponse, tags=["Session"])
    async def create_session(request: SessionRequest):
        """Create a new session"""
        session = SessionState(
            user_id=request.user_id,
            workspace=request.workspace,
            preferences=request.preferences or {},
        )
        sessions[session.id] = session

        return SessionResponse(session=session)

    @app.get("/session/{session_id}", response_model=SessionState, tags=["Session"])
    async def get_session(session_id: UUID):
        """Get session by ID"""
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        return sessions[session_id]

    @app.put("/session/{session_id}", response_model=SessionState, tags=["Session"])
    async def update_session(session_id: UUID, session_data: SessionState):
        """Update session data"""
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        sessions[session_id] = session_data
        return session_data

    @app.delete("/session/{session_id}", tags=["Session"])
    async def delete_session(session_id: UUID):
        """Delete a session"""
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        del sessions[session_id]
        return {"message": "Session deleted"}

    # Collaboration endpoints
    @app.post("/collaboration/users", tags=["Collaboration"])
    async def create_user(
        username: str,
        email: str,
        display_name: str,
        password: str,
        role: str = "collaborator",
    ):
        """Create a new user"""
        try:
            from cli.collaboration.user_manager import UserRole

            user_role = UserRole(role)
            user = collaboration_manager.create_user(
                username,
                email,
                display_name,
                password,
                user_role,
            )
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "display_name": user.display_name,
                "role": user.role.value,
                "created_at": user.created_at.isoformat(),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/collaboration/auth/login", tags=["Collaboration"])
    async def login_user(username: str, password: str):
        """Authenticate user and get session token"""
        token = collaboration_manager.authenticate_user(username, password)
        if not token:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"token": token, "message": "Login successful"}

    @app.post("/collaboration/auth/guest", tags=["Collaboration"])
    async def create_guest_user(display_name: str):
        """Create a temporary guest user"""
        user = collaboration_manager.create_guest_user(display_name)
        return {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role.value,
            "session_token": user.session_token,
        }

    @app.get("/collaboration/users/online", tags=["Collaboration"])
    async def get_online_users():
        """Get currently online users"""
        users = collaboration_manager.get_online_users()
        return [
            {
                "id": user.id,
                "username": user.username,
                "display_name": user.display_name,
                "role": user.role.value,
                "last_active": user.last_active.isoformat(),
            }
            for user in users
        ]

    @app.post("/collaboration/sessions", tags=["Collaboration"])
    async def create_collaboration_session(
        token: str,
        name: str,
        description: str = "",
        session_type: str = "general",
        max_participants: int = 10,
        allow_guests: bool = True,
    ):
        """Create a new collaboration session"""
        user = collaboration_manager.authenticate_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        try:
            from cli.collaboration.session_sharing import SessionSettings
            from cli.collaboration.session_sharing import SessionType

            session_type_enum = SessionType(session_type)
            settings = SessionSettings(
                max_participants=max_participants,
                allow_guests=allow_guests,
            )

            session = collaboration_manager.create_session(
                user,
                name,
                description,
                session_type_enum,
                settings,
            )
            return {
                "id": session.id,
                "name": session.name,
                "description": session.description,
                "invite_code": session.invite_code,
                "owner_id": session.owner_id,
                "created_at": session.created_at.isoformat(),
                "participants": len(session.participants),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/collaboration/sessions/join", tags=["Collaboration"])
    async def join_collaboration_session(invite_code: str, token: str):
        """Join a collaboration session"""
        user = collaboration_manager.authenticate_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        session = await collaboration_manager.join_session(invite_code, user)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or cannot join",
            )

        return {
            "id": session.id,
            "name": session.name,
            "description": session.description,
            "participants": [
                {
                    "user_id": p.user_id,
                    "display_name": p.display_name,
                    "role": p.role.value,
                    "is_online": p.is_online,
                }
                for p in session.participants.values()
            ],
        }

    @app.post("/collaboration/sessions/{session_id}/leave", tags=["Collaboration"])
    async def leave_collaboration_session(session_id: str, token: str):
        """Leave a collaboration session"""
        user = collaboration_manager.authenticate_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        success = await collaboration_manager.leave_session(session_id, user)
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Session not found or not a participant",
            )

        return {"message": "Left session successfully"}

    @app.get("/collaboration/sessions/active", tags=["Collaboration"])
    async def get_active_sessions():
        """Get all active collaboration sessions"""
        sessions = collaboration_manager.get_active_sessions()
        return [
            {
                "id": session.id,
                "name": session.name,
                "description": session.description,
                "session_type": session.session_type.value,
                "participants": len(session.participants),
                "online_participants": len(session.get_online_participants()),
                "created_at": session.created_at.isoformat(),
            }
            for session in sessions
        ]

    @app.post("/collaboration/sessions/{session_id}/messages", tags=["Collaboration"])
    async def send_message(
        session_id: str,
        token: str,
        content: str,
        message_type: str = "text",
    ):
        """Send a chat message"""
        user = collaboration_manager.authenticate_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        try:
            from cli.collaboration.communication import MessageType

            msg_type = MessageType(message_type)
            message = await collaboration_manager.send_message(
                session_id,
                user,
                content,
                msg_type,
            )
            if not message:
                raise HTTPException(status_code=403, detail="Cannot send message")

            return {
                "id": message.id,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "user": message.display_name,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/collaboration/sessions/{session_id}/messages", tags=["Collaboration"])
    async def get_messages(
        session_id: str,
        token: str,
        limit: int = 50,
        offset: int = 0,
    ):
        """Get chat messages for a session"""
        user = collaboration_manager.authenticate_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Verify user is in session
        session = collaboration_manager.get_session(session_id)
        if not session or not session.is_participant(user.id):
            raise HTTPException(status_code=403, detail="Not a session participant")

        messages = collaboration_manager.get_messages(session_id, limit, offset)
        return [
            {
                "id": msg.id,
                "content": msg.content,
                "message_type": msg.message_type.value,
                "timestamp": msg.timestamp.isoformat(),
                "user": msg.display_name,
                "reactions": msg.reactions,
            }
            for msg in messages
        ]

    @app.post(
        "/collaboration/sessions/{session_id}/annotations",
        tags=["Collaboration"],
    )
    async def create_annotation(
        session_id: str,
        token: str,
        content: str,
        annotation_type: str,
        target: dict,
        position: dict,
    ):
        """Create an annotation"""
        user = collaboration_manager.authenticate_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        try:
            from cli.collaboration.communication import AnnotationType

            ann_type = AnnotationType(annotation_type)
            annotation = await collaboration_manager.create_annotation(
                session_id,
                user,
                content,
                ann_type,
                target,
                position,
            )
            if not annotation:
                raise HTTPException(status_code=403, detail="Cannot create annotation")

            return {
                "id": annotation.id,
                "content": annotation.content,
                "annotation_type": annotation.annotation_type.value,
                "timestamp": annotation.timestamp.isoformat(),
                "user": annotation.display_name,
                "resolved": annotation.resolved,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/collaboration/sessions/{session_id}/annotations", tags=["Collaboration"])
    async def get_annotations(
        session_id: str,
        token: str,
        resolved: Optional[bool] = None,
    ):
        """Get annotations for a session"""
        user = collaboration_manager.authenticate_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Verify user is in session
        session = collaboration_manager.get_session(session_id)
        if not session or not session.is_participant(user.id):
            raise HTTPException(status_code=403, detail="Not a session participant")

        annotations = collaboration_manager.get_annotations(session_id, resolved)
        return [
            {
                "id": ann.id,
                "content": ann.content,
                "annotation_type": ann.annotation_type.value,
                "timestamp": ann.timestamp.isoformat(),
                "user": ann.display_name,
                "resolved": ann.resolved,
                "target": ann.target,
                "position": ann.position,
            }
            for ann in annotations
        ]

    @app.get("/collaboration/stats", tags=["Collaboration"])
    async def get_collaboration_stats():
        """Get collaboration system statistics"""
        return collaboration_manager.get_collaboration_stats()

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time communication"""
        await websocket.accept()
        try:
            await handle_websocket_connection(websocket)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    return app


def validate_analysis_config(config: AnalysisConfig) -> ValidationResult:
    """Validate analysis configuration"""
    errors = []
    warnings = []
    suggestions = []

    # Basic validation
    if not config.target:
        errors.append("Target is required")

    if not config.solver:
        errors.append("Solver is required")

    # Solver-specific validation would go here
    if config.solver == SolverType.XFOIL and config.analysis_type.value != "airfoil":
        errors.append("XFoil solver can only be used for airfoil analysis")

    # Parameter validation would be more comprehensive in real implementation
    if not config.parameters:
        warnings.append("No parameters specified, using defaults")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        suggestions=suggestions,
    )


async def execute_analysis(config: AnalysisConfig, result_id: UUID):
    """Execute an analysis (mock implementation)"""
    try:
        result = analyses[result_id]
        result.status = AnalysisStatus.RUNNING
        result.started_at = result.created_at

        # Simulate analysis execution with progress updates
        for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
            await asyncio.sleep(1)  # Simulate work
            await websocket_manager.send_progress_update(
                str(result_id),
                progress,
                f"Analysis {progress:.0%} complete",
            )

        # Simulate successful completion
        result.status = AnalysisStatus.COMPLETED
        result.completed_at = result.started_at
        result.data = {
            "results": f"Mock results for {config.analysis_type} analysis",
            "solver_used": config.solver,
            "target": config.target,
        }

        # Send completion notification
        await websocket_manager.send_analysis_result(
            str(result_id),
            result.model_dump(),
        )

    except Exception as e:
        logger.error(f"Analysis execution failed: {e}")
        result = analyses[result_id]
        result.status = AnalysisStatus.FAILED
        result.error_message = str(e)

        await websocket_manager.send_error_notification(
            str(result_id),
            f"Analysis failed: {e}",
        )


async def execute_workflow_async(workflow: Workflow, execution_id: UUID):
    """Execute a workflow asynchronously (mock implementation)"""
    try:
        execution = workflow_executions[execution_id]
        execution.status = AnalysisStatus.RUNNING
        execution.started_at = execution.created_at

        # Execute each step
        for step in workflow.steps:
            execution.current_step = step.id

            # Mock step execution
            await asyncio.sleep(2)

            # Create mock result for this step
            step_result = AnalysisResult(
                id=step.analysis_config.id,
                config_id=step.analysis_config.id,
                status=AnalysisStatus.COMPLETED,
                data={"step_result": f"Mock result for step {step.name}"},
            )
            execution.step_results[step.id] = step_result

        execution.status = AnalysisStatus.COMPLETED
        execution.completed_at = execution.started_at

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        execution = workflow_executions[execution_id]
        execution.status = AnalysisStatus.FAILED
        execution.error_message = str(e)
