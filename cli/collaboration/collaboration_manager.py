"""
Collaboration Manager

This module provides a unified interface for all collaboration features,
integrating user management, session sharing, state synchronization,
and communication systems.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from uuid import uuid4

from .communication import AnnotationType
from .communication import CommunicationSystem
from .communication import MessageType
from .conflict_resolution import ConflictResolver
from .conflict_resolution import ResolutionStrategy
from .session_recording import EventType as RecordingEventType
from .session_recording import SessionPlayer
from .session_recording import SessionRecorder
from .session_sharing import CollaborationSession
from .session_sharing import SessionManager
from .session_sharing import SessionSettings
from .session_sharing import SessionType
from .state_sync import StateChangeType
from .state_sync import StateSynchronizer
from .user_manager import User
from .user_manager import UserManager
from .user_manager import UserRole


class CollaborationManager:
    """Main manager for all collaboration features"""

    def __init__(self, data_dir: str = "~/.icarus/collaboration"):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.user_manager = UserManager(data_dir)
        self.session_manager = SessionManager(data_dir)
        self.state_synchronizer = StateSynchronizer(self.session_manager)
        self.communication_system = CommunicationSystem(self.session_manager, data_dir)
        self.conflict_resolver = ConflictResolver(self.session_manager, data_dir)
        self.session_recorder = SessionRecorder(self.session_manager, data_dir)
        self.session_player = SessionPlayer(self.session_recorder)

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        self.logger.info("Collaboration manager initialized")

    async def start(self):
        """Start the collaboration manager and background tasks"""
        if self._running:
            return

        self._running = True

        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._background_cleanup())

        self.logger.info("Collaboration manager started")

    async def stop(self):
        """Stop the collaboration manager and cleanup"""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Collaboration manager stopped")

    # User Management Methods

    def create_user(
        self,
        username: str,
        email: str,
        display_name: str,
        password: str,
        role: UserRole = UserRole.COLLABORATOR,
    ) -> User:
        """Create a new user"""
        return self.user_manager.create_user(
            username,
            email,
            display_name,
            password,
            role,
        )

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token"""
        return self.user_manager.authenticate_user(username, password)

    def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate using session token"""
        return self.user_manager.authenticate_token(token)

    def create_guest_user(self, display_name: str) -> User:
        """Create a temporary guest user"""
        return self.user_manager.create_guest_user(display_name)

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.user_manager.get_user(user_id)

    def get_online_users(self) -> List[User]:
        """Get currently online users"""
        return self.user_manager.get_online_users()

    # Session Management Methods

    def create_session(
        self,
        owner: User,
        name: str,
        description: str = "",
        session_type: SessionType = SessionType.GENERAL,
        settings: Optional[SessionSettings] = None,
    ) -> CollaborationSession:
        """Create a new collaboration session"""
        session = self.session_manager.create_session(
            owner,
            name,
            description,
            session_type,
            settings,
        )

        # Initialize state for the session
        self.state_synchronizer.initialize_session_state(session.id)

        return session

    async def join_session(
        self,
        invite_code: str,
        user: User,
    ) -> Optional[CollaborationSession]:
        """Join a session using invite code"""
        session = self.session_manager.join_session(invite_code, user)

        if session:
            # Send welcome message
            await self.communication_system.send_message(
                session_id=session.id,
                user=user,
                content=f"{user.display_name} joined the session",
                message_type=MessageType.SYSTEM,
            )

            # Initialize user's state in the session
            await self.state_synchronizer.create_state_change(
                session_id=session.id,
                user_id=user.id,
                change_type=StateChangeType.UI_INTERACTION,
                path="participants",
                old_value=None,
                new_value={"action": "joined", "user": user.display_name},
            )

        return session

    async def leave_session(self, session_id: str, user: User) -> bool:
        """Leave a session"""
        success = self.session_manager.leave_session(session_id, user.id)

        if success:
            # Send goodbye message
            await self.communication_system.send_message(
                session_id=session_id,
                user=user,
                content=f"{user.display_name} left the session",
                message_type=MessageType.SYSTEM,
            )

        return success

    async def end_session(self, session_id: str, user: User) -> bool:
        """End a session (owner only)"""
        success = self.session_manager.end_session(session_id, user.id)

        if success:
            # Cleanup state and communication data
            self.state_synchronizer.cleanup_session_state(session_id)
            self.communication_system.cleanup_session_data(session_id)

        return success

    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get session by ID"""
        return self.session_manager.get_session(session_id)

    def get_session_by_invite_code(
        self,
        invite_code: str,
    ) -> Optional[CollaborationSession]:
        """Get session by invite code"""
        return self.session_manager.get_session_by_invite_code(invite_code)

    def get_user_sessions(self, user: User) -> List[CollaborationSession]:
        """Get all sessions where user is a participant"""
        return self.session_manager.get_user_sessions(user.id)

    def get_active_sessions(self) -> List[CollaborationSession]:
        """Get all active sessions"""
        return self.session_manager.get_active_sessions()

    # State Synchronization Methods

    async def sync_state_change(
        self,
        session_id: str,
        user: User,
        change_type: StateChangeType,
        path: str,
        old_value: Any,
        new_value: Any,
        metadata: Dict = None,
    ) -> bool:
        """Synchronize a state change across session participants"""
        change = await self.state_synchronizer.create_state_change(
            session_id=session_id,
            user_id=user.id,
            change_type=change_type,
            path=path,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata,
        )

        return change is not None

    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """Get current state for a session"""
        return self.state_synchronizer.get_session_state(session_id)

    def get_state_history(self, session_id: str, limit: int = 100) -> List:
        """Get state change history for a session"""
        return self.state_synchronizer.get_state_history(session_id, limit)

    # Communication Methods

    async def send_message(
        self,
        session_id: str,
        user: User,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        reply_to: Optional[str] = None,
    ) -> Optional[Any]:
        """Send a chat message"""
        return await self.communication_system.send_message(
            session_id=session_id,
            user=user,
            content=content,
            message_type=message_type,
            reply_to=reply_to,
        )

    async def create_annotation(
        self,
        session_id: str,
        user: User,
        content: str,
        annotation_type: AnnotationType,
        target: Dict,
        position: Dict,
    ) -> Optional[Any]:
        """Create an annotation"""
        return await self.communication_system.create_annotation(
            session_id=session_id,
            user=user,
            content=content,
            annotation_type=annotation_type,
            target=target,
            position=position,
        )

    def get_messages(self, session_id: str, limit: int = 50, offset: int = 0) -> List:
        """Get messages for a session"""
        return self.communication_system.get_messages(session_id, limit, offset)

    def get_annotations(self, session_id: str, resolved: Optional[bool] = None) -> List:
        """Get annotations for a session"""
        return self.communication_system.get_annotations(session_id, resolved)

    def search_messages(self, session_id: str, query: str, limit: int = 20) -> List:
        """Search messages in a session"""
        return self.communication_system.search_messages(session_id, query, limit)

    # Analysis Integration Methods

    async def share_analysis_result(
        self,
        session_id: str,
        user: User,
        analysis_id: str,
        result_data: Dict,
    ) -> bool:
        """Share an analysis result with session participants"""
        # Update shared state
        await self.sync_state_change(
            session_id=session_id,
            user=user,
            change_type=StateChangeType.RESULT_UPDATE,
            path=f"results.{analysis_id}",
            old_value=None,
            new_value=result_data,
        )

        # Send notification message
        await self.send_message(
            session_id=session_id,
            user=user,
            content=f"Shared analysis result: {analysis_id}",
            message_type=MessageType.ANALYSIS_RESULT,
        )

        return True

    async def start_collaborative_analysis(
        self,
        session_id: str,
        user: User,
        analysis_config: Dict,
    ) -> bool:
        """Start a collaborative analysis session"""
        # Update shared state
        analysis_id = analysis_config.get("id", "unknown")

        await self.sync_state_change(
            session_id=session_id,
            user=user,
            change_type=StateChangeType.ANALYSIS_UPDATE,
            path=f"active_analyses.{analysis_id}",
            old_value=None,
            new_value=analysis_config,
        )

        # Send notification
        await self.send_message(
            session_id=session_id,
            user=user,
            content=f"Started collaborative analysis: {analysis_config.get('name', analysis_id)}",
            message_type=MessageType.SYSTEM,
        )

        return True

    async def update_analysis_progress(
        self,
        session_id: str,
        analysis_id: str,
        progress: float,
        message: str = "",
    ) -> bool:
        """Update analysis progress for all session participants"""
        # Create a system user for progress updates
        from datetime import datetime

        from .user_manager import User
        from .user_manager import UserRole

        system_user = User(
            id="system",
            username="system",
            email="",
            display_name="System",
            role=UserRole.ADMIN,
            permissions=set(),
            created_at=datetime.now(),
            last_active=datetime.now(),
        )

        # Update shared state
        await self.sync_state_change(
            session_id=session_id,
            user=system_user,
            change_type=StateChangeType.ANALYSIS_UPDATE,
            path=f"active_analyses.{analysis_id}.progress",
            old_value=None,
            new_value={"progress": progress, "message": message},
        )

        return True

    # Utility Methods

    async def update_participant_activity(
        self,
        session_id: str,
        user: User,
        screen: Optional[str] = None,
        cursor_position: Optional[Dict] = None,
    ) -> bool:
        """Update participant activity information"""
        success = self.session_manager.update_participant_activity(
            session_id,
            user.id,
            screen,
            cursor_position,
        )

        if success and cursor_position:
            # Sync cursor position
            await self.sync_state_change(
                session_id=session_id,
                user=user,
                change_type=StateChangeType.CURSOR_MOVE,
                path=f"cursors.{user.id}",
                old_value=None,
                new_value=cursor_position,
            )

        return success

    # Real-time Collaboration Methods

    async def detect_and_resolve_conflicts(
        self,
        session_id: str,
        changes: List,
    ) -> bool:
        """Detect and automatically resolve conflicts"""
        from .conflict_resolution import ConflictingChange

        # Convert changes to ConflictingChange objects
        conflicting_changes = []
        for change in changes:
            conflicting_change = ConflictingChange(
                id=str(uuid4()) if not hasattr(change, "id") else change.id,
                user_id=change.user_id,
                username=getattr(change, "username", "Unknown"),
                timestamp=change.timestamp,
                change_type=str(change.change_type),
                path=change.path,
                old_value=change.old_value,
                new_value=change.new_value,
                metadata=getattr(change, "metadata", {}),
            )
            conflicting_changes.append(conflicting_change)

        # Detect conflicts
        conflict = await self.conflict_resolver.detect_conflict(
            session_id,
            conflicting_changes,
        )

        if conflict:
            # Try automatic resolution
            success = await self.conflict_resolver.resolve_conflict(
                conflict.id,
                ResolutionStrategy.LAST_WRITER_WINS,
            )

            if success:
                # Notify participants
                await self.send_notification(
                    session_id=session_id,
                    notification_type="conflict_resolved",
                    title="Conflict Resolved",
                    message=f"Conflict automatically resolved: {conflict.description}",
                    data={"conflict_id": conflict.id, "resolution": "last_writer_wins"},
                )

            return success

        return True

    async def send_notification(
        self,
        session_id: str,
        notification_type: str,
        title: str,
        message: str,
        data: Dict = None,
    ):
        """Send a notification to session participants"""
        try:
            from cli.api.websocket import websocket_manager

            await websocket_manager.send_notification(
                session_id,
                notification_type,
                title,
                message,
                data,
            )
        except ImportError:
            # Fallback: send as system message
            system_user = User(
                id="system",
                username="system",
                email="",
                display_name="System",
                role=UserRole.ADMIN,
                permissions=set(),
                created_at=datetime.now(),
                last_active=datetime.now(),
            )

            await self.send_message(
                session_id=session_id,
                user=system_user,
                content=f"{title}: {message}",
                message_type=MessageType.NOTIFICATION,
            )

    def get_active_conflicts(self, session_id: str) -> List:
        """Get active conflicts for a session"""
        return self.conflict_resolver.get_active_conflicts(session_id)

    def get_resolved_conflicts(self, session_id: str) -> List:
        """Get resolved conflicts for a session"""
        return self.conflict_resolver.get_resolved_conflicts(session_id)

    async def resolve_conflict_manually(
        self,
        conflict_id: str,
        resolution_strategy: ResolutionStrategy,
        resolver_user_id: str,
    ) -> bool:
        """Manually resolve a conflict"""
        return await self.conflict_resolver.resolve_conflict(
            conflict_id,
            resolution_strategy,
            resolver_user_id,
        )

    async def start_collaborative_vote(
        self,
        conflict_id: str,
        choices: List[Dict],
    ) -> bool:
        """Start a collaborative vote to resolve a conflict"""
        return await self.conflict_resolver.start_collaborative_vote(
            conflict_id,
            choices,
        )

    async def cast_vote(self, conflict_id: str, user_id: str, choice_id: str) -> bool:
        """Cast a vote for conflict resolution"""
        return await self.conflict_resolver.cast_vote(conflict_id, user_id, choice_id)

    # Session Recording Methods

    async def start_recording(self, session_id: str, user_id: str) -> Optional[Any]:
        """Start recording a collaboration session"""
        recording = await self.session_recorder.start_recording(session_id, user_id)

        if recording:
            # Record the start event
            await self.session_recorder.record_event(
                session_id=session_id,
                event_type=RecordingEventType.USER_JOIN,
                user_id=user_id,
                data={"action": "recording_started", "recording_id": recording.id},
            )

            # Notify participants
            await self.send_notification(
                session_id=session_id,
                notification_type="recording_started",
                title="Recording Started",
                message="Session recording has been started",
                data={"recording_id": recording.id},
            )

        return recording

    async def stop_recording(self, session_id: str, user_id: str) -> bool:
        """Stop recording a session"""
        success = await self.session_recorder.stop_recording(session_id, user_id)

        if success:
            # Notify participants
            await self.send_notification(
                session_id=session_id,
                notification_type="recording_stopped",
                title="Recording Stopped",
                message="Session recording has been stopped and saved",
                data={},
            )

        return success

    async def pause_recording(self, session_id: str, user_id: str) -> bool:
        """Pause recording a session"""
        return await self.session_recorder.pause_recording(session_id, user_id)

    async def resume_recording(self, session_id: str, user_id: str) -> bool:
        """Resume recording a session"""
        return await self.session_recorder.resume_recording(session_id, user_id)

    def get_recordings(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """Get list of available recordings"""
        return self.session_recorder.get_recordings(session_id, user_id)

    async def load_recording(self, recording_id: str) -> Optional[Any]:
        """Load a recording for playback"""
        return await self.session_recorder.load_recording(recording_id)

    async def play_recording(self, recording_id: str) -> bool:
        """Start playback of a recording"""
        success = await self.session_player.load_recording(recording_id)
        if success:
            return await self.session_player.play()
        return False

    async def pause_playback(self) -> bool:
        """Pause recording playback"""
        return await self.session_player.pause()

    async def stop_playback(self) -> bool:
        """Stop recording playback"""
        return await self.session_player.stop()

    def get_playback_info(self) -> Dict:
        """Get current playback information"""
        return self.session_player.get_playback_info()

    # Enhanced Event Recording

    async def record_user_action(
        self,
        session_id: str,
        user: User,
        action: str,
        data: Dict = None,
    ):
        """Record a user action for session recording"""
        if session_id in self.session_recorder.active_recordings:
            await self.session_recorder.record_event(
                session_id=session_id,
                event_type=RecordingEventType.CUSTOM,
                user_id=user.id,
                username=user.username,
                data={"action": action, "user_data": data or {}},
            )

    async def record_analysis_event(
        self,
        session_id: str,
        event_type: str,
        analysis_id: str,
        data: Dict = None,
    ):
        """Record an analysis-related event"""
        if session_id in self.session_recorder.active_recordings:
            event_type_map = {
                "start": RecordingEventType.ANALYSIS_START,
                "progress": RecordingEventType.ANALYSIS_PROGRESS,
                "complete": RecordingEventType.ANALYSIS_COMPLETE,
            }

            recording_event_type = event_type_map.get(
                event_type,
                RecordingEventType.CUSTOM,
            )

            await self.session_recorder.record_event(
                session_id=session_id,
                event_type=recording_event_type,
                data={"analysis_id": analysis_id, "event_data": data or {}},
            )

    def get_collaboration_stats(self) -> Dict:
        """Get comprehensive collaboration statistics"""
        user_stats = self.user_manager.get_user_stats()
        session_stats = self.session_manager.get_session_stats()
        sync_stats = self.state_synchronizer.get_synchronization_stats()
        comm_stats = self.communication_system.get_communication_stats()
        conflict_stats = self.conflict_resolver.get_conflict_stats()
        recording_stats = self.session_recorder.get_recording_stats()

        return {
            "users": user_stats,
            "sessions": session_stats,
            "synchronization": sync_stats,
            "communication": comm_stats,
            "conflicts": conflict_stats,
            "recordings": recording_stats,
            "timestamp": datetime.now().isoformat(),
        }

    async def _background_cleanup(self):
        """Background task for periodic cleanup"""
        while self._running:
            try:
                # Wait for 1 hour
                await asyncio.sleep(3600)

                if not self._running:
                    break

                # Perform cleanup tasks
                self.user_manager.cleanup_expired_sessions()
                self.session_manager.cleanup_inactive_sessions()

                self.logger.info("Performed background cleanup")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {e}")

    # Context manager support

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# Global collaboration manager instance
collaboration_manager: Optional[CollaborationManager] = None


def get_collaboration_manager(
    data_dir: str = "~/.icarus/collaboration",
) -> CollaborationManager:
    """Get or create the global collaboration manager instance"""
    global collaboration_manager

    if collaboration_manager is None:
        collaboration_manager = CollaborationManager(data_dir)

    return collaboration_manager


async def initialize_collaboration(
    data_dir: str = "~/.icarus/collaboration",
) -> CollaborationManager:
    """Initialize and start the collaboration system"""
    manager = get_collaboration_manager(data_dir)
    await manager.start()
    return manager


async def shutdown_collaboration():
    """Shutdown the collaboration system"""
    global collaboration_manager

    if collaboration_manager:
        await collaboration_manager.stop()
        collaboration_manager = None
