"""
Test Real-time Collaboration Features

This module tests the implementation of real-time collaboration features including
WebSocket-based updates, conflict resolution, notifications, and session recording.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from api.websocket import RealTimeEventType
from api.websocket import RealTimeUpdate
from api.websocket import WebSocketManager
from collaboration.collaboration_manager import CollaborationManager
from collaboration.conflict_resolution import ConflictType
from collaboration.conflict_resolution import ResolutionStrategy
from collaboration.session_sharing import SessionType
from collaboration.user_manager import User
from collaboration.user_manager import UserRole


class TestRealTimeCollaboration:
    """Test suite for real-time collaboration features"""

    @pytest.fixture
    async def collaboration_manager(self, tmp_path):
        """Create a collaboration manager for testing"""
        manager = CollaborationManager(str(tmp_path / "collaboration"))
        await manager.start()
        yield manager
        await manager.stop()

    @pytest.fixture
    def test_users(self):
        """Create test users"""
        owner = User(
            id="user1",
            username="owner",
            email="owner@test.com",
            display_name="Session Owner",
            role=UserRole.ADMIN,
            permissions=set(),
            created_at=datetime.now(),
            last_active=datetime.now(),
        )

        collaborator = User(
            id="user2",
            username="collaborator",
            email="collab@test.com",
            display_name="Collaborator",
            role=UserRole.COLLABORATOR,
            permissions=set(),
            created_at=datetime.now(),
            last_active=datetime.now(),
        )

        return owner, collaborator

    @pytest.fixture
    async def test_session(self, collaboration_manager, test_users):
        """Create a test collaboration session"""
        owner, _ = test_users
        session = collaboration_manager.create_session(
            owner=owner,
            name="Test Session",
            description="Test collaboration session",
            session_type=SessionType.ANALYSIS,
        )
        return session

    async def test_websocket_real_time_updates(self):
        """Test WebSocket-based real-time updates"""
        websocket_manager = WebSocketManager()

        # Create mock WebSocket connections
        mock_websocket1 = Mock()
        mock_websocket2 = Mock()

        # Add connections
        conn1 = await websocket_manager.add_connection(mock_websocket1, "session1")
        conn2 = await websocket_manager.add_connection(mock_websocket2, "session2")

        # Authenticate connections
        await websocket_manager.authenticate_connection("session1", "user1")
        await websocket_manager.authenticate_connection("session2", "user2")

        # Join collaboration room
        await websocket_manager.join_collaboration_room("session1", "room1")
        await websocket_manager.join_collaboration_room("session2", "room1")

        # Create real-time update
        update = RealTimeUpdate(
            id="update1",
            event_type=RealTimeEventType.CURSOR_MOVE,
            user_id="user1",
            session_id="room1",
            timestamp=datetime.now(),
            data={"x": 100, "y": 200},
        )

        # Send update
        with (
            patch.object(conn1, "send_message", new_callable=AsyncMock) as mock_send1,
            patch.object(conn2, "send_message", new_callable=AsyncMock) as mock_send2,
        ):
            await websocket_manager.send_real_time_update(update)

            # Verify update was sent to other participants (excluding originator)
            mock_send2.assert_called_once()
            mock_send1.assert_not_called()  # Originator excluded

        # Clean up
        await websocket_manager.remove_connection("session1")
        await websocket_manager.remove_connection("session2")

    async def test_conflict_detection_and_resolution(
        self,
        collaboration_manager,
        test_users,
        test_session,
    ):
        """Test conflict detection and resolution system"""
        owner, collaborator = test_users
        session = test_session

        # Join collaborator to session
        await collaboration_manager.join_session(session.invite_code, collaborator)

        # Create conflicting changes
        from collaboration.conflict_resolution import ConflictingChange

        change1 = ConflictingChange(
            id="change1",
            user_id=owner.id,
            username=owner.username,
            timestamp=datetime.now(),
            change_type="parameter_change",
            path="analysis.mesh_density",
            old_value=0.1,
            new_value=0.2,
        )

        change2 = ConflictingChange(
            id="change2",
            user_id=collaborator.id,
            username=collaborator.username,
            timestamp=datetime.now(),
            change_type="parameter_change",
            path="analysis.mesh_density",
            old_value=0.1,
            new_value=0.3,
        )

        # Detect conflict
        conflict = await collaboration_manager.conflict_resolver.detect_conflict(
            session.id,
            [change1, change2],
        )

        assert conflict is not None
        assert conflict.conflict_type == ConflictType.PARAMETER_CONFLICT
        assert len(conflict.changes) == 2

        # Resolve conflict automatically
        success = await collaboration_manager.resolve_conflict_manually(
            conflict.id,
            ResolutionStrategy.LAST_WRITER_WINS,
            owner.id,
        )

        assert success

        # Verify conflict is resolved
        resolved_conflicts = collaboration_manager.get_resolved_conflicts(session.id)
        assert len(resolved_conflicts) == 1
        assert resolved_conflicts[0].id == conflict.id

    async def test_collaborative_voting(
        self,
        collaboration_manager,
        test_users,
        test_session,
    ):
        """Test collaborative voting for conflict resolution"""
        owner, collaborator = test_users
        session = test_session

        # Join collaborator to session
        await collaboration_manager.join_session(session.invite_code, collaborator)

        # Create a conflict
        from collaboration.conflict_resolution import Conflict
        from collaboration.conflict_resolution import ConflictingChange
        from collaboration.conflict_resolution import ConflictSeverity
        from collaboration.conflict_resolution import ConflictType

        changes = [
            ConflictingChange(
                id="change1",
                user_id=owner.id,
                username=owner.username,
                timestamp=datetime.now(),
                change_type="parameter_change",
                path="analysis.solver",
                old_value="xfoil",
                new_value="avl",
            ),
            ConflictingChange(
                id="change2",
                user_id=collaborator.id,
                username=collaborator.username,
                timestamp=datetime.now(),
                change_type="parameter_change",
                path="analysis.solver",
                old_value="xfoil",
                new_value="genuvp",
            ),
        ]

        conflict = Conflict(
            id="conflict1",
            session_id=session.id,
            conflict_type=ConflictType.PARAMETER_CONFLICT,
            severity=ConflictSeverity.MEDIUM,
            path="analysis.solver",
            description="Solver selection conflict",
            changes=changes,
            detected_at=datetime.now(),
        )

        # Add to active conflicts
        collaboration_manager.conflict_resolver.active_conflicts[session.id] = [
            conflict,
        ]

        # Start collaborative vote
        choices = [
            {"id": "choice1", "label": "Use AVL", "value": "avl"},
            {"id": "choice2", "label": "Use GenuVP", "value": "genuvp"},
        ]

        success = await collaboration_manager.start_collaborative_vote(
            conflict.id,
            choices,
        )
        assert success

        # Cast votes
        vote1_success = await collaboration_manager.cast_vote(
            conflict.id,
            owner.id,
            "choice1",
        )
        vote2_success = await collaboration_manager.cast_vote(
            conflict.id,
            collaborator.id,
            "choice2",
        )

        assert vote1_success
        assert vote2_success

        # Verify votes were recorded
        assert conflict.votes[owner.id] == "choice1"
        assert conflict.votes[collaborator.id] == "choice2"

    async def test_session_recording_and_playback(
        self,
        collaboration_manager,
        test_users,
        test_session,
    ):
        """Test session recording and playback functionality"""
        owner, collaborator = test_users
        session = test_session

        # Start recording
        recording = await collaboration_manager.start_recording(session.id, owner.id)
        assert recording is not None
        assert recording.session_id == session.id
        assert recording.status.value == "recording"

        # Join collaborator to session (should be recorded)
        await collaboration_manager.join_session(session.invite_code, collaborator)

        # Record some events
        await collaboration_manager.record_user_action(
            session.id,
            owner,
            "parameter_change",
            {"parameter": "mesh_density", "value": 0.2},
        )

        await collaboration_manager.record_analysis_event(
            session.id,
            "start",
            "analysis1",
            {"solver": "xfoil", "airfoil": "naca0012"},
        )

        # Wait a bit for events to be recorded
        await asyncio.sleep(0.1)

        # Stop recording
        success = await collaboration_manager.stop_recording(session.id, owner.id)
        assert success

        # Get recordings list
        recordings = collaboration_manager.get_recordings(session_id=session.id)
        assert len(recordings) >= 1

        # Load and play recording
        loaded_recording = await collaboration_manager.load_recording(recording.id)
        assert loaded_recording is not None
        assert len(loaded_recording.events) > 0

        # Test playback
        playback_success = await collaboration_manager.play_recording(recording.id)
        assert playback_success

        # Get playback info
        playback_info = collaboration_manager.get_playback_info()
        assert playback_info["recording_id"] == recording.id
        assert playback_info["is_playing"] == True

        # Stop playback
        stop_success = await collaboration_manager.stop_playback()
        assert stop_success

    async def test_notification_system(
        self,
        collaboration_manager,
        test_users,
        test_session,
    ):
        """Test notification system"""
        owner, collaborator = test_users
        session = test_session

        # Join collaborator to session
        await collaboration_manager.join_session(session.invite_code, collaborator)

        # Mock WebSocket manager to capture notifications
        with patch("cli.api.websocket.websocket_manager") as mock_ws_manager:
            mock_ws_manager.send_notification = AsyncMock()

            # Send notification
            await collaboration_manager.send_notification(
                session_id=session.id,
                notification_type="test_notification",
                title="Test Notification",
                message="This is a test notification",
                data={"test_key": "test_value"},
            )

            # Verify notification was sent
            mock_ws_manager.send_notification.assert_called_once_with(
                session.id,
                "test_notification",
                "Test Notification",
                "This is a test notification",
                {"test_key": "test_value"},
            )

    async def test_real_time_state_synchronization(
        self,
        collaboration_manager,
        test_users,
        test_session,
    ):
        """Test real-time state synchronization"""
        owner, collaborator = test_users
        session = test_session

        # Join collaborator to session
        await collaboration_manager.join_session(session.invite_code, collaborator)

        # Sync state change
        success = await collaboration_manager.sync_state_change(
            session_id=session.id,
            user=owner,
            change_type="PARAMETER_CHANGE",
            path="analysis.mesh_density",
            old_value=0.1,
            new_value=0.2,
            metadata={"source": "ui_input"},
        )

        assert success

        # Get session state
        session_state = collaboration_manager.get_session_state(session.id)
        assert session_state is not None
        assert "analysis" in session_state
        assert session_state["analysis"]["mesh_density"] == 0.2

        # Get state history
        state_history = collaboration_manager.get_state_history(session.id)
        assert len(state_history) > 0

        # Verify the change is in history
        latest_change = state_history[-1]
        assert latest_change.path == "analysis.mesh_density"
        assert latest_change.new_value == 0.2
        assert latest_change.user_id == owner.id

    async def test_comprehensive_statistics(
        self,
        collaboration_manager,
        test_users,
        test_session,
    ):
        """Test comprehensive collaboration statistics"""
        owner, collaborator = test_users
        session = test_session

        # Join collaborator and perform some actions
        await collaboration_manager.join_session(session.invite_code, collaborator)

        # Send some messages
        await collaboration_manager.send_message(session.id, owner, "Hello!")
        await collaboration_manager.send_message(session.id, collaborator, "Hi there!")

        # Create an annotation
        await collaboration_manager.create_annotation(
            session.id,
            owner,
            "This needs attention",
            "NOTE",
            {"type": "parameter"},
            {"x": 100, "y": 200},
        )

        # Start recording
        await collaboration_manager.start_recording(session.id, owner.id)
        await asyncio.sleep(0.1)
        await collaboration_manager.stop_recording(session.id, owner.id)

        # Get comprehensive stats
        stats = collaboration_manager.get_collaboration_stats()

        # Verify stats structure
        assert "users" in stats
        assert "sessions" in stats
        assert "synchronization" in stats
        assert "communication" in stats
        assert "conflicts" in stats
        assert "recordings" in stats
        assert "timestamp" in stats

        # Verify some basic counts
        assert stats["sessions"]["active_sessions"] >= 1
        assert stats["communication"]["total_messages"] >= 2
        assert stats["communication"]["total_annotations"] >= 1
        assert stats["recordings"]["total_recordings"] >= 1


async def test_integration_scenario():
    """Test a complete integration scenario"""
    # Create collaboration manager
    manager = CollaborationManager("/tmp/test_collaboration")
    await manager.start()

    try:
        # Create users
        from collaboration.user_manager import Permission

        owner = manager.create_user(
            "owner",
            "owner@test.com",
            "Owner",
            "password123",
            UserRole.ADMIN,
        )
        collab1 = manager.create_user(
            "collab1",
            "collab1@test.com",
            "Collaborator 1",
            "password123",
            UserRole.COLLABORATOR,
        )
        collab2 = manager.create_user(
            "collab2",
            "collab2@test.com",
            "Collaborator 2",
            "password123",
            UserRole.COLLABORATOR,
        )

        # Set permissions
        owner.permissions.add(Permission.CREATE_SESSION)
        owner.permissions.add(Permission.SEND_MESSAGES)
        owner.permissions.add(Permission.CREATE_ANNOTATIONS)

        collab1.permissions.add(Permission.SEND_MESSAGES)
        collab1.permissions.add(Permission.CREATE_ANNOTATIONS)

        collab2.permissions.add(Permission.SEND_MESSAGES)
        collab2.permissions.add(Permission.CREATE_ANNOTATIONS)

        # Create session
        session = manager.create_session(
            owner=owner,
            name="Integration Test Session",
            description="Testing full collaboration workflow",
        )

        # Start recording
        recording = await manager.start_recording(session.id, owner.id)
        assert recording is not None

        # Join collaborators
        await manager.join_session(session.invite_code, collab1)
        await manager.join_session(session.invite_code, collab2)

        # Simulate collaborative work
        await manager.send_message(session.id, owner, "Let's start the analysis")
        await manager.send_message(session.id, collab1, "I'll handle the mesh settings")
        await manager.send_message(session.id, collab2, "I'll configure the solver")

        # Simulate parameter changes that might conflict
        await manager.sync_state_change(
            session.id,
            collab1,
            "PARAMETER_CHANGE",
            "mesh.density",
            0.1,
            0.2,
        )

        await manager.sync_state_change(
            session.id,
            collab2,
            "PARAMETER_CHANGE",
            "mesh.density",
            0.1,
            0.3,
        )

        # Start analysis
        await manager.start_collaborative_analysis(
            session.id,
            owner,
            {"id": "test_analysis", "name": "NACA 0012 Analysis", "solver": "xfoil"},
        )

        # Update progress
        await manager.update_analysis_progress(
            session.id,
            "test_analysis",
            0.5,
            "Mesh generation complete",
        )
        await manager.update_analysis_progress(
            session.id,
            "test_analysis",
            1.0,
            "Analysis complete",
        )

        # Share results
        await manager.share_analysis_result(
            session.id,
            owner,
            "test_analysis",
            {"cl": 0.8, "cd": 0.02, "cm": -0.1},
        )

        # Create annotations
        await manager.create_annotation(
            session.id,
            collab1,
            "Great results!",
            "NOTE",
            {"analysis": "test_analysis"},
            {"x": 0, "y": 0},
        )

        # Stop recording
        await manager.stop_recording(session.id, owner.id)

        # Get final statistics
        stats = manager.get_collaboration_stats()

        print("Integration Test Results:")
        print(f"- Active sessions: {stats['sessions']['active_sessions']}")
        print(f"- Total messages: {stats['communication']['total_messages']}")
        print(f"- Total annotations: {stats['communication']['total_annotations']}")
        print(f"- Total recordings: {stats['recordings']['total_recordings']}")
        print(f"- State changes: {stats['synchronization']['total_state_changes']}")

        # Verify minimum expected activity
        assert stats["sessions"]["active_sessions"] >= 1
        assert stats["communication"]["total_messages"] >= 3
        assert stats["communication"]["total_annotations"] >= 1
        assert stats["recordings"]["total_recordings"] >= 1

        print("✅ Integration test passed!")

    finally:
        await manager.stop()


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_integration_scenario())
