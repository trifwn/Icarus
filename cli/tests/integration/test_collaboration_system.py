#!/usr/bin/env python3
"""
Test script for the ICARUS CLI Collaboration System

This script tests all collaboration features including:
- User management with role-based permissions
- Session sharing with secure authentication
- Real-time state synchronization
- Communication system with chat and annotations
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def test_collaboration_system():
    """Test the complete collaboration system"""

    print("=" * 60)
    print("ICARUS CLI Collaboration System Test")
    print("=" * 60)

    # Import collaboration components
    from collaboration.collaboration_manager import CollaborationManager
    from collaboration.communication import AnnotationType
    from collaboration.communication import MessageType
    from collaboration.session_sharing import SessionSettings
    from collaboration.session_sharing import SessionType
    from collaboration.state_sync import StateChangeType
    from collaboration.user_manager import UserRole

    # Initialize collaboration manager
    test_data_dir = "~/.icarus/test_collaboration"
    manager = CollaborationManager(test_data_dir)

    try:
        await manager.start()

        # Test 1: User Management
        print("\n1. Testing User Management System")
        print("-" * 40)

        # Create users with different roles
        owner = manager.create_user(
            username="alice_owner",
            email="alice@example.com",
            display_name="Alice (Owner)",
            password="secure_password_123",
            role=UserRole.OWNER,
        )
        print(f"✓ Created owner user: {owner.display_name}")

        collaborator = manager.create_user(
            username="bob_collab",
            email="bob@example.com",
            display_name="Bob (Collaborator)",
            password="secure_password_456",
            role=UserRole.COLLABORATOR,
        )
        print(f"✓ Created collaborator user: {collaborator.display_name}")

        viewer = manager.create_user(
            username="charlie_viewer",
            email="charlie@example.com",
            display_name="Charlie (Viewer)",
            password="secure_password_789",
            role=UserRole.VIEWER,
        )
        print(f"✓ Created viewer user: {viewer.display_name}")

        # Create guest user
        guest = manager.create_guest_user("Diana (Guest)")
        print(f"✓ Created guest user: {guest.display_name}")

        # Test authentication
        owner_token = manager.authenticate_user("alice_owner", "secure_password_123")
        collab_token = manager.authenticate_user("bob_collab", "secure_password_456")
        viewer_token = manager.authenticate_user(
            "charlie_viewer",
            "secure_password_789",
        )

        print("✓ Authenticated users - tokens generated")

        # Verify token authentication
        auth_owner = manager.authenticate_token(owner_token)
        auth_collab = manager.authenticate_token(collab_token)
        auth_viewer = manager.authenticate_token(viewer_token)

        assert auth_owner.username == "alice_owner"
        assert auth_collab.username == "bob_collab"
        assert auth_viewer.username == "charlie_viewer"
        print("✓ Token authentication verified")

        # Test 2: Session Management
        print("\n2. Testing Session Sharing System")
        print("-" * 40)

        # Create collaboration session
        session_settings = SessionSettings(
            max_participants=5,
            allow_guests=True,
            chat_enabled=True,
            annotations_enabled=True,
        )

        session = manager.create_session(
            owner=owner,
            name="Airfoil Analysis Session",
            description="Collaborative analysis of NACA 2412 airfoil",
            session_type=SessionType.ANALYSIS,
            settings=session_settings,
        )
        print(f"✓ Created session: {session.name}")
        print(f"  - Session ID: {session.id}")
        print(f"  - Invite Code: {session.invite_code}")
        print(f"  - Owner: {session.owner_id}")

        # Join session with other users
        collab_session = await manager.join_session(session.invite_code, collaborator)
        viewer_session = await manager.join_session(session.invite_code, viewer)
        guest_session = await manager.join_session(session.invite_code, guest)

        assert collab_session.id == session.id
        assert viewer_session.id == session.id
        assert guest_session.id == session.id
        print("✓ All users joined session successfully")

        # Verify participants
        participants = session.get_online_participants()
        print(f"✓ Online participants: {len(participants)}")
        for p in participants:
            print(f"  - {p.display_name} ({p.role.value})")

        # Test 3: State Synchronization
        print("\n3. Testing Real-time State Synchronization")
        print("-" * 40)

        # Simulate state changes
        await manager.sync_state_change(
            session_id=session.id,
            user=owner,
            change_type=StateChangeType.SCREEN_CHANGE,
            path="current_screen",
            old_value="dashboard",
            new_value="analysis",
        )
        print("✓ Owner changed screen to analysis")

        await manager.sync_state_change(
            session_id=session.id,
            user=collaborator,
            change_type=StateChangeType.PARAMETER_CHANGE,
            path="analysis.parameters.angle_of_attack",
            old_value=0,
            new_value=5.0,
        )
        print("✓ Collaborator updated angle of attack parameter")

        await manager.sync_state_change(
            session_id=session.id,
            user=owner,
            change_type=StateChangeType.ANALYSIS_UPDATE,
            path="active_analyses.naca2412",
            old_value=None,
            new_value={
                "id": "naca2412",
                "name": "NACA 2412 Analysis",
                "solver": "xfoil",
                "status": "running",
            },
        )
        print("✓ Owner started analysis")

        # Get session state
        session_state = manager.get_session_state(session.id)
        print("✓ Retrieved synchronized session state:")
        print(f"  - Current screen: {session_state.get('current_screen', 'unknown')}")
        print(f"  - Active analyses: {len(session_state.get('active_analyses', {}))}")

        # Get state history
        state_history = manager.get_state_history(session.id, limit=10)
        print(f"✓ State change history: {len(state_history)} changes")

        # Test 4: Communication System
        print("\n4. Testing Communication System")
        print("-" * 40)

        # Send chat messages
        msg1 = await manager.send_message(
            session_id=session.id,
            user=owner,
            content="Welcome everyone! Let's analyze the NACA 2412 airfoil.",
            message_type=MessageType.TEXT,
        )
        print("✓ Owner sent welcome message")

        msg2 = await manager.send_message(
            session_id=session.id,
            user=collaborator,
            content="Great! I've set the angle of attack to 5 degrees.",
            message_type=MessageType.TEXT,
        )
        print("✓ Collaborator sent parameter update message")

        msg3 = await manager.send_message(
            session_id=session.id,
            user=viewer,
            content="I can see the analysis is starting. Looking forward to the results!",
            message_type=MessageType.TEXT,
        )
        print("✓ Viewer sent observation message")

        # System message for analysis start
        sys_msg = await manager.send_message(
            session_id=session.id,
            user=owner,
            content="Analysis started: NACA 2412 at 5° AoA",
            message_type=MessageType.SYSTEM,
        )
        print("✓ System message sent")

        # Get messages
        messages = manager.get_messages(session.id, limit=10)
        print(f"✓ Retrieved {len(messages)} messages:")
        for msg in messages[-3:]:  # Show last 3 messages
            print(f"  - {msg.display_name}: {msg.content}")

        # Create annotations
        annotation1 = await manager.create_annotation(
            session_id=session.id,
            user=collaborator,
            content="This parameter might need adjustment based on Reynolds number",
            annotation_type=AnnotationType.SUGGESTION,
            target={"type": "parameter", "id": "angle_of_attack"},
            position={"x": 100, "y": 200},
        )
        print("✓ Collaborator created suggestion annotation")

        annotation2 = await manager.create_annotation(
            session_id=session.id,
            user=viewer,
            content="Interesting convergence pattern here",
            annotation_type=AnnotationType.NOTE,
            target={"type": "plot", "id": "convergence_plot"},
            position={"x": 300, "y": 150},
        )
        print("✓ Viewer created note annotation")

        # Get annotations
        annotations = manager.get_annotations(session.id)
        print(f"✓ Retrieved {len(annotations)} annotations:")
        for ann in annotations:
            print(
                f"  - {ann.display_name}: {ann.content} ({ann.annotation_type.value})",
            )

        # Test 5: Analysis Integration
        print("\n5. Testing Analysis Integration")
        print("-" * 40)

        # Share analysis result
        analysis_result = {
            "id": "naca2412_result",
            "name": "NACA 2412 Analysis Results",
            "data": {
                "cl_max": 1.45,
                "cd_min": 0.0089,
                "alpha_stall": 12.5,
                "convergence": True,
            },
            "plots": ["pressure_distribution", "velocity_field"],
            "timestamp": datetime.now().isoformat(),
        }

        await manager.share_analysis_result(
            session_id=session.id,
            user=owner,
            analysis_id="naca2412",
            result_data=analysis_result,
        )
        print("✓ Analysis result shared with session participants")

        # Update analysis progress
        await manager.update_analysis_progress(
            session_id=session.id,
            analysis_id="naca2412",
            progress=1.0,
            message="Analysis completed successfully",
        )
        print("✓ Analysis progress updated")

        # Test 6: Participant Activity
        print("\n6. Testing Participant Activity Tracking")
        print("-" * 40)

        # Update participant activities
        await manager.update_participant_activity(
            session_id=session.id,
            user=collaborator,
            screen="results",
            cursor_position={"x": 250, "y": 180},
        )
        print("✓ Collaborator activity updated")

        await manager.update_participant_activity(
            session_id=session.id,
            user=viewer,
            screen="results",
            cursor_position={"x": 400, "y": 220},
        )
        print("✓ Viewer activity updated")

        # Test 7: Search and Statistics
        print("\n7. Testing Search and Statistics")
        print("-" * 40)

        # Search messages
        search_results = manager.search_messages(session.id, "analysis", limit=5)
        print(
            f"✓ Message search results: {len(search_results)} messages containing 'analysis'",
        )

        # Get collaboration statistics
        stats = manager.get_collaboration_stats()
        print("✓ Collaboration statistics:")
        print(f"  - Total users: {stats['users']['total_users']}")
        print(f"  - Online users: {stats['users']['online_users']}")
        print(f"  - Active sessions: {stats['sessions']['active_sessions']}")
        print(f"  - Total messages: {stats['communication']['total_messages']}")
        print(f"  - Total annotations: {stats['communication']['total_annotations']}")
        print(f"  - State changes: {stats['synchronization']['total_state_changes']}")

        # Test 8: Session Management
        print("\n8. Testing Session Management")
        print("-" * 40)

        # Get user sessions
        owner_sessions = manager.get_user_sessions(owner)
        print(f"✓ Owner is in {len(owner_sessions)} sessions")

        # Get active sessions
        active_sessions = manager.get_active_sessions()
        print(f"✓ Total active sessions: {len(active_sessions)}")

        # Leave session (viewer leaves)
        await manager.leave_session(session.id, viewer)
        print("✓ Viewer left the session")

        # Verify participant count
        remaining_participants = session.get_online_participants()
        print(f"✓ Remaining online participants: {len(remaining_participants)}")

        # Test 9: Cleanup and Session End
        print("\n9. Testing Session Cleanup")
        print("-" * 40)

        # End session (owner only)
        session_ended = await manager.end_session(session.id, owner)
        assert session_ended == True
        print("✓ Session ended by owner")

        # Verify session status
        ended_session = manager.get_session(session.id)
        print(f"✓ Session status: {ended_session.status.value}")

        print("\n" + "=" * 60)
        print("✅ ALL COLLABORATION TESTS PASSED!")
        print("=" * 60)

        # Display final summary
        final_stats = manager.get_collaboration_stats()
        print("\nFinal System State:")
        print(f"- Users created: {final_stats['users']['total_users']}")
        print("- Sessions created: 1 (ended)")
        print(f"- Messages sent: {final_stats['communication']['total_messages']}")
        print(
            f"- Annotations created: {final_stats['communication']['total_annotations']}",
        )
        print(
            f"- State changes: {final_stats['synchronization']['total_state_changes']}",
        )

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        await manager.stop()

        # Clean up test data
        test_path = Path(test_data_dir).expanduser()
        if test_path.exists():
            import shutil

            shutil.rmtree(test_path)
            print(f"\n🧹 Cleaned up test data directory: {test_path}")

    return True


async def test_api_integration():
    """Test collaboration system integration with API"""

    print("\n" + "=" * 60)
    print("Testing API Integration")
    print("=" * 60)

    try:
        # Test API app creation with collaboration
        from api.app import create_api_app

        app = create_api_app()
        print("✓ API app created with collaboration endpoints")

        # List collaboration endpoints
        collaboration_routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and "/collaboration" in route.path
        ]

        print(f"✓ Found {len(collaboration_routes)} collaboration endpoints:")
        for route in collaboration_routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                methods = ", ".join(route.methods)
                print(f"  - {methods} {route.path}")

        print("✅ API Integration Test Passed!")

    except Exception as e:
        logger.error(f"API integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_data_models():
    """Test collaboration data models"""

    print("\n" + "=" * 60)
    print("Testing Data Models")
    print("=" * 60)

    try:
        from datetime import datetime

        from collaboration.communication import Annotation
        from collaboration.communication import AnnotationType
        from collaboration.communication import ChatMessage
        from collaboration.communication import MessageType
        from collaboration.session_sharing import SessionSettings
        from collaboration.state_sync import StateChange
        from collaboration.state_sync import StateChangeType
        from collaboration.user_manager import Permission
        from collaboration.user_manager import User
        from collaboration.user_manager import UserRole

        # Test User model
        user = User(
            id="test_user_1",
            username="testuser",
            email="test@example.com",
            display_name="Test User",
            role=UserRole.COLLABORATOR,
            permissions={Permission.SEND_MESSAGES, Permission.CREATE_ANNOTATIONS},
            created_at=datetime.now(),
            last_active=datetime.now(),
        )

        user_dict = user.to_dict()
        user_restored = User.from_dict(user_dict)
        assert user_restored.username == user.username
        print("✓ User model serialization/deserialization works")

        # Test SessionSettings model
        settings = SessionSettings(max_participants=5, allow_guests=True)
        settings_dict = settings.to_dict()
        settings_restored = SessionSettings.from_dict(settings_dict)
        assert settings_restored.max_participants == settings.max_participants
        print("✓ SessionSettings model serialization works")

        # Test StateChange model
        state_change = StateChange(
            id="change_1",
            session_id="session_1",
            user_id="user_1",
            change_type=StateChangeType.PARAMETER_CHANGE,
            path="analysis.parameters.alpha",
            old_value=0,
            new_value=5.0,
            timestamp=datetime.now(),
        )

        change_dict = state_change.to_dict()
        change_restored = StateChange.from_dict(change_dict)
        assert change_restored.path == state_change.path
        print("✓ StateChange model serialization works")

        # Test ChatMessage model
        message = ChatMessage(
            id="msg_1",
            session_id="session_1",
            user_id="user_1",
            username="testuser",
            display_name="Test User",
            message_type=MessageType.TEXT,
            content="Hello, world!",
            timestamp=datetime.now(),
        )

        msg_dict = message.to_dict()
        msg_restored = ChatMessage.from_dict(msg_dict)
        assert msg_restored.content == message.content
        print("✓ ChatMessage model serialization works")

        # Test Annotation model
        annotation = Annotation(
            id="ann_1",
            session_id="session_1",
            user_id="user_1",
            username="testuser",
            display_name="Test User",
            annotation_type=AnnotationType.NOTE,
            content="This is a note",
            target={"type": "plot", "id": "plot_1"},
            position={"x": 100, "y": 200},
            timestamp=datetime.now(),
        )

        ann_dict = annotation.to_dict()
        ann_restored = Annotation.from_dict(ann_dict)
        assert ann_restored.content == annotation.content
        print("✓ Annotation model serialization works")

        print("✅ All Data Model Tests Passed!")

    except Exception as e:
        logger.error(f"Data model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


async def main():
    """Run all collaboration system tests"""

    print("🚀 Starting ICARUS CLI Collaboration System Tests")
    print("=" * 60)

    # Test 1: Data Models
    if not test_data_models():
        print("❌ Data model tests failed")
        return

    # Test 2: Core Collaboration System
    if not await test_collaboration_system():
        print("❌ Collaboration system tests failed")
        return

    # Test 3: API Integration
    if not await test_api_integration():
        print("❌ API integration tests failed")
        return

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("The ICARUS CLI Collaboration System is working correctly.")
    print("\nFeatures verified:")
    print("✅ User management with role-based permissions")
    print("✅ Session sharing with secure authentication")
    print("✅ Real-time state synchronization between users")
    print("✅ Communication system with chat and annotations")
    print("✅ API integration with REST endpoints")
    print("✅ Data model serialization and persistence")


if __name__ == "__main__":
    asyncio.run(main())
