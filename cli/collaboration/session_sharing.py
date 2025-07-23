"""
Session Sharing System with Secure Authentication

This module handles collaborative session creation, management, and secure
sharing between multiple users.
"""

import json
import logging
import secrets
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from uuid import uuid4

from .user_manager import Permission
from .user_manager import User
from .user_manager import UserRole


class SessionStatus(str, Enum):
    """Status of a collaboration session"""

    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ARCHIVED = "archived"


class SessionType(str, Enum):
    """Type of collaboration session"""

    ANALYSIS = "analysis"
    WORKFLOW = "workflow"
    DESIGN_REVIEW = "design_review"
    TRAINING = "training"
    GENERAL = "general"


@dataclass
class SessionSettings:
    """Settings for a collaboration session"""

    max_participants: int = 10
    allow_guests: bool = True
    require_approval: bool = False
    auto_save_interval: int = 300  # seconds
    session_timeout: int = 3600  # seconds
    recording_enabled: bool = False
    chat_enabled: bool = True
    annotations_enabled: bool = True
    screen_sharing_enabled: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SessionSettings":
        return cls(**data)


@dataclass
class SessionParticipant:
    """Information about a session participant"""

    user_id: str
    username: str
    display_name: str
    role: UserRole
    joined_at: datetime
    last_active: datetime
    is_online: bool = True
    cursor_position: Optional[Dict] = None
    current_screen: str = "dashboard"

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role.value,
            "joined_at": self.joined_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "is_online": self.is_online,
            "cursor_position": self.cursor_position,
            "current_screen": self.current_screen,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SessionParticipant":
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            display_name=data["display_name"],
            role=UserRole(data["role"]),
            joined_at=datetime.fromisoformat(data["joined_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            is_online=data.get("is_online", True),
            cursor_position=data.get("cursor_position"),
            current_screen=data.get("current_screen", "dashboard"),
        )


@dataclass
class CollaborationSession:
    """A collaborative session with multiple participants"""

    id: str
    name: str
    description: str
    session_type: SessionType
    owner_id: str
    status: SessionStatus
    settings: SessionSettings
    participants: Dict[str, SessionParticipant]
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    last_activity: datetime
    invite_code: str
    shared_state: Dict = None

    def __post_init__(self):
        if self.shared_state is None:
            self.shared_state = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "session_type": self.session_type.value,
            "owner_id": self.owner_id,
            "status": self.status.value,
            "settings": self.settings.to_dict(),
            "participants": {uid: p.to_dict() for uid, p in self.participants.items()},
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "last_activity": self.last_activity.isoformat(),
            "invite_code": self.invite_code,
            "shared_state": self.shared_state,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CollaborationSession":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            session_type=SessionType(data["session_type"]),
            owner_id=data["owner_id"],
            status=SessionStatus(data["status"]),
            settings=SessionSettings.from_dict(data["settings"]),
            participants={
                uid: SessionParticipant.from_dict(p)
                for uid, p in data["participants"].items()
            },
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            ended_at=datetime.fromisoformat(data["ended_at"])
            if data.get("ended_at")
            else None,
            last_activity=datetime.fromisoformat(data["last_activity"]),
            invite_code=data["invite_code"],
            shared_state=data.get("shared_state", {}),
        )

    def is_participant(self, user_id: str) -> bool:
        """Check if user is a participant"""
        return user_id in self.participants

    def get_participant(self, user_id: str) -> Optional[SessionParticipant]:
        """Get participant by user ID"""
        return self.participants.get(user_id)

    def get_online_participants(self) -> List[SessionParticipant]:
        """Get list of online participants"""
        return [p for p in self.participants.values() if p.is_online]

    def can_user_join(self, user: User) -> bool:
        """Check if user can join the session"""
        if self.status != SessionStatus.ACTIVE:
            return False

        if len(self.participants) >= self.settings.max_participants:
            return False

        if user.role == UserRole.GUEST and not self.settings.allow_guests:
            return False

        return True

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()


class SessionManager:
    """Manages collaboration sessions"""

    def __init__(self, data_dir: str = "~/.icarus/collaboration"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sessions_file = self.data_dir / "sessions.json"
        self.active_sessions_file = self.data_dir / "active_sessions.json"

        self.sessions: Dict[str, CollaborationSession] = {}
        self.active_sessions: Dict[str, str] = {}  # invite_code -> session_id

        self.logger = logging.getLogger(__name__)

        # Load existing sessions
        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from storage"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file) as f:
                    data = json.load(f)
                    for session_data in data.get("sessions", []):
                        session = CollaborationSession.from_dict(session_data)
                        self.sessions[session.id] = session
                        if session.status == SessionStatus.ACTIVE:
                            self.active_sessions[session.invite_code] = session.id

                self.logger.info(f"Loaded {len(self.sessions)} sessions")
        except Exception as e:
            self.logger.error(f"Failed to load sessions: {e}")

    def _save_sessions(self):
        """Save sessions to storage"""
        try:
            data = {
                "sessions": [session.to_dict() for session in self.sessions.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.sessions_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save sessions: {e}")

    def _generate_invite_code(self) -> str:
        """Generate a unique invite code"""
        while True:
            code = secrets.token_hex(4).upper()
            if code not in self.active_sessions:
                return code

    def create_session(
        self,
        owner: User,
        name: str,
        description: str = "",
        session_type: SessionType = SessionType.GENERAL,
        settings: Optional[SessionSettings] = None,
    ) -> CollaborationSession:
        """Create a new collaboration session"""
        if not owner.has_permission(Permission.CREATE_SESSION):
            raise PermissionError("User does not have permission to create sessions")

        session_id = str(uuid4())
        invite_code = self._generate_invite_code()

        if settings is None:
            settings = SessionSettings()

        session = CollaborationSession(
            id=session_id,
            name=name,
            description=description,
            session_type=session_type,
            owner_id=owner.id,
            status=SessionStatus.ACTIVE,
            settings=settings,
            participants={},
            created_at=datetime.now(),
            started_at=datetime.now(),
            ended_at=None,
            last_activity=datetime.now(),
            invite_code=invite_code,
        )

        # Add owner as first participant
        owner_participant = SessionParticipant(
            user_id=owner.id,
            username=owner.username,
            display_name=owner.display_name,
            role=owner.role,
            joined_at=datetime.now(),
            last_active=datetime.now(),
        )
        session.participants[owner.id] = owner_participant

        self.sessions[session_id] = session
        self.active_sessions[invite_code] = session_id
        self._save_sessions()

        self.logger.info(
            f"Created session: {name} (ID: {session_id}, Code: {invite_code})",
        )
        return session

    def join_session(
        self,
        invite_code: str,
        user: User,
    ) -> Optional[CollaborationSession]:
        """Join a session using invite code"""
        if invite_code not in self.active_sessions:
            return None

        session_id = self.active_sessions[invite_code]
        session = self.sessions.get(session_id)

        if not session or not session.can_user_join(user):
            return None

        # Check if user is already a participant
        if session.is_participant(user.id):
            # Update participant status
            participant = session.get_participant(user.id)
            participant.is_online = True
            participant.last_active = datetime.now()
        else:
            # Add new participant
            participant = SessionParticipant(
                user_id=user.id,
                username=user.username,
                display_name=user.display_name,
                role=user.role,
                joined_at=datetime.now(),
                last_active=datetime.now(),
            )
            session.participants[user.id] = participant

        session.update_activity()
        self._save_sessions()

        self.logger.info(f"User {user.username} joined session {session.name}")
        return session

    def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a session"""
        session = self.sessions.get(session_id)
        if not session or not session.is_participant(user_id):
            return False

        participant = session.get_participant(user_id)
        participant.is_online = False
        participant.last_active = datetime.now()

        session.update_activity()
        self._save_sessions()

        self.logger.info(f"User {participant.username} left session {session.name}")
        return True

    def end_session(self, session_id: str, user_id: str) -> bool:
        """End a session (owner only)"""
        session = self.sessions.get(session_id)
        if not session or session.owner_id != user_id:
            return False

        session.status = SessionStatus.ENDED
        session.ended_at = datetime.now()
        session.update_activity()

        # Remove from active sessions
        if session.invite_code in self.active_sessions:
            del self.active_sessions[session.invite_code]

        # Mark all participants as offline
        for participant in session.participants.values():
            participant.is_online = False

        self._save_sessions()

        self.logger.info(f"Session ended: {session.name}")
        return True

    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def get_session_by_invite_code(
        self,
        invite_code: str,
    ) -> Optional[CollaborationSession]:
        """Get session by invite code"""
        if invite_code in self.active_sessions:
            session_id = self.active_sessions[invite_code]
            return self.sessions.get(session_id)
        return None

    def get_user_sessions(self, user_id: str) -> List[CollaborationSession]:
        """Get all sessions where user is a participant"""
        user_sessions = []
        for session in self.sessions.values():
            if session.is_participant(user_id):
                user_sessions.append(session)
        return user_sessions

    def get_active_sessions(self) -> List[CollaborationSession]:
        """Get all active sessions"""
        return [
            session
            for session in self.sessions.values()
            if session.status == SessionStatus.ACTIVE
        ]

    def update_participant_activity(
        self,
        session_id: str,
        user_id: str,
        screen: Optional[str] = None,
        cursor_position: Optional[Dict] = None,
    ) -> bool:
        """Update participant activity"""
        session = self.sessions.get(session_id)
        if not session or not session.is_participant(user_id):
            return False

        participant = session.get_participant(user_id)
        participant.last_active = datetime.now()
        participant.is_online = True

        if screen:
            participant.current_screen = screen

        if cursor_position:
            participant.cursor_position = cursor_position

        session.update_activity()
        return True

    def update_session_state(
        self,
        session_id: str,
        state_key: str,
        state_value,
    ) -> bool:
        """Update shared session state"""
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.shared_state[state_key] = state_value
        session.update_activity()
        return True

    def get_session_state(self, session_id: str, state_key: str = None):
        """Get shared session state"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        if state_key:
            return session.shared_state.get(state_key)
        return session.shared_state

    def cleanup_inactive_sessions(self, timeout_hours: int = 24):
        """Clean up inactive sessions"""
        cutoff_time = datetime.now() - timedelta(hours=timeout_hours)
        inactive_sessions = []

        for session in self.sessions.values():
            if (
                session.status == SessionStatus.ACTIVE
                and session.last_activity < cutoff_time
            ):
                inactive_sessions.append(session)

        for session in inactive_sessions:
            session.status = SessionStatus.ENDED
            session.ended_at = datetime.now()

            # Remove from active sessions
            if session.invite_code in self.active_sessions:
                del self.active_sessions[session.invite_code]

            # Mark all participants as offline
            for participant in session.participants.values():
                participant.is_online = False

        if inactive_sessions:
            self._save_sessions()
            self.logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")

    def get_session_stats(self) -> Dict:
        """Get session statistics"""
        active_sessions = self.get_active_sessions()
        total_participants = sum(len(s.participants) for s in active_sessions)
        online_participants = sum(
            len(s.get_online_participants()) for s in active_sessions
        )

        session_types = {}
        for session_type in SessionType:
            session_types[session_type.value] = len(
                [s for s in active_sessions if s.session_type == session_type],
            )

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "total_participants": total_participants,
            "online_participants": online_participants,
            "session_types": session_types,
            "average_participants_per_session": (
                total_participants / len(active_sessions) if active_sessions else 0
            ),
        }
