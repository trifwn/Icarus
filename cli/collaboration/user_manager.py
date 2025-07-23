"""
User Management System with Role-Based Permissions

This module handles user authentication, authorization, and role-based
access control for collaboration features.
"""

import hashlib
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from uuid import uuid4


class UserRole(str, Enum):
    """User roles with different permission levels"""

    OWNER = "owner"  # Full control over session
    ADMIN = "admin"  # Can manage users and settings
    COLLABORATOR = "collaborator"  # Can edit and run analyses
    VIEWER = "viewer"  # Read-only access
    GUEST = "guest"  # Limited temporary access


class Permission(str, Enum):
    """Specific permissions that can be granted to users"""

    # Session management
    CREATE_SESSION = "create_session"
    DELETE_SESSION = "delete_session"
    MODIFY_SESSION_SETTINGS = "modify_session_settings"

    # User management
    INVITE_USERS = "invite_users"
    REMOVE_USERS = "remove_users"
    CHANGE_USER_ROLES = "change_user_roles"

    # Analysis operations
    RUN_ANALYSIS = "run_analysis"
    MODIFY_ANALYSIS = "modify_analysis"
    DELETE_ANALYSIS = "delete_analysis"

    # Data operations
    IMPORT_DATA = "import_data"
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"

    # Workflow operations
    CREATE_WORKFLOW = "create_workflow"
    MODIFY_WORKFLOW = "modify_workflow"
    EXECUTE_WORKFLOW = "execute_workflow"
    DELETE_WORKFLOW = "delete_workflow"

    # Communication
    SEND_MESSAGES = "send_messages"
    CREATE_ANNOTATIONS = "create_annotations"
    MODERATE_CHAT = "moderate_chat"


@dataclass
class User:
    """User data model"""

    id: str
    username: str
    email: str
    display_name: str
    role: UserRole
    permissions: Set[Permission]
    created_at: datetime
    last_active: datetime
    is_active: bool = True
    session_token: Optional[str] = None
    password_hash: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "is_active": self.is_active,
            "session_token": self.session_token,
            "password_hash": self.password_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "User":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            username=data["username"],
            email=data["email"],
            display_name=data["display_name"],
            role=UserRole(data["role"]),
            permissions={Permission(p) for p in data.get("permissions", [])},
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            is_active=data.get("is_active", True),
            session_token=data.get("session_token"),
            password_hash=data.get("password_hash"),
        )

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        return permission in self.permissions

    def can_perform_action(self, action: str) -> bool:
        """Check if user can perform a specific action"""
        action_permissions = {
            "create_session": [Permission.CREATE_SESSION],
            "delete_session": [Permission.DELETE_SESSION],
            "invite_user": [Permission.INVITE_USERS],
            "remove_user": [Permission.REMOVE_USERS],
            "run_analysis": [Permission.RUN_ANALYSIS],
            "modify_analysis": [Permission.MODIFY_ANALYSIS],
            "create_workflow": [Permission.CREATE_WORKFLOW],
            "execute_workflow": [Permission.EXECUTE_WORKFLOW],
            "send_message": [Permission.SEND_MESSAGES],
            "create_annotation": [Permission.CREATE_ANNOTATIONS],
        }

        required_permissions = action_permissions.get(action, [])
        return any(self.has_permission(perm) for perm in required_permissions)


class UserManager:
    """Manages users, authentication, and permissions"""

    # Default permissions for each role
    ROLE_PERMISSIONS = {
        UserRole.OWNER: {
            Permission.CREATE_SESSION,
            Permission.DELETE_SESSION,
            Permission.MODIFY_SESSION_SETTINGS,
            Permission.INVITE_USERS,
            Permission.REMOVE_USERS,
            Permission.CHANGE_USER_ROLES,
            Permission.RUN_ANALYSIS,
            Permission.MODIFY_ANALYSIS,
            Permission.DELETE_ANALYSIS,
            Permission.IMPORT_DATA,
            Permission.EXPORT_DATA,
            Permission.DELETE_DATA,
            Permission.CREATE_WORKFLOW,
            Permission.MODIFY_WORKFLOW,
            Permission.EXECUTE_WORKFLOW,
            Permission.DELETE_WORKFLOW,
            Permission.SEND_MESSAGES,
            Permission.CREATE_ANNOTATIONS,
            Permission.MODERATE_CHAT,
        },
        UserRole.ADMIN: {
            Permission.MODIFY_SESSION_SETTINGS,
            Permission.INVITE_USERS,
            Permission.REMOVE_USERS,
            Permission.RUN_ANALYSIS,
            Permission.MODIFY_ANALYSIS,
            Permission.DELETE_ANALYSIS,
            Permission.IMPORT_DATA,
            Permission.EXPORT_DATA,
            Permission.CREATE_WORKFLOW,
            Permission.MODIFY_WORKFLOW,
            Permission.EXECUTE_WORKFLOW,
            Permission.DELETE_WORKFLOW,
            Permission.SEND_MESSAGES,
            Permission.CREATE_ANNOTATIONS,
            Permission.MODERATE_CHAT,
        },
        UserRole.COLLABORATOR: {
            Permission.RUN_ANALYSIS,
            Permission.MODIFY_ANALYSIS,
            Permission.IMPORT_DATA,
            Permission.EXPORT_DATA,
            Permission.CREATE_WORKFLOW,
            Permission.MODIFY_WORKFLOW,
            Permission.EXECUTE_WORKFLOW,
            Permission.SEND_MESSAGES,
            Permission.CREATE_ANNOTATIONS,
        },
        UserRole.VIEWER: {
            Permission.EXPORT_DATA,
            Permission.SEND_MESSAGES,
            Permission.CREATE_ANNOTATIONS,
        },
        UserRole.GUEST: {Permission.SEND_MESSAGES},
    }

    def __init__(self, data_dir: str = "~/.icarus/collaboration"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.users_file = self.data_dir / "users.json"
        self.sessions_file = self.data_dir / "user_sessions.json"

        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, str] = {}  # token -> user_id

        self.logger = logging.getLogger(__name__)

        # Load existing users
        self._load_users()

    def _load_users(self):
        """Load users from storage"""
        try:
            if self.users_file.exists():
                with open(self.users_file) as f:
                    data = json.load(f)
                    for user_data in data.get("users", []):
                        user = User.from_dict(user_data)
                        self.users[user.id] = user
                self.logger.info(f"Loaded {len(self.users)} users")
        except Exception as e:
            self.logger.error(f"Failed to load users: {e}")

    def _save_users(self):
        """Save users to storage"""
        try:
            data = {
                "users": [user.to_dict() for user in self.users.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.users_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save users: {e}")

    def _hash_password(self, password: str) -> str:
        """Hash a password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100000,
        )
        return f"{salt}:{password_hash.hex()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        try:
            salt, hash_hex = password_hash.split(":")
            password_hash_check = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode(),
                salt.encode(),
                100000,
            )
            return password_hash_check.hex() == hash_hex
        except Exception:
            return False

    def _generate_session_token(self) -> str:
        """Generate a secure session token"""
        return secrets.token_urlsafe(32)

    def create_user(
        self,
        username: str,
        email: str,
        display_name: str,
        password: str,
        role: UserRole = UserRole.COLLABORATOR,
    ) -> User:
        """Create a new user"""
        # Check if username or email already exists
        for user in self.users.values():
            if user.username == username:
                raise ValueError(f"Username '{username}' already exists")
            if user.email == email:
                raise ValueError(f"Email '{email}' already exists")

        user = User(
            id=str(uuid4()),
            username=username,
            email=email,
            display_name=display_name,
            role=role,
            permissions=self.ROLE_PERMISSIONS.get(role, set()),
            created_at=datetime.now(),
            last_active=datetime.now(),
            password_hash=self._hash_password(password),
        )

        self.users[user.id] = user
        self._save_users()

        self.logger.info(f"Created user: {username} ({role.value})")
        return user

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token"""
        for user in self.users.values():
            if user.username == username and user.is_active:
                if user.password_hash and self._verify_password(
                    password,
                    user.password_hash,
                ):
                    # Generate session token
                    token = self._generate_session_token()
                    user.session_token = token
                    user.last_active = datetime.now()

                    self.active_sessions[token] = user.id
                    self._save_users()

                    self.logger.info(f"User authenticated: {username}")
                    return token

        self.logger.warning(f"Authentication failed for: {username}")
        return None

    def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate using session token"""
        if token in self.active_sessions:
            user_id = self.active_sessions[token]
            user = self.users.get(user_id)
            if user and user.is_active and user.session_token == token:
                user.last_active = datetime.now()
                return user

        return None

    def logout_user(self, token: str) -> bool:
        """Logout user by invalidating session token"""
        if token in self.active_sessions:
            user_id = self.active_sessions[token]
            user = self.users.get(user_id)
            if user:
                user.session_token = None
                del self.active_sessions[token]
                self._save_users()
                self.logger.info(f"User logged out: {user.username}")
                return True

        return False

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def update_user_role(
        self,
        user_id: str,
        new_role: UserRole,
        admin_user_id: str,
    ) -> bool:
        """Update user role (requires admin permissions)"""
        admin_user = self.users.get(admin_user_id)
        if not admin_user or not admin_user.has_permission(
            Permission.CHANGE_USER_ROLES,
        ):
            return False

        user = self.users.get(user_id)
        if not user:
            return False

        user.role = new_role
        user.permissions = self.ROLE_PERMISSIONS.get(new_role, set())
        self._save_users()

        self.logger.info(f"Updated user role: {user.username} -> {new_role.value}")
        return True

    def deactivate_user(self, user_id: str, admin_user_id: str) -> bool:
        """Deactivate a user (requires admin permissions)"""
        admin_user = self.users.get(admin_user_id)
        if not admin_user or not admin_user.has_permission(Permission.REMOVE_USERS):
            return False

        user = self.users.get(user_id)
        if not user:
            return False

        user.is_active = False
        if user.session_token and user.session_token in self.active_sessions:
            del self.active_sessions[user.session_token]
        user.session_token = None

        self._save_users()

        self.logger.info(f"Deactivated user: {user.username}")
        return True

    def get_active_users(self) -> List[User]:
        """Get all active users"""
        return [user for user in self.users.values() if user.is_active]

    def get_online_users(self) -> List[User]:
        """Get currently online users (with active sessions)"""
        online_user_ids = set(self.active_sessions.values())
        return [
            user
            for user in self.users.values()
            if user.id in online_user_ids and user.is_active
        ]

    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired session tokens"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_tokens = []

        for token, user_id in self.active_sessions.items():
            user = self.users.get(user_id)
            if not user or user.last_active < cutoff_time:
                expired_tokens.append(token)

        for token in expired_tokens:
            user_id = self.active_sessions[token]
            user = self.users.get(user_id)
            if user:
                user.session_token = None
            del self.active_sessions[token]

        if expired_tokens:
            self._save_users()
            self.logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")

    def create_guest_user(self, display_name: str) -> User:
        """Create a temporary guest user"""
        guest_id = f"guest_{secrets.token_hex(8)}"

        user = User(
            id=guest_id,
            username=guest_id,
            email="",
            display_name=display_name,
            role=UserRole.GUEST,
            permissions=self.ROLE_PERMISSIONS[UserRole.GUEST],
            created_at=datetime.now(),
            last_active=datetime.now(),
            session_token=self._generate_session_token(),
        )

        # Don't save guest users to persistent storage
        self.active_sessions[user.session_token] = user.id

        self.logger.info(f"Created guest user: {display_name}")
        return user

    def get_user_stats(self) -> Dict:
        """Get user statistics"""
        active_users = self.get_active_users()
        online_users = self.get_online_users()

        role_counts = {}
        for role in UserRole:
            role_counts[role.value] = len([u for u in active_users if u.role == role])

        return {
            "total_users": len(self.users),
            "active_users": len(active_users),
            "online_users": len(online_users),
            "role_distribution": role_counts,
            "active_sessions": len(self.active_sessions),
        }
