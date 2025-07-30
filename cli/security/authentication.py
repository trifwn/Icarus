"""
Authentication Manager for ICARUS CLI Security System.

Handles user authentication, session management, and token validation.
Integrates with the existing user management system and adds enhanced security features.
"""

import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from ..collaboration.user_manager import User
from ..collaboration.user_manager import UserManager
from .crypto import CryptoManager


class AuthMethod(str, Enum):
    """Authentication methods supported by the system."""

    PASSWORD = "password"
    TOKEN = "token"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    LDAP = "ldap"
    OAUTH = "oauth"


class SessionStatus(str, Enum):
    """Session status values."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class AuthSession:
    """Authentication session data."""

    session_id: str
    user_id: str
    auth_method: AuthMethod
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    permissions: List[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["last_accessed"] = self.last_accessed.isoformat()
        data["expires_at"] = self.expires_at.isoformat()
        data["auth_method"] = self.auth_method.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "AuthSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            auth_method=AuthMethod(data["auth_method"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            status=SessionStatus(data.get("status", "active")),
            permissions=data.get("permissions", []),
        )

    def is_valid(self) -> bool:
        """Check if session is valid and not expired."""
        return self.status == SessionStatus.ACTIVE and datetime.now() < self.expires_at

    def refresh(self, extend_hours: int = 24):
        """Refresh session expiration time."""
        self.last_accessed = datetime.now()
        self.expires_at = datetime.now() + timedelta(hours=extend_hours)


@dataclass
class AuthConfig:
    """Authentication configuration."""

    session_timeout_hours: int = 24
    max_sessions_per_user: int = 5
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    enable_2fa: bool = False
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    enable_session_encryption: bool = True


class AuthenticationManager:
    """Manages authentication, sessions, and security policies."""

    def __init__(
        self,
        user_manager: UserManager,
        crypto_manager: CryptoManager,
        data_dir: str = "~/.icarus/security",
    ):
        """
        Initialize authentication manager.

        Args:
            user_manager: User management system
            crypto_manager: Cryptographic operations manager
            data_dir: Directory for security data storage
        """
        self.user_manager = user_manager
        self.crypto_manager = crypto_manager

        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sessions_file = self.data_dir / "sessions.json"
        self.auth_config_file = self.data_dir / "auth_config.json"
        self.failed_attempts_file = self.data_dir / "failed_attempts.json"

        # Configuration
        self.config = AuthConfig()

        # Runtime data
        self.active_sessions: Dict[str, AuthSession] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id

        self.logger = logging.getLogger(__name__)

        # Load existing data
        self._load_sessions()
        self._load_config()
        self._load_failed_attempts()

    def _load_sessions(self):
        """Load active sessions from storage."""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file) as f:
                    data = json.load(f)

                for session_data in data.get("sessions", []):
                    session = AuthSession.from_dict(session_data)
                    if session.is_valid():
                        self.active_sessions[session.session_id] = session

                # Load API keys
                self.api_keys = data.get("api_keys", {})

                self.logger.info(f"Loaded {len(self.active_sessions)} active sessions")
        except Exception as e:
            self.logger.error(f"Failed to load sessions: {e}")

    def _save_sessions(self):
        """Save active sessions to storage."""
        try:
            data = {
                "sessions": [
                    session.to_dict() for session in self.active_sessions.values()
                ],
                "api_keys": self.api_keys,
                "updated_at": datetime.now().isoformat(),
            }

            if self.config.enable_session_encryption:
                # Encrypt session data
                encrypted_data = self.crypto_manager.encrypt_data(json.dumps(data))
                with open(self.sessions_file, "w") as f:
                    json.dump({"encrypted": True, "data": encrypted_data}, f)
            else:
                with open(self.sessions_file, "w") as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save sessions: {e}")

    def _load_config(self):
        """Load authentication configuration."""
        try:
            if self.auth_config_file.exists():
                with open(self.auth_config_file) as f:
                    data = json.load(f)

                for key, value in data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
        except Exception as e:
            self.logger.error(f"Failed to load auth config: {e}")

    def _save_config(self):
        """Save authentication configuration."""
        try:
            with open(self.auth_config_file, "w") as f:
                json.dump(asdict(self.config), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save auth config: {e}")

    def _load_failed_attempts(self):
        """Load failed login attempts."""
        try:
            if self.failed_attempts_file.exists():
                with open(self.failed_attempts_file) as f:
                    data = json.load(f)

                for username, attempts in data.items():
                    self.failed_attempts[username] = [
                        datetime.fromisoformat(attempt) for attempt in attempts
                    ]
        except Exception as e:
            self.logger.error(f"Failed to load failed attempts: {e}")

    def _save_failed_attempts(self):
        """Save failed login attempts."""
        try:
            data = {}
            for username, attempts in self.failed_attempts.items():
                data[username] = [attempt.isoformat() for attempt in attempts]

            with open(self.failed_attempts_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save failed attempts: {e}")

    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[str]:
        """
        Authenticate user with username and password.

        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Session token if authentication successful, None otherwise
        """
        # Check if user is locked out
        if self._is_user_locked_out(username):
            self.logger.warning(f"User {username} is locked out")
            return None

        # Authenticate with user manager
        user = self.user_manager.get_user_by_username(username)
        if not user or not user.is_active:
            self._record_failed_attempt(username)
            return None

        # Verify password
        if not user.password_hash:
            self._record_failed_attempt(username)
            return None

        try:
            salt, stored_hash = user.password_hash.split(":")
            if not self.crypto_manager.verify_password(password, stored_hash, salt):
                self._record_failed_attempt(username)
                return None
        except Exception:
            self._record_failed_attempt(username)
            return None

        # Clear failed attempts on successful login
        if username in self.failed_attempts:
            del self.failed_attempts[username]
            self._save_failed_attempts()

        # Create session
        session = self._create_session(
            user.id,
            AuthMethod.PASSWORD,
            ip_address,
            user_agent,
        )

        self.logger.info(f"User authenticated: {username}")
        return session.session_id

    def authenticate_token(self, token: str) -> Optional[User]:
        """
        Authenticate using session token.

        Args:
            token: Session token

        Returns:
            User object if token is valid, None otherwise
        """
        session = self.active_sessions.get(token)
        if not session or not session.is_valid():
            return None

        # Refresh session
        session.refresh()
        self._save_sessions()

        # Get user
        user = self.user_manager.get_user(session.user_id)
        if not user or not user.is_active:
            self.revoke_session(token)
            return None

        return user

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """
        Authenticate using API key.

        Args:
            api_key: API key

        Returns:
            User object if API key is valid, None otherwise
        """
        user_id = self.api_keys.get(api_key)
        if not user_id:
            return None

        user = self.user_manager.get_user(user_id)
        if not user or not user.is_active:
            # Remove invalid API key
            del self.api_keys[api_key]
            self._save_sessions()
            return None

        return user

    def _create_session(
        self,
        user_id: str,
        auth_method: AuthMethod,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuthSession:
        """Create new authentication session."""
        # Clean up old sessions for user
        self._cleanup_user_sessions(user_id)

        # Create new session
        session = AuthSession(
            session_id=self.crypto_manager.generate_secure_token(),
            user_id=user_id,
            auth_method=auth_method,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now()
            + timedelta(hours=self.config.session_timeout_hours),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.active_sessions[session.session_id] = session
        self._save_sessions()

        return session

    def _cleanup_user_sessions(self, user_id: str):
        """Clean up old sessions for a user."""
        user_sessions = [
            session
            for session in self.active_sessions.values()
            if session.user_id == user_id
        ]

        # Sort by last accessed time
        user_sessions.sort(key=lambda s: s.last_accessed, reverse=True)

        # Keep only the most recent sessions
        sessions_to_remove = user_sessions[self.config.max_sessions_per_user :]

        for session in sessions_to_remove:
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]

    def _is_user_locked_out(self, username: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        if username not in self.failed_attempts:
            return False

        # Clean up old attempts
        cutoff_time = datetime.now() - timedelta(
            minutes=self.config.lockout_duration_minutes,
        )
        recent_attempts = [
            attempt
            for attempt in self.failed_attempts[username]
            if attempt > cutoff_time
        ]

        self.failed_attempts[username] = recent_attempts

        return len(recent_attempts) >= self.config.max_login_attempts

    def _record_failed_attempt(self, username: str):
        """Record a failed login attempt."""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []

        self.failed_attempts[username].append(datetime.now())
        self._save_failed_attempts()

        self.logger.warning(f"Failed login attempt for user: {username}")

    def revoke_session(self, session_id: str) -> bool:
        """
        Revoke a session.

        Args:
            session_id: Session ID to revoke

        Returns:
            True if session was revoked, False if not found
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.status = SessionStatus.REVOKED
            del self.active_sessions[session_id]
            self._save_sessions()

            self.logger.info(f"Session revoked: {session_id}")
            return True

        return False

    def revoke_all_user_sessions(self, user_id: str) -> int:
        """
        Revoke all sessions for a user.

        Args:
            user_id: User ID

        Returns:
            Number of sessions revoked
        """
        sessions_to_revoke = [
            session_id
            for session_id, session in self.active_sessions.items()
            if session.user_id == user_id
        ]

        for session_id in sessions_to_revoke:
            self.revoke_session(session_id)

        return len(sessions_to_revoke)

    def generate_api_key(self, user_id: str) -> str:
        """
        Generate API key for a user.

        Args:
            user_id: User ID

        Returns:
            Generated API key
        """
        api_key = self.crypto_manager.generate_api_key()
        self.api_keys[api_key] = user_id
        self._save_sessions()

        self.logger.info(f"Generated API key for user: {user_id}")
        return api_key

    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.

        Args:
            api_key: API key to revoke

        Returns:
            True if key was revoked, False if not found
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            self._save_sessions()
            self.logger.info(f"API key revoked: {api_key[:10]}...")
            return True

        return False

    def get_user_sessions(self, user_id: str) -> List[AuthSession]:
        """
        Get all active sessions for a user.

        Args:
            user_id: User ID

        Returns:
            List of active sessions
        """
        return [
            session
            for session in self.active_sessions.values()
            if session.user_id == user_id and session.is_valid()
        ]

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        expired_sessions = [
            session_id
            for session_id, session in self.active_sessions.items()
            if not session.is_valid()
        ]

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            self._save_sessions()
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password strength according to policy.

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if len(password) < self.config.password_min_length:
            issues.append(
                f"Password must be at least {self.config.password_min_length} characters",
            )

        if self.config.password_require_uppercase and not any(
            c.isupper() for c in password
        ):
            issues.append("Password must contain at least one uppercase letter")

        if self.config.password_require_numbers and not any(
            c.isdigit() for c in password
        ):
            issues.append("Password must contain at least one number")

        if self.config.password_require_special and not any(
            c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password
        ):
            issues.append("Password must contain at least one special character")

        return len(issues) == 0, issues

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get session information.

        Args:
            session_id: Session ID

        Returns:
            Session information dictionary or None
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        user = self.user_manager.get_user(session.user_id)

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "username": user.username if user else "Unknown",
            "auth_method": session.auth_method.value,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "ip_address": session.ip_address,
            "user_agent": session.user_agent,
            "status": session.status.value,
        }

    def get_auth_stats(self) -> Dict:
        """Get authentication statistics."""
        return {
            "active_sessions": len(self.active_sessions),
            "api_keys": len(self.api_keys),
            "failed_attempts": sum(
                len(attempts) for attempts in self.failed_attempts.values()
            ),
            "locked_users": sum(
                1
                for username in self.failed_attempts.keys()
                if self._is_user_locked_out(username)
            ),
            "config": asdict(self.config),
        }
