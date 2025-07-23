"""
Collaboration System for ICARUS CLI

This module provides real-time collaboration features including:
- User management with role-based permissions
- Session sharing with secure authentication
- Real-time state synchronization
- Communication system with chat and annotations
"""

from .communication import Annotation
from .communication import ChatMessage
from .communication import CommunicationSystem
from .session_sharing import CollaborationSession
from .session_sharing import SessionManager
from .state_sync import StateSynchronizer
from .user_manager import User
from .user_manager import UserManager
from .user_manager import UserRole

__all__ = [
    "UserManager",
    "User",
    "UserRole",
    "CollaborationSession",
    "SessionManager",
    "StateSynchronizer",
    "CommunicationSystem",
    "ChatMessage",
    "Annotation",
]
