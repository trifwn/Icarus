"""
Security and Authentication System for ICARUS CLI

This module provides comprehensive security features including:
- Data encryption for sensitive information
- Role-based access control for collaboration features
- Audit logging system for security monitoring
- Plugin security scanning and validation
"""

from .audit_logger import AuditLogger
from .authentication import AuthenticationManager
from .authorization import AuthorizationManager
from .crypto import CryptoManager
from .security_manager import SecurityManager

__all__ = [
    "AuthenticationManager",
    "AuthorizationManager",
    "AuditLogger",
    "CryptoManager",
    "SecurityManager",
]
