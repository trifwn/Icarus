"""
Main Security Manager for ICARUS CLI.

Coordinates all security components including authentication, authorization,
audit logging, encryption, and plugin security. Provides a unified interface
for all security operations.
"""

import logging
from datetime import datetime
from typing import Dict
from typing import Optional

from ..collaboration.user_manager import User
from ..collaboration.user_manager import UserManager
from ..plugins.models import PluginInfo
from ..plugins.security import PluginSecurity
from .audit_logger import AuditEventType
from .audit_logger import AuditLogger
from .audit_logger import AuditSeverity
from .authentication import AuthenticationManager
from .authorization import AccessLevel
from .authorization import AuthorizationManager
from .authorization import ResourceType
from .crypto import CryptoManager


class SecurityManager:
    """Main security manager coordinating all security components."""

    def __init__(
        self,
        user_manager: UserManager,
        data_dir: str = "~/.icarus/security",
        enable_encryption: bool = True,
    ):
        """
        Initialize security manager.

        Args:
            user_manager: User management system
            data_dir: Directory for security data storage
            enable_encryption: Whether to enable encryption features
        """
        self.data_dir = data_dir
        self.enable_encryption = enable_encryption

        # Initialize crypto manager
        self.crypto_manager = CryptoManager() if enable_encryption else None

        # Initialize audit logger
        self.audit_logger = AuditLogger(
            data_dir=data_dir,
            crypto_manager=self.crypto_manager,
            enable_encryption=enable_encryption,
        )

        # Initialize authentication manager
        self.auth_manager = AuthenticationManager(
            user_manager=user_manager,
            crypto_manager=self.crypto_manager,
            data_dir=data_dir,
        )

        # Initialize authorization manager
        self.authz_manager = AuthorizationManager(data_dir=data_dir)

        # Initialize plugin security
        self.plugin_security = PluginSecurity()

        self.user_manager = user_manager
        self.logger = logging.getLogger(__name__)

        # Log system startup
        self.audit_logger.log_event(
            AuditEventType.SYSTEM_STARTUP,
            "Security system initialized",
            severity=AuditSeverity.LOW,
            details={"encryption_enabled": enable_encryption, "data_dir": data_dir},
        )

    # Authentication methods
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
            Session token if successful, None otherwise
        """
        session_token = self.auth_manager.authenticate_user(
            username,
            password,
            ip_address,
            user_agent,
        )

        if session_token:
            user = self.user_manager.get_user_by_username(username)
            self.audit_logger.log_event(
                AuditEventType.LOGIN_SUCCESS,
                f"User {username} logged in successfully",
                severity=AuditSeverity.LOW,
                user_id=user.id if user else None,
                username=username,
                session_id=session_token,
                ip_address=ip_address,
                user_agent=user_agent,
            )
        else:
            self.audit_logger.log_event(
                AuditEventType.LOGIN_FAILURE,
                f"Failed login attempt for user {username}",
                success=False,
                severity=AuditSeverity.MEDIUM,
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
            )

        return session_token

    def authenticate_token(self, token: str) -> Optional[User]:
        """
        Authenticate using session token.

        Args:
            token: Session token

        Returns:
            User object if valid, None otherwise
        """
        return self.auth_manager.authenticate_token(token)

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """
        Authenticate using API key.

        Args:
            api_key: API key

        Returns:
            User object if valid, None otherwise
        """
        return self.auth_manager.authenticate_api_key(api_key)

    def logout_user(self, session_token: str) -> bool:
        """
        Logout user by revoking session.

        Args:
            session_token: Session token to revoke

        Returns:
            True if successful, False otherwise
        """
        session_info = self.auth_manager.get_session_info(session_token)
        success = self.auth_manager.revoke_session(session_token)

        if success and session_info:
            self.audit_logger.log_event(
                AuditEventType.LOGOUT,
                "User logged out",
                severity=AuditSeverity.LOW,
                user_id=session_info["user_id"],
                username=session_info["username"],
                session_id=session_token,
            )

        return success

    # Authorization methods
    def check_access(
        self,
        user: User,
        resource_type: str,
        resource_id: str,
        required_access: str,
        session_token: Optional[str] = None,
    ) -> bool:
        """
        Check if user has required access to a resource.

        Args:
            user: User requesting access
            resource_type: Type of resource
            resource_id: Resource identifier
            required_access: Required access level
            session_token: Optional session token for audit

        Returns:
            True if access granted, False otherwise
        """
        try:
            resource_type_enum = ResourceType(resource_type)
            access_level_enum = AccessLevel(required_access)
        except ValueError as e:
            self.logger.error(f"Invalid resource type or access level: {e}")
            return False

        has_access = self.authz_manager.check_access(
            user,
            resource_type_enum,
            resource_id,
            access_level_enum,
        )

        # Log access attempt
        event_type = (
            AuditEventType.ACCESS_GRANTED
            if has_access
            else AuditEventType.ACCESS_DENIED
        )
        severity = AuditSeverity.LOW if has_access else AuditSeverity.MEDIUM

        self.audit_logger.log_event(
            event_type,
            f"Access {'granted' if has_access else 'denied'} to {resource_type}:{resource_id}",
            success=has_access,
            severity=severity,
            user_id=user.id,
            username=user.username,
            session_id=session_token,
            resource_type=resource_type,
            resource_id=resource_id,
            details={"required_access": required_access, "user_role": user.role.value},
        )

        return has_access

    def grant_permission(
        self,
        admin_user: User,
        target_user_id: str,
        resource_type: str,
        resource_id: str,
        access_level: str,
        expires_at: Optional[datetime] = None,
    ) -> bool:
        """
        Grant permission to user for a resource.

        Args:
            admin_user: User granting the permission (must have admin rights)
            target_user_id: User to grant permission to
            resource_type: Type of resource
            resource_id: Resource identifier
            access_level: Access level to grant
            expires_at: Optional expiration time

        Returns:
            True if permission granted, False otherwise
        """
        # Check if admin user has permission to grant access
        if not self.check_access(admin_user, "user_management", "permissions", "admin"):
            return False

        try:
            resource_type_enum = ResourceType(resource_type)
            access_level_enum = AccessLevel(access_level)
        except ValueError:
            return False

        success = self.authz_manager.grant_permission(
            target_user_id,
            resource_type_enum,
            resource_id,
            access_level_enum,
            admin_user.id,
            expires_at,
        )

        if success:
            target_user = self.user_manager.get_user(target_user_id)
            self.audit_logger.log_event(
                AuditEventType.PERMISSION_GRANTED,
                f"Permission granted to user {target_user.username if target_user else target_user_id}",
                severity=AuditSeverity.MEDIUM,
                user_id=admin_user.id,
                username=admin_user.username,
                resource_type=resource_type,
                resource_id=resource_id,
                details={
                    "target_user_id": target_user_id,
                    "access_level": access_level,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                },
            )

        return success

    def revoke_permission(
        self,
        admin_user: User,
        target_user_id: str,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """
        Revoke permission from user for a resource.

        Args:
            admin_user: User revoking the permission
            target_user_id: User to revoke permission from
            resource_type: Type of resource
            resource_id: Resource identifier

        Returns:
            True if permission revoked, False otherwise
        """
        # Check if admin user has permission to revoke access
        if not self.check_access(admin_user, "user_management", "permissions", "admin"):
            return False

        try:
            resource_type_enum = ResourceType(resource_type)
        except ValueError:
            return False

        success = self.authz_manager.revoke_permission(
            target_user_id,
            resource_type_enum,
            resource_id,
        )

        if success:
            target_user = self.user_manager.get_user(target_user_id)
            self.audit_logger.log_event(
                AuditEventType.PERMISSION_REVOKED,
                f"Permission revoked from user {target_user.username if target_user else target_user_id}",
                severity=AuditSeverity.MEDIUM,
                user_id=admin_user.id,
                username=admin_user.username,
                resource_type=resource_type,
                resource_id=resource_id,
                details={"target_user_id": target_user_id},
            )

        return success

    # Plugin security methods
    def validate_plugin_security(self, plugin_info: PluginInfo, user: User) -> bool:
        """
        Validate plugin security before loading.

        Args:
            plugin_info: Plugin information
            user: User attempting to load plugin

        Returns:
            True if plugin is safe to load, False otherwise
        """
        is_safe = self.plugin_security.validate_plugin_security(plugin_info)

        event_type = (
            AuditEventType.PLUGIN_LOADED if is_safe else AuditEventType.PLUGIN_BLOCKED
        )
        severity = AuditSeverity.LOW if is_safe else AuditSeverity.HIGH

        self.audit_logger.log_event(
            event_type,
            f"Plugin {plugin_info.manifest.name} {'validated' if is_safe else 'blocked'}",
            success=is_safe,
            severity=severity,
            user_id=user.id,
            username=user.username,
            resource_type="plugin",
            resource_id=plugin_info.id,
            details={
                "plugin_name": plugin_info.manifest.name,
                "plugin_version": plugin_info.manifest.version,
                "security_level": plugin_info.manifest.security_level.value,
                "risk_level": self.plugin_security.get_plugin_risk_level(plugin_info),
            },
        )

        return is_safe

    def trust_plugin(self, plugin_id: str, admin_user: User) -> bool:
        """
        Mark a plugin as trusted.

        Args:
            plugin_id: Plugin identifier
            admin_user: Admin user trusting the plugin

        Returns:
            True if successful, False otherwise
        """
        if not self.check_access(admin_user, "plugin", plugin_id, "admin"):
            return False

        self.plugin_security.trust_plugin(plugin_id)

        self.audit_logger.log_event(
            AuditEventType.PLUGIN_LOADED,
            f"Plugin {plugin_id} marked as trusted",
            severity=AuditSeverity.MEDIUM,
            user_id=admin_user.id,
            username=admin_user.username,
            resource_type="plugin",
            resource_id=plugin_id,
            details={"action": "trust_plugin"},
        )

        return True

    def block_plugin(self, plugin_id: str, admin_user: User) -> bool:
        """
        Block a plugin from loading.

        Args:
            plugin_id: Plugin identifier
            admin_user: Admin user blocking the plugin

        Returns:
            True if successful, False otherwise
        """
        if not self.check_access(admin_user, "plugin", plugin_id, "admin"):
            return False

        self.plugin_security.block_plugin(plugin_id)

        self.audit_logger.log_event(
            AuditEventType.PLUGIN_BLOCKED,
            f"Plugin {plugin_id} blocked",
            severity=AuditSeverity.HIGH,
            user_id=admin_user.id,
            username=admin_user.username,
            resource_type="plugin",
            resource_id=plugin_id,
            details={"action": "block_plugin"},
        )

        return True

    # Encryption methods
    def encrypt_sensitive_data(self, data: str) -> Optional[str]:
        """
        Encrypt sensitive data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data or None if encryption disabled
        """
        if not self.crypto_manager:
            return data

        return self.crypto_manager.encrypt_data(data)

    def decrypt_sensitive_data(self, encrypted_data: str) -> Optional[str]:
        """
        Decrypt sensitive data.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted data or None if encryption disabled
        """
        if not self.crypto_manager:
            return encrypted_data

        try:
            return self.crypto_manager.decrypt_data(encrypted_data)
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            return None

    def encrypt_configuration(self, config_data: Dict) -> Dict:
        """
        Encrypt sensitive configuration fields.

        Args:
            config_data: Configuration dictionary

        Returns:
            Configuration with sensitive fields encrypted
        """
        if not self.crypto_manager:
            return config_data

        return self.crypto_manager.encrypt_sensitive_config(config_data)

    def decrypt_configuration(self, encrypted_config: Dict) -> Dict:
        """
        Decrypt sensitive configuration fields.

        Args:
            encrypted_config: Configuration with encrypted fields

        Returns:
            Configuration with sensitive fields decrypted
        """
        if not self.crypto_manager:
            return encrypted_config

        return self.crypto_manager.decrypt_sensitive_config(encrypted_config)

    # Audit and monitoring methods
    def log_security_event(
        self,
        event_type: str,
        action: str,
        user: Optional[User] = None,
        session_token: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        success: bool = True,
        severity: str = "low",
        details: Optional[Dict] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """
        Log a security event.

        Args:
            event_type: Type of security event
            action: Description of the action
            user: User associated with the event
            session_token: Session token
            resource_type: Type of resource involved
            resource_id: Resource identifier
            success: Whether the action was successful
            severity: Event severity (low, medium, high, critical)
            details: Additional event details
            error_message: Error message if action failed

        Returns:
            Event ID
        """
        try:
            event_type_enum = AuditEventType(event_type)
            severity_enum = AuditSeverity(severity)
        except ValueError:
            event_type_enum = AuditEventType.SYSTEM_STARTUP  # Default
            severity_enum = AuditSeverity.LOW

        return self.audit_logger.log_event(
            event_type_enum,
            action,
            success=success,
            severity=severity_enum,
            user_id=user.id if user else None,
            username=user.username if user else None,
            session_id=session_token,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            error_message=error_message,
        )

    def get_security_summary(self, days: int = 7) -> Dict:
        """
        Get comprehensive security summary.

        Args:
            days: Number of days to analyze

        Returns:
            Security summary dictionary
        """
        audit_summary = self.audit_logger.get_security_summary(days)
        auth_stats = self.auth_manager.get_auth_stats()
        authz_stats = self.authz_manager.get_authorization_stats()

        return {
            "period_days": days,
            "audit_summary": audit_summary,
            "authentication": auth_stats,
            "authorization": authz_stats,
            "encryption_enabled": self.enable_encryption,
            "system_health": {
                "active_sessions": auth_stats.get("active_sessions", 0),
                "failed_logins": audit_summary.get("failed_logins", 0),
                "security_violations": audit_summary.get("security_violations", 0),
                "access_denied_count": audit_summary.get("access_denied_count", 0),
            },
        }

    def cleanup_security_data(self):
        """Clean up old security data and expired sessions."""
        # Clean up expired sessions
        self.auth_manager.cleanup_expired_sessions()

        # Clean up expired permissions
        self.authz_manager.cleanup_expired_permissions()

        # Clean up old audit events
        self.audit_logger.cleanup_old_events()

        self.audit_logger.log_event(
            AuditEventType.SYSTEM_STARTUP,
            "Security data cleanup completed",
            severity=AuditSeverity.LOW,
        )

    def export_security_report(self, filepath: str, format: str = "json") -> bool:
        """
        Export comprehensive security report.

        Args:
            filepath: Output file path
            format: Export format (json, csv)

        Returns:
            True if export successful, False otherwise
        """
        try:
            security_summary = self.get_security_summary(30)  # 30-day summary

            if format.lower() == "json":
                import json

                with open(filepath, "w") as f:
                    json.dump(security_summary, f, indent=2, default=str)
            else:
                return self.audit_logger.export_audit_log(filepath, format)

            self.audit_logger.log_event(
                AuditEventType.AUDIT_LOG_ACCESSED,
                f"Security report exported to {filepath}",
                severity=AuditSeverity.MEDIUM,
                details={"format": format},
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to export security report: {e}")
            return False

    def shutdown(self):
        """Shutdown security system gracefully."""
        self.audit_logger.log_event(
            AuditEventType.SYSTEM_SHUTDOWN,
            "Security system shutting down",
            severity=AuditSeverity.LOW,
        )

        # Perform final cleanup
        self.cleanup_security_data()

        self.logger.info("Security system shutdown complete")
