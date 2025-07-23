"""
Audit Logging System for ICARUS CLI Security.

Provides comprehensive audit logging for security monitoring, compliance,
and forensic analysis. Tracks all security-relevant events and user actions.
"""

import json
import logging
import sqlite3
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .crypto import CryptoManager


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SESSION_REVOKED = "session_revoked"
    PASSWORD_CHANGED = "password_changed"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_CHANGED = "role_changed"

    # Resource events
    RESOURCE_CREATED = "resource_created"
    RESOURCE_ACCESSED = "resource_accessed"
    RESOURCE_MODIFIED = "resource_modified"
    RESOURCE_DELETED = "resource_deleted"
    RESOURCE_SHARED = "resource_shared"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_BLOCKED = "plugin_blocked"

    # Security events
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ENCRYPTION_KEY_ROTATED = "encryption_key_rotated"
    AUDIT_LOG_ACCESSED = "audit_log_accessed"

    # Data events
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"
    DATA_BACKUP_CREATED = "data_backup_created"
    DATA_RESTORED = "data_restored"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""

    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            severity=AuditSeverity(data["severity"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data.get("user_id"),
            username=data.get("username"),
            session_id=data.get("session_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            action=data["action"],
            details=data.get("details", {}),
            success=data["success"],
            error_message=data.get("error_message"),
        )


class AuditLogger:
    """Comprehensive audit logging system."""

    def __init__(
        self,
        data_dir: str = "~/.icarus/security",
        crypto_manager: Optional[CryptoManager] = None,
        enable_encryption: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            data_dir: Directory for audit data storage
            crypto_manager: Cryptographic operations manager
            enable_encryption: Whether to encrypt audit logs
        """
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.crypto_manager = crypto_manager
        self.enable_encryption = enable_encryption and crypto_manager is not None

        # Database for structured audit logs
        self.db_path = self.data_dir / "audit.db"

        # JSON file for backup/export
        self.json_backup_path = self.data_dir / "audit_backup.json"

        # Configuration
        self.max_log_age_days = 365  # Keep logs for 1 year
        self.max_log_size_mb = 100  # Rotate logs at 100MB
        self.enable_real_time_alerts = True

        self.logger = logging.getLogger(__name__)

        # Initialize database
        self._initialize_database()

        # Event counters for anomaly detection
        self.event_counters: Dict[str, int] = {}
        self.last_counter_reset = datetime.now()

    def _initialize_database(self):
        """Initialize SQLite database for audit logs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT UNIQUE NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        user_id TEXT,
                        username TEXT,
                        session_id TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        resource_type TEXT,
                        resource_id TEXT,
                        action TEXT NOT NULL,
                        details TEXT,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create indexes for better query performance
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)",
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)",
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)",
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_resource ON audit_events(resource_type, resource_id)",
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_success ON audit_events(success)",
                )

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to initialize audit database: {e}")

    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        success: bool = True,
        severity: AuditSeverity = AuditSeverity.LOW,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            action: Description of the action
            success: Whether the action was successful
            severity: Event severity level
            user_id: User ID associated with the event
            username: Username associated with the event
            session_id: Session ID associated with the event
            ip_address: Client IP address
            user_agent: Client user agent
            resource_type: Type of resource involved
            resource_id: ID of resource involved
            details: Additional event details
            error_message: Error message if action failed

        Returns:
            Event ID
        """
        # Generate unique event ID
        event_id = (
            self.crypto_manager.generate_secure_token(16)
            if self.crypto_manager
            else f"evt_{datetime.now().timestamp()}"
        )

        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            user_id=user_id,
            username=username,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            success=success,
            error_message=error_message,
        )

        # Store event in database
        self._store_event(event)

        # Update event counters
        self._update_event_counters(event)

        # Check for suspicious activity
        if self.enable_real_time_alerts:
            self._check_for_anomalies(event)

        # Log to standard logger as well
        log_level = self._get_log_level(severity)
        self.logger.log(
            log_level,
            f"AUDIT: {event_type.value} - {action} - User: {username or 'N/A'} - Success: {success}",
        )

        return event_id

    def _store_event(self, event: AuditEvent):
        """Store audit event in database."""
        try:
            details_json = json.dumps(event.details) if event.details else None

            # Encrypt sensitive data if encryption is enabled
            if self.enable_encryption and self.crypto_manager:
                if event.user_id:
                    event.user_id = self.crypto_manager.encrypt_data(event.user_id)
                if event.session_id:
                    event.session_id = self.crypto_manager.encrypt_data(
                        event.session_id,
                    )
                if details_json:
                    details_json = self.crypto_manager.encrypt_data(details_json)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, event_type, severity, timestamp, user_id, username,
                        session_id, ip_address, user_agent, resource_type, resource_id,
                        action, details, success, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.event_id,
                        event.event_type.value,
                        event.severity.value,
                        event.timestamp.isoformat(),
                        event.user_id,
                        event.username,
                        event.session_id,
                        event.ip_address,
                        event.user_agent,
                        event.resource_type,
                        event.resource_id,
                        event.action,
                        details_json,
                        event.success,
                        event.error_message,
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to store audit event: {e}")

    def _update_event_counters(self, event: AuditEvent):
        """Update event counters for anomaly detection."""
        # Reset counters daily
        if datetime.now() - self.last_counter_reset > timedelta(days=1):
            self.event_counters.clear()
            self.last_counter_reset = datetime.now()

        # Count events by type and user
        event_key = f"{event.event_type.value}"
        self.event_counters[event_key] = self.event_counters.get(event_key, 0) + 1

        if event.user_id:
            user_key = f"user_{event.user_id}_{event.event_type.value}"
            self.event_counters[user_key] = self.event_counters.get(user_key, 0) + 1

    def _check_for_anomalies(self, event: AuditEvent):
        """Check for suspicious activity patterns."""
        # Define thresholds for different event types
        thresholds = {
            AuditEventType.LOGIN_FAILURE: 5,
            AuditEventType.ACCESS_DENIED: 10,
            AuditEventType.SECURITY_VIOLATION: 1,
            AuditEventType.SUSPICIOUS_ACTIVITY: 1,
        }

        threshold = thresholds.get(event.event_type, 50)
        event_key = f"{event.event_type.value}"

        if self.event_counters.get(event_key, 0) >= threshold:
            # Log suspicious activity
            self.log_event(
                AuditEventType.SUSPICIOUS_ACTIVITY,
                f"High frequency of {event.event_type.value} events detected",
                severity=AuditSeverity.HIGH,
                details={
                    "original_event_type": event.event_type.value,
                    "event_count": self.event_counters[event_key],
                    "threshold": threshold,
                },
            )

    def _get_log_level(self, severity: AuditSeverity) -> int:
        """Convert audit severity to logging level."""
        severity_map = {
            AuditSeverity.LOW: logging.INFO,
            AuditSeverity.MEDIUM: logging.WARNING,
            AuditSeverity.HIGH: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }
        return severity_map.get(severity, logging.INFO)

    def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        success: Optional[bool] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """
        Query audit events with filters.

        Args:
            start_time: Start time for query
            end_time: End time for query
            event_types: List of event types to filter by
            user_id: User ID to filter by
            username: Username to filter by
            resource_type: Resource type to filter by
            resource_id: Resource ID to filter by
            success: Success status to filter by
            severity: Severity level to filter by
            limit: Maximum number of events to return

        Returns:
            List of audit events
        """
        try:
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            if event_types:
                placeholders = ",".join("?" * len(event_types))
                query += f" AND event_type IN ({placeholders})"
                params.extend([et.value for et in event_types])

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            if username:
                query += " AND username = ?"
                params.append(username)

            if resource_type:
                query += " AND resource_type = ?"
                params.append(resource_type)

            if resource_id:
                query += " AND resource_id = ?"
                params.append(resource_id)

            if success is not None:
                query += " AND success = ?"
                params.append(success)

            if severity:
                query += " AND severity = ?"
                params.append(severity.value)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            events = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)

                for row in cursor.fetchall():
                    # Decrypt sensitive data if needed
                    user_id_decrypted = row["user_id"]
                    session_id_decrypted = row["session_id"]
                    details_decrypted = row["details"]

                    if self.enable_encryption and self.crypto_manager:
                        try:
                            if user_id_decrypted:
                                user_id_decrypted = self.crypto_manager.decrypt_data(
                                    user_id_decrypted,
                                )
                            if session_id_decrypted:
                                session_id_decrypted = self.crypto_manager.decrypt_data(
                                    session_id_decrypted,
                                )
                            if details_decrypted:
                                details_decrypted = self.crypto_manager.decrypt_data(
                                    details_decrypted,
                                )
                        except Exception:
                            # If decryption fails, use original data
                            pass

                    event = AuditEvent(
                        event_id=row["event_id"],
                        event_type=AuditEventType(row["event_type"]),
                        severity=AuditSeverity(row["severity"]),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        user_id=user_id_decrypted,
                        username=row["username"],
                        session_id=session_id_decrypted,
                        ip_address=row["ip_address"],
                        user_agent=row["user_agent"],
                        resource_type=row["resource_type"],
                        resource_id=row["resource_id"],
                        action=row["action"],
                        details=json.loads(details_decrypted)
                        if details_decrypted
                        else {},
                        success=bool(row["success"]),
                        error_message=row["error_message"],
                    )
                    events.append(event)

            return events

        except Exception as e:
            self.logger.error(f"Failed to query audit events: {e}")
            return []

    def get_security_summary(self, days: int = 7) -> Dict:
        """
        Get security summary for the specified number of days.

        Args:
            days: Number of days to analyze

        Returns:
            Security summary dictionary
        """
        start_time = datetime.now() - timedelta(days=days)
        events = self.query_events(start_time=start_time, limit=10000)

        summary = {
            "period_days": days,
            "total_events": len(events),
            "events_by_type": {},
            "events_by_severity": {},
            "failed_logins": 0,
            "access_denied_count": 0,
            "security_violations": 0,
            "unique_users": set(),
            "unique_ips": set(),
            "top_users": {},
            "top_resources": {},
        }

        for event in events:
            # Count by type
            event_type = event.event_type.value
            summary["events_by_type"][event_type] = (
                summary["events_by_type"].get(event_type, 0) + 1
            )

            # Count by severity
            severity = event.severity.value
            summary["events_by_severity"][severity] = (
                summary["events_by_severity"].get(severity, 0) + 1
            )

            # Count specific security events
            if event.event_type == AuditEventType.LOGIN_FAILURE:
                summary["failed_logins"] += 1
            elif event.event_type == AuditEventType.ACCESS_DENIED:
                summary["access_denied_count"] += 1
            elif event.event_type == AuditEventType.SECURITY_VIOLATION:
                summary["security_violations"] += 1

            # Track unique users and IPs
            if event.username:
                summary["unique_users"].add(event.username)
                summary["top_users"][event.username] = (
                    summary["top_users"].get(event.username, 0) + 1
                )

            if event.ip_address:
                summary["unique_ips"].add(event.ip_address)

            # Track resource access
            if event.resource_type and event.resource_id:
                resource_key = f"{event.resource_type}:{event.resource_id}"
                summary["top_resources"][resource_key] = (
                    summary["top_resources"].get(resource_key, 0) + 1
                )

        # Convert sets to counts
        summary["unique_users"] = len(summary["unique_users"])
        summary["unique_ips"] = len(summary["unique_ips"])

        # Get top 10 users and resources
        summary["top_users"] = dict(
            sorted(summary["top_users"].items(), key=lambda x: x[1], reverse=True)[:10],
        )
        summary["top_resources"] = dict(
            sorted(summary["top_resources"].items(), key=lambda x: x[1], reverse=True)[
                :10
            ],
        )

        return summary

    def export_audit_log(self, filepath: str, format: str = "json") -> bool:
        """
        Export audit log to file.

        Args:
            filepath: Output file path
            format: Export format (json, csv)

        Returns:
            True if export successful, False otherwise
        """
        try:
            events = self.query_events(limit=100000)  # Export all events

            if format.lower() == "json":
                with open(filepath, "w") as f:
                    json.dump([event.to_dict() for event in events], f, indent=2)
            elif format.lower() == "csv":
                import csv

                with open(filepath, "w", newline="") as f:
                    if events:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=events[0].to_dict().keys(),
                        )
                        writer.writeheader()
                        for event in events:
                            writer.writerow(event.to_dict())
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Log the export
            self.log_event(
                AuditEventType.AUDIT_LOG_ACCESSED,
                f"Audit log exported to {filepath}",
                severity=AuditSeverity.MEDIUM,
                details={"format": format, "event_count": len(events)},
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to export audit log: {e}")
            return False

    def cleanup_old_events(self):
        """Clean up old audit events based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_log_age_days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (cutoff_date.isoformat(),),
                )
                deleted_count = cursor.rowcount
                conn.commit()

            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old audit events")

                # Log the cleanup
                self.log_event(
                    AuditEventType.SYSTEM_STARTUP,  # Using system event type
                    f"Cleaned up {deleted_count} old audit events",
                    details={
                        "cutoff_date": cutoff_date.isoformat(),
                        "retention_days": self.max_log_age_days,
                    },
                )

        except Exception as e:
            self.logger.error(f"Failed to cleanup old audit events: {e}")

    def get_audit_stats(self) -> Dict:
        """Get audit logging statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM audit_events")
                total_events = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM audit_events WHERE user_id IS NOT NULL",
                )
                unique_users = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type ORDER BY COUNT(*) DESC LIMIT 10",
                )
                top_event_types = dict(cursor.fetchall())

            return {
                "total_events": total_events,
                "unique_users": unique_users,
                "top_event_types": top_event_types,
                "database_size_mb": self.db_path.stat().st_size / (1024 * 1024)
                if self.db_path.exists()
                else 0,
                "encryption_enabled": self.enable_encryption,
                "retention_days": self.max_log_age_days,
            }

        except Exception as e:
            self.logger.error(f"Failed to get audit stats: {e}")
            return {}
