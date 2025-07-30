"""
Authorization Manager for ICARUS CLI Security System.

Handles role-based access control, permission management, and resource authorization.
Integrates with the authentication system to provide comprehensive access control.
"""

import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from ..collaboration.user_manager import Permission
from ..collaboration.user_manager import User
from ..collaboration.user_manager import UserRole


class ResourceType(str, Enum):
    """Types of resources that can be protected."""

    SESSION = "session"
    ANALYSIS = "analysis"
    WORKFLOW = "workflow"
    DATA = "data"
    PLUGIN = "plugin"
    CONFIGURATION = "configuration"
    USER_MANAGEMENT = "user_management"
    SYSTEM = "system"


class AccessLevel(str, Enum):
    """Access levels for resources."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    OWNER = "owner"


@dataclass
class ResourcePermission:
    """Permission for a specific resource."""

    resource_type: ResourceType
    resource_id: str
    access_level: AccessLevel
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    conditions: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["resource_type"] = self.resource_type.value
        data["access_level"] = self.access_level.value
        data["granted_at"] = self.granted_at.isoformat()
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "ResourcePermission":
        """Create from dictionary."""
        return cls(
            resource_type=ResourceType(data["resource_type"]),
            resource_id=data["resource_id"],
            access_level=AccessLevel(data["access_level"]),
            granted_by=data["granted_by"],
            granted_at=datetime.fromisoformat(data["granted_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            conditions=data.get("conditions"),
        )

    def is_valid(self) -> bool:
        """Check if permission is still valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


@dataclass
class AccessPolicy:
    """Access control policy for a resource type."""

    resource_type: ResourceType
    default_permissions: Dict[UserRole, AccessLevel]
    required_permissions: List[Permission]
    allow_owner_override: bool = True
    require_explicit_grant: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "resource_type": self.resource_type.value,
            "default_permissions": {
                role.value: level.value
                for role, level in self.default_permissions.items()
            },
            "required_permissions": [perm.value for perm in self.required_permissions],
            "allow_owner_override": self.allow_owner_override,
            "require_explicit_grant": self.require_explicit_grant,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AccessPolicy":
        """Create from dictionary."""
        return cls(
            resource_type=ResourceType(data["resource_type"]),
            default_permissions={
                UserRole(role): AccessLevel(level)
                for role, level in data["default_permissions"].items()
            },
            required_permissions=[
                Permission(perm) for perm in data["required_permissions"]
            ],
            allow_owner_override=data.get("allow_owner_override", True),
            require_explicit_grant=data.get("require_explicit_grant", False),
        )


class AuthorizationManager:
    """Manages authorization, access control, and permissions."""

    def __init__(self, data_dir: str = "~/.icarus/security"):
        """
        Initialize authorization manager.

        Args:
            data_dir: Directory for security data storage
        """
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.policies_file = self.data_dir / "access_policies.json"
        self.permissions_file = self.data_dir / "resource_permissions.json"

        # Access policies for different resource types
        self.access_policies: Dict[ResourceType, AccessPolicy] = {}

        # Explicit resource permissions
        self.resource_permissions: Dict[
            str,
            List[ResourcePermission],
        ] = {}  # user_id -> permissions

        # Resource ownership tracking
        self.resource_owners: Dict[str, str] = {}  # resource_id -> user_id

        self.logger = logging.getLogger(__name__)

        # Initialize default policies
        self._initialize_default_policies()

        # Load existing data
        self._load_policies()
        self._load_permissions()

    def _initialize_default_policies(self):
        """Initialize default access policies for different resource types."""

        # Session management policy
        self.access_policies[ResourceType.SESSION] = AccessPolicy(
            resource_type=ResourceType.SESSION,
            default_permissions={
                UserRole.OWNER: AccessLevel.OWNER,
                UserRole.ADMIN: AccessLevel.ADMIN,
                UserRole.COLLABORATOR: AccessLevel.READ,
                UserRole.VIEWER: AccessLevel.READ,
                UserRole.GUEST: AccessLevel.NONE,
            },
            required_permissions=[
                Permission.CREATE_SESSION,
                Permission.MODIFY_SESSION_SETTINGS,
            ],
        )

        # Analysis policy
        self.access_policies[ResourceType.ANALYSIS] = AccessPolicy(
            resource_type=ResourceType.ANALYSIS,
            default_permissions={
                UserRole.OWNER: AccessLevel.OWNER,
                UserRole.ADMIN: AccessLevel.ADMIN,
                UserRole.COLLABORATOR: AccessLevel.WRITE,
                UserRole.VIEWER: AccessLevel.READ,
                UserRole.GUEST: AccessLevel.NONE,
            },
            required_permissions=[Permission.RUN_ANALYSIS, Permission.MODIFY_ANALYSIS],
        )

        # Workflow policy
        self.access_policies[ResourceType.WORKFLOW] = AccessPolicy(
            resource_type=ResourceType.WORKFLOW,
            default_permissions={
                UserRole.OWNER: AccessLevel.OWNER,
                UserRole.ADMIN: AccessLevel.ADMIN,
                UserRole.COLLABORATOR: AccessLevel.WRITE,
                UserRole.VIEWER: AccessLevel.READ,
                UserRole.GUEST: AccessLevel.NONE,
            },
            required_permissions=[
                Permission.CREATE_WORKFLOW,
                Permission.EXECUTE_WORKFLOW,
            ],
        )

        # Data policy
        self.access_policies[ResourceType.DATA] = AccessPolicy(
            resource_type=ResourceType.DATA,
            default_permissions={
                UserRole.OWNER: AccessLevel.OWNER,
                UserRole.ADMIN: AccessLevel.ADMIN,
                UserRole.COLLABORATOR: AccessLevel.WRITE,
                UserRole.VIEWER: AccessLevel.READ,
                UserRole.GUEST: AccessLevel.NONE,
            },
            required_permissions=[Permission.IMPORT_DATA, Permission.EXPORT_DATA],
        )

        # Plugin policy
        self.access_policies[ResourceType.PLUGIN] = AccessPolicy(
            resource_type=ResourceType.PLUGIN,
            default_permissions={
                UserRole.OWNER: AccessLevel.OWNER,
                UserRole.ADMIN: AccessLevel.ADMIN,
                UserRole.COLLABORATOR: AccessLevel.READ,
                UserRole.VIEWER: AccessLevel.READ,
                UserRole.GUEST: AccessLevel.NONE,
            },
            required_permissions=[],
            require_explicit_grant=True,  # Plugins require explicit permission
        )

        # User management policy
        self.access_policies[ResourceType.USER_MANAGEMENT] = AccessPolicy(
            resource_type=ResourceType.USER_MANAGEMENT,
            default_permissions={
                UserRole.OWNER: AccessLevel.OWNER,
                UserRole.ADMIN: AccessLevel.ADMIN,
                UserRole.COLLABORATOR: AccessLevel.NONE,
                UserRole.VIEWER: AccessLevel.NONE,
                UserRole.GUEST: AccessLevel.NONE,
            },
            required_permissions=[
                Permission.INVITE_USERS,
                Permission.CHANGE_USER_ROLES,
            ],
        )

        # System policy
        self.access_policies[ResourceType.SYSTEM] = AccessPolicy(
            resource_type=ResourceType.SYSTEM,
            default_permissions={
                UserRole.OWNER: AccessLevel.OWNER,
                UserRole.ADMIN: AccessLevel.READ,
                UserRole.COLLABORATOR: AccessLevel.NONE,
                UserRole.VIEWER: AccessLevel.NONE,
                UserRole.GUEST: AccessLevel.NONE,
            },
            required_permissions=[],
        )

    def _load_policies(self):
        """Load access policies from storage."""
        try:
            if self.policies_file.exists():
                with open(self.policies_file) as f:
                    data = json.load(f)

                for policy_data in data.get("policies", []):
                    policy = AccessPolicy.from_dict(policy_data)
                    self.access_policies[policy.resource_type] = policy

                self.logger.info(f"Loaded {len(self.access_policies)} access policies")
        except Exception as e:
            self.logger.error(f"Failed to load access policies: {e}")

    def _save_policies(self):
        """Save access policies to storage."""
        try:
            data = {
                "policies": [
                    policy.to_dict() for policy in self.access_policies.values()
                ],
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.policies_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save access policies: {e}")

    def _load_permissions(self):
        """Load resource permissions from storage."""
        try:
            if self.permissions_file.exists():
                with open(self.permissions_file) as f:
                    data = json.load(f)

                # Load resource permissions
                for user_id, perms_data in data.get("permissions", {}).items():
                    permissions = [
                        ResourcePermission.from_dict(perm) for perm in perms_data
                    ]
                    self.resource_permissions[user_id] = permissions

                # Load resource owners
                self.resource_owners = data.get("owners", {})

                self.logger.info(
                    f"Loaded permissions for {len(self.resource_permissions)} users",
                )
        except Exception as e:
            self.logger.error(f"Failed to load resource permissions: {e}")

    def _save_permissions(self):
        """Save resource permissions to storage."""
        try:
            data = {
                "permissions": {
                    user_id: [perm.to_dict() for perm in permissions]
                    for user_id, permissions in self.resource_permissions.items()
                },
                "owners": self.resource_owners,
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.permissions_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save resource permissions: {e}")

    def check_access(
        self,
        user: User,
        resource_type: ResourceType,
        resource_id: str,
        required_access: AccessLevel,
    ) -> bool:
        """
        Check if user has required access to a resource.

        Args:
            user: User requesting access
            resource_type: Type of resource
            resource_id: Specific resource identifier
            required_access: Required access level

        Returns:
            True if access is granted, False otherwise
        """
        # Check if user is active
        if not user.is_active:
            return False

        # Check resource ownership
        if self._is_resource_owner(user.id, resource_id):
            return True

        # Get access policy for resource type
        policy = self.access_policies.get(resource_type)
        if not policy:
            self.logger.warning(
                f"No access policy found for resource type: {resource_type}",
            )
            return False

        # Check explicit permissions first
        explicit_access = self._check_explicit_permission(
            user.id,
            resource_type,
            resource_id,
        )
        if explicit_access and self._access_level_sufficient(
            explicit_access,
            required_access,
        ):
            return True

        # Check if explicit grant is required
        if policy.require_explicit_grant and not explicit_access:
            return False

        # Check default role-based permissions
        default_access = policy.default_permissions.get(user.role, AccessLevel.NONE)
        if not self._access_level_sufficient(default_access, required_access):
            return False

        # Check required permissions
        if policy.required_permissions:
            if not any(
                user.has_permission(perm) for perm in policy.required_permissions
            ):
                return False

        return True

    def _is_resource_owner(self, user_id: str, resource_id: str) -> bool:
        """Check if user owns the resource."""
        return self.resource_owners.get(resource_id) == user_id

    def _check_explicit_permission(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
    ) -> Optional[AccessLevel]:
        """Check explicit permission for user on specific resource."""
        user_permissions = self.resource_permissions.get(user_id, [])

        for permission in user_permissions:
            if (
                permission.resource_type == resource_type
                and permission.resource_id == resource_id
                and permission.is_valid()
            ):
                return permission.access_level

        return None

    def _access_level_sufficient(
        self,
        granted_level: AccessLevel,
        required_level: AccessLevel,
    ) -> bool:
        """Check if granted access level is sufficient for required level."""
        level_hierarchy = {
            AccessLevel.NONE: 0,
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.EXECUTE: 3,
            AccessLevel.ADMIN: 4,
            AccessLevel.OWNER: 5,
        }

        return level_hierarchy.get(granted_level, 0) >= level_hierarchy.get(
            required_level,
            0,
        )

    def grant_permission(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
        access_level: AccessLevel,
        granted_by: str,
        expires_at: Optional[datetime] = None,
    ) -> bool:
        """
        Grant explicit permission to user for a resource.

        Args:
            user_id: User to grant permission to
            resource_type: Type of resource
            resource_id: Specific resource identifier
            access_level: Access level to grant
            granted_by: User ID who granted the permission
            expires_at: Optional expiration time

        Returns:
            True if permission was granted, False otherwise
        """
        permission = ResourcePermission(
            resource_type=resource_type,
            resource_id=resource_id,
            access_level=access_level,
            granted_by=granted_by,
            granted_at=datetime.now(),
            expires_at=expires_at,
        )

        if user_id not in self.resource_permissions:
            self.resource_permissions[user_id] = []

        # Remove existing permission for same resource if exists
        self.resource_permissions[user_id] = [
            perm
            for perm in self.resource_permissions[user_id]
            if not (
                perm.resource_type == resource_type and perm.resource_id == resource_id
            )
        ]

        # Add new permission
        self.resource_permissions[user_id].append(permission)
        self._save_permissions()

        self.logger.info(
            f"Granted {access_level.value} access to {resource_type.value}:{resource_id} for user {user_id}",
        )
        return True

    def revoke_permission(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
    ) -> bool:
        """
        Revoke explicit permission from user for a resource.

        Args:
            user_id: User to revoke permission from
            resource_type: Type of resource
            resource_id: Specific resource identifier

        Returns:
            True if permission was revoked, False if not found
        """
        if user_id not in self.resource_permissions:
            return False

        original_count = len(self.resource_permissions[user_id])

        self.resource_permissions[user_id] = [
            perm
            for perm in self.resource_permissions[user_id]
            if not (
                perm.resource_type == resource_type and perm.resource_id == resource_id
            )
        ]

        if len(self.resource_permissions[user_id]) < original_count:
            self._save_permissions()
            self.logger.info(
                f"Revoked access to {resource_type.value}:{resource_id} from user {user_id}",
            )
            return True

        return False

    def set_resource_owner(self, resource_id: str, user_id: str):
        """
        Set owner of a resource.

        Args:
            resource_id: Resource identifier
            user_id: User ID of the owner
        """
        self.resource_owners[resource_id] = user_id
        self._save_permissions()

        self.logger.info(f"Set owner of resource {resource_id} to user {user_id}")

    def get_user_permissions(self, user_id: str) -> List[ResourcePermission]:
        """
        Get all explicit permissions for a user.

        Args:
            user_id: User ID

        Returns:
            List of resource permissions
        """
        return self.resource_permissions.get(user_id, [])

    def get_resource_permissions(
        self,
        resource_type: ResourceType,
        resource_id: str,
    ) -> List[Dict]:
        """
        Get all permissions for a specific resource.

        Args:
            resource_type: Type of resource
            resource_id: Resource identifier

        Returns:
            List of permission information
        """
        permissions = []

        for user_id, user_permissions in self.resource_permissions.items():
            for permission in user_permissions:
                if (
                    permission.resource_type == resource_type
                    and permission.resource_id == resource_id
                    and permission.is_valid()
                ):
                    permissions.append(
                        {
                            "user_id": user_id,
                            "access_level": permission.access_level.value,
                            "granted_by": permission.granted_by,
                            "granted_at": permission.granted_at.isoformat(),
                            "expires_at": permission.expires_at.isoformat()
                            if permission.expires_at
                            else None,
                        },
                    )

        return permissions

    def cleanup_expired_permissions(self):
        """Clean up expired permissions."""
        total_removed = 0

        for user_id in self.resource_permissions:
            original_count = len(self.resource_permissions[user_id])

            self.resource_permissions[user_id] = [
                perm for perm in self.resource_permissions[user_id] if perm.is_valid()
            ]

            removed = original_count - len(self.resource_permissions[user_id])
            total_removed += removed

        if total_removed > 0:
            self._save_permissions()
            self.logger.info(f"Cleaned up {total_removed} expired permissions")

    def get_access_summary(self, user: User) -> Dict:
        """
        Get access summary for a user.

        Args:
            user: User to get summary for

        Returns:
            Dictionary with access information
        """
        summary = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [perm.value for perm in user.permissions],
            "explicit_permissions": [],
            "owned_resources": [],
        }

        # Add explicit permissions
        user_permissions = self.resource_permissions.get(user.id, [])
        for permission in user_permissions:
            if permission.is_valid():
                summary["explicit_permissions"].append(
                    {
                        "resource_type": permission.resource_type.value,
                        "resource_id": permission.resource_id,
                        "access_level": permission.access_level.value,
                        "expires_at": permission.expires_at.isoformat()
                        if permission.expires_at
                        else None,
                    },
                )

        # Add owned resources
        for resource_id, owner_id in self.resource_owners.items():
            if owner_id == user.id:
                summary["owned_resources"].append(resource_id)

        return summary

    def update_access_policy(
        self,
        resource_type: ResourceType,
        policy: AccessPolicy,
        admin_user_id: str,
    ) -> bool:
        """
        Update access policy for a resource type.

        Args:
            resource_type: Resource type to update policy for
            policy: New access policy
            admin_user_id: User ID of admin making the change

        Returns:
            True if policy was updated, False otherwise
        """
        self.access_policies[resource_type] = policy
        self._save_policies()

        self.logger.info(
            f"Updated access policy for {resource_type.value} by user {admin_user_id}",
        )
        return True

    def get_authorization_stats(self) -> Dict:
        """Get authorization statistics."""
        total_permissions = sum(
            len(perms) for perms in self.resource_permissions.values()
        )

        return {
            "total_policies": len(self.access_policies),
            "total_explicit_permissions": total_permissions,
            "total_resource_owners": len(self.resource_owners),
            "users_with_permissions": len(self.resource_permissions),
            "resource_types": [rt.value for rt in self.access_policies.keys()],
        }
