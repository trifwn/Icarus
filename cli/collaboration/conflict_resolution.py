"""
Conflict Resolution System for Real-time Collaboration

This module handles detection and resolution of conflicts that occur when
multiple users simultaneously edit the same data or parameters.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from uuid import uuid4

from .session_sharing import SessionManager
from .user_manager import UserRole


class ConflictType(str, Enum):
    """Types of conflicts that can occur"""

    PARAMETER_CONFLICT = "parameter_conflict"
    TEXT_EDIT_CONFLICT = "text_edit_conflict"
    ANALYSIS_CONFIG_CONFLICT = "analysis_config_conflict"
    WORKFLOW_CONFLICT = "workflow_conflict"
    DATA_CONFLICT = "data_conflict"
    UI_STATE_CONFLICT = "ui_state_conflict"


class ConflictSeverity(str, Enum):
    """Severity levels for conflicts"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts"""

    LAST_WRITER_WINS = "last_writer_wins"
    FIRST_WRITER_WINS = "first_writer_wins"
    MERGE_CHANGES = "merge_changes"
    MANUAL_RESOLUTION = "manual_resolution"
    OWNER_PRIORITY = "owner_priority"
    ROLE_BASED = "role_based"
    COLLABORATIVE_VOTE = "collaborative_vote"


class ConflictStatus(str, Enum):
    """Status of conflict resolution"""

    DETECTED = "detected"
    PENDING = "pending"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    FAILED = "failed"


@dataclass
class ConflictingChange:
    """Represents a single conflicting change"""

    id: str
    user_id: str
    username: str
    timestamp: datetime
    change_type: str
    path: str
    old_value: Any
    new_value: Any
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.username,
            "timestamp": self.timestamp.isoformat(),
            "change_type": self.change_type,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConflictingChange":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            username=data["username"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            change_type=data["change_type"],
            path=data["path"],
            old_value=data["old_value"],
            new_value=data["new_value"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conflict:
    """Represents a conflict between multiple changes"""

    id: str
    session_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    path: str
    description: str
    changes: List[ConflictingChange]
    detected_at: datetime
    status: ConflictStatus = ConflictStatus.DETECTED
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolved_value: Any = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""
    votes: Dict[str, str] = None  # user_id -> choice_id for voting

    def __post_init__(self):
        if self.votes is None:
            self.votes = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "path": self.path,
            "description": self.description,
            "changes": [change.to_dict() for change in self.changes],
            "detected_at": self.detected_at.isoformat(),
            "status": self.status.value,
            "resolution_strategy": self.resolution_strategy.value
            if self.resolution_strategy
            else None,
            "resolved_value": self.resolved_value,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "votes": self.votes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Conflict":
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            conflict_type=ConflictType(data["conflict_type"]),
            severity=ConflictSeverity(data["severity"]),
            path=data["path"],
            description=data["description"],
            changes=[ConflictingChange.from_dict(c) for c in data["changes"]],
            detected_at=datetime.fromisoformat(data["detected_at"]),
            status=ConflictStatus(data.get("status", "detected")),
            resolution_strategy=ResolutionStrategy(data["resolution_strategy"])
            if data.get("resolution_strategy")
            else None,
            resolved_value=data.get("resolved_value"),
            resolved_by=data.get("resolved_by"),
            resolved_at=datetime.fromisoformat(data["resolved_at"])
            if data.get("resolved_at")
            else None,
            resolution_notes=data.get("resolution_notes", ""),
            votes=data.get("votes", {}),
        )


class ConflictResolver:
    """Handles conflict detection and resolution"""

    def __init__(
        self,
        session_manager: SessionManager,
        data_dir: str = "~/.icarus/collaboration",
    ):
        self.session_manager = session_manager
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.conflicts_file = self.data_dir / "conflicts.json"

        # In-memory storage
        self.active_conflicts: Dict[str, List[Conflict]] = {}  # session_id -> conflicts
        self.resolved_conflicts: Dict[
            str,
            List[Conflict],
        ] = {}  # session_id -> conflicts

        # Resolution handlers
        self.resolution_handlers: Dict[ResolutionStrategy, Callable] = {}
        self.conflict_detectors: Dict[ConflictType, Callable] = {}

        # Settings
        self.conflict_timeout = 300  # 5 minutes
        self.auto_resolve_threshold = 10  # seconds
        self.voting_timeout = 60  # seconds

        self.logger = logging.getLogger(__name__)

        # Setup default handlers
        self._setup_default_handlers()

        # Load existing conflicts
        self._load_conflicts()

    def _setup_default_handlers(self):
        """Setup default conflict resolution handlers"""
        self.resolution_handlers[ResolutionStrategy.LAST_WRITER_WINS] = (
            self._resolve_last_writer_wins
        )
        self.resolution_handlers[ResolutionStrategy.FIRST_WRITER_WINS] = (
            self._resolve_first_writer_wins
        )
        self.resolution_handlers[ResolutionStrategy.OWNER_PRIORITY] = (
            self._resolve_owner_priority
        )
        self.resolution_handlers[ResolutionStrategy.ROLE_BASED] = (
            self._resolve_role_based
        )
        self.resolution_handlers[ResolutionStrategy.MERGE_CHANGES] = (
            self._resolve_merge_changes
        )

        self.conflict_detectors[ConflictType.PARAMETER_CONFLICT] = (
            self._detect_parameter_conflict
        )
        self.conflict_detectors[ConflictType.TEXT_EDIT_CONFLICT] = (
            self._detect_text_edit_conflict
        )
        self.conflict_detectors[ConflictType.ANALYSIS_CONFIG_CONFLICT] = (
            self._detect_analysis_config_conflict
        )

    def _load_conflicts(self):
        """Load conflicts from storage"""
        try:
            if self.conflicts_file.exists():
                with open(self.conflicts_file) as f:
                    data = json.load(f)

                    for session_id, conflicts_data in data.get(
                        "active_conflicts",
                        {},
                    ).items():
                        self.active_conflicts[session_id] = [
                            Conflict.from_dict(conflict_data)
                            for conflict_data in conflicts_data
                        ]

                    for session_id, conflicts_data in data.get(
                        "resolved_conflicts",
                        {},
                    ).items():
                        self.resolved_conflicts[session_id] = [
                            Conflict.from_dict(conflict_data)
                            for conflict_data in conflicts_data
                        ]

                self.logger.info(
                    f"Loaded conflicts for {len(self.active_conflicts)} sessions",
                )
        except Exception as e:
            self.logger.error(f"Failed to load conflicts: {e}")

    def _save_conflicts(self):
        """Save conflicts to storage"""
        try:
            data = {
                "active_conflicts": {
                    session_id: [conflict.to_dict() for conflict in conflicts]
                    for session_id, conflicts in self.active_conflicts.items()
                },
                "resolved_conflicts": {
                    session_id: [conflict.to_dict() for conflict in conflicts]
                    for session_id, conflicts in self.resolved_conflicts.items()
                },
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.conflicts_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save conflicts: {e}")

    async def detect_conflict(
        self,
        session_id: str,
        changes: List[ConflictingChange],
    ) -> Optional[Conflict]:
        """Detect if changes create a conflict"""
        if len(changes) < 2:
            return None

        # Group changes by path
        path_groups = {}
        for change in changes:
            if change.path not in path_groups:
                path_groups[change.path] = []
            path_groups[change.path].append(change)

        # Check each path for conflicts
        for path, path_changes in path_groups.items():
            if len(path_changes) < 2:
                continue

            # Determine conflict type
            conflict_type = self._determine_conflict_type(path_changes)

            # Check if changes actually conflict
            if await self._changes_conflict(path_changes, conflict_type):
                # Create conflict
                conflict = Conflict(
                    id=str(uuid4()),
                    session_id=session_id,
                    conflict_type=conflict_type,
                    severity=self._determine_severity(path_changes, conflict_type),
                    path=path,
                    description=self._generate_conflict_description(
                        path_changes,
                        conflict_type,
                    ),
                    changes=path_changes,
                    detected_at=datetime.now(),
                )

                # Store conflict
                if session_id not in self.active_conflicts:
                    self.active_conflicts[session_id] = []
                self.active_conflicts[session_id].append(conflict)
                self._save_conflicts()

                self.logger.info(
                    f"Conflict detected in session {session_id}: {conflict.description}",
                )
                return conflict

        return None

    def _determine_conflict_type(
        self,
        changes: List[ConflictingChange],
    ) -> ConflictType:
        """Determine the type of conflict based on changes"""
        change_types = {change.change_type for change in changes}

        if "parameter_change" in change_types:
            return ConflictType.PARAMETER_CONFLICT
        elif "text_edit" in change_types:
            return ConflictType.TEXT_EDIT_CONFLICT
        elif "analysis_config" in change_types:
            return ConflictType.ANALYSIS_CONFIG_CONFLICT
        elif "workflow_update" in change_types:
            return ConflictType.WORKFLOW_CONFLICT
        else:
            return ConflictType.DATA_CONFLICT

    async def _changes_conflict(
        self,
        changes: List[ConflictingChange],
        conflict_type: ConflictType,
    ) -> bool:
        """Check if changes actually conflict with each other"""
        detector = self.conflict_detectors.get(conflict_type)
        if detector:
            return await detector(changes)

        # Default: changes to same path within time window conflict
        if len(changes) < 2:
            return False

        # Sort by timestamp
        sorted_changes = sorted(changes, key=lambda c: c.timestamp)

        # Check if changes are within conflict window
        time_diff = (
            sorted_changes[-1].timestamp - sorted_changes[0].timestamp
        ).total_seconds()
        return time_diff <= self.auto_resolve_threshold

    async def _detect_parameter_conflict(
        self,
        changes: List[ConflictingChange],
    ) -> bool:
        """Detect parameter conflicts"""
        # Parameters conflict if they change the same parameter to different values
        if len(changes) < 2:
            return False

        values = {str(change.new_value) for change in changes}
        return len(values) > 1

    async def _detect_text_edit_conflict(
        self,
        changes: List[ConflictingChange],
    ) -> bool:
        """Detect text editing conflicts"""
        # Text edits conflict if they overlap in range
        for i, change1 in enumerate(changes):
            for change2 in changes[i + 1 :]:
                range1 = change1.metadata.get("range", {})
                range2 = change2.metadata.get("range", {})

                if range1 and range2:
                    if self._ranges_overlap(range1, range2):
                        return True

        return False

    async def _detect_analysis_config_conflict(
        self,
        changes: List[ConflictingChange],
    ) -> bool:
        """Detect analysis configuration conflicts"""
        # Analysis configs conflict if they change critical parameters differently
        critical_params = {"solver", "mesh_density", "convergence_criteria"}

        for param in critical_params:
            param_changes = [c for c in changes if param in str(c.path)]
            if len(param_changes) > 1:
                values = {str(c.new_value) for c in param_changes}
                if len(values) > 1:
                    return True

        return False

    def _ranges_overlap(self, range1: Dict, range2: Dict) -> bool:
        """Check if two text ranges overlap"""
        start1, end1 = range1.get("start", 0), range1.get("end", 0)
        start2, end2 = range2.get("start", 0), range2.get("end", 0)

        return not (end1 <= start2 or end2 <= start1)

    def _determine_severity(
        self,
        changes: List[ConflictingChange],
        conflict_type: ConflictType,
    ) -> ConflictSeverity:
        """Determine conflict severity"""
        if conflict_type == ConflictType.ANALYSIS_CONFIG_CONFLICT:
            return ConflictSeverity.HIGH
        elif conflict_type == ConflictType.WORKFLOW_CONFLICT:
            return ConflictSeverity.MEDIUM
        elif len(changes) > 3:
            return ConflictSeverity.HIGH
        else:
            return ConflictSeverity.LOW

    def _generate_conflict_description(
        self,
        changes: List[ConflictingChange],
        conflict_type: ConflictType,
    ) -> str:
        """Generate a human-readable conflict description"""
        users = [change.username for change in changes]
        path = changes[0].path

        if conflict_type == ConflictType.PARAMETER_CONFLICT:
            return f"Parameter conflict at '{path}' between {', '.join(users)}"
        elif conflict_type == ConflictType.TEXT_EDIT_CONFLICT:
            return f"Text editing conflict at '{path}' between {', '.join(users)}"
        elif conflict_type == ConflictType.ANALYSIS_CONFIG_CONFLICT:
            return f"Analysis configuration conflict at '{path}' between {', '.join(users)}"
        else:
            return f"Data conflict at '{path}' between {', '.join(users)}"

    async def resolve_conflict(
        self,
        conflict_id: str,
        strategy: ResolutionStrategy,
        resolver_user_id: str = None,
    ) -> bool:
        """Resolve a conflict using the specified strategy"""
        # Find the conflict
        conflict = None
        session_id = None

        for sid, conflicts in self.active_conflicts.items():
            for c in conflicts:
                if c.id == conflict_id:
                    conflict = c
                    session_id = sid
                    break
            if conflict:
                break

        if not conflict:
            return False

        conflict.status = ConflictStatus.RESOLVING
        conflict.resolution_strategy = strategy

        # Execute resolution handler
        handler = self.resolution_handlers.get(strategy)
        if handler:
            try:
                resolved_value = await handler(conflict, session_id)

                if resolved_value is not None:
                    conflict.resolved_value = resolved_value
                    conflict.resolved_by = resolver_user_id
                    conflict.resolved_at = datetime.now()
                    conflict.status = ConflictStatus.RESOLVED

                    # Move to resolved conflicts
                    self.active_conflicts[session_id].remove(conflict)
                    if session_id not in self.resolved_conflicts:
                        self.resolved_conflicts[session_id] = []
                    self.resolved_conflicts[session_id].append(conflict)

                    self._save_conflicts()

                    self.logger.info(
                        f"Conflict {conflict_id} resolved using {strategy.value}",
                    )
                    return True
                else:
                    conflict.status = ConflictStatus.FAILED

            except Exception as e:
                self.logger.error(f"Error resolving conflict {conflict_id}: {e}")
                conflict.status = ConflictStatus.FAILED

        return False

    async def _resolve_last_writer_wins(
        self,
        conflict: Conflict,
        session_id: str,
    ) -> Any:
        """Resolve conflict by using the most recent change"""
        latest_change = max(conflict.changes, key=lambda c: c.timestamp)
        return latest_change.new_value

    async def _resolve_first_writer_wins(
        self,
        conflict: Conflict,
        session_id: str,
    ) -> Any:
        """Resolve conflict by using the earliest change"""
        earliest_change = min(conflict.changes, key=lambda c: c.timestamp)
        return earliest_change.new_value

    async def _resolve_owner_priority(self, conflict: Conflict, session_id: str) -> Any:
        """Resolve conflict by prioritizing session owner's change"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None

        # Find owner's change
        for change in conflict.changes:
            if change.user_id == session.owner_id:
                return change.new_value

        # Fallback to last writer wins
        return await self._resolve_last_writer_wins(conflict, session_id)

    async def _resolve_role_based(self, conflict: Conflict, session_id: str) -> Any:
        """Resolve conflict based on user roles"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None

        # Priority order: ADMIN > MODERATOR > COLLABORATOR > GUEST
        role_priority = {
            UserRole.ADMIN: 4,
            UserRole.MODERATOR: 3,
            UserRole.COLLABORATOR: 2,
            UserRole.GUEST: 1,
        }

        # Find change from highest priority role
        best_change = None
        best_priority = 0

        for change in conflict.changes:
            participant = session.get_participant(change.user_id)
            if participant:
                priority = role_priority.get(participant.role, 0)
                if priority > best_priority:
                    best_priority = priority
                    best_change = change

        return best_change.new_value if best_change else None

    async def _resolve_merge_changes(self, conflict: Conflict, session_id: str) -> Any:
        """Attempt to merge conflicting changes"""
        if conflict.conflict_type == ConflictType.TEXT_EDIT_CONFLICT:
            return await self._merge_text_edits(conflict.changes)
        elif conflict.conflict_type == ConflictType.PARAMETER_CONFLICT:
            return await self._merge_parameters(conflict.changes)
        else:
            # Fallback to last writer wins for non-mergeable conflicts
            return await self._resolve_last_writer_wins(conflict, session_id)

    async def _merge_text_edits(self, changes: List[ConflictingChange]) -> str:
        """Merge text editing changes"""
        # Simple merge: concatenate all new text
        merged_text = ""
        for change in sorted(changes, key=lambda c: c.timestamp):
            if isinstance(change.new_value, str):
                merged_text += change.new_value + " "

        return merged_text.strip()

    async def _merge_parameters(self, changes: List[ConflictingChange]) -> Any:
        """Merge parameter changes"""
        # For numeric parameters, use average
        numeric_values = []
        for change in changes:
            try:
                numeric_values.append(float(change.new_value))
            except (ValueError, TypeError):
                pass

        if numeric_values:
            return sum(numeric_values) / len(numeric_values)

        # For non-numeric, use most recent
        latest_change = max(changes, key=lambda c: c.timestamp)
        return latest_change.new_value

    async def start_collaborative_vote(
        self,
        conflict_id: str,
        choices: List[Dict],
    ) -> bool:
        """Start a collaborative vote to resolve a conflict"""
        # Find the conflict
        conflict = None
        session_id = None

        for sid, conflicts in self.active_conflicts.items():
            for c in conflicts:
                if c.id == conflict_id:
                    conflict = c
                    session_id = sid
                    break
            if conflict:
                break

        if not conflict:
            return False

        conflict.status = ConflictStatus.PENDING
        conflict.resolution_strategy = ResolutionStrategy.COLLABORATIVE_VOTE
        conflict.metadata["voting_choices"] = choices
        conflict.metadata["voting_started"] = datetime.now().isoformat()
        conflict.votes = {}

        self._save_conflicts()

        # Schedule vote timeout
        asyncio.create_task(self._handle_vote_timeout(conflict_id, session_id))

        return True

    async def cast_vote(self, conflict_id: str, user_id: str, choice_id: str) -> bool:
        """Cast a vote for conflict resolution"""
        # Find the conflict
        conflict = None
        session_id = None

        for sid, conflicts in self.active_conflicts.items():
            for c in conflicts:
                if c.id == conflict_id:
                    conflict = c
                    session_id = sid
                    break
            if conflict:
                break

        if not conflict or conflict.status != ConflictStatus.PENDING:
            return False

        # Validate choice
        choices = conflict.metadata.get("voting_choices", [])
        valid_choices = [choice["id"] for choice in choices]

        if choice_id not in valid_choices:
            return False

        # Cast vote
        conflict.votes[user_id] = choice_id
        self._save_conflicts()

        # Check if all participants have voted
        session = self.session_manager.get_session(session_id)
        if session:
            online_participants = len(session.get_online_participants())
            if len(conflict.votes) >= online_participants:
                await self._finalize_vote(conflict_id, session_id)

        return True

    async def _handle_vote_timeout(self, conflict_id: str, session_id: str):
        """Handle voting timeout"""
        await asyncio.sleep(self.voting_timeout)
        await self._finalize_vote(conflict_id, session_id)

    async def _finalize_vote(self, conflict_id: str, session_id: str):
        """Finalize voting and resolve conflict"""
        # Find the conflict
        conflict = None

        for c in self.active_conflicts.get(session_id, []):
            if c.id == conflict_id:
                conflict = c
                break

        if not conflict or conflict.status != ConflictStatus.PENDING:
            return

        # Count votes
        vote_counts = {}
        for choice_id in conflict.votes.values():
            vote_counts[choice_id] = vote_counts.get(choice_id, 0) + 1

        if not vote_counts:
            # No votes, use last writer wins
            await self.resolve_conflict(
                conflict_id,
                ResolutionStrategy.LAST_WRITER_WINS,
            )
            return

        # Find winning choice
        winning_choice_id = max(vote_counts.keys(), key=lambda k: vote_counts[k])

        # Find corresponding value
        choices = conflict.metadata.get("voting_choices", [])
        winning_value = None

        for choice in choices:
            if choice["id"] == winning_choice_id:
                winning_value = choice["value"]
                break

        if winning_value is not None:
            conflict.resolved_value = winning_value
            conflict.resolved_by = "collaborative_vote"
            conflict.resolved_at = datetime.now()
            conflict.status = ConflictStatus.RESOLVED
            conflict.resolution_notes = f"Resolved by vote: {vote_counts}"

            # Move to resolved conflicts
            self.active_conflicts[session_id].remove(conflict)
            if session_id not in self.resolved_conflicts:
                self.resolved_conflicts[session_id] = []
            self.resolved_conflicts[session_id].append(conflict)

            self._save_conflicts()

    def get_active_conflicts(self, session_id: str) -> List[Conflict]:
        """Get active conflicts for a session"""
        return self.active_conflicts.get(session_id, [])

    def get_resolved_conflicts(self, session_id: str) -> List[Conflict]:
        """Get resolved conflicts for a session"""
        return self.resolved_conflicts.get(session_id, [])

    def get_conflict_by_id(self, conflict_id: str) -> Optional[Conflict]:
        """Get a specific conflict by ID"""
        for conflicts in self.active_conflicts.values():
            for conflict in conflicts:
                if conflict.id == conflict_id:
                    return conflict

        for conflicts in self.resolved_conflicts.values():
            for conflict in conflicts:
                if conflict.id == conflict_id:
                    return conflict

        return None

    def cleanup_session_conflicts(self, session_id: str):
        """Clean up conflicts for an ended session"""
        if session_id in self.active_conflicts:
            del self.active_conflicts[session_id]

        if session_id in self.resolved_conflicts:
            del self.resolved_conflicts[session_id]

        self._save_conflicts()

        self.logger.info(f"Cleaned up conflicts for session: {session_id}")

    def get_conflict_stats(self) -> Dict:
        """Get conflict resolution statistics"""
        total_active = sum(
            len(conflicts) for conflicts in self.active_conflicts.values()
        )
        total_resolved = sum(
            len(conflicts) for conflicts in self.resolved_conflicts.values()
        )

        # Count by type
        type_counts = {}
        for conflict_type in ConflictType:
            count = 0
            for conflicts in self.active_conflicts.values():
                count += len([c for c in conflicts if c.conflict_type == conflict_type])
            for conflicts in self.resolved_conflicts.values():
                count += len([c for c in conflicts if c.conflict_type == conflict_type])
            type_counts[conflict_type.value] = count

        # Count by resolution strategy
        strategy_counts = {}
        for conflicts in self.resolved_conflicts.values():
            for conflict in conflicts:
                if conflict.resolution_strategy:
                    strategy = conflict.resolution_strategy.value
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            "active_conflicts": total_active,
            "resolved_conflicts": total_resolved,
            "total_conflicts": total_active + total_resolved,
            "conflict_types": type_counts,
            "resolution_strategies": strategy_counts,
            "resolution_rate": (
                total_resolved / (total_active + total_resolved)
                if (total_active + total_resolved) > 0
                else 0
            ),
        }
