"""
Communication System with Chat and Annotations

This module provides real-time communication features including chat,
annotations, and collaborative commenting for analysis sessions.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from uuid import uuid4

try:
    from cli.api.websocket import WebSocketMessage
    from cli.api.websocket import websocket_manager
except ImportError:
    # Fallback for testing without full API setup
    websocket_manager = None
    WebSocketMessage = None
from .session_sharing import SessionManager
from .user_manager import Permission
from .user_manager import User


class MessageType(str, Enum):
    """Types of chat messages"""

    TEXT = "text"
    SYSTEM = "system"
    NOTIFICATION = "notification"
    ANALYSIS_RESULT = "analysis_result"
    FILE_SHARE = "file_share"
    SCREEN_SHARE = "screen_share"
    EMOJI_REACTION = "emoji_reaction"


class AnnotationType(str, Enum):
    """Types of annotations"""

    NOTE = "note"
    QUESTION = "question"
    SUGGESTION = "suggestion"
    WARNING = "warning"
    ERROR = "error"
    HIGHLIGHT = "highlight"
    DRAWING = "drawing"


class MessageStatus(str, Enum):
    """Status of messages"""

    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    EDITED = "edited"
    DELETED = "deleted"


@dataclass
class ChatMessage:
    """A chat message in a collaboration session"""

    id: str
    session_id: str
    user_id: str
    username: str
    display_name: str
    message_type: MessageType
    content: str
    timestamp: datetime
    status: MessageStatus = MessageStatus.SENT
    reply_to: Optional[str] = None
    attachments: List[Dict] = None
    reactions: Dict[str, List[str]] = None  # emoji -> list of user_ids
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []
        if self.reactions is None:
            self.reactions = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "reply_to": self.reply_to,
            "attachments": self.attachments,
            "reactions": self.reactions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChatMessage":
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            user_id=data["user_id"],
            username=data["username"],
            display_name=data["display_name"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=MessageStatus(data.get("status", "sent")),
            reply_to=data.get("reply_to"),
            attachments=data.get("attachments", []),
            reactions=data.get("reactions", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Annotation:
    """An annotation on analysis results or interface elements"""

    id: str
    session_id: str
    user_id: str
    username: str
    display_name: str
    annotation_type: AnnotationType
    content: str
    target: Dict[str, Any]  # What the annotation is attached to
    position: Dict[str, Any]  # Position information (x, y, etc.)
    timestamp: datetime
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    replies: List[str] = None  # List of message IDs that are replies
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.replies is None:
            self.replies = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "annotation_type": self.annotation_type.value,
            "content": self.content,
            "target": self.target,
            "position": self.position,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "replies": self.replies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Annotation":
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            user_id=data["user_id"],
            username=data["username"],
            display_name=data["display_name"],
            annotation_type=AnnotationType(data["annotation_type"]),
            content=data["content"],
            target=data["target"],
            position=data["position"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            resolved=data.get("resolved", False),
            resolved_by=data.get("resolved_by"),
            resolved_at=datetime.fromisoformat(data["resolved_at"])
            if data.get("resolved_at")
            else None,
            replies=data.get("replies", []),
            metadata=data.get("metadata", {}),
        )


class CommunicationSystem:
    """Manages chat and annotation features for collaboration"""

    def __init__(
        self,
        session_manager: SessionManager,
        data_dir: str = "~/.icarus/collaboration",
    ):
        self.session_manager = session_manager
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.messages_file = self.data_dir / "messages.json"
        self.annotations_file = self.data_dir / "annotations.json"

        # In-memory storage
        self.messages: Dict[str, List[ChatMessage]] = {}  # session_id -> messages
        self.annotations: Dict[str, List[Annotation]] = {}  # session_id -> annotations

        # Settings
        self.max_messages_per_session = 1000
        self.max_message_length = 2000
        self.allowed_file_types = {".txt", ".json", ".csv", ".png", ".jpg", ".pdf"}
        self.max_file_size = 10 * 1024 * 1024  # 10MB

        self.logger = logging.getLogger(__name__)

        # Load existing data
        self._load_messages()
        self._load_annotations()

    def _load_messages(self):
        """Load messages from storage"""
        try:
            if self.messages_file.exists():
                with open(self.messages_file) as f:
                    data = json.load(f)
                    for session_id, message_list in data.get("messages", {}).items():
                        self.messages[session_id] = [
                            ChatMessage.from_dict(msg_data) for msg_data in message_list
                        ]
                self.logger.info(f"Loaded messages for {len(self.messages)} sessions")
        except Exception as e:
            self.logger.error(f"Failed to load messages: {e}")

    def _save_messages(self):
        """Save messages to storage"""
        try:
            data = {
                "messages": {
                    session_id: [msg.to_dict() for msg in messages]
                    for session_id, messages in self.messages.items()
                },
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.messages_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save messages: {e}")

    def _load_annotations(self):
        """Load annotations from storage"""
        try:
            if self.annotations_file.exists():
                with open(self.annotations_file) as f:
                    data = json.load(f)
                    for session_id, annotation_list in data.get(
                        "annotations",
                        {},
                    ).items():
                        self.annotations[session_id] = [
                            Annotation.from_dict(ann_data)
                            for ann_data in annotation_list
                        ]
                self.logger.info(
                    f"Loaded annotations for {len(self.annotations)} sessions",
                )
        except Exception as e:
            self.logger.error(f"Failed to load annotations: {e}")

    def _save_annotations(self):
        """Save annotations to storage"""
        try:
            data = {
                "annotations": {
                    session_id: [ann.to_dict() for ann in annotations]
                    for session_id, annotations in self.annotations.items()
                },
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.annotations_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save annotations: {e}")

    async def send_message(
        self,
        session_id: str,
        user: User,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        reply_to: Optional[str] = None,
        attachments: List[Dict] = None,
    ) -> Optional[ChatMessage]:
        """Send a chat message"""
        # Validate permissions
        if not user.has_permission(Permission.SEND_MESSAGES):
            self.logger.warning(
                f"User {user.username} lacks permission to send messages",
            )
            return None

        # Validate session
        session = self.session_manager.get_session(session_id)
        if not session or not session.is_participant(user.id):
            self.logger.warning(f"User {user.username} not in session {session_id}")
            return None

        # Validate message content
        if len(content) > self.max_message_length:
            self.logger.warning(
                f"Message too long: {len(content)} > {self.max_message_length}",
            )
            return None

        # Create message
        message = ChatMessage(
            id=str(uuid4()),
            session_id=session_id,
            user_id=user.id,
            username=user.username,
            display_name=user.display_name,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            reply_to=reply_to,
            attachments=attachments or [],
        )

        # Add to session messages
        if session_id not in self.messages:
            self.messages[session_id] = []

        self.messages[session_id].append(message)

        # Maintain message limit
        if len(self.messages[session_id]) > self.max_messages_per_session:
            self.messages[session_id] = self.messages[session_id][
                -self.max_messages_per_session :
            ]

        # Save to storage
        self._save_messages()

        # Broadcast to session participants
        await self._broadcast_message(message)

        self.logger.info(f"Message sent by {user.username} in session {session_id}")
        return message

    async def edit_message(self, message_id: str, user: User, new_content: str) -> bool:
        """Edit an existing message"""
        # Find the message
        for session_id, messages in self.messages.items():
            for message in messages:
                if message.id == message_id and message.user_id == user.id:
                    # Update message
                    message.content = new_content
                    message.status = MessageStatus.EDITED
                    message.metadata["edited_at"] = datetime.now().isoformat()

                    # Save and broadcast
                    self._save_messages()
                    await self._broadcast_message_update(message, "edited")

                    return True

        return False

    async def delete_message(self, message_id: str, user: User) -> bool:
        """Delete a message"""
        # Find the message
        for session_id, messages in self.messages.items():
            for i, message in enumerate(messages):
                if message.id == message_id:
                    # Check permissions (owner or moderator)
                    session = self.session_manager.get_session(session_id)
                    if (
                        message.user_id == user.id
                        or user.has_permission(Permission.MODERATE_CHAT)
                        or (session and session.owner_id == user.id)
                    ):
                        # Mark as deleted instead of removing
                        message.status = MessageStatus.DELETED
                        message.content = "[Message deleted]"
                        message.metadata["deleted_at"] = datetime.now().isoformat()
                        message.metadata["deleted_by"] = user.id

                        # Save and broadcast
                        self._save_messages()
                        await self._broadcast_message_update(message, "deleted")

                        return True

        return False

    async def add_reaction(self, message_id: str, user: User, emoji: str) -> bool:
        """Add an emoji reaction to a message"""
        # Find the message
        for session_id, messages in self.messages.items():
            for message in messages:
                if message.id == message_id:
                    # Validate session participation
                    session = self.session_manager.get_session(session_id)
                    if not session or not session.is_participant(user.id):
                        return False

                    # Add reaction
                    if emoji not in message.reactions:
                        message.reactions[emoji] = []

                    if user.id not in message.reactions[emoji]:
                        message.reactions[emoji].append(user.id)

                        # Save and broadcast
                        self._save_messages()
                        await self._broadcast_message_update(
                            message,
                            "reaction_added",
                            {"emoji": emoji, "user_id": user.id},
                        )

                        return True

        return False

    async def remove_reaction(self, message_id: str, user: User, emoji: str) -> bool:
        """Remove an emoji reaction from a message"""
        # Find the message
        for session_id, messages in self.messages.items():
            for message in messages:
                if message.id == message_id:
                    # Remove reaction
                    if (
                        emoji in message.reactions
                        and user.id in message.reactions[emoji]
                    ):
                        message.reactions[emoji].remove(user.id)

                        # Clean up empty reaction lists
                        if not message.reactions[emoji]:
                            del message.reactions[emoji]

                        # Save and broadcast
                        self._save_messages()
                        await self._broadcast_message_update(
                            message,
                            "reaction_removed",
                            {"emoji": emoji, "user_id": user.id},
                        )

                        return True

        return False

    async def create_annotation(
        self,
        session_id: str,
        user: User,
        content: str,
        annotation_type: AnnotationType,
        target: Dict,
        position: Dict,
    ) -> Optional[Annotation]:
        """Create an annotation"""
        # Validate permissions
        if not user.has_permission(Permission.CREATE_ANNOTATIONS):
            self.logger.warning(
                f"User {user.username} lacks permission to create annotations",
            )
            return None

        # Validate session
        session = self.session_manager.get_session(session_id)
        if not session or not session.is_participant(user.id):
            self.logger.warning(f"User {user.username} not in session {session_id}")
            return None

        # Create annotation
        annotation = Annotation(
            id=str(uuid4()),
            session_id=session_id,
            user_id=user.id,
            username=user.username,
            display_name=user.display_name,
            annotation_type=annotation_type,
            content=content,
            target=target,
            position=position,
            timestamp=datetime.now(),
        )

        # Add to session annotations
        if session_id not in self.annotations:
            self.annotations[session_id] = []

        self.annotations[session_id].append(annotation)

        # Save to storage
        self._save_annotations()

        # Broadcast to session participants
        await self._broadcast_annotation(annotation, "created")

        self.logger.info(
            f"Annotation created by {user.username} in session {session_id}",
        )
        return annotation

    async def resolve_annotation(self, annotation_id: str, user: User) -> bool:
        """Resolve an annotation"""
        # Find the annotation
        for session_id, annotations in self.annotations.items():
            for annotation in annotations:
                if annotation.id == annotation_id:
                    # Check permissions (creator, moderator, or session owner)
                    session = self.session_manager.get_session(session_id)
                    if (
                        annotation.user_id == user.id
                        or user.has_permission(Permission.MODERATE_CHAT)
                        or (session and session.owner_id == user.id)
                    ):
                        annotation.resolved = True
                        annotation.resolved_by = user.id
                        annotation.resolved_at = datetime.now()

                        # Save and broadcast
                        self._save_annotations()
                        await self._broadcast_annotation(annotation, "resolved")

                        return True

        return False

    async def reply_to_annotation(
        self,
        annotation_id: str,
        user: User,
        content: str,
    ) -> Optional[ChatMessage]:
        """Reply to an annotation with a message"""
        # Find the annotation
        for session_id, annotations in self.annotations.items():
            for annotation in annotations:
                if annotation.id == annotation_id:
                    # Create a reply message
                    message = await self.send_message(
                        session_id=session_id,
                        user=user,
                        content=content,
                        message_type=MessageType.TEXT,
                        reply_to=annotation_id,
                    )

                    if message:
                        # Add to annotation replies
                        annotation.replies.append(message.id)
                        self._save_annotations()

                        return message

        return None

    def get_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ChatMessage]:
        """Get messages for a session"""
        messages = self.messages.get(session_id, [])
        start = max(0, len(messages) - offset - limit)
        end = len(messages) - offset if offset > 0 else len(messages)
        return messages[start:end]

    def get_annotations(
        self,
        session_id: str,
        resolved: Optional[bool] = None,
    ) -> List[Annotation]:
        """Get annotations for a session"""
        annotations = self.annotations.get(session_id, [])

        if resolved is not None:
            annotations = [ann for ann in annotations if ann.resolved == resolved]

        return annotations

    def search_messages(
        self,
        session_id: str,
        query: str,
        limit: int = 20,
    ) -> List[ChatMessage]:
        """Search messages in a session"""
        messages = self.messages.get(session_id, [])
        query_lower = query.lower()

        matching_messages = [
            msg
            for msg in messages
            if (
                query_lower in msg.content.lower()
                or query_lower in msg.username.lower()
                or query_lower in msg.display_name.lower()
            )
        ]

        return matching_messages[-limit:] if limit else matching_messages

    async def _broadcast_message(self, message: ChatMessage):
        """Broadcast a new message to session participants"""
        if websocket_manager is None or WebSocketMessage is None:
            # Skip broadcasting if websocket is not available
            return

        ws_message = WebSocketMessage(
            type="chat_message",
            payload=message.to_dict(),
            session_id=message.session_id,
        )

        await websocket_manager.broadcast_to_room(message.session_id, ws_message)

    async def _broadcast_message_update(
        self,
        message: ChatMessage,
        action: str,
        data: Dict = None,
    ):
        """Broadcast a message update to session participants"""
        if websocket_manager is None or WebSocketMessage is None:
            # Skip broadcasting if websocket is not available
            return

        payload = message.to_dict()
        payload["action"] = action
        if data:
            payload.update(data)

        ws_message = WebSocketMessage(
            type="chat_message_update",
            payload=payload,
            session_id=message.session_id,
        )

        await websocket_manager.broadcast_to_room(message.session_id, ws_message)

    async def _broadcast_annotation(self, annotation: Annotation, action: str):
        """Broadcast an annotation event to session participants"""
        if websocket_manager is None or WebSocketMessage is None:
            # Skip broadcasting if websocket is not available
            return

        payload = annotation.to_dict()
        payload["action"] = action

        ws_message = WebSocketMessage(
            type="annotation_event",
            payload=payload,
            session_id=annotation.session_id,
        )

        await websocket_manager.broadcast_to_room(annotation.session_id, ws_message)

    def cleanup_session_data(self, session_id: str):
        """Clean up communication data for an ended session"""
        if session_id in self.messages:
            del self.messages[session_id]

        if session_id in self.annotations:
            del self.annotations[session_id]

        self._save_messages()
        self._save_annotations()

        self.logger.info(f"Cleaned up communication data for session: {session_id}")

    def get_communication_stats(self) -> Dict:
        """Get communication statistics"""
        total_messages = sum(len(messages) for messages in self.messages.values())
        total_annotations = sum(
            len(annotations) for annotations in self.annotations.values()
        )
        resolved_annotations = sum(
            len([ann for ann in annotations if ann.resolved])
            for annotations in self.annotations.values()
        )

        message_types = {}
        for message_type in MessageType:
            count = sum(
                len([msg for msg in messages if msg.message_type == message_type])
                for messages in self.messages.values()
            )
            message_types[message_type.value] = count

        annotation_types = {}
        for annotation_type in AnnotationType:
            count = sum(
                len(
                    [
                        ann
                        for ann in annotations
                        if ann.annotation_type == annotation_type
                    ],
                )
                for annotations in self.annotations.values()
            )
            annotation_types[annotation_type.value] = count

        return {
            "active_sessions": len(self.messages),
            "total_messages": total_messages,
            "total_annotations": total_annotations,
            "resolved_annotations": resolved_annotations,
            "pending_annotations": total_annotations - resolved_annotations,
            "message_types": message_types,
            "annotation_types": annotation_types,
            "average_messages_per_session": (
                total_messages / len(self.messages) if self.messages else 0
            ),
        }
