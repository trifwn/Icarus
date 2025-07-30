"""
Session Recording and Playback System

This module provides capabilities to record collaboration sessions and play them back
for review, training, and analysis purposes.
"""

import asyncio
import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from uuid import uuid4

from .session_sharing import SessionManager


class RecordingStatus(str, Enum):
    """Status of a recording session"""

    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class EventType(str, Enum):
    """Types of events that can be recorded"""

    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    CHAT_MESSAGE = "chat_message"
    ANNOTATION = "annotation"
    STATE_CHANGE = "state_change"
    SCREEN_CHANGE = "screen_change"
    ANALYSIS_START = "analysis_start"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETE = "analysis_complete"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    CUSTOM = "custom"


class PlaybackSpeed(str, Enum):
    """Playback speed options"""

    QUARTER = "0.25x"
    HALF = "0.5x"
    NORMAL = "1x"
    DOUBLE = "2x"
    QUADRUPLE = "4x"
    MAX = "max"


@dataclass
class RecordedEvent:
    """Represents a single recorded event"""

    id: str
    timestamp: datetime
    event_type: EventType
    user_id: Optional[str]
    username: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "username": self.username,
            "data": self.data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RecordedEvent":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=EventType(data["event_type"]),
            user_id=data.get("user_id"),
            username=data.get("username"),
            data=data["data"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionRecording:
    """Represents a complete session recording"""

    id: str
    session_id: str
    session_name: str
    started_at: datetime
    ended_at: Optional[datetime]
    duration: Optional[timedelta]
    status: RecordingStatus
    participants: List[Dict[str, str]]  # user info
    events: List[RecordedEvent]
    metadata: Dict[str, Any] = None
    file_path: Optional[str] = None
    compressed_size: Optional[int] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "session_name": self.session_name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration": str(self.duration) if self.duration else None,
            "status": self.status.value,
            "participants": self.participants,
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
            "file_path": self.file_path,
            "compressed_size": self.compressed_size,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SessionRecording":
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            session_name=data["session_name"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"])
            if data.get("ended_at")
            else None,
            duration=timedelta(seconds=float(data["duration"].split(":")[-1]))
            if data.get("duration")
            else None,
            status=RecordingStatus(data["status"]),
            participants=data["participants"],
            events=[RecordedEvent.from_dict(e) for e in data["events"]],
            metadata=data.get("metadata", {}),
            file_path=data.get("file_path"),
            compressed_size=data.get("compressed_size"),
        )


class SessionRecorder:
    """Records collaboration sessions for playback"""

    def __init__(
        self,
        session_manager: SessionManager,
        data_dir: str = "~/.icarus/collaboration",
    ):
        self.session_manager = session_manager
        self.data_dir = Path(data_dir).expanduser()
        self.recordings_dir = self.data_dir / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self.recordings_index_file = self.data_dir / "recordings_index.json"

        # Active recordings
        self.active_recordings: Dict[
            str,
            SessionRecording,
        ] = {}  # session_id -> recording
        self.recording_tasks: Dict[str, asyncio.Task] = {}  # session_id -> task

        # Completed recordings index
        self.recordings_index: Dict[str, Dict] = {}  # recording_id -> metadata

        # Settings
        self.max_events_per_recording = 10000
        self.auto_compress = True
        self.retention_days = 30

        self.logger = logging.getLogger(__name__)

        # Load recordings index
        self._load_recordings_index()

    def _load_recordings_index(self):
        """Load recordings index from storage"""
        try:
            if self.recordings_index_file.exists():
                with open(self.recordings_index_file) as f:
                    self.recordings_index = json.load(f)

                self.logger.info(
                    f"Loaded {len(self.recordings_index)} recording entries",
                )
        except Exception as e:
            self.logger.error(f"Failed to load recordings index: {e}")

    def _save_recordings_index(self):
        """Save recordings index to storage"""
        try:
            with open(self.recordings_index_file, "w") as f:
                json.dump(self.recordings_index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save recordings index: {e}")

    async def start_recording(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[SessionRecording]:
        """Start recording a collaboration session"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None

        # Check if already recording
        if session_id in self.active_recordings:
            return self.active_recordings[session_id]

        # Check permissions (only owner or admin can start recording)
        if session.owner_id != user_id:
            participant = session.get_participant(user_id)
            if not participant or participant.role.value not in ["admin", "moderator"]:
                return None

        # Create recording
        recording = SessionRecording(
            id=str(uuid4()),
            session_id=session_id,
            session_name=session.name,
            started_at=datetime.now(),
            ended_at=None,
            duration=None,
            status=RecordingStatus.RECORDING,
            participants=[
                {
                    "user_id": p.user_id,
                    "username": p.username,
                    "display_name": p.display_name,
                    "role": p.role.value,
                }
                for p in session.participants.values()
            ],
            events=[],
        )

        self.active_recordings[session_id] = recording

        # Start recording task
        self.recording_tasks[session_id] = asyncio.create_task(
            self._recording_loop(session_id),
        )

        self.logger.info(f"Started recording session {session_id}")
        return recording

    async def stop_recording(self, session_id: str, user_id: str) -> bool:
        """Stop recording a session"""
        if session_id not in self.active_recordings:
            return False

        recording = self.active_recordings[session_id]

        # Check permissions
        session = self.session_manager.get_session(session_id)
        if session and session.owner_id != user_id:
            participant = session.get_participant(user_id)
            if not participant or participant.role.value not in ["admin", "moderator"]:
                return False

        # Stop recording
        recording.status = RecordingStatus.STOPPED
        recording.ended_at = datetime.now()
        recording.duration = recording.ended_at - recording.started_at

        # Cancel recording task
        if session_id in self.recording_tasks:
            self.recording_tasks[session_id].cancel()
            del self.recording_tasks[session_id]

        # Process and save recording
        await self._process_recording(recording)

        # Remove from active recordings
        del self.active_recordings[session_id]

        self.logger.info(f"Stopped recording session {session_id}")
        return True

    async def pause_recording(self, session_id: str, user_id: str) -> bool:
        """Pause recording a session"""
        if session_id not in self.active_recordings:
            return False

        recording = self.active_recordings[session_id]

        # Check permissions
        session = self.session_manager.get_session(session_id)
        if session and session.owner_id != user_id:
            participant = session.get_participant(user_id)
            if not participant or participant.role.value not in ["admin", "moderator"]:
                return False

        recording.status = RecordingStatus.PAUSED
        return True

    async def resume_recording(self, session_id: str, user_id: str) -> bool:
        """Resume recording a session"""
        if session_id not in self.active_recordings:
            return False

        recording = self.active_recordings[session_id]

        # Check permissions
        session = self.session_manager.get_session(session_id)
        if session and session.owner_id != user_id:
            participant = session.get_participant(user_id)
            if not participant or participant.role.value not in ["admin", "moderator"]:
                return False

        recording.status = RecordingStatus.RECORDING
        return True

    async def record_event(
        self,
        session_id: str,
        event_type: EventType,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        data: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ):
        """Record a single event"""
        if session_id not in self.active_recordings:
            return

        recording = self.active_recordings[session_id]

        if recording.status != RecordingStatus.RECORDING:
            return

        # Create event
        event = RecordedEvent(
            id=str(uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            username=username,
            data=data or {},
            metadata=metadata or {},
        )

        recording.events.append(event)

        # Check event limit
        if len(recording.events) >= self.max_events_per_recording:
            await self.stop_recording(session_id, recording.participants[0]["user_id"])

    async def _recording_loop(self, session_id: str):
        """Background loop for recording session events"""
        try:
            while session_id in self.active_recordings:
                recording = self.active_recordings[session_id]

                if recording.status == RecordingStatus.RECORDING:
                    # Record periodic state snapshots
                    await self._record_state_snapshot(session_id)

                # Wait before next iteration
                await asyncio.sleep(10)  # Record state every 10 seconds

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in recording loop for session {session_id}: {e}")

    async def _record_state_snapshot(self, session_id: str):
        """Record a snapshot of current session state"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return

        # Record participant states
        for participant in session.participants.values():
            if participant.is_online:
                await self.record_event(
                    session_id=session_id,
                    event_type=EventType.STATE_CHANGE,
                    user_id=participant.user_id,
                    username=participant.username,
                    data={
                        "type": "participant_state",
                        "current_screen": participant.current_screen,
                        "cursor_position": participant.cursor_position,
                        "last_active": participant.last_active.isoformat(),
                    },
                )

    async def _process_recording(self, recording: SessionRecording):
        """Process and save a completed recording"""
        recording.status = RecordingStatus.PROCESSING

        try:
            # Generate file path
            filename = f"recording_{recording.id}_{recording.started_at.strftime('%Y%m%d_%H%M%S')}.json"
            file_path = self.recordings_dir / filename

            # Save recording data
            recording_data = recording.to_dict()

            if self.auto_compress:
                # Save as compressed file
                compressed_path = file_path.with_suffix(".json.gz")
                with gzip.open(compressed_path, "wt", encoding="utf-8") as f:
                    json.dump(recording_data, f, indent=2)

                recording.file_path = str(compressed_path)
                recording.compressed_size = compressed_path.stat().st_size
            else:
                # Save as regular JSON
                with open(file_path, "w") as f:
                    json.dump(recording_data, f, indent=2)

                recording.file_path = str(file_path)

            # Update index
            self.recordings_index[recording.id] = {
                "id": recording.id,
                "session_id": recording.session_id,
                "session_name": recording.session_name,
                "started_at": recording.started_at.isoformat(),
                "ended_at": recording.ended_at.isoformat()
                if recording.ended_at
                else None,
                "duration": str(recording.duration) if recording.duration else None,
                "participants": recording.participants,
                "event_count": len(recording.events),
                "file_path": recording.file_path,
                "compressed_size": recording.compressed_size,
                "created_at": datetime.now().isoformat(),
            }

            self._save_recordings_index()

            recording.status = RecordingStatus.READY

            self.logger.info(
                f"Processed recording {recording.id} with {len(recording.events)} events",
            )

        except Exception as e:
            self.logger.error(f"Error processing recording {recording.id}: {e}")
            recording.status = RecordingStatus.ERROR

    def get_recordings(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """Get list of available recordings"""
        recordings = list(self.recordings_index.values())

        if session_id:
            recordings = [r for r in recordings if r["session_id"] == session_id]

        if user_id:
            recordings = [
                r
                for r in recordings
                if any(p["user_id"] == user_id for p in r["participants"])
            ]

        # Sort by creation date (newest first)
        recordings.sort(key=lambda r: r["created_at"], reverse=True)

        return recordings

    async def load_recording(self, recording_id: str) -> Optional[SessionRecording]:
        """Load a recording from storage"""
        if recording_id not in self.recordings_index:
            return None

        recording_info = self.recordings_index[recording_id]
        file_path = Path(recording_info["file_path"])

        if not file_path.exists():
            return None

        try:
            if file_path.suffix == ".gz":
                # Load compressed file
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    recording_data = json.load(f)
            else:
                # Load regular JSON
                with open(file_path) as f:
                    recording_data = json.load(f)

            return SessionRecording.from_dict(recording_data)

        except Exception as e:
            self.logger.error(f"Error loading recording {recording_id}: {e}")
            return None

    def delete_recording(self, recording_id: str, user_id: str) -> bool:
        """Delete a recording"""
        if recording_id not in self.recordings_index:
            return False

        recording_info = self.recordings_index[recording_id]

        # Check permissions (only participants can delete)
        if not any(p["user_id"] == user_id for p in recording_info["participants"]):
            return False

        try:
            # Delete file
            file_path = Path(recording_info["file_path"])
            if file_path.exists():
                file_path.unlink()

            # Remove from index
            del self.recordings_index[recording_id]
            self._save_recordings_index()

            self.logger.info(f"Deleted recording {recording_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting recording {recording_id}: {e}")
            return False

    def cleanup_old_recordings(self):
        """Clean up old recordings based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        to_delete = []
        for recording_id, recording_info in self.recordings_index.items():
            created_at = datetime.fromisoformat(recording_info["created_at"])
            if created_at < cutoff_date:
                to_delete.append(recording_id)

        for recording_id in to_delete:
            # Use system user for cleanup
            self.delete_recording(recording_id, "system")

        if to_delete:
            self.logger.info(f"Cleaned up {len(to_delete)} old recordings")

    def get_recording_stats(self) -> Dict:
        """Get recording statistics"""
        total_recordings = len(self.recordings_index)
        active_recordings = len(self.active_recordings)

        total_events = sum(
            r.get("event_count", 0) for r in self.recordings_index.values()
        )
        total_size = sum(
            r.get("compressed_size", 0) for r in self.recordings_index.values()
        )

        # Calculate average duration
        durations = []
        for recording_info in self.recordings_index.values():
            if recording_info.get("duration"):
                try:
                    # Parse duration string (e.g., "1:23:45")
                    time_parts = recording_info["duration"].split(":")
                    if len(time_parts) == 3:
                        hours, minutes, seconds = map(float, time_parts)
                        total_seconds = hours * 3600 + minutes * 60 + seconds
                        durations.append(total_seconds)
                except:
                    pass

        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_recordings": total_recordings,
            "active_recordings": active_recordings,
            "total_events": total_events,
            "total_size_bytes": total_size,
            "average_duration_seconds": avg_duration,
            "recordings_by_month": self._get_recordings_by_month(),
        }

    def _get_recordings_by_month(self) -> Dict[str, int]:
        """Get recording counts by month"""
        monthly_counts = {}

        for recording_info in self.recordings_index.values():
            try:
                created_at = datetime.fromisoformat(recording_info["created_at"])
                month_key = created_at.strftime("%Y-%m")
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
            except:
                pass

        return monthly_counts


class SessionPlayer:
    """Plays back recorded collaboration sessions"""

    def __init__(self, recorder: SessionRecorder):
        self.recorder = recorder
        self.logger = logging.getLogger(__name__)

        # Playback state
        self.current_recording: Optional[SessionRecording] = None
        self.current_position: int = 0
        self.playback_speed: PlaybackSpeed = PlaybackSpeed.NORMAL
        self.is_playing: bool = False
        self.playback_task: Optional[asyncio.Task] = None

        # Event handlers
        self.event_handlers: Dict[EventType, List] = {}

    def register_event_handler(self, event_type: EventType, handler):
        """Register a handler for playback events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def load_recording(self, recording_id: str) -> bool:
        """Load a recording for playback"""
        recording = await self.recorder.load_recording(recording_id)
        if not recording:
            return False

        self.current_recording = recording
        self.current_position = 0
        self.is_playing = False

        return True

    async def play(self) -> bool:
        """Start playback"""
        if not self.current_recording or self.is_playing:
            return False

        self.is_playing = True
        self.playback_task = asyncio.create_task(self._playback_loop())

        return True

    async def pause(self) -> bool:
        """Pause playback"""
        if not self.is_playing:
            return False

        self.is_playing = False
        if self.playback_task:
            self.playback_task.cancel()
            self.playback_task = None

        return True

    async def stop(self) -> bool:
        """Stop playback"""
        if self.playback_task:
            self.playback_task.cancel()
            self.playback_task = None

        self.is_playing = False
        self.current_position = 0

        return True

    async def seek(self, position: int) -> bool:
        """Seek to a specific position"""
        if not self.current_recording:
            return False

        if 0 <= position < len(self.current_recording.events):
            self.current_position = position
            return True

        return False

    def set_speed(self, speed: PlaybackSpeed):
        """Set playback speed"""
        self.playback_speed = speed

    async def _playback_loop(self):
        """Main playback loop"""
        try:
            if not self.current_recording:
                return

            events = self.current_recording.events
            start_time = datetime.now()
            recording_start = events[0].timestamp if events else datetime.now()

            while self.is_playing and self.current_position < len(events):
                event = events[self.current_position]

                # Calculate when this event should be played
                event_offset = (event.timestamp - recording_start).total_seconds()

                # Apply speed multiplier
                speed_multiplier = self._get_speed_multiplier()
                if speed_multiplier > 0:
                    adjusted_offset = event_offset / speed_multiplier

                    # Wait until it's time to play this event
                    elapsed = (datetime.now() - start_time).total_seconds()
                    wait_time = adjusted_offset - elapsed

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)

                # Execute event
                await self._execute_event(event)

                self.current_position += 1

            # Playback finished
            self.is_playing = False

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in playback loop: {e}")
            self.is_playing = False

    def _get_speed_multiplier(self) -> float:
        """Get speed multiplier for current playback speed"""
        speed_map = {
            PlaybackSpeed.QUARTER: 0.25,
            PlaybackSpeed.HALF: 0.5,
            PlaybackSpeed.NORMAL: 1.0,
            PlaybackSpeed.DOUBLE: 2.0,
            PlaybackSpeed.QUADRUPLE: 4.0,
            PlaybackSpeed.MAX: 0,  # No delay
        }
        return speed_map.get(self.playback_speed, 1.0)

    async def _execute_event(self, event: RecordedEvent):
        """Execute a recorded event during playback"""
        # Execute registered handlers
        handlers = self.event_handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")

    def get_playback_info(self) -> Dict:
        """Get current playback information"""
        if not self.current_recording:
            return {}

        total_events = len(self.current_recording.events)
        progress = (self.current_position / total_events) if total_events > 0 else 0

        current_event = None
        if 0 <= self.current_position < total_events:
            current_event = self.current_recording.events[
                self.current_position
            ].to_dict()

        return {
            "recording_id": self.current_recording.id,
            "session_name": self.current_recording.session_name,
            "total_events": total_events,
            "current_position": self.current_position,
            "progress": progress,
            "is_playing": self.is_playing,
            "playback_speed": self.playback_speed.value,
            "current_event": current_event,
            "duration": str(self.current_recording.duration)
            if self.current_recording.duration
            else None,
        }

    async def get_events_in_range(self, start: int, end: int) -> List[RecordedEvent]:
        """Get events in a specific range"""
        if not self.current_recording:
            return []

        events = self.current_recording.events
        start = max(0, start)
        end = min(len(events), end)

        return events[start:end]

    async def search_events(
        self,
        query: str,
        event_types: List[EventType] = None,
    ) -> List[int]:
        """Search for events matching a query"""
        if not self.current_recording:
            return []

        matching_positions = []
        query_lower = query.lower()

        for i, event in enumerate(self.current_recording.events):
            if event_types and event.event_type not in event_types:
                continue

            # Search in event data
            event_text = json.dumps(event.data).lower()
            if query_lower in event_text:
                matching_positions.append(i)

            # Search in username
            if event.username and query_lower in event.username.lower():
                matching_positions.append(i)

        return matching_positions
