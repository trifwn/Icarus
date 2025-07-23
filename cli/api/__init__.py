"""
ICARUS CLI API Layer

This module provides the REST API and WebSocket interfaces for the ICARUS CLI,
designed to support both the current TUI and future web-based interfaces.
"""

from .adapters import TextualUIAdapter
from .adapters import UIAdapter
from .app import create_api_app
from .models import *
from .websocket import WebSocketManager

__all__ = ["create_api_app", "UIAdapter", "TextualUIAdapter", "WebSocketManager"]
