"""ICARUS CLI Application Framework

This module provides the core application framework for the ICARUS CLI,
including the main application controller, screen management, and event system.
"""

from .event_system import EventSystem
from .main_app import IcarusApp
from .screen_manager import ScreenManager
from .state_manager import StateManager

__all__ = [
    "IcarusApp",
    "ScreenManager",
    "EventSystem",
    "StateManager",
]
