"""Screen Transition System for ICARUS CLI

Provides smooth transitions between different screens and views in the TUI,
with support for various transition effects and animations.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from textual.app import App
from textual.screen import Screen
from textual.widget import Widget


class TransitionType(Enum):
    """Types of screen transitions."""

    FADE = "fade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    INSTANT = "instant"


class TransitionDirection(Enum):
    """Direction of transition."""

    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class TransitionConfig:
    """Configuration for screen transitions."""

    type: TransitionType = TransitionType.FADE
    duration: float = 0.3  # Duration in seconds
    easing: str = "ease_in_out"  # Easing function
    direction: TransitionDirection = TransitionDirection.FORWARD

    # Advanced options
    overlay: bool = False  # Whether to overlay screens during transition
    blur_background: bool = False  # Blur background during transition
    scale_factor: float = 1.0  # Scale factor for zoom transitions

    # Callbacks
    on_start: Optional[Callable[[], None]] = None
    on_complete: Optional[Callable[[], None]] = None
    on_cancel: Optional[Callable[[], None]] = None


class TransitionState(Enum):
    """State of a transition."""

    IDLE = "idle"
    PREPARING = "preparing"
    ANIMATING = "animating"
    COMPLETING = "completing"
    CANCELLED = "cancelled"


class ScreenTransition:
    """Represents an active screen transition."""

    def __init__(
        self,
        from_screen: Optional[Screen],
        to_screen: Screen,
        config: TransitionConfig,
    ):
        self.from_screen = from_screen
        self.to_screen = to_screen
        self.config = config
        self.state = TransitionState.IDLE
        self.progress = 0.0
        self.start_time: Optional[float] = None
        self._animation_task: Optional[asyncio.Task] = None

    async def execute(self, app: App) -> bool:
        """Execute the transition."""
        try:
            self.state = TransitionState.PREPARING

            # Call start callback
            if self.config.on_start:
                self.config.on_start()

            # Prepare screens for transition
            await self._prepare_transition(app)

            # Execute the animation
            self.state = TransitionState.ANIMATING
            await self._animate_transition(app)

            # Complete the transition
            self.state = TransitionState.COMPLETING
            await self._complete_transition(app)

            # Call completion callback
            if self.config.on_complete:
                self.config.on_complete()

            return True

        except asyncio.CancelledError:
            self.state = TransitionState.CANCELLED
            if self.config.on_cancel:
                self.config.on_cancel()
            return False
        except Exception as e:
            self.state = TransitionState.CANCELLED
            # Log error
            print(f"Transition error: {e}")
            return False

    async def _prepare_transition(self, app: App) -> None:
        """Prepare screens for transition."""
        # Setup initial states based on transition type
        if self.config.type == TransitionType.FADE:
            if self.from_screen:
                self.from_screen.styles.opacity = 1.0
            self.to_screen.styles.opacity = 0.0

        elif self.config.type in [
            TransitionType.SLIDE_LEFT,
            TransitionType.SLIDE_RIGHT,
        ]:
            # Note: Textual has limited support for transforms
            # These would be simulated through layout changes
            pass

        elif self.config.type in [TransitionType.SLIDE_UP, TransitionType.SLIDE_DOWN]:
            pass

        # Mount the new screen if not already mounted
        if self.to_screen not in app.screen_stack:
            app.push_screen(self.to_screen)

    async def _animate_transition(self, app: App) -> None:
        """Execute the transition animation."""
        import time

        self.start_time = time.time()
        steps = max(10, int(self.config.duration * 30))  # 30 FPS
        step_duration = self.config.duration / steps

        for step in range(steps + 1):
            if self.state == TransitionState.CANCELLED:
                break

            # Calculate progress with easing
            raw_progress = step / steps
            self.progress = self._apply_easing(raw_progress)

            # Apply transition effects
            await self._apply_transition_step(app)

            # Wait for next frame
            if step < steps:
                await asyncio.sleep(step_duration)

    def _apply_easing(self, progress: float) -> float:
        """Apply easing function to progress."""
        if self.config.easing == "linear":
            return progress
        elif self.config.easing == "ease_in":
            return progress * progress
        elif self.config.easing == "ease_out":
            return 1 - (1 - progress) * (1 - progress)
        elif self.config.easing == "ease_in_out":
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - 2 * (1 - progress) * (1 - progress)
        else:
            return progress

    async def _apply_transition_step(self, app: App) -> None:
        """Apply a single step of the transition."""
        if self.config.type == TransitionType.FADE:
            if self.from_screen:
                self.from_screen.styles.opacity = 1.0 - self.progress
            self.to_screen.styles.opacity = self.progress

        elif self.config.type == TransitionType.SLIDE_LEFT:
            # Simulate slide by adjusting layout
            # This is limited in Textual but we can try
            pass

        elif self.config.type == TransitionType.ZOOM_IN:
            # Limited zoom support in terminal
            pass

        # Force refresh
        app.refresh()

    async def _complete_transition(self, app: App) -> None:
        """Complete the transition."""
        # Ensure final states are set
        if self.from_screen:
            self.from_screen.styles.opacity = 0.0
        self.to_screen.styles.opacity = 1.0

        # Remove the old screen from stack if needed
        if self.from_screen and self.from_screen in app.screen_stack:
            # Don't pop if it's the base screen
            if len(app.screen_stack) > 1:
                try:
                    app.pop_screen()
                except:
                    pass

        app.refresh()

    def cancel(self) -> None:
        """Cancel the transition."""
        self.state = TransitionState.CANCELLED
        if self._animation_task:
            self._animation_task.cancel()


class ScreenTransitionManager:
    """Manages screen transitions for the ICARUS CLI."""

    def __init__(self, app: App):
        self.app = app
        self._active_transition: Optional[ScreenTransition] = None
        self._transition_history: List[Dict[str, Any]] = []
        self._default_config = TransitionConfig()

        # Predefined transition configurations
        self._transition_presets: Dict[str, TransitionConfig] = {
            "quick_fade": TransitionConfig(
                type=TransitionType.FADE,
                duration=0.15,
                easing="ease_out",
            ),
            "smooth_fade": TransitionConfig(
                type=TransitionType.FADE,
                duration=0.3,
                easing="ease_in_out",
            ),
            "slide_forward": TransitionConfig(
                type=TransitionType.SLIDE_LEFT,
                duration=0.25,
                easing="ease_out",
                direction=TransitionDirection.FORWARD,
            ),
            "slide_back": TransitionConfig(
                type=TransitionType.SLIDE_RIGHT,
                duration=0.25,
                easing="ease_out",
                direction=TransitionDirection.BACKWARD,
            ),
            "instant": TransitionConfig(type=TransitionType.INSTANT, duration=0.0),
        }

    async def transition_to_screen(
        self,
        screen: Screen,
        config: Optional[TransitionConfig] = None,
        preset: Optional[str] = None,
    ) -> bool:
        """Transition to a new screen."""

        # Cancel any active transition
        if self._active_transition:
            self._active_transition.cancel()
            await asyncio.sleep(0.1)  # Brief pause

        # Determine configuration
        if preset and preset in self._transition_presets:
            transition_config = self._transition_presets[preset]
        elif config:
            transition_config = config
        else:
            transition_config = self._default_config

        # Get current screen
        current_screen = self.app.screen if hasattr(self.app, "screen") else None

        # Create and execute transition
        self._active_transition = ScreenTransition(
            from_screen=current_screen,
            to_screen=screen,
            config=transition_config,
        )

        # Record transition in history
        self._transition_history.append(
            {
                "from": current_screen.__class__.__name__ if current_screen else "None",
                "to": screen.__class__.__name__,
                "config": transition_config,
                "timestamp": asyncio.get_event_loop().time(),
            },
        )

        # Keep history limited
        if len(self._transition_history) > 50:
            self._transition_history = self._transition_history[-50:]

        # Execute transition
        success = await self._active_transition.execute(self.app)
        self._active_transition = None

        return success

    def set_default_config(self, config: TransitionConfig) -> None:
        """Set the default transition configuration."""
        self._default_config = config

    def add_preset(self, name: str, config: TransitionConfig) -> None:
        """Add a custom transition preset."""
        self._transition_presets[name] = config

    def get_preset(self, name: str) -> Optional[TransitionConfig]:
        """Get a transition preset by name."""
        return self._transition_presets.get(name)

    def get_available_presets(self) -> List[str]:
        """Get list of available transition presets."""
        return list(self._transition_presets.keys())

    def is_transitioning(self) -> bool:
        """Check if a transition is currently active."""
        return (
            self._active_transition is not None
            and self._active_transition.state == TransitionState.ANIMATING
        )

    def get_transition_progress(self) -> float:
        """Get progress of current transition (0.0 to 1.0)."""
        if self._active_transition:
            return self._active_transition.progress
        return 0.0

    def cancel_current_transition(self) -> None:
        """Cancel the current transition if active."""
        if self._active_transition:
            self._active_transition.cancel()

    def get_transition_history(self) -> List[Dict[str, Any]]:
        """Get the transition history."""
        return self._transition_history.copy()

    def clear_history(self) -> None:
        """Clear the transition history."""
        self._transition_history.clear()


class AnimatedWidget(Widget):
    """Base class for widgets with animation support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._animations: Dict[str, asyncio.Task] = {}

    async def animate_property(
        self,
        property_name: str,
        target_value: Any,
        duration: float = 0.3,
        easing: str = "ease_in_out",
    ) -> None:
        """Animate a widget property."""

        # Cancel existing animation for this property
        if property_name in self._animations:
            self._animations[property_name].cancel()

        # Start new animation
        self._animations[property_name] = asyncio.create_task(
            self._animate_property_task(property_name, target_value, duration, easing),
        )

    async def _animate_property_task(
        self,
        property_name: str,
        target_value: Any,
        duration: float,
        easing: str,
    ) -> None:
        """Task for animating a property."""
        try:
            # Get current value
            current_value = getattr(self, property_name, 0)

            # Calculate steps
            steps = max(10, int(duration * 30))  # 30 FPS
            step_duration = duration / steps

            for step in range(steps + 1):
                # Calculate progress with easing
                raw_progress = step / steps
                progress = self._apply_easing(raw_progress, easing)

                # Interpolate value
                if isinstance(current_value, (int, float)) and isinstance(
                    target_value,
                    (int, float),
                ):
                    value = current_value + (target_value - current_value) * progress
                    setattr(self, property_name, value)

                # Refresh widget
                self.refresh()

                # Wait for next frame
                if step < steps:
                    await asyncio.sleep(step_duration)

        except asyncio.CancelledError:
            pass
        finally:
            # Clean up
            if property_name in self._animations:
                del self._animations[property_name]

    def _apply_easing(self, progress: float, easing: str) -> float:
        """Apply easing function to progress."""
        if easing == "linear":
            return progress
        elif easing == "ease_in":
            return progress * progress
        elif easing == "ease_out":
            return 1 - (1 - progress) * (1 - progress)
        elif easing == "ease_in_out":
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - 2 * (1 - progress) * (1 - progress)
        else:
            return progress

    def stop_all_animations(self) -> None:
        """Stop all active animations."""
        for animation in self._animations.values():
            animation.cancel()
        self._animations.clear()
