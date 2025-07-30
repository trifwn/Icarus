"""Animation System for ICARUS CLI

Provides animation capabilities for TUI components with support for
various animation types and easing functions.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from textual.widget import Widget


class AnimationType(Enum):
    """Types of animations available."""

    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SLIDE_IN = "slide_in"
    SLIDE_OUT = "slide_out"
    PULSE = "pulse"
    BOUNCE = "bounce"
    SHAKE = "shake"
    GLOW = "glow"
    TYPING = "typing"
    PROGRESS = "progress"


class EasingFunction(Enum):
    """Easing functions for animations."""

    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EASE_IN_QUAD = "ease_in_quad"
    EASE_OUT_QUAD = "ease_out_quad"
    EASE_IN_OUT_QUAD = "ease_in_out_quad"
    EASE_IN_CUBIC = "ease_in_cubic"
    EASE_OUT_CUBIC = "ease_out_cubic"
    EASE_IN_OUT_CUBIC = "ease_in_out_cubic"
    BOUNCE = "bounce"
    ELASTIC = "elastic"


@dataclass
class AnimationConfig:
    """Configuration for an animation."""

    type: AnimationType
    duration: float = 1.0  # Duration in seconds
    easing: EasingFunction = EasingFunction.EASE_IN_OUT
    delay: float = 0.0  # Delay before starting
    repeat: int = 1  # Number of repetitions (0 = infinite)
    reverse: bool = False  # Reverse animation on completion

    # Animation-specific parameters
    start_value: Optional[float] = None
    end_value: Optional[float] = None
    amplitude: float = 1.0  # For bounce, shake, etc.
    frequency: float = 1.0  # For oscillating animations

    # Callbacks
    on_start: Optional[Callable[[], None]] = None
    on_update: Optional[Callable[[float], None]] = None
    on_complete: Optional[Callable[[], None]] = None
    on_cancel: Optional[Callable[[], None]] = None


class Animation:
    """Represents an active animation."""

    def __init__(
        self,
        widget: Widget,
        config: AnimationConfig,
        property_name: Optional[str] = None,
    ):
        self.widget = widget
        self.config = config
        self.property_name = property_name
        self.start_time: Optional[float] = None
        self.current_repetition = 0
        self.is_running = False
        self.is_paused = False
        self.progress = 0.0
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the animation."""
        if self.is_running:
            return

        self.is_running = True
        self._task = asyncio.create_task(self._animate())

    async def _animate(self) -> None:
        """Main animation loop."""
        try:
            # Initial delay
            if self.config.delay > 0:
                await asyncio.sleep(self.config.delay)

            # Call start callback
            if self.config.on_start:
                self.config.on_start()

            # Animation loop
            while (
                self.current_repetition < self.config.repeat or self.config.repeat == 0
            ):
                await self._animate_cycle()

                self.current_repetition += 1

                # Break if not infinite repeat
                if (
                    self.config.repeat != 0
                    and self.current_repetition >= self.config.repeat
                ):
                    break

            # Call completion callback
            if self.config.on_complete:
                self.config.on_complete()

        except asyncio.CancelledError:
            if self.config.on_cancel:
                self.config.on_cancel()
        finally:
            self.is_running = False

    async def _animate_cycle(self) -> None:
        """Animate a single cycle."""
        self.start_time = time.time()
        steps = max(30, int(self.config.duration * 60))  # 60 FPS target
        step_duration = self.config.duration / steps

        for step in range(steps + 1):
            if not self.is_running:
                break

            # Wait if paused
            while self.is_paused and self.is_running:
                await asyncio.sleep(0.1)

            # Calculate progress
            raw_progress = step / steps
            self.progress = self._apply_easing(raw_progress)

            # Apply animation
            await self._apply_animation_step()

            # Call update callback
            if self.config.on_update:
                self.config.on_update(self.progress)

            # Wait for next frame
            if step < steps:
                await asyncio.sleep(step_duration)

        # Reverse animation if configured
        if self.config.reverse:
            await self._animate_reverse_cycle()

    async def _animate_reverse_cycle(self) -> None:
        """Animate reverse cycle."""
        steps = max(30, int(self.config.duration * 60))
        step_duration = self.config.duration / steps

        for step in range(steps + 1):
            if not self.is_running:
                break

            while self.is_paused and self.is_running:
                await asyncio.sleep(0.1)

            raw_progress = 1.0 - (step / steps)
            self.progress = self._apply_easing(raw_progress)

            await self._apply_animation_step()

            if self.config.on_update:
                self.config.on_update(self.progress)

            if step < steps:
                await asyncio.sleep(step_duration)

    def _apply_easing(self, progress: float) -> float:
        """Apply easing function to progress."""
        easing = self.config.easing

        if easing == EasingFunction.LINEAR:
            return progress
        elif easing == EasingFunction.EASE_IN:
            return progress * progress
        elif easing == EasingFunction.EASE_OUT:
            return 1 - (1 - progress) * (1 - progress)
        elif easing == EasingFunction.EASE_IN_OUT:
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - 2 * (1 - progress) * (1 - progress)
        elif easing == EasingFunction.EASE_IN_QUAD:
            return progress * progress
        elif easing == EasingFunction.EASE_OUT_QUAD:
            return 1 - (1 - progress) * (1 - progress)
        elif easing == EasingFunction.EASE_IN_OUT_QUAD:
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - 2 * (1 - progress) * (1 - progress)
        elif easing == EasingFunction.EASE_IN_CUBIC:
            return progress * progress * progress
        elif easing == EasingFunction.EASE_OUT_CUBIC:
            return 1 - (1 - progress) ** 3
        elif easing == EasingFunction.EASE_IN_OUT_CUBIC:
            if progress < 0.5:
                return 4 * progress * progress * progress
            else:
                return 1 - 4 * (1 - progress) ** 3
        elif easing == EasingFunction.BOUNCE:
            return self._bounce_easing(progress)
        elif easing == EasingFunction.ELASTIC:
            return self._elastic_easing(progress)
        else:
            return progress

    def _bounce_easing(self, progress: float) -> float:
        """Bounce easing function."""
        if progress < 1 / 2.75:
            return 7.5625 * progress * progress
        elif progress < 2 / 2.75:
            progress -= 1.5 / 2.75
            return 7.5625 * progress * progress + 0.75
        elif progress < 2.5 / 2.75:
            progress -= 2.25 / 2.75
            return 7.5625 * progress * progress + 0.9375
        else:
            progress -= 2.625 / 2.75
            return 7.5625 * progress * progress + 0.984375

    def _elastic_easing(self, progress: float) -> float:
        """Elastic easing function."""
        if progress == 0 or progress == 1:
            return progress

        import math

        period = 0.3
        amplitude = 1.0
        s = period / 4

        return -(
            amplitude
            * (2 ** (10 * (progress - 1)))
            * math.sin((progress - 1 - s) * (2 * math.pi) / period)
        )

    async def _apply_animation_step(self) -> None:
        """Apply a single animation step."""
        animation_type = self.config.type

        if animation_type == AnimationType.FADE_IN:
            await self._apply_fade_in()
        elif animation_type == AnimationType.FADE_OUT:
            await self._apply_fade_out()
        elif animation_type == AnimationType.PULSE:
            await self._apply_pulse()
        elif animation_type == AnimationType.BOUNCE:
            await self._apply_bounce()
        elif animation_type == AnimationType.SHAKE:
            await self._apply_shake()
        elif animation_type == AnimationType.GLOW:
            await self._apply_glow()
        elif animation_type == AnimationType.TYPING:
            await self._apply_typing()
        elif animation_type == AnimationType.PROGRESS:
            await self._apply_progress()

    async def _apply_fade_in(self) -> None:
        """Apply fade in animation."""
        # Limited opacity support in Textual
        if hasattr(self.widget.styles, "opacity"):
            self.widget.styles.opacity = self.progress
        else:
            # Simulate with color changes or visibility
            if self.progress > 0.5:
                self.widget.display = True

    async def _apply_fade_out(self) -> None:
        """Apply fade out animation."""
        if hasattr(self.widget.styles, "opacity"):
            self.widget.styles.opacity = 1.0 - self.progress
        else:
            if self.progress > 0.5:
                self.widget.display = False

    async def _apply_pulse(self) -> None:
        """Apply pulse animation."""
        import math

        pulse_value = 0.5 + 0.5 * math.sin(
            self.progress * 2 * math.pi * self.config.frequency,
        )

        # Apply pulse effect (could be opacity, scale, or color)
        if hasattr(self.widget.styles, "opacity"):
            self.widget.styles.opacity = pulse_value

    async def _apply_bounce(self) -> None:
        """Apply bounce animation."""
        # Simulate bounce with margin or padding changes
        bounce_offset = int(
            self.config.amplitude * abs(self._bounce_easing(self.progress)),
        )

        # Apply bounce effect
        if hasattr(self.widget.styles, "margin_top"):
            self.widget.styles.margin_top = bounce_offset

    async def _apply_shake(self) -> None:
        """Apply shake animation."""
        import math

        shake_offset = int(
            self.config.amplitude
            * math.sin(self.progress * 2 * math.pi * self.config.frequency * 10),
        )

        # Apply shake effect
        if hasattr(self.widget.styles, "margin_left"):
            self.widget.styles.margin_left = shake_offset

    async def _apply_glow(self) -> None:
        """Apply glow animation."""
        # Simulate glow with border or background changes
        import math

        glow_intensity = 0.5 + 0.5 * math.sin(
            self.progress * 2 * math.pi * self.config.frequency,
        )

        # Apply glow effect through CSS classes
        if glow_intensity > 0.7:
            self.widget.add_class("glow-high")
            self.widget.remove_class("glow-low")
        elif glow_intensity > 0.3:
            self.widget.add_class("glow-medium")
            self.widget.remove_class("glow-high")
            self.widget.remove_class("glow-low")
        else:
            self.widget.add_class("glow-low")
            self.widget.remove_class("glow-high")
            self.widget.remove_class("glow-medium")

    async def _apply_typing(self) -> None:
        """Apply typing animation."""
        if hasattr(self.widget, "renderable") and hasattr(
            self.widget.renderable,
            "plain",
        ):
            full_text = str(self.widget.renderable.plain)
            visible_length = int(len(full_text) * self.progress)
            visible_text = full_text[:visible_length]

            # Add cursor if not complete
            if self.progress < 1.0:
                visible_text += "_"

            self.widget.update(visible_text)

    async def _apply_progress(self) -> None:
        """Apply progress animation."""
        if self.property_name and hasattr(self.widget, self.property_name):
            start_val = self.config.start_value or 0
            end_val = self.config.end_value or 100
            current_val = start_val + (end_val - start_val) * self.progress
            setattr(self.widget, self.property_name, current_val)

    def pause(self) -> None:
        """Pause the animation."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume the animation."""
        self.is_paused = False

    def stop(self) -> None:
        """Stop the animation."""
        self.is_running = False
        if self._task:
            self._task.cancel()


class AnimationManager:
    """Manages animations for widgets."""

    def __init__(self):
        self._animations: Dict[str, Animation] = {}
        self._widget_animations: Dict[Widget, List[str]] = {}

    async def animate(
        self,
        widget: Widget,
        config: AnimationConfig,
        animation_id: Optional[str] = None,
        property_name: Optional[str] = None,
    ) -> str:
        """Start an animation on a widget."""

        # Generate animation ID if not provided
        if animation_id is None:
            animation_id = f"{id(widget)}_{config.type.value}_{time.time()}"

        # Stop existing animation with same ID
        if animation_id in self._animations:
            self._animations[animation_id].stop()

        # Create and start animation
        animation = Animation(widget, config, property_name)
        self._animations[animation_id] = animation

        # Track animations per widget
        if widget not in self._widget_animations:
            self._widget_animations[widget] = []
        self._widget_animations[widget].append(animation_id)

        # Start animation
        await animation.start()

        return animation_id

    def stop_animation(self, animation_id: str) -> None:
        """Stop a specific animation."""
        if animation_id in self._animations:
            self._animations[animation_id].stop()
            del self._animations[animation_id]

    def stop_widget_animations(self, widget: Widget) -> None:
        """Stop all animations for a widget."""
        if widget in self._widget_animations:
            for animation_id in self._widget_animations[widget]:
                if animation_id in self._animations:
                    self._animations[animation_id].stop()
                    del self._animations[animation_id]
            del self._widget_animations[widget]

    def stop_all_animations(self) -> None:
        """Stop all active animations."""
        for animation in self._animations.values():
            animation.stop()
        self._animations.clear()
        self._widget_animations.clear()

    def pause_animation(self, animation_id: str) -> None:
        """Pause a specific animation."""
        if animation_id in self._animations:
            self._animations[animation_id].pause()

    def resume_animation(self, animation_id: str) -> None:
        """Resume a specific animation."""
        if animation_id in self._animations:
            self._animations[animation_id].resume()

    def get_animation_progress(self, animation_id: str) -> float:
        """Get progress of a specific animation."""
        if animation_id in self._animations:
            return self._animations[animation_id].progress
        return 0.0

    def is_animation_running(self, animation_id: str) -> bool:
        """Check if an animation is running."""
        if animation_id in self._animations:
            return self._animations[animation_id].is_running
        return False

    def get_active_animations(self) -> List[str]:
        """Get list of active animation IDs."""
        return [aid for aid, anim in self._animations.items() if anim.is_running]

    def get_widget_animations(self, widget: Widget) -> List[str]:
        """Get animations for a specific widget."""
        return self._widget_animations.get(widget, []).copy()

    # Convenience methods for common animations
    async def fade_in(
        self,
        widget: Widget,
        duration: float = 0.5,
        easing: EasingFunction = EasingFunction.EASE_OUT,
    ) -> str:
        """Fade in a widget."""
        config = AnimationConfig(
            type=AnimationType.FADE_IN,
            duration=duration,
            easing=easing,
        )
        return await self.animate(widget, config)

    async def fade_out(
        self,
        widget: Widget,
        duration: float = 0.5,
        easing: EasingFunction = EasingFunction.EASE_IN,
    ) -> str:
        """Fade out a widget."""
        config = AnimationConfig(
            type=AnimationType.FADE_OUT,
            duration=duration,
            easing=easing,
        )
        return await self.animate(widget, config)

    async def pulse(
        self,
        widget: Widget,
        duration: float = 1.0,
        repeat: int = 0,
        frequency: float = 1.0,
    ) -> str:
        """Make a widget pulse."""
        config = AnimationConfig(
            type=AnimationType.PULSE,
            duration=duration,
            repeat=repeat,
            frequency=frequency,
            easing=EasingFunction.EASE_IN_OUT,
        )
        return await self.animate(widget, config)

    async def shake(
        self,
        widget: Widget,
        duration: float = 0.5,
        amplitude: float = 2.0,
        frequency: float = 10.0,
    ) -> str:
        """Shake a widget."""
        config = AnimationConfig(
            type=AnimationType.SHAKE,
            duration=duration,
            amplitude=amplitude,
            frequency=frequency,
            easing=EasingFunction.EASE_OUT,
        )
        return await self.animate(widget, config)

    async def type_text(
        self,
        widget: Widget,
        duration: float = 2.0,
        easing: EasingFunction = EasingFunction.LINEAR,
    ) -> str:
        """Animate typing text."""
        config = AnimationConfig(
            type=AnimationType.TYPING,
            duration=duration,
            easing=easing,
        )
        return await self.animate(widget, config)
