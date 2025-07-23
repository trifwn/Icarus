"""Base Widget Library for ICARUS CLI

Provides a comprehensive set of reusable UI components specifically designed
for aerospace applications and terminal interfaces.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Log
from textual.widgets import ProgressBar
from textual.widgets import Static
from textual.widgets import Tree


class ButtonVariant(Enum):
    """Button style variants."""

    PRIMARY = "primary"
    SECONDARY = "default"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "primary"
    OUTLINE = "default"


class InputType(Enum):
    """Input field types."""

    TEXT = "text"
    NUMBER = "number"
    EMAIL = "email"
    PASSWORD = "password"
    SEARCH = "search"


@dataclass
class ValidationRule:
    """Validation rule for form inputs."""

    name: str
    validator: Callable[[str], bool]
    error_message: str


class AerospaceButton(Button):
    """Enhanced button with aerospace-specific styling and behavior."""

    def __init__(
        self,
        label: str,
        variant: ButtonVariant = ButtonVariant.PRIMARY,
        icon: Optional[str] = None,
        tooltip: Optional[str] = None,
        disabled: bool = False,
        **kwargs,
    ):
        # Set the Textual button variant
        kwargs["variant"] = variant.value
        super().__init__(label, disabled=disabled, **kwargs)
        self.custom_variant = variant
        self.icon = icon
        self.tooltip = tooltip
        self._setup_styling()

    def _setup_styling(self) -> None:
        """Setup button styling based on variant."""
        variant_classes = {
            ButtonVariant.PRIMARY: "btn-primary",
            ButtonVariant.SECONDARY: "btn-secondary",
            ButtonVariant.SUCCESS: "btn-success",
            ButtonVariant.WARNING: "btn-warning",
            ButtonVariant.ERROR: "btn-error",
            ButtonVariant.INFO: "btn-info",
            ButtonVariant.OUTLINE: "btn-outline",
        }

        self.add_class(variant_classes.get(self.custom_variant, "btn-primary"))

        if self.icon:
            self.add_class("btn-with-icon")


class ValidatedInput(Container):
    """Input field with built-in validation and error display."""

    class ValidationChanged(Message):
        """Message sent when validation state changes."""

        def __init__(self, is_valid: bool, errors: List[str]) -> None:
            self.is_valid = is_valid
            self.errors = errors
            super().__init__()

    def __init__(
        self,
        label: str,
        placeholder: str = "",
        input_type: InputType = InputType.TEXT,
        validation_rules: Optional[List[ValidationRule]] = None,
        required: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_text = label
        self.placeholder = placeholder
        self.input_type = input_type
        self.validation_rules = validation_rules or []
        self.required = required
        self.is_valid = reactive(True)
        self.errors = reactive([])

        if required:
            self.validation_rules.append(
                ValidationRule(
                    "required",
                    lambda x: bool(x.strip()),
                    "This field is required",
                ),
            )

    def compose(self) -> ComposeResult:
        yield Label(self.label_text, classes="input-label")
        yield Input(
            placeholder=self.placeholder,
            id=f"{self.id}_input" if self.id else "input",
            classes="validated-input",
        )
        yield Label(
            "",
            id=f"{self.id}_error" if self.id else "error",
            classes="error-message",
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes and validate."""
        if event.input.id == f"{self.id}_input":
            self.validate_input(event.value)

    def validate_input(self, value: str) -> None:
        """Validate input value against all rules."""
        errors = []

        for rule in self.validation_rules:
            if not rule.validator(value):
                errors.append(rule.error_message)

        self.is_valid = len(errors) == 0
        self.errors = errors

        # Update error display
        error_label = self.query_one(
            f"#{self.id}_error" if self.id else "#error",
            Label,
        )
        if errors:
            error_label.update("\n".join(errors))
            error_label.add_class("error-visible")
        else:
            error_label.update("")
            error_label.remove_class("error-visible")

        # Send validation message
        self.post_message(self.ValidationChanged(self.is_valid, errors))

    @property
    def value(self) -> str:
        """Get the current input value."""
        input_widget = self.query_one(
            f"#{self.id}_input" if self.id else "Input",
            Input,
        )
        return input_widget.value

    @value.setter
    def value(self, new_value: str) -> None:
        """Set the input value."""
        input_widget = self.query_one(
            f"#{self.id}_input" if self.id else "Input",
            Input,
        )
        input_widget.value = new_value
        self.validate_input(new_value)


class AerospaceProgressBar(Container):
    """Enhanced progress bar with status text and aerospace styling."""

    def __init__(
        self,
        total: float = 100.0,
        show_percentage: bool = True,
        show_eta: bool = False,
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.total = total
        self.show_percentage = show_percentage
        self.show_eta = show_eta
        self.label_text = label
        self.current_progress = reactive(0.0)
        self.status_text = reactive("")

    def compose(self) -> ComposeResult:
        if self.label_text:
            yield Label(self.label_text, classes="progress-label")

        yield ProgressBar(id="progress_bar", total=self.total)

        if self.show_percentage or self.show_eta:
            yield Label("0%", id="progress_status", classes="progress-status")

    def update_progress(self, progress: float, status: Optional[str] = None) -> None:
        """Update progress value and status."""
        self.current_progress = progress

        progress_bar = self.query_one("#progress_bar", ProgressBar)
        progress_bar.update(progress=progress)

        if self.show_percentage or self.show_eta:
            status_label = self.query_one("#progress_status", Label)

            status_parts = []
            if self.show_percentage:
                percentage = (progress / self.total) * 100
                status_parts.append(f"{percentage:.1f}%")

            if status:
                status_parts.append(status)

            status_label.update(" | ".join(status_parts))


class StatusIndicator(Static):
    """Status indicator with color-coded states."""

    class StatusType(Enum):
        SUCCESS = "success"
        WARNING = "warning"
        ERROR = "error"
        INFO = "info"
        LOADING = "loading"
        IDLE = "idle"

    def __init__(
        self,
        status: StatusType = StatusType.IDLE,
        message: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.status = reactive(status)
        self.message = reactive(message)

    def render(self) -> str:
        """Render the status indicator."""
        status_symbols = {
            self.StatusType.SUCCESS: "✓",
            self.StatusType.WARNING: "⚠",
            self.StatusType.ERROR: "✗",
            self.StatusType.INFO: "ℹ",
            self.StatusType.LOADING: "⟳",
            self.StatusType.IDLE: "○",
        }

        symbol = status_symbols.get(self.status, "○")
        return f"{symbol} {self.message}"

    def watch_status(self, status: StatusType) -> None:
        """Update styling when status changes."""
        # Remove all status classes
        for status_type in self.StatusType:
            self.remove_class(f"status-{status_type.value}")

        # Add current status class
        self.add_class(f"status-{status.value}")

    def set_status(self, status: StatusType, message: str = "") -> None:
        """Set status and message."""
        self.status = status
        self.message = message


class AerospaceDataTable(DataTable):
    """Enhanced data table with aerospace-specific features."""

    def __init__(
        self,
        show_header: bool = True,
        show_row_numbers: bool = False,
        sortable: bool = True,
        filterable: bool = False,
        **kwargs,
    ):
        super().__init__(show_header=show_header, **kwargs)
        self.show_row_numbers = show_row_numbers
        self.sortable = sortable
        self.filterable = filterable
        self._sort_column: Optional[str] = None
        self._sort_reverse: bool = False
        self._filter_text: str = ""

    def add_aerospace_columns(self, columns: List[Dict[str, Any]]) -> None:
        """Add columns with aerospace-specific formatting."""
        for col in columns:
            self.add_column(
                col["label"],
                width=col.get("width"),
                key=col.get("key", col["label"].lower().replace(" ", "_")),
            )

    def add_aerospace_row(
        self,
        data: Dict[str, Any],
        key: Optional[str] = None,
    ) -> None:
        """Add a row with aerospace data formatting."""
        formatted_data = []

        for column in self.columns:
            value = data.get(column.key, "")

            # Format aerospace-specific values
            if isinstance(value, float):
                if "angle" in column.key.lower() or "deg" in str(value):
                    formatted_data.append(f"{value:.2f}°")
                elif "mach" in column.key.lower():
                    formatted_data.append(f"M{value:.3f}")
                elif "altitude" in column.key.lower() or "height" in column.key.lower():
                    formatted_data.append(f"{value:,.0f} ft")
                elif "speed" in column.key.lower() or "velocity" in column.key.lower():
                    formatted_data.append(f"{value:.1f} kt")
                else:
                    formatted_data.append(f"{value:.3f}")
            else:
                formatted_data.append(str(value))

        self.add_row(*formatted_data, key=key)


class FormContainer(Container):
    """Container for forms with built-in validation and submission."""

    class FormSubmitted(Message):
        """Message sent when form is submitted."""

        def __init__(self, data: Dict[str, Any], is_valid: bool) -> None:
            self.data = data
            self.is_valid = is_valid
            super().__init__()

    def __init__(
        self,
        title: str,
        submit_label: str = "Submit",
        cancel_label: str = "Cancel",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.submit_label = submit_label
        self.cancel_label = cancel_label
        self.form_fields: List[ValidatedInput] = []
        self.is_valid = reactive(True)

    def compose(self) -> ComposeResult:
        yield Label(self.title, classes="form-title")
        yield Container(id="form_fields", classes="form-fields")

        with Horizontal(classes="form-actions"):
            yield AerospaceButton(
                self.submit_label,
                variant=ButtonVariant.SUCCESS,
                id="submit_btn",
            )
            yield AerospaceButton(
                self.cancel_label,
                variant=ButtonVariant.SECONDARY,
                id="cancel_btn",
            )

    def add_field(self, field: ValidatedInput) -> None:
        """Add a form field."""
        self.form_fields.append(field)
        fields_container = self.query_one("#form_fields", Container)
        fields_container.mount(field)

    def on_validated_input_validation_changed(
        self,
        event: ValidatedInput.ValidationChanged,
    ) -> None:
        """Handle field validation changes."""
        # Check if all fields are valid
        all_valid = all(field.is_valid for field in self.form_fields)
        self.is_valid = all_valid

        # Enable/disable submit button
        submit_btn = self.query_one("#submit_btn", AerospaceButton)
        submit_btn.disabled = not all_valid

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "submit_btn":
            self.submit_form()
        elif event.button.id == "cancel_btn":
            self.cancel_form()

    def submit_form(self) -> None:
        """Submit the form."""
        form_data = {}
        is_valid = True

        for field in self.form_fields:
            field_id = field.id or f"field_{len(form_data)}"
            form_data[field_id] = field.value
            if not field.is_valid:
                is_valid = False

        self.post_message(self.FormSubmitted(form_data, is_valid))

    def cancel_form(self) -> None:
        """Cancel the form."""
        # Clear all fields
        for field in self.form_fields:
            field.value = ""


class AerospaceTree(Tree):
    """Enhanced tree widget for aerospace data structures."""

    def __init__(self, label: str, **kwargs):
        super().__init__(label, **kwargs)
        self.node_icons = {
            "aircraft": "✈",
            "wing": "🛩",
            "engine": "⚙",
            "analysis": "📊",
            "result": "📈",
            "folder": "📁",
            "file": "📄",
        }

    def add_aerospace_node(
        self,
        parent_node,
        label: str,
        node_type: str = "file",
        data: Optional[Any] = None,
    ):
        """Add a node with aerospace-specific icon."""
        icon = self.node_icons.get(node_type, "•")
        full_label = f"{icon} {label}"
        return parent_node.add(full_label, data=data)


class NotificationPanel(Container):
    """Panel for displaying notifications and alerts."""

    class NotificationType(Enum):
        INFO = "info"
        SUCCESS = "success"
        WARNING = "warning"
        ERROR = "error"

    def __init__(self, max_notifications: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.max_notifications = max_notifications
        self.notifications: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Label("Notifications", classes="panel-title")
        yield Log(id="notification_log", classes="notification-log")

    def add_notification(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a notification to the panel."""
        import time

        if timestamp is None:
            timestamp = time.strftime("%H:%M:%S")

        notification = {
            "message": message,
            "type": notification_type,
            "timestamp": timestamp,
        }

        self.notifications.append(notification)

        # Limit number of notifications
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications :]

        # Add to log
        log = self.query_one("#notification_log", Log)
        type_symbol = {
            self.NotificationType.INFO: "ℹ",
            self.NotificationType.SUCCESS: "✓",
            self.NotificationType.WARNING: "⚠",
            self.NotificationType.ERROR: "✗",
        }.get(notification_type, "•")

        log.write(f"[{timestamp}] {type_symbol} {message}")

    def clear_notifications(self) -> None:
        """Clear all notifications."""
        self.notifications.clear()
        log = self.query_one("#notification_log", Log)
        log.clear()
