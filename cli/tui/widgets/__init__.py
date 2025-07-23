"""TUI Widgets Package for ICARUS CLI

This package contains custom Textual widgets that integrate with the ICARUS CLI core framework.
"""

from .analysis_widget import AnalysisWidget

# Import new base widgets
from .base_widgets import AerospaceButton
from .base_widgets import AerospaceDataTable
from .base_widgets import AerospaceProgressBar
from .base_widgets import AerospaceTree
from .base_widgets import ButtonVariant
from .base_widgets import FormContainer
from .base_widgets import InputType
from .base_widgets import NotificationPanel
from .base_widgets import StatusIndicator
from .base_widgets import ValidatedInput
from .base_widgets import ValidationRule
from .notification_widget import NotificationWidget
from .progress_widget import ProgressWidget
from .results_widget import ResultsWidget
from .session_widget import SessionWidget
from .workflow_widget import WorkflowWidget

__all__ = [
    "SessionWidget",
    "WorkflowWidget",
    "AnalysisWidget",
    "ResultsWidget",
    "ProgressWidget",
    "NotificationWidget",
    # Base widgets
    "AerospaceButton",
    "ValidatedInput",
    "AerospaceProgressBar",
    "StatusIndicator",
    "AerospaceDataTable",
    "FormContainer",
    "AerospaceTree",
    "NotificationPanel",
    "ButtonVariant",
    "InputType",
    "ValidationRule",
]
