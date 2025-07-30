"""Contextual Help System

This module provides contextual help and documentation throughout the ICARUS CLI,
making it easy for users to get assistance when needed.
"""

import json
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set


class HelpTopicType(Enum):
    """Types of help topics."""

    FEATURE = "feature"
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    TROUBLESHOOTING = "troubleshooting"
    REFERENCE = "reference"


@dataclass
class HelpTopic:
    """Individual help topic with content and metadata."""

    id: str
    title: str
    content: str
    topic_type: HelpTopicType
    category: str
    tags: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "topic_type": self.topic_type.value,
            "category": self.category,
            "tags": self.tags,
            "related_topics": self.related_topics,
            "examples": self.examples,
            "see_also": self.see_also,
            "last_updated": self.last_updated,
        }


@dataclass
class ContextualHelp:
    """Context-specific help for UI elements."""

    element_id: str
    title: str
    description: str
    quick_help: str
    detailed_help: Optional[str] = None
    related_topics: List[str] = field(default_factory=list)
    keyboard_shortcuts: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "element_id": self.element_id,
            "title": self.title,
            "description": self.description,
            "quick_help": self.quick_help,
            "detailed_help": self.detailed_help,
            "related_topics": self.related_topics,
            "keyboard_shortcuts": self.keyboard_shortcuts,
        }


class HelpSystem:
    """Manages contextual help and documentation system."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("cli/learning/data")
        self.help_topics: Dict[str, HelpTopic] = {}
        self.contextual_help: Dict[str, ContextualHelp] = {}
        self.search_index: Dict[str, Set[str]] = {}

        # Initialize built-in help content
        self._initialize_help_content()
        self._build_search_index()

    def _initialize_help_content(self) -> None:
        """Initialize built-in help topics and contextual help."""
        # Core interface help
        self._create_interface_help()

        # Analysis help
        self._create_analysis_help()

        # Workflow help
        self._create_workflow_help()

        # Troubleshooting help
        self._create_troubleshooting_help()

    def _create_interface_help(self) -> None:
        """Create help topics for interface elements."""
        # Dashboard help
        dashboard_topic = HelpTopic(
            id="dashboard_overview",
            title="Dashboard Overview",
            content="""The Dashboard is your main workspace in ICARUS CLI.

Key Features:
• Recent analyses and results
• Quick access to common tasks
• System status and notifications
• Project overview and statistics

The dashboard provides a centralized view of your work and quick access to frequently used features.""",
            topic_type=HelpTopicType.FEATURE,
            category="Interface",
            tags=["dashboard", "workspace", "overview"],
            examples=[
                {
                    "title": "Accessing Recent Work",
                    "description": "Click on any item in the Recent Analyses section to quickly return to previous work.",
                },
            ],
        )
        self.help_topics[dashboard_topic.id] = dashboard_topic

        # Navigation help
        navigation_topic = HelpTopic(
            id="navigation_basics",
            title="Navigation Basics",
            content="""Navigate the ICARUS CLI efficiently using these methods:

Keyboard Navigation:
• Arrow keys: Move between menu items
• Tab/Shift+Tab: Navigate forward/backward
• Enter: Select current item
• Escape: Go back or cancel
• Number keys: Quick selection in menus

Mouse Navigation:
• Click to select items
• Scroll to navigate long lists
• Right-click for context menus (where available)

Global Shortcuts:
• Ctrl+H: Show help
• Ctrl+Q: Quit application
• F5: Refresh current screen
• F1: Context-sensitive help""",
            topic_type=HelpTopicType.PROCEDURE,
            category="Interface",
            tags=["navigation", "keyboard", "shortcuts"],
            related_topics=["keyboard_shortcuts"],
        )
        self.help_topics[navigation_topic.id] = navigation_topic

        # Contextual help for dashboard
        dashboard_help = ContextualHelp(
            element_id="dashboard",
            title="Dashboard",
            description="Main workspace showing recent work and quick access to features",
            quick_help="Your central workspace for managing analyses and projects",
            detailed_help="The dashboard provides an overview of your recent work, system status, and quick access to commonly used features. Use it to navigate to different parts of ICARUS or resume previous work.",
            related_topics=["dashboard_overview", "navigation_basics"],
            keyboard_shortcuts=[
                {"key": "1-9", "action": "Quick access to menu items"},
                {"key": "F5", "action": "Refresh dashboard"},
            ],
        )
        self.contextual_help[dashboard_help.element_id] = dashboard_help

    def _create_analysis_help(self) -> None:
        """Create help topics for analysis features."""
        # Airfoil analysis help
        airfoil_topic = HelpTopic(
            id="airfoil_analysis",
            title="Airfoil Analysis",
            content="""Analyze 2D airfoil performance using various solvers.

Supported Analysis Types:
• Viscous analysis with XFoil
• Inviscid panel methods
• Polar generation over angle of attack ranges
• Pressure distribution analysis

Key Parameters:
• Reynolds Number: Affects boundary layer behavior
• Mach Number: For compressibility effects
• Angle of Attack: Single value or range
• Transition Location: For boundary layer transition

The analysis generates lift, drag, and moment coefficients along with detailed flow information.""",
            topic_type=HelpTopicType.FEATURE,
            category="Analysis",
            tags=["airfoil", "xfoil", "2d", "analysis"],
            examples=[
                {
                    "title": "Basic Polar Analysis",
                    "description": "Set Reynolds number to 1e6, angle range -5° to 15°, and run XFoil analysis",
                },
                {
                    "title": "High-Speed Analysis",
                    "description": "Include Mach number effects for transonic airfoil analysis",
                },
            ],
            related_topics=["xfoil_parameters", "polar_interpretation"],
        )
        self.help_topics[airfoil_topic.id] = airfoil_topic

        # XFoil parameters help
        xfoil_params_topic = HelpTopic(
            id="xfoil_parameters",
            title="XFoil Parameters",
            content="""Understanding XFoil analysis parameters:

Reynolds Number:
• Typical values: 1e4 to 1e8
• Higher Re = thinner boundary layer
• Affects stall characteristics and drag

Mach Number:
• 0.0 to 0.8 for XFoil validity
• Compressibility effects above M=0.3
• Critical Mach number varies by airfoil

Angle of Attack:
• Single value: -20° to +20° typical range
• Range analysis: specify start, end, increment
• Stall limits depend on airfoil and Re

Transition:
• Free transition: Natural boundary layer transition
• Fixed transition: Specify x/c location
• Affects drag and separation characteristics""",
            topic_type=HelpTopicType.REFERENCE,
            category="Analysis",
            tags=["xfoil", "parameters", "reynolds", "mach"],
            related_topics=["airfoil_analysis", "polar_interpretation"],
        )
        self.help_topics[xfoil_params_topic.id] = xfoil_params_topic

        # Contextual help for analysis screen
        analysis_help = ContextualHelp(
            element_id="analysis_screen",
            title="Analysis Configuration",
            description="Configure and run aerodynamic analyses",
            quick_help="Set up analysis parameters and run simulations",
            detailed_help="The analysis screen allows you to configure various types of aerodynamic analyses. Select your analysis type, configure parameters, and run simulations to generate results.",
            related_topics=["airfoil_analysis", "airplane_analysis"],
            keyboard_shortcuts=[
                {"key": "Ctrl+R", "action": "Run analysis"},
                {"key": "Ctrl+S", "action": "Save configuration"},
                {"key": "F1", "action": "Parameter help"},
            ],
        )
        self.contextual_help[analysis_help.element_id] = analysis_help

    def _create_workflow_help(self) -> None:
        """Create help topics for workflow features."""
        workflow_topic = HelpTopic(
            id="workflow_basics",
            title="Workflow System",
            content="""Create and manage complex analysis workflows.

Workflow Components:
• Analysis Steps: Individual analysis tasks
• Data Flow: How results pass between steps
• Conditions: Logic for branching workflows
• Parameters: Variables that can be swept

Benefits:
• Automate repetitive analyses
• Ensure consistent procedures
• Parameter studies and optimization
• Reproducible results

Workflows can be saved, shared, and reused across projects.""",
            topic_type=HelpTopicType.CONCEPT,
            category="Workflow",
            tags=["workflow", "automation", "analysis"],
            examples=[
                {
                    "title": "Airfoil Optimization",
                    "description": "Create workflow to optimize airfoil shape for maximum L/D ratio",
                },
                {
                    "title": "Parameter Study",
                    "description": "Sweep Reynolds number and angle of attack for sensitivity analysis",
                },
            ],
            related_topics=["workflow_builder", "parameter_studies"],
        )
        self.help_topics[workflow_topic.id] = workflow_topic

    def _create_troubleshooting_help(self) -> None:
        """Create troubleshooting help topics."""
        convergence_topic = HelpTopic(
            id="convergence_issues",
            title="Analysis Convergence Issues",
            content="""Troubleshooting analysis convergence problems:

Common Causes:
• Extreme operating conditions (high angle of attack, low Re)
• Poor geometry quality (sharp corners, discontinuities)
• Inappropriate solver settings
• Insufficient computational resources

Solutions:
• Reduce angle of attack range near stall
• Check airfoil geometry for issues
• Adjust solver tolerance settings
• Try different solver algorithms
• Increase iteration limits

For XFoil specifically:
• Use viscous analysis for realistic results
• Enable transition prediction
• Adjust panel density for complex geometries""",
            topic_type=HelpTopicType.TROUBLESHOOTING,
            category="Analysis",
            tags=["convergence", "troubleshooting", "xfoil", "errors"],
            related_topics=["xfoil_parameters", "error_solutions"],
        )
        self.help_topics[convergence_topic.id] = convergence_topic

    def _build_search_index(self) -> None:
        """Build search index for help topics."""
        self.search_index.clear()

        for topic_id, topic in self.help_topics.items():
            # Index title words
            title_words = topic.title.lower().split()
            for word in title_words:
                if word not in self.search_index:
                    self.search_index[word] = set()
                self.search_index[word].add(topic_id)

            # Index content words (first 100 words to avoid huge index)
            content_words = topic.content.lower().split()[:100]
            for word in content_words:
                if len(word) > 3:  # Skip short words
                    if word not in self.search_index:
                        self.search_index[word] = set()
                    self.search_index[word].add(topic_id)

            # Index tags
            for tag in topic.tags:
                if tag not in self.search_index:
                    self.search_index[tag] = set()
                self.search_index[tag].add(topic_id)

    def search_help(self, query: str, max_results: int = 10) -> List[HelpTopic]:
        """Search help topics by query string."""
        query_words = query.lower().split()
        topic_scores: Dict[str, int] = {}

        for word in query_words:
            # Exact matches
            if word in self.search_index:
                for topic_id in self.search_index[word]:
                    topic_scores[topic_id] = topic_scores.get(topic_id, 0) + 3

            # Partial matches
            for index_word, topic_ids in self.search_index.items():
                if word in index_word or index_word in word:
                    for topic_id in topic_ids:
                        topic_scores[topic_id] = topic_scores.get(topic_id, 0) + 1

        # Sort by relevance score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top results
        results = []
        for topic_id, score in sorted_topics[:max_results]:
            if topic_id in self.help_topics:
                results.append(self.help_topics[topic_id])

        return results

    def get_contextual_help(self, element_id: str) -> Optional[ContextualHelp]:
        """Get contextual help for a UI element."""
        return self.contextual_help.get(element_id)

    def get_help_topic(self, topic_id: str) -> Optional[HelpTopic]:
        """Get a specific help topic."""
        return self.help_topics.get(topic_id)

    def get_topics_by_category(self, category: str) -> List[HelpTopic]:
        """Get all help topics in a category."""
        return [
            topic
            for topic in self.help_topics.values()
            if topic.category.lower() == category.lower()
        ]

    def get_related_topics(self, topic_id: str) -> List[HelpTopic]:
        """Get topics related to the specified topic."""
        if topic_id not in self.help_topics:
            return []

        topic = self.help_topics[topic_id]
        related = []

        for related_id in topic.related_topics:
            if related_id in self.help_topics:
                related.append(self.help_topics[related_id])

        return related

    def add_help_topic(self, topic: HelpTopic) -> None:
        """Add a new help topic."""
        self.help_topics[topic.id] = topic
        self._build_search_index()  # Rebuild index

    def add_contextual_help(self, help_item: ContextualHelp) -> None:
        """Add contextual help for a UI element."""
        self.contextual_help[help_item.element_id] = help_item

    def get_all_categories(self) -> List[str]:
        """Get all available help categories."""
        categories = set()
        for topic in self.help_topics.values():
            categories.add(topic.category)
        return sorted(list(categories))

    def save_help_data(self, filepath: Path = None) -> None:
        """Save help data to file."""
        if filepath is None:
            filepath = self.data_dir / "help_data.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "help_topics": {
                tid: topic.to_dict() for tid, topic in self.help_topics.items()
            },
            "contextual_help": {
                eid: help_item.to_dict()
                for eid, help_item in self.contextual_help.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_help_data(self, filepath: Path = None) -> None:
        """Load help data from file."""
        if filepath is None:
            filepath = self.data_dir / "help_data.json"

        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)

            # Load help topics
            for topic_id, topic_data in data.get("help_topics", {}).items():
                topic = HelpTopic(
                    id=topic_data["id"],
                    title=topic_data["title"],
                    content=topic_data["content"],
                    topic_type=HelpTopicType(topic_data["topic_type"]),
                    category=topic_data["category"],
                    tags=topic_data.get("tags", []),
                    related_topics=topic_data.get("related_topics", []),
                    examples=topic_data.get("examples", []),
                    see_also=topic_data.get("see_also", []),
                    last_updated=topic_data.get("last_updated"),
                )
                self.help_topics[topic_id] = topic

            # Load contextual help
            for element_id, help_data in data.get("contextual_help", {}).items():
                help_item = ContextualHelp(
                    element_id=help_data["element_id"],
                    title=help_data["title"],
                    description=help_data["description"],
                    quick_help=help_data["quick_help"],
                    detailed_help=help_data.get("detailed_help"),
                    related_topics=help_data.get("related_topics", []),
                    keyboard_shortcuts=help_data.get("keyboard_shortcuts", []),
                )
                self.contextual_help[element_id] = help_item

            self._build_search_index()
