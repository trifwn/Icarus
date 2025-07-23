"""Guided Tutorial System

This module provides a comprehensive tutorial system for new users to learn
ICARUS capabilities through interactive guided tours.
"""

import json
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional


class TutorialStepType(Enum):
    """Types of tutorial steps."""

    INTRODUCTION = "introduction"
    DEMONSTRATION = "demonstration"
    INTERACTION = "interaction"
    EXPLANATION = "explanation"
    PRACTICE = "practice"
    SUMMARY = "summary"


@dataclass
class TutorialStep:
    """Individual step in a tutorial."""

    id: str
    title: str
    content: str
    step_type: TutorialStepType
    duration_estimate: int  # seconds
    prerequisites: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    validation: Optional[Callable] = None
    hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "step_type": self.step_type.value,
            "duration_estimate": self.duration_estimate,
            "prerequisites": self.prerequisites,
            "actions": self.actions,
            "hints": self.hints,
        }


@dataclass
class Tutorial:
    """Complete tutorial with multiple steps."""

    id: str
    title: str
    description: str
    category: str
    difficulty: str  # beginner, intermediate, advanced
    estimated_duration: int  # minutes
    steps: List[TutorialStep] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tutorial to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "difficulty": self.difficulty,
            "estimated_duration": self.estimated_duration,
            "steps": [step.to_dict() for step in self.steps],
            "prerequisites": self.prerequisites,
            "learning_objectives": self.learning_objectives,
        }


class TutorialSystem:
    """Manages guided tutorials for new users."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("cli/learning/data")
        self.tutorials: Dict[str, Tutorial] = {}
        self.user_progress: Dict[str, Dict[str, Any]] = {}
        self.current_tutorial: Optional[str] = None
        self.current_step: int = 0

        # Initialize built-in tutorials
        self._initialize_tutorials()

    def _initialize_tutorials(self) -> None:
        """Initialize built-in tutorials."""
        # Welcome Tutorial
        welcome_tutorial = self._create_welcome_tutorial()
        self.tutorials[welcome_tutorial.id] = welcome_tutorial

        # Airfoil Analysis Tutorial
        airfoil_tutorial = self._create_airfoil_tutorial()
        self.tutorials[airfoil_tutorial.id] = airfoil_tutorial

        # Airplane Analysis Tutorial
        airplane_tutorial = self._create_airplane_tutorial()
        self.tutorials[airplane_tutorial.id] = airplane_tutorial

        # Workflow Tutorial
        workflow_tutorial = self._create_workflow_tutorial()
        self.tutorials[workflow_tutorial.id] = workflow_tutorial

    def _create_welcome_tutorial(self) -> Tutorial:
        """Create the welcome tutorial for first-time users."""
        steps = [
            TutorialStep(
                id="welcome_intro",
                title="Welcome to ICARUS",
                content="""Welcome to ICARUS - Advanced Aircraft Design & Analysis!

ICARUS is a comprehensive aerodynamics software suite that helps you:
• Analyze airfoil performance
• Design and analyze complete aircraft
• Run optimization studies
• Create complex analysis workflows

This tutorial will guide you through the key features and help you get started.""",
                step_type=TutorialStepType.INTRODUCTION,
                duration_estimate=30,
            ),
            TutorialStep(
                id="interface_overview",
                title="Interface Overview",
                content="""Let's explore the ICARUS CLI interface:

• Dashboard: Your main workspace with quick access to recent work
• Analysis: Configure and run aerodynamic analyses
• Results: View and visualize analysis results
• Workflow: Create and manage complex analysis workflows
• Settings: Customize your ICARUS experience

Use Ctrl+H anytime to access help, or F5 to refresh the current screen.""",
                step_type=TutorialStepType.DEMONSTRATION,
                duration_estimate=45,
                actions=[
                    {"type": "highlight_navigation", "target": "sidebar"},
                    {"type": "show_keybindings"},
                ],
            ),
            TutorialStep(
                id="navigation_practice",
                title="Navigation Practice",
                content="""Now let's practice navigating the interface.

Try using the arrow keys or Tab to move between menu items.
Press Enter to select an item, or use the number keys for quick access.

Navigate to the Analysis section to continue.""",
                step_type=TutorialStepType.INTERACTION,
                duration_estimate=60,
                actions=[{"type": "wait_for_navigation", "target": "analysis"}],
                hints=[
                    "Use arrow keys to navigate",
                    "Press Enter to select",
                    "Look for the Analysis option in the menu",
                ],
            ),
            TutorialStep(
                id="help_system",
                title="Getting Help",
                content="""ICARUS provides comprehensive help throughout the interface:

• Press Ctrl+H for general help
• Press F1 for contextual help on the current screen
• Look for the (?) icon next to complex features
• Error messages include explanations and solutions

The help system is searchable and includes examples for all features.""",
                step_type=TutorialStepType.EXPLANATION,
                duration_estimate=30,
            ),
            TutorialStep(
                id="welcome_summary",
                title="Tutorial Complete",
                content="""Congratulations! You've completed the welcome tutorial.

You've learned:
• How to navigate the ICARUS interface
• Where to find key features
• How to access help when needed

Next steps:
• Try the Airfoil Analysis tutorial to learn about aerodynamic analysis
• Explore the Dashboard to see what ICARUS can do
• Check out the Settings to customize your experience""",
                step_type=TutorialStepType.SUMMARY,
                duration_estimate=30,
            ),
        ]

        return Tutorial(
            id="welcome",
            title="Welcome to ICARUS",
            description="Introduction to ICARUS CLI for new users",
            category="Getting Started",
            difficulty="beginner",
            estimated_duration=5,
            steps=steps,
            learning_objectives=[
                "Navigate the ICARUS interface",
                "Access help and documentation",
                "Understand key features and capabilities",
            ],
        )

    def _create_airfoil_tutorial(self) -> Tutorial:
        """Create airfoil analysis tutorial."""
        steps = [
            TutorialStep(
                id="airfoil_intro",
                title="Airfoil Analysis Introduction",
                content="""Airfoil analysis is fundamental to aircraft design.

In this tutorial, you'll learn to:
• Load and analyze airfoil geometries
• Configure analysis parameters
• Run XFoil simulations
• Interpret polar plots and performance data

We'll analyze a NACA 2412 airfoil as an example.""",
                step_type=TutorialStepType.INTRODUCTION,
                duration_estimate=45,
            ),
            TutorialStep(
                id="airfoil_selection",
                title="Selecting an Airfoil",
                content="""First, let's select an airfoil to analyze.

ICARUS includes a database of common airfoils, or you can import custom geometries.

Navigate to Analysis > Airfoil Analysis and select NACA 2412 from the database.""",
                step_type=TutorialStepType.INTERACTION,
                duration_estimate=60,
                actions=[
                    {"type": "navigate_to", "target": "analysis.airfoil"},
                    {"type": "wait_for_selection", "target": "naca2412"},
                ],
            ),
            TutorialStep(
                id="analysis_parameters",
                title="Configuring Analysis Parameters",
                content="""Now let's set up the analysis parameters:

• Reynolds Number: Controls viscous effects (try 1e6)
• Angle of Attack Range: -5° to 15° in 1° increments
• Mach Number: 0.1 for low-speed analysis
• Solver: XFoil for 2D viscous analysis

These parameters determine the operating conditions for your analysis.""",
                step_type=TutorialStepType.DEMONSTRATION,
                duration_estimate=90,
            ),
            TutorialStep(
                id="run_analysis",
                title="Running the Analysis",
                content="""Time to run your first analysis!

Click 'Run Analysis' to start the XFoil simulation.
You'll see progress updates as the solver works through each angle of attack.

The analysis typically takes 30-60 seconds depending on your system.""",
                step_type=TutorialStepType.INTERACTION,
                duration_estimate=120,
                actions=[
                    {"type": "wait_for_analysis_start"},
                    {"type": "show_progress"},
                ],
            ),
            TutorialStep(
                id="interpret_results",
                title="Interpreting Results",
                content="""Excellent! Your analysis is complete.

The results show:
• Lift Coefficient (Cl) vs Angle of Attack
• Drag Coefficient (Cd) vs Angle of Attack
• Lift-to-Drag Ratio (L/D) vs Angle of Attack
• Pressure distributions at specific angles

Key insights:
• Maximum L/D occurs around 4-6° angle of attack
• Stall begins around 12-14° for this airfoil
• Minimum drag occurs near 0° angle of attack""",
                step_type=TutorialStepType.EXPLANATION,
                duration_estimate=120,
            ),
        ]

        return Tutorial(
            id="airfoil_analysis",
            title="Airfoil Analysis Basics",
            description="Learn to analyze airfoil performance using XFoil",
            category="Analysis",
            difficulty="beginner",
            estimated_duration=10,
            steps=steps,
            prerequisites=["welcome"],
            learning_objectives=[
                "Select and load airfoil geometries",
                "Configure analysis parameters",
                "Run XFoil simulations",
                "Interpret aerodynamic performance data",
            ],
        )

    def _create_airplane_tutorial(self) -> Tutorial:
        """Create airplane analysis tutorial."""
        steps = [
            TutorialStep(
                id="airplane_intro",
                title="Airplane Analysis Introduction",
                content="""Airplane analysis extends airfoil concepts to complete aircraft.

You'll learn to:
• Define aircraft geometry
• Set up 3D analysis with AVL or GenuVP
• Analyze stability and control characteristics
• Generate flight envelope data

We'll analyze a simple wing-body configuration.""",
                step_type=TutorialStepType.INTRODUCTION,
                duration_estimate=60,
            ),
            TutorialStep(
                id="geometry_definition",
                title="Defining Aircraft Geometry",
                content="""Aircraft geometry includes:
• Wing planform (span, chord, sweep, twist)
• Fuselage shape and size
• Control surfaces (elevator, rudder, ailerons)
• Mass and balance properties

ICARUS provides templates for common configurations,
or you can import from CAD software.""",
                step_type=TutorialStepType.DEMONSTRATION,
                duration_estimate=90,
            ),
        ]

        return Tutorial(
            id="airplane_analysis",
            title="Airplane Analysis Basics",
            description="Learn to analyze complete aircraft configurations",
            category="Analysis",
            difficulty="intermediate",
            estimated_duration=15,
            steps=steps,
            prerequisites=["airfoil_analysis"],
            learning_objectives=[
                "Define aircraft geometry",
                "Configure 3D analysis parameters",
                "Interpret stability and performance results",
            ],
        )

    def _create_workflow_tutorial(self) -> Tutorial:
        """Create workflow tutorial."""
        steps = [
            TutorialStep(
                id="workflow_intro",
                title="Workflow System Introduction",
                content="""Workflows automate complex analysis sequences.

Benefits:
• Consistent analysis procedures
• Automated parameter studies
• Reproducible results
• Time savings for repetitive tasks

We'll create a simple airfoil optimization workflow.""",
                step_type=TutorialStepType.INTRODUCTION,
                duration_estimate=45,
            ),
        ]

        return Tutorial(
            id="workflow_basics",
            title="Workflow System Basics",
            description="Learn to create and manage analysis workflows",
            category="Advanced",
            difficulty="intermediate",
            estimated_duration=12,
            steps=steps,
            prerequisites=["airfoil_analysis"],
            learning_objectives=[
                "Understand workflow concepts",
                "Create simple workflows",
                "Manage workflow execution",
            ],
        )

    def get_available_tutorials(self, user_level: str = "beginner") -> List[Tutorial]:
        """Get tutorials appropriate for user level."""
        tutorials = []
        for tutorial in self.tutorials.values():
            if tutorial.difficulty == user_level or user_level == "all":
                tutorials.append(tutorial)
        return sorted(tutorials, key=lambda t: t.estimated_duration)

    def start_tutorial(self, tutorial_id: str, user_id: str = "default") -> bool:
        """Start a tutorial for a user."""
        if tutorial_id not in self.tutorials:
            return False

        self.current_tutorial = tutorial_id
        self.current_step = 0

        # Initialize user progress
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}

        self.user_progress[user_id][tutorial_id] = {
            "started_at": None,
            "current_step": 0,
            "completed_steps": [],
            "status": "in_progress",
        }

        return True

    def get_current_step(self) -> Optional[TutorialStep]:
        """Get the current tutorial step."""
        if not self.current_tutorial:
            return None

        tutorial = self.tutorials[self.current_tutorial]
        if self.current_step < len(tutorial.steps):
            return tutorial.steps[self.current_step]
        return None

    def advance_step(self, user_id: str = "default") -> bool:
        """Advance to the next tutorial step."""
        if not self.current_tutorial:
            return False

        tutorial = self.tutorials[self.current_tutorial]

        # Mark current step as completed
        if (
            user_id in self.user_progress
            and self.current_tutorial in self.user_progress[user_id]
        ):
            progress = self.user_progress[user_id][self.current_tutorial]
            progress["completed_steps"].append(self.current_step)

        self.current_step += 1

        # Check if tutorial is complete
        if self.current_step >= len(tutorial.steps):
            self._complete_tutorial(user_id)
            return False

        # Update progress
        if (
            user_id in self.user_progress
            and self.current_tutorial in self.user_progress[user_id]
        ):
            self.user_progress[user_id][self.current_tutorial]["current_step"] = (
                self.current_step
            )

        return True

    def _complete_tutorial(self, user_id: str) -> None:
        """Mark tutorial as completed."""
        if (
            user_id in self.user_progress
            and self.current_tutorial in self.user_progress[user_id]
        ):
            progress = self.user_progress[user_id][self.current_tutorial]
            progress["status"] = "completed"
            progress["completed_at"] = None  # Would use datetime in real implementation

        self.current_tutorial = None
        self.current_step = 0

    def get_user_progress(self, user_id: str = "default") -> Dict[str, Any]:
        """Get user's tutorial progress."""
        return self.user_progress.get(user_id, {})

    def suggest_next_tutorial(self, user_id: str = "default") -> Optional[Tutorial]:
        """Suggest the next tutorial based on user progress."""
        progress = self.get_user_progress(user_id)
        completed_tutorials = [
            tid for tid, data in progress.items() if data.get("status") == "completed"
        ]

        # Find tutorials with satisfied prerequisites
        for tutorial in self.tutorials.values():
            if tutorial.id in completed_tutorials:
                continue

            # Check if prerequisites are met
            if all(prereq in completed_tutorials for prereq in tutorial.prerequisites):
                return tutorial

        return None

    def save_progress(self, filepath: Path = None) -> None:
        """Save user progress to file."""
        if filepath is None:
            filepath = self.data_dir / "tutorial_progress.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.user_progress, f, indent=2)

    def load_progress(self, filepath: Path = None) -> None:
        """Load user progress from file."""
        if filepath is None:
            filepath = self.data_dir / "tutorial_progress.json"

        if filepath.exists():
            with open(filepath) as f:
                self.user_progress = json.load(f)
