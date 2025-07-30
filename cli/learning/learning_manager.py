"""Learning Manager

This module provides a unified interface to all learning systems and integrates
them with the main ICARUS CLI application.
"""

from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from .documentation import DocumentationSystem
from .documentation import SearchableDoc
from .error_system import ErrorExplanation
from .error_system import ErrorExplanationSystem
from .help_system import ContextualHelp
from .help_system import HelpSystem
from .help_system import HelpTopic
from .progress_tracker import Achievement
from .progress_tracker import ProgressTracker
from .tutorial_system import Tutorial
from .tutorial_system import TutorialSystem


class LearningManager:
    """Unified manager for all learning and help systems."""

    def __init__(self, data_dir: Path = None, user_id: str = "default"):
        self.data_dir = data_dir or Path("cli/learning/data")
        self.user_id = user_id

        # Initialize all learning systems
        self.tutorial_system = TutorialSystem(self.data_dir)
        self.help_system = HelpSystem(self.data_dir)
        self.error_system = ErrorExplanationSystem(self.data_dir)
        self.progress_tracker = ProgressTracker(self.data_dir)
        self.documentation = DocumentationSystem(self.data_dir)

        # Event callbacks for integration with main app
        self.on_tutorial_completed: Optional[Callable] = None
        self.on_achievement_earned: Optional[Callable] = None
        self.on_help_requested: Optional[Callable] = None
        self.on_error_explained: Optional[Callable] = None

        # Load existing data
        self._load_all_data()

    def _load_all_data(self) -> None:
        """Load all learning data from files."""
        try:
            self.tutorial_system.load_progress()
            self.help_system.load_help_data()
            self.error_system.load_error_data()
            self.progress_tracker.load_progress()
            self.documentation.load_documentation()
        except Exception as e:
            # Log error but continue - systems will use defaults
            print(f"Warning: Could not load some learning data: {e}")

    def save_all_data(self) -> None:
        """Save all learning data to files."""
        try:
            self.tutorial_system.save_progress()
            self.help_system.save_help_data()
            self.error_system.save_error_data()
            self.progress_tracker.save_progress()
            self.documentation.save_documentation()
        except Exception as e:
            print(f"Error saving learning data: {e}")

    # Tutorial System Interface
    def get_available_tutorials(self, difficulty: str = "all") -> List[Tutorial]:
        """Get available tutorials for the user."""
        return self.tutorial_system.get_available_tutorials(difficulty)

    def start_tutorial(self, tutorial_id: str) -> bool:
        """Start a tutorial for the current user."""
        success = self.tutorial_system.start_tutorial(tutorial_id, self.user_id)
        if success:
            # Record activity in progress tracker
            self.progress_tracker.record_analysis(self.user_id, "tutorial_started")
        return success

    def complete_tutorial(self, tutorial_id: str) -> List[Achievement]:
        """Complete a tutorial and award achievements."""
        achievements = self.progress_tracker.complete_tutorial(
            self.user_id,
            tutorial_id,
        )

        # Trigger callback if set
        if self.on_tutorial_completed:
            self.on_tutorial_completed(tutorial_id, achievements)

        # Trigger achievement callback
        if achievements and self.on_achievement_earned:
            for achievement in achievements:
                self.on_achievement_earned(achievement)

        return achievements

    def get_current_tutorial_step(self):
        """Get the current tutorial step."""
        return self.tutorial_system.get_current_step()

    def advance_tutorial_step(self) -> bool:
        """Advance to the next tutorial step."""
        return self.tutorial_system.advance_step(self.user_id)

    def get_suggested_tutorial(self) -> Optional[Tutorial]:
        """Get a suggested next tutorial for the user."""
        return self.tutorial_system.suggest_next_tutorial(self.user_id)

    # Help System Interface
    def search_help(self, query: str, max_results: int = 10) -> List[HelpTopic]:
        """Search help topics."""
        results = self.help_system.search_help(query, max_results)

        # Record help usage
        self.progress_tracker.record_analysis(self.user_id, "help_searched")

        # Trigger callback if set
        if self.on_help_requested:
            self.on_help_requested(query, results)

        return results

    def get_contextual_help(self, element_id: str) -> Optional[ContextualHelp]:
        """Get contextual help for a UI element."""
        help_item = self.help_system.get_contextual_help(element_id)

        if help_item:
            # Record contextual help usage
            self.progress_tracker.record_analysis(self.user_id, "contextual_help_used")

        return help_item

    def get_help_topic(self, topic_id: str) -> Optional[HelpTopic]:
        """Get a specific help topic."""
        return self.help_system.get_help_topic(topic_id)

    def get_help_categories(self) -> List[str]:
        """Get all help categories."""
        return self.help_system.get_all_categories()

    # Error System Interface
    def explain_error(
        self,
        error_message: str,
        context: Dict[str, Any] = None,
    ) -> Optional[ErrorExplanation]:
        """Get explanation for an error message."""
        explanation = self.error_system.explain_error(error_message, context)

        if explanation:
            # Record error explanation usage
            self.progress_tracker.record_analysis(self.user_id, "error_explained")

            # Trigger callback if set
            if self.on_error_explained:
                self.on_error_explained(error_message, explanation)

        return explanation

    def record_error_resolution(self, error_type: str) -> List[Achievement]:
        """Record that user successfully resolved an error."""
        achievements = self.progress_tracker.record_error_resolution(
            self.user_id,
            error_type,
        )

        # Trigger achievement callback
        if achievements and self.on_achievement_earned:
            for achievement in achievements:
                self.on_achievement_earned(achievement)

        return achievements

    def get_common_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common errors from history."""
        return self.error_system.get_common_errors(limit=limit)

    # Progress Tracking Interface
    def get_user_progress(self):
        """Get current user's progress."""
        return self.progress_tracker.get_user_progress(self.user_id)

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get user's learning statistics."""
        return self.progress_tracker.get_learning_statistics(self.user_id)

    def get_recommended_next_steps(self) -> List[Dict[str, Any]]:
        """Get recommended next learning steps."""
        return self.progress_tracker.get_recommended_next_steps(self.user_id)

    def record_analysis_completion(
        self,
        analysis_type: str,
        success: bool = True,
    ) -> List[Achievement]:
        """Record completion of an analysis."""
        achievements = self.progress_tracker.record_analysis(
            self.user_id,
            analysis_type,
            success,
        )

        # Trigger achievement callback
        if achievements and self.on_achievement_earned:
            for achievement in achievements:
                self.on_achievement_earned(achievement)

        return achievements

    # Documentation Interface
    def search_documentation(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[SearchableDoc]:
        """Search documentation."""
        results = self.documentation.search(query, max_results)

        # Record documentation usage
        self.progress_tracker.record_analysis(self.user_id, "documentation_searched")

        return [doc for doc, score in results]

    def get_document(self, doc_id: str) -> Optional[SearchableDoc]:
        """Get a specific document."""
        return self.documentation.get_document(doc_id)

    def get_documentation_categories(self) -> List[str]:
        """Get all documentation categories."""
        return self.documentation.get_all_categories()

    def get_related_documents(self, doc_id: str) -> List[SearchableDoc]:
        """Get documents related to the specified document."""
        return self.documentation.get_related_documents(doc_id)

    # Integrated Features
    def show_welcome_for_new_user(self) -> Dict[str, Any]:
        """Show welcome information for new users (Requirement 3.1)."""
        progress = self.get_user_progress()

        # Check if this is a new user
        is_new_user = (
            len(progress.completed_tutorials) == 0 and len(progress.activity_log) == 0
        )

        welcome_info = {
            "is_new_user": is_new_user,
            "welcome_tutorial": None,
            "quick_start_tips": [],
            "recommended_first_steps": [],
        }

        if is_new_user:
            # Get welcome tutorial
            welcome_tutorial = self.tutorial_system.tutorials.get("welcome")
            if welcome_tutorial:
                welcome_info["welcome_tutorial"] = welcome_tutorial

            # Provide quick start tips
            welcome_info["quick_start_tips"] = [
                "Press Ctrl+H anytime for help",
                "Use F1 for context-sensitive help",
                "Start with the Welcome tutorial to learn the basics",
                "Try the Airfoil Analysis tutorial for your first analysis",
            ]

            # Recommend first steps
            welcome_info["recommended_first_steps"] = [
                {
                    "title": "Take the Welcome Tour",
                    "description": "Learn the interface basics",
                    "action": "start_tutorial",
                    "target": "welcome",
                },
                {
                    "title": "Try Your First Analysis",
                    "description": "Analyze a simple airfoil",
                    "action": "start_tutorial",
                    "target": "airfoil_analysis",
                },
                {
                    "title": "Explore the Help System",
                    "description": "Learn how to get assistance",
                    "action": "show_help",
                    "target": "help_system",
                },
            ]

        return welcome_info

    def provide_contextual_assistance(
        self,
        current_screen: str,
        user_action: str = None,
    ) -> Dict[str, Any]:
        """Provide contextual help and suggestions (Requirement 3.2)."""
        assistance = {
            "contextual_help": None,
            "related_topics": [],
            "quick_tips": [],
            "suggested_actions": [],
        }

        # Get contextual help for current screen
        contextual_help = self.get_contextual_help(current_screen)
        if contextual_help:
            assistance["contextual_help"] = contextual_help

            # Get related help topics
            for topic_id in contextual_help.related_topics:
                topic = self.get_help_topic(topic_id)
                if topic:
                    assistance["related_topics"].append(topic)

        # Provide screen-specific tips
        screen_tips = {
            "dashboard": [
                "Recent work appears in the main area",
                "Use the sidebar to navigate to different features",
                "Check notifications for important updates",
            ],
            "analysis": [
                "Select your analysis type first",
                "Hover over parameters for explanations",
                "Use the preview to check your setup",
            ],
            "analysis_screen": [
                "Configure analysis parameters carefully",
                "Check solver availability before running",
                "Save configurations for reuse",
            ],
            "results": [
                "Click on plots to interact with them",
                "Use export options to save your work",
                "Compare results using the comparison tools",
            ],
        }

        assistance["quick_tips"] = screen_tips.get(
            current_screen,
            [
                "Press Ctrl+H for help",
                "Use F1 for context-sensitive help",
                "Check the documentation for detailed information",
            ],
        )

        # Suggest relevant actions based on user progress
        progress = self.get_user_progress()
        if len(progress.completed_tutorials) == 0:
            assistance["suggested_actions"].append(
                {
                    "title": "Start with a tutorial",
                    "description": "Learn the basics with guided tutorials",
                    "action": "show_tutorials",
                },
            )

        return assistance

    def handle_error_with_education(
        self,
        error_message: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Handle errors with educational explanations (Requirement 3.3)."""
        response = {
            "error_explanation": None,
            "solutions": [],
            "learning_opportunity": None,
            "prevention_tips": [],
            "related_help": [],
        }

        # Get error explanation
        explanation = self.explain_error(error_message, context)
        if explanation:
            response["error_explanation"] = explanation
            response["solutions"] = explanation.solutions
            response["prevention_tips"] = explanation.prevention_tips

            # Find related help topics
            for resource in explanation.learning_resources:
                help_topic = self.get_help_topic(resource)
                if help_topic:
                    response["related_help"].append(help_topic)

            # Create learning opportunity
            response["learning_opportunity"] = {
                "title": f"Learn more about {explanation.category.value.replace('_', ' ')}",
                "description": f"Understanding {explanation.title.lower()} can help prevent similar issues",
                "suggested_topics": explanation.learning_resources,
            }

        return response

    def track_learning_progress_and_suggest_next(self) -> Dict[str, Any]:
        """Track progress and suggest next steps (Requirement 3.4)."""
        progress = self.get_user_progress()
        stats = self.get_learning_statistics()
        recommendations = self.get_recommended_next_steps()

        progress_info = {
            "current_level": progress.skill_level.value,
            "total_points": progress.total_points,
            "completion_rate": stats["completion_rate"],
            "learning_streak": stats["learning_streak"],
            "recent_achievements": progress.achievements[-3:]
            if progress.achievements
            else [],
            "next_recommendations": recommendations,
            "milestone_progress": self._calculate_milestone_progress(progress, stats),
        }

        return progress_info

    def _calculate_milestone_progress(self, progress, stats) -> Dict[str, Any]:
        """Calculate progress toward learning milestones."""
        milestones = {
            "first_tutorial": {
                "title": "Complete First Tutorial",
                "current": len(progress.completed_tutorials),
                "target": 1,
                "completed": len(progress.completed_tutorials) >= 1,
            },
            "tutorial_master": {
                "title": "Complete All Beginner Tutorials",
                "current": len(progress.completed_tutorials),
                "target": 3,  # Assuming 3 beginner tutorials
                "completed": len(progress.completed_tutorials) >= 3,
            },
            "analysis_veteran": {
                "title": "Complete 10 Analyses",
                "current": stats["activity_counts"].get("analysis_completed", 0),
                "target": 10,
                "completed": stats["activity_counts"].get("analysis_completed", 0)
                >= 10,
            },
            "consistent_learner": {
                "title": "7-Day Learning Streak",
                "current": stats["learning_streak"],
                "target": 7,
                "completed": stats["learning_streak"] >= 7,
            },
        }

        return milestones

    def provide_searchable_help(self, query: str) -> Dict[str, Any]:
        """Provide comprehensive searchable help (Requirement 3.5)."""
        search_results = {
            "help_topics": self.search_help(query, max_results=5),
            "documentation": self.search_documentation(query, max_results=5),
            "tutorials": [],
            "examples": [],
            "total_results": 0,
        }

        # Search tutorials
        for tutorial in self.get_available_tutorials("all"):
            if (
                query.lower() in tutorial.title.lower()
                or query.lower() in tutorial.description.lower()
                or any(
                    query.lower() in obj.lower() for obj in tutorial.learning_objectives
                )
            ):
                search_results["tutorials"].append(tutorial)

        # Limit tutorial results
        search_results["tutorials"] = search_results["tutorials"][:3]

        # Find examples in documentation
        for doc in search_results["documentation"]:
            if doc.examples:
                for example in doc.examples:
                    # Include examples from relevant documents, even if query doesn't match exactly
                    search_results["examples"].append(
                        {
                            "title": example.title,
                            "description": example.description,
                            "source_doc": doc.title,
                            "example": example,
                        },
                    )

        # If no examples found in search results, add some general examples
        if not search_results["examples"]:
            # Get examples from any documentation
            all_docs = [
                self.documentation.get_document(doc_id)
                for doc_id in self.documentation.documents.keys()
            ]
            for doc in all_docs:
                if doc and doc.examples:
                    for example in doc.examples[:1]:  # Just first example from each doc
                        search_results["examples"].append(
                            {
                                "title": example.title,
                                "description": example.description,
                                "source_doc": doc.title,
                                "example": example,
                            },
                        )
                        if len(search_results["examples"]) >= 3:
                            break
                if len(search_results["examples"]) >= 3:
                    break

        # Limit example results
        search_results["examples"] = search_results["examples"][:3]

        # Calculate total results
        search_results["total_results"] = (
            len(search_results["help_topics"])
            + len(search_results["documentation"])
            + len(search_results["tutorials"])
            + len(search_results["examples"])
        )

        return search_results

    # Utility Methods
    def set_user_id(self, user_id: str) -> None:
        """Set the current user ID."""
        self.user_id = user_id

    def register_callbacks(
        self,
        on_tutorial_completed: Callable = None,
        on_achievement_earned: Callable = None,
        on_help_requested: Callable = None,
        on_error_explained: Callable = None,
    ) -> None:
        """Register callbacks for learning events."""
        if on_tutorial_completed:
            self.on_tutorial_completed = on_tutorial_completed
        if on_achievement_earned:
            self.on_achievement_earned = on_achievement_earned
        if on_help_requested:
            self.on_help_requested = on_help_requested
        if on_error_explained:
            self.on_error_explained = on_error_explained

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all learning systems."""
        return {
            "tutorial_system": {
                "total_tutorials": len(self.tutorial_system.tutorials),
                "user_progress": len(self.tutorial_system.user_progress),
            },
            "help_system": {
                "total_topics": len(self.help_system.help_topics),
                "total_contextual_help": len(self.help_system.contextual_help),
            },
            "error_system": {
                "total_explanations": len(self.error_system.error_explanations),
                "error_history_size": len(self.error_system.error_history),
            },
            "progress_tracker": {
                "total_modules": len(self.progress_tracker.learning_modules),
                "total_achievements": len(self.progress_tracker.achievements),
                "total_users": len(self.progress_tracker.user_progress),
            },
            "documentation": {
                "total_documents": len(self.documentation.documents),
                "search_index_size": len(self.documentation.search_index),
            },
        }
