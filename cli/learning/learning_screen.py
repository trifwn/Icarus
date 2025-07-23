"""Learning Screen

This module provides the main learning interface that integrates tutorials,
help system, error explanations, and progress tracking.
"""

from typing import Optional

try:
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Container
    from textual.containers import Horizontal
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.widgets import Button
    from textual.widgets import DataTable
    from textual.widgets import Input
    from textual.widgets import Label
    from textual.widgets import Markdown
    from textual.widgets import ProgressBar
    from textual.widgets import Static
    from textual.widgets import TabPane
    from textual.widgets import Tabs
    from textual.widgets import Tree

    TEXTUAL_AVAILABLE = True
except ImportError:
    # Mock classes when Textual is not available
    class Screen:
        def __init__(self, **kwargs):
            pass

        def compose(self):
            return []

        async def on_mount(self):
            pass

    class Container:
        pass

    class Static:
        pass

    class Button:
        pass

    class Input:
        pass

    class Tree:
        pass

    class Tabs:
        pass

    class TabPane:
        pass

    TEXTUAL_AVAILABLE = False

from .documentation import DocumentationSystem
from .error_system import ErrorExplanationSystem
from .help_system import HelpSystem
from .progress_tracker import ProgressTracker
from .tutorial_system import Tutorial
from .tutorial_system import TutorialSystem


class LearningScreen(Screen):
    """Main learning and help screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("f1", "help", "Help"),
        Binding("ctrl+s", "search", "Search"),
        Binding("ctrl+t", "tutorials", "Tutorials"),
        Binding("ctrl+p", "progress", "Progress"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize learning systems
        self.tutorial_system = TutorialSystem()
        self.help_system = HelpSystem()
        self.error_system = ErrorExplanationSystem()
        self.progress_tracker = ProgressTracker()
        self.documentation = DocumentationSystem()

        # Current state
        self.current_tutorial: Optional[Tutorial] = None
        self.current_step: int = 0
        self.user_id = "default"  # Would be set from app context

    def compose(self) -> ComposeResult:
        """Compose the learning screen layout."""
        if not TEXTUAL_AVAILABLE:
            yield Static("Learning system requires Textual framework")
            return

        with Container(id="learning_container"):
            # Header with title and search
            with Horizontal(id="learning_header"):
                yield Static("🎓 Learning Center", id="learning_title")
                yield Input(
                    placeholder="Search help and documentation...",
                    id="search_input",
                )
                yield Button("Search", id="search_button")

            # Main content with tabs
            with Tabs(id="learning_tabs"):
                with TabPane("Tutorials", id="tutorials_tab"):
                    yield self._create_tutorials_content()

                with TabPane("Help", id="help_tab"):
                    yield self._create_help_content()

                with TabPane("Documentation", id="docs_tab"):
                    yield self._create_documentation_content()

                with TabPane("Progress", id="progress_tab"):
                    yield self._create_progress_content()

                with TabPane("Troubleshooting", id="troubleshooting_tab"):
                    yield self._create_troubleshooting_content()

    def _create_tutorials_content(self) -> Container:
        """Create tutorials tab content."""
        with Container(id="tutorials_content"):
            with Horizontal(id="tutorials_layout"):
                # Tutorial list
                with Vertical(id="tutorial_list", classes="sidebar"):
                    yield Static("Available Tutorials", classes="section_title")
                    yield Tree("Tutorials", id="tutorial_tree")

                    yield Static("Current Progress", classes="section_title")
                    yield ProgressBar(id="tutorial_progress")
                    yield Static("0% Complete", id="progress_text")

                # Tutorial content
                with Vertical(id="tutorial_content", classes="main_content"):
                    yield Static("Select a tutorial to begin", id="tutorial_display")

                    with Horizontal(id="tutorial_controls"):
                        yield Button("Previous", id="prev_step", disabled=True)
                        yield Button("Next", id="next_step", disabled=True)
                        yield Button("Start Tutorial", id="start_tutorial")
                        yield Button("Complete Step", id="complete_step", disabled=True)

        return Container()

    def _create_help_content(self) -> Container:
        """Create help tab content."""
        with Container(id="help_content"):
            with Horizontal(id="help_layout"):
                # Help categories
                with Vertical(id="help_categories", classes="sidebar"):
                    yield Static("Help Categories", classes="section_title")
                    yield Tree("Categories", id="help_tree")

                # Help content
                with Vertical(id="help_display", classes="main_content"):
                    yield Static("Select a help topic", id="help_text")

                    with Horizontal(id="help_controls"):
                        yield Button("Related Topics", id="related_topics")
                        yield Button("Examples", id="show_examples")

        return Container()

    def _create_documentation_content(self) -> Container:
        """Create documentation tab content."""
        with Container(id="docs_content"):
            with Horizontal(id="docs_layout"):
                # Documentation categories
                with Vertical(id="docs_categories", classes="sidebar"):
                    yield Static("Documentation", classes="section_title")
                    yield Tree("Documents", id="docs_tree")

                # Documentation content
                with Vertical(id="docs_display", classes="main_content"):
                    yield Markdown(
                        "# Welcome to ICARUS Documentation\n\nSelect a document to view its content.",
                        id="docs_markdown",
                    )

        return Container()

    def _create_progress_content(self) -> Container:
        """Create progress tab content."""
        with Container(id="progress_content"):
            with Vertical(id="progress_layout"):
                # Progress overview
                with Horizontal(id="progress_overview"):
                    with Vertical(id="progress_stats"):
                        yield Static("Learning Progress", classes="section_title")
                        yield Static("Level: Beginner", id="skill_level")
                        yield Static("Points: 0", id="total_points")
                        yield Static("Streak: 0 days", id="learning_streak")

                    with Vertical(id="achievements_preview"):
                        yield Static("Recent Achievements", classes="section_title")
                        yield Static("No achievements yet", id="recent_achievements")

                # Detailed progress
                with Horizontal(id="progress_details"):
                    with Vertical(id="modules_progress"):
                        yield Static("Learning Modules", classes="section_title")
                        yield DataTable(id="modules_table")

                    with Vertical(id="recommendations"):
                        yield Static("Recommended Next Steps", classes="section_title")
                        yield Static("Complete the welcome tutorial", id="next_steps")

        return Container()

    def _create_troubleshooting_content(self) -> Container:
        """Create troubleshooting tab content."""
        with Container(id="troubleshooting_content"):
            with Horizontal(id="troubleshooting_layout"):
                # Common errors
                with Vertical(id="common_errors", classes="sidebar"):
                    yield Static("Common Issues", classes="section_title")
                    yield Tree("Error Categories", id="error_tree")

                # Error solutions
                with Vertical(id="error_solutions", classes="main_content"):
                    yield Static(
                        "Select an error type for solutions",
                        id="error_display",
                    )

                    with Horizontal(id="error_controls"):
                        yield Button("Try Solution", id="try_solution")
                        yield Button("More Help", id="more_help")

        return Container()

    async def on_mount(self) -> None:
        """Initialize the learning screen."""
        await self._populate_tutorials()
        await self._populate_help()
        await self._populate_documentation()
        await self._populate_progress()
        await self._populate_troubleshooting()

    async def _populate_tutorials(self) -> None:
        """Populate the tutorials tree."""
        if not TEXTUAL_AVAILABLE:
            return

        tutorial_tree = self.query_one("#tutorial_tree", Tree)

        # Group tutorials by category
        categories = {}
        for tutorial in self.tutorial_system.get_available_tutorials("all"):
            if tutorial.category not in categories:
                categories[tutorial.category] = []
            categories[tutorial.category].append(tutorial)

        # Add to tree
        for category, tutorials in categories.items():
            category_node = tutorial_tree.root.add(category)
            for tutorial in tutorials:
                difficulty_icon = {
                    "beginner": "🟢",
                    "intermediate": "🟡",
                    "advanced": "🔴",
                }.get(tutorial.difficulty, "⚪")
                tutorial_node = category_node.add(f"{difficulty_icon} {tutorial.title}")
                tutorial_node.data = tutorial

    async def _populate_help(self) -> None:
        """Populate the help categories tree."""
        if not TEXTUAL_AVAILABLE:
            return

        help_tree = self.query_one("#help_tree", Tree)

        # Add help categories
        categories = self.help_system.get_all_categories()
        for category in categories:
            category_node = help_tree.root.add(category)
            topics = self.help_system.get_topics_by_category(category)
            for topic in topics:
                topic_node = category_node.add(topic.title)
                topic_node.data = topic

    async def _populate_documentation(self) -> None:
        """Populate the documentation tree."""
        if not TEXTUAL_AVAILABLE:
            return

        docs_tree = self.query_one("#docs_tree", Tree)

        # Add documentation categories
        categories = self.documentation.get_all_categories()
        for category in categories:
            category_node = docs_tree.root.add(category)
            docs = self.documentation.get_documents_by_category(category)
            for doc in docs:
                difficulty_icon = {
                    "beginner": "📗",
                    "intermediate": "📘",
                    "advanced": "📕",
                }.get(doc.difficulty, "📄")
                doc_node = category_node.add(f"{difficulty_icon} {doc.title}")
                doc_node.data = doc

    async def _populate_progress(self) -> None:
        """Populate the progress information."""
        if not TEXTUAL_AVAILABLE:
            return

        # Get user progress
        progress = self.progress_tracker.get_user_progress(self.user_id)
        stats = self.progress_tracker.get_learning_statistics(self.user_id)

        # Update progress display
        self.query_one("#skill_level", Static).update(
            f"Level: {progress.skill_level.value.title()}",
        )
        self.query_one("#total_points", Static).update(
            f"Points: {progress.total_points}",
        )
        self.query_one("#learning_streak", Static).update(
            f"Streak: {stats['learning_streak']} days",
        )

        # Show recent achievements
        if progress.achievements:
            recent = progress.achievements[-3:]  # Last 3 achievements
            achievement_text = "\n".join([f"{ach.icon} {ach.title}" for ach in recent])
            self.query_one("#recent_achievements", Static).update(achievement_text)

        # Populate modules table
        modules_table = self.query_one("#modules_table", DataTable)
        modules_table.add_columns("Module", "Status", "Progress")

        for module_id, module in self.progress_tracker.learning_modules.items():
            status = (
                "✅ Complete"
                if module_id in progress.completed_modules
                else "⏳ Available"
            )
            progress_pct = "100%" if module_id in progress.completed_modules else "0%"
            modules_table.add_row(module.title, status, progress_pct)

        # Show recommendations
        recommendations = self.progress_tracker.get_recommended_next_steps(self.user_id)
        if recommendations:
            next_step = recommendations[0]
            self.query_one("#next_steps", Static).update(
                f"📚 {next_step['title']}\n{next_step['description'][:100]}...",
            )

    async def _populate_troubleshooting(self) -> None:
        """Populate the troubleshooting tree."""
        if not TEXTUAL_AVAILABLE:
            return

        error_tree = self.query_one("#error_tree", Tree)

        # Add error categories
        categories = self.error_system.get_error_categories()
        for category in categories:
            category_node = error_tree.root.add(
                category.value.replace("_", " ").title(),
            )
            category_node.data = category

    async def on_button_pressed(self, event) -> None:
        """Handle button presses."""
        if not TEXTUAL_AVAILABLE:
            return

        button_id = event.button.id

        if button_id == "search_button":
            await self._perform_search()
        elif button_id == "start_tutorial":
            await self._start_selected_tutorial()
        elif button_id == "next_step":
            await self._next_tutorial_step()
        elif button_id == "prev_step":
            await self._prev_tutorial_step()
        elif button_id == "complete_step":
            await self._complete_tutorial_step()

    async def _perform_search(self) -> None:
        """Perform search across all learning content."""
        search_input = self.query_one("#search_input", Input)
        query = search_input.value.strip()

        if not query:
            return

        # Search documentation
        doc_results = self.documentation.search(query, max_results=5)

        # Search help topics
        help_results = self.help_system.search_help(query, max_results=5)

        # Display results (would implement results display)
        # For now, just switch to documentation tab and show first result
        if doc_results:
            await self._show_document(doc_results[0][0])

    async def _start_selected_tutorial(self) -> None:
        """Start the selected tutorial."""
        tutorial_tree = self.query_one("#tutorial_tree", Tree)
        if tutorial_tree.cursor_node and hasattr(tutorial_tree.cursor_node, "data"):
            tutorial = tutorial_tree.cursor_node.data
            if isinstance(tutorial, Tutorial):
                self.tutorial_system.start_tutorial(tutorial.id, self.user_id)
                self.current_tutorial = tutorial
                self.current_step = 0
                await self._display_current_tutorial_step()

    async def _display_current_tutorial_step(self) -> None:
        """Display the current tutorial step."""
        if not self.current_tutorial:
            return

        step = self.tutorial_system.get_current_step()
        if step:
            # Update tutorial display
            content = f"# {step.title}\n\n{step.content}"
            if step.hints:
                content += "\n\n## Hints\n" + "\n".join(
                    [f"💡 {hint}" for hint in step.hints],
                )

            self.query_one("#tutorial_display", Static).update(content)

            # Update progress
            progress = (self.current_step / len(self.current_tutorial.steps)) * 100
            self.query_one("#tutorial_progress", ProgressBar).update(progress=progress)
            self.query_one("#progress_text", Static).update(f"{progress:.0f}% Complete")

            # Update button states
            self.query_one("#prev_step", Button).disabled = self.current_step == 0
            self.query_one("#next_step", Button).disabled = False
            self.query_one("#complete_step", Button).disabled = False
            self.query_one("#start_tutorial", Button).disabled = True

    async def _next_tutorial_step(self) -> None:
        """Advance to next tutorial step."""
        if self.tutorial_system.advance_step(self.user_id):
            self.current_step += 1
            await self._display_current_tutorial_step()
        else:
            # Tutorial completed
            await self._complete_tutorial()

    async def _prev_tutorial_step(self) -> None:
        """Go back to previous tutorial step."""
        if self.current_step > 0:
            self.current_step -= 1
            self.tutorial_system.current_step = self.current_step
            await self._display_current_tutorial_step()

    async def _complete_tutorial_step(self) -> None:
        """Mark current step as completed."""
        # In a real implementation, this would validate step completion
        await self._next_tutorial_step()

    async def _complete_tutorial(self) -> None:
        """Complete the current tutorial."""
        if self.current_tutorial:
            # Award achievements and update progress
            achievements = self.progress_tracker.complete_tutorial(
                self.user_id,
                self.current_tutorial.id,
            )

            # Show completion message
            completion_msg = f"🎉 Congratulations! You've completed the '{self.current_tutorial.title}' tutorial!"
            if achievements:
                completion_msg += "\n\nNew achievements unlocked:\n" + "\n".join(
                    [f"{ach.icon} {ach.title}" for ach in achievements],
                )

            self.query_one("#tutorial_display", Static).update(completion_msg)

            # Reset tutorial state
            self.current_tutorial = None
            self.current_step = 0

            # Update progress display
            await self._populate_progress()

    async def _show_document(self, document) -> None:
        """Show a documentation document."""
        # Switch to documentation tab
        tabs = self.query_one("#learning_tabs", Tabs)
        tabs.active = "docs_tab"

        # Display document content
        content = f"# {document.title}\n\n{document.content}"
        if document.examples:
            content += "\n\n## Examples\n"
            for example in document.examples:
                content += f"\n### {example.title}\n{example.description}\n\n```{example.language}\n{example.code}\n```\n"

        self.query_one("#docs_markdown", Markdown).update(content)

    def action_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()

    def action_help(self) -> None:
        """Show contextual help."""
        # Switch to help tab
        if TEXTUAL_AVAILABLE:
            tabs = self.query_one("#learning_tabs", Tabs)
            tabs.active = "help_tab"

    def action_search(self) -> None:
        """Focus search input."""
        if TEXTUAL_AVAILABLE:
            search_input = self.query_one("#search_input", Input)
            search_input.focus()

    def action_tutorials(self) -> None:
        """Switch to tutorials tab."""
        if TEXTUAL_AVAILABLE:
            tabs = self.query_one("#learning_tabs", Tabs)
            tabs.active = "tutorials_tab"

    def action_progress(self) -> None:
        """Switch to progress tab."""
        if TEXTUAL_AVAILABLE:
            tabs = self.query_one("#learning_tabs", Tabs)
            tabs.active = "progress_tab"
