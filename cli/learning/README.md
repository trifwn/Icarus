# ICARUS Learning and Help System

This module implements a comprehensive learning and help system for the ICARUS CLI, providing guided tutorials, contextual help, error explanations, and progress tracking.

## Features

### 🎓 Guided Tutorial System
- Interactive tutorials for new users
- Step-by-step guidance through ICARUS features
- Progress tracking and validation
- Multiple difficulty levels (beginner, intermediate, advanced)

### ❓ Contextual Help System
- Context-sensitive help for UI elements
- Searchable help topics with examples
- Cross-referenced documentation
- Quick tips and keyboard shortcuts

### 🔧 Error Explanation System
- Educational error explanations with solutions
- Common error patterns and fixes
- Prevention tips and best practices
- Learning opportunities from errors

### 📊 Progress Tracking
- User skill level assessment
- Achievement system with points and badges
- Learning streak tracking
- Personalized recommendations

### 📖 Documentation System
- Comprehensive searchable documentation
- Code examples with explanations
- Cross-referenced topics
- Multiple document types (guides, references, FAQs)

## Requirements Compliance

This system fulfills the following requirements from the ICARUS CLI specification:

### Requirement 3.1: Guided Tour for New Users
✅ **WHEN new users first launch the CLI THEN the system SHALL offer an optional guided tour of key features**

- Detects new users automatically
- Offers welcome tutorial with interface overview
- Provides quick start tips and recommendations

### Requirement 3.2: Contextual Help and Documentation Links
✅ **WHEN users access any feature THEN the system SHALL provide contextual help and documentation links**

- Context-sensitive help for all UI elements
- Related topic suggestions
- Quick tips for current screen
- Keyboard shortcut references

### Requirement 3.3: Educational Error Explanations and Solutions
✅ **WHEN users encounter errors THEN the system SHALL provide educational explanations and suggested solutions**

- Pattern-based error recognition
- Educational explanations with context
- Multiple solution strategies
- Prevention tips and learning resources

### Requirement 3.4: Progress Tracking and Next Steps
✅ **WHEN users complete tutorials THEN the system SHALL track progress and suggest next steps**

- Tutorial completion tracking
- Achievement system with points
- Skill level progression
- Personalized learning recommendations

### Requirement 3.5: Searchable Documentation with Examples
✅ **IF users request help THEN the system SHALL provide searchable documentation with examples**

- Full-text search across all content
- Code examples with explanations
- Multiple content types (help, docs, tutorials)
- Related content suggestions

## Architecture

```
cli/learning/
├── __init__.py                 # Module exports
├── learning_manager.py         # Unified interface to all systems
├── tutorial_system.py          # Guided tutorials
├── help_system.py             # Contextual help
├── error_system.py            # Error explanations
├── progress_tracker.py        # Progress and achievements
├── documentation.py           # Searchable documentation
├── learning_screen.py         # TUI interface
├── test_learning_system.py    # Comprehensive tests
├── data/                      # Persistent data storage
└── README.md                  # This file
```

## Usage

### Basic Usage

```python
from cli.learning.learning_manager import LearningManager

# Initialize learning system
learning_manager = LearningManager(user_id="user123")

# Check for new user
welcome_info = learning_manager.show_welcome_for_new_user()
if welcome_info['is_new_user']:
    # Show welcome tutorial
    learning_manager.start_tutorial("welcome")

# Get contextual help
assistance = learning_manager.provide_contextual_assistance("analysis_screen")
print(assistance['quick_tips'])

# Handle errors educationally
error_response = learning_manager.handle_error_with_education("XFoil convergence failed")
print(error_response['solutions'])

# Track progress
progress = learning_manager.track_learning_progress_and_suggest_next()
print(f"User level: {progress['current_level']}")
print(f"Points: {progress['total_points']}")

# Search help
results = learning_manager.provide_searchable_help("airfoil analysis")
print(f"Found {results['total_results']} results")
```

### Integration with Main App

```python
from cli.learning.learning_manager import LearningManager

class IcarusApp:
    def __init__(self):
        self.learning_manager = LearningManager()

        # Register callbacks
        self.learning_manager.register_callbacks(
            on_tutorial_completed=self._on_tutorial_completed,
            on_achievement_earned=self._on_achievement_earned,
            on_help_requested=self._on_help_requested,
            on_error_explained=self._on_error_explained
        )

    def _on_achievement_earned(self, achievement):
        # Show notification
        self.show_notification(f"🏆 {achievement.title}")
```

## Built-in Content

### Tutorials
- **Welcome to ICARUS**: Interface basics and navigation
- **Airfoil Analysis Basics**: First analysis walkthrough
- **Airplane Analysis Basics**: 3D analysis introduction
- **Workflow System Basics**: Automation and workflows

### Help Topics
- Interface navigation and shortcuts
- Analysis parameter explanations
- Solver configuration guides
- Troubleshooting common issues

### Error Explanations
- XFoil convergence failures
- File and data errors
- Configuration problems
- System resource issues

### Documentation
- Getting started guide
- Interface reference
- Analysis tutorials
- FAQ and troubleshooting
- Code examples

## Testing

Run the comprehensive test suite:

```bash
python cli/learning/test_learning_system.py
```

This tests all components and verifies requirements compliance.

## Data Persistence

The learning system automatically saves and loads:
- User progress and achievements
- Tutorial completion status
- Error history and patterns
- Help usage statistics

Data is stored in JSON format in the `cli/learning/data/` directory.

## Extensibility

### Adding New Tutorials

```python
from cli.learning.tutorial_system import Tutorial, TutorialStep, TutorialStepType

# Create tutorial steps
steps = [
    TutorialStep(
        id="step1",
        title="Introduction",
        content="Welcome to this tutorial...",
        step_type=TutorialStepType.INTRODUCTION,
        duration_estimate=30
    )
]

# Create tutorial
tutorial = Tutorial(
    id="my_tutorial",
    title="My Custom Tutorial",
    description="Learn about custom features",
    category="Advanced",
    difficulty="intermediate",
    estimated_duration=20,
    steps=steps
)

# Add to system
learning_manager.tutorial_system.tutorials[tutorial.id] = tutorial
```

### Adding Help Topics

```python
from cli.learning.help_system import HelpTopic, HelpTopicType

topic = HelpTopic(
    id="my_feature_help",
    title="My Feature Help",
    content="This feature allows you to...",
    topic_type=HelpTopicType.FEATURE,
    category="Features",
    tags=["feature", "help"],
    examples=[{
        "title": "Basic Usage",
        "description": "How to use this feature"
    }]
)

learning_manager.help_system.add_help_topic(topic)
```

### Adding Error Explanations

```python
from cli.learning.error_system import ErrorExplanation, ErrorSolution, ErrorCategory, SolutionType

solution = ErrorSolution(
    id="fix_my_error",
    title="Fix My Error",
    description="This solution fixes the error by...",
    solution_type=SolutionType.QUICK_FIX,
    steps=["Step 1", "Step 2", "Step 3"]
)

explanation = ErrorExplanation(
    error_pattern=r"my.*error.*pattern",
    title="My Error Type",
    explanation="This error occurs when...",
    category=ErrorCategory.USER_INPUT,
    solutions=[solution]
)

learning_manager.error_system.add_error_explanation(explanation)
```

## Performance

The learning system is designed for efficiency:
- Lazy loading of content
- Indexed search for fast queries
- Minimal memory footprint
- Asynchronous operations where possible

## Future Enhancements

Potential improvements:
- Video tutorials and interactive demos
- Community-contributed content
- Multi-language support
- Advanced analytics and insights
- Integration with external learning platforms
- Collaborative learning features

## Contributing

To contribute to the learning system:
1. Add new content following existing patterns
2. Update tests to cover new functionality
3. Document new features and APIs
4. Ensure requirements compliance
5. Test thoroughly with real users

The learning system is a key differentiator for ICARUS CLI, making advanced aerodynamics software accessible to users of all skill levels.
