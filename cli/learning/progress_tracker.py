"""Progress Tracking System

This module tracks user progress through learning modules and tutorials,
providing personalized learning paths and achievement tracking.
"""

import json
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set


class SkillLevel(Enum):
    """User skill levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AchievementType(Enum):
    """Types of achievements."""

    TUTORIAL_COMPLETION = "tutorial_completion"
    ANALYSIS_MILESTONE = "analysis_milestone"
    FEATURE_MASTERY = "feature_mastery"
    PROBLEM_SOLVING = "problem_solving"
    CONSISTENCY = "consistency"
    EXPLORATION = "exploration"


@dataclass
class Achievement:
    """User achievement."""

    id: str
    title: str
    description: str
    achievement_type: AchievementType
    icon: str
    points: int
    requirements: Dict[str, Any] = field(default_factory=dict)
    unlocked_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "achievement_type": self.achievement_type.value,
            "icon": self.icon,
            "points": self.points,
            "requirements": self.requirements,
            "unlocked_at": self.unlocked_at,
        }


@dataclass
class LearningModule:
    """A learning module with progress tracking."""

    id: str
    title: str
    description: str
    category: str
    skill_level: SkillLevel
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    estimated_duration: int = 0  # minutes
    activities: List[str] = field(default_factory=list)
    assessments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "skill_level": self.skill_level.value,
            "prerequisites": self.prerequisites,
            "learning_objectives": self.learning_objectives,
            "estimated_duration": self.estimated_duration,
            "activities": self.activities,
            "assessments": self.assessments,
        }


@dataclass
class UserProgress:
    """User's learning progress."""

    user_id: str
    skill_level: SkillLevel = SkillLevel.BEGINNER
    total_points: int = 0
    completed_tutorials: Set[str] = field(default_factory=set)
    completed_modules: Set[str] = field(default_factory=set)
    achievements: List[Achievement] = field(default_factory=list)
    activity_log: List[Dict[str, Any]] = field(default_factory=list)
    learning_path: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "skill_level": self.skill_level.value,
            "total_points": self.total_points,
            "completed_tutorials": list(self.completed_tutorials),
            "completed_modules": list(self.completed_modules),
            "achievements": [ach.to_dict() for ach in self.achievements],
            "activity_log": self.activity_log,
            "learning_path": self.learning_path,
            "preferences": self.preferences,
        }


class ProgressTracker:
    """Tracks user progress through learning modules and tutorials."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("cli/learning/data")
        self.learning_modules: Dict[str, LearningModule] = {}
        self.user_progress: Dict[str, UserProgress] = {}
        self.achievements: Dict[str, Achievement] = {}

        # Initialize built-in content
        self._initialize_learning_modules()
        self._initialize_achievements()

    def _initialize_learning_modules(self) -> None:
        """Initialize built-in learning modules."""
        # Interface Basics Module
        interface_module = LearningModule(
            id="interface_basics",
            title="Interface Basics",
            description="Learn to navigate the ICARUS CLI interface efficiently",
            category="Getting Started",
            skill_level=SkillLevel.BEGINNER,
            learning_objectives=[
                "Navigate using keyboard and mouse",
                "Access help and documentation",
                "Customize interface settings",
                "Use keyboard shortcuts effectively",
            ],
            estimated_duration=15,
            activities=[
                "Complete welcome tutorial",
                "Practice navigation exercises",
                "Customize theme and layout",
                "Learn essential keyboard shortcuts",
            ],
            assessments=[
                {
                    "type": "practical",
                    "description": "Navigate to Analysis screen using only keyboard",
                    "points": 10,
                },
                {
                    "type": "knowledge",
                    "description": "Identify 5 keyboard shortcuts",
                    "points": 5,
                },
            ],
        )
        self.learning_modules[interface_module.id] = interface_module

        # Airfoil Analysis Module
        airfoil_module = LearningModule(
            id="airfoil_analysis_fundamentals",
            title="Airfoil Analysis Fundamentals",
            description="Master 2D airfoil analysis using XFoil and other solvers",
            category="Analysis",
            skill_level=SkillLevel.BEGINNER,
            prerequisites=["interface_basics"],
            learning_objectives=[
                "Understand airfoil geometry and nomenclature",
                "Configure XFoil analysis parameters",
                "Interpret polar plots and performance data",
                "Troubleshoot common analysis issues",
            ],
            estimated_duration=45,
            activities=[
                "Complete airfoil analysis tutorial",
                "Analyze 3 different airfoil types",
                "Create and interpret polar plots",
                "Solve convergence problems",
            ],
            assessments=[
                {
                    "type": "practical",
                    "description": "Analyze NACA 0012 airfoil and identify stall angle",
                    "points": 20,
                },
                {
                    "type": "interpretation",
                    "description": "Explain differences between symmetric and cambered airfoils",
                    "points": 15,
                },
            ],
        )
        self.learning_modules[airfoil_module.id] = airfoil_module

        # Advanced Analysis Module
        advanced_module = LearningModule(
            id="advanced_analysis_techniques",
            title="Advanced Analysis Techniques",
            description="Learn advanced analysis methods and optimization",
            category="Analysis",
            skill_level=SkillLevel.INTERMEDIATE,
            prerequisites=["airfoil_analysis_fundamentals"],
            learning_objectives=[
                "Perform 3D aircraft analysis",
                "Set up parameter studies",
                "Use optimization algorithms",
                "Validate results against experimental data",
            ],
            estimated_duration=90,
            activities=[
                "Complete airplane analysis tutorial",
                "Set up multi-point optimization",
                "Validate analysis results",
                "Create custom analysis workflows",
            ],
        )
        self.learning_modules[advanced_module.id] = advanced_module

        # Workflow Mastery Module
        workflow_module = LearningModule(
            id="workflow_mastery",
            title="Workflow System Mastery",
            description="Create and manage complex analysis workflows",
            category="Automation",
            skill_level=SkillLevel.INTERMEDIATE,
            prerequisites=["airfoil_analysis_fundamentals"],
            learning_objectives=[
                "Design efficient analysis workflows",
                "Automate repetitive tasks",
                "Handle workflow errors and recovery",
                "Share and collaborate on workflows",
            ],
            estimated_duration=60,
            activities=[
                "Create basic workflow",
                "Build parameter sweep workflow",
                "Implement error handling",
                "Share workflow with team",
            ],
        )
        self.learning_modules[workflow_module.id] = workflow_module

    def _initialize_achievements(self) -> None:
        """Initialize built-in achievements."""
        achievements = [
            Achievement(
                id="first_steps",
                title="First Steps",
                description="Complete your first tutorial",
                achievement_type=AchievementType.TUTORIAL_COMPLETION,
                icon="🎯",
                points=10,
                requirements={"tutorials_completed": 1},
            ),
            Achievement(
                id="tutorial_master",
                title="Tutorial Master",
                description="Complete all beginner tutorials",
                achievement_type=AchievementType.TUTORIAL_COMPLETION,
                icon="🎓",
                points=50,
                requirements={"beginner_tutorials_completed": "all"},
            ),
            Achievement(
                id="first_analysis",
                title="First Analysis",
                description="Run your first successful analysis",
                achievement_type=AchievementType.ANALYSIS_MILESTONE,
                icon="🔬",
                points=15,
                requirements={"analyses_completed": 1},
            ),
            Achievement(
                id="analysis_veteran",
                title="Analysis Veteran",
                description="Complete 50 analyses",
                achievement_type=AchievementType.ANALYSIS_MILESTONE,
                icon="⚡",
                points=100,
                requirements={"analyses_completed": 50},
            ),
            Achievement(
                id="problem_solver",
                title="Problem Solver",
                description="Successfully resolve 10 analysis errors",
                achievement_type=AchievementType.PROBLEM_SOLVING,
                icon="🔧",
                points=75,
                requirements={"errors_resolved": 10},
            ),
            Achievement(
                id="consistent_learner",
                title="Consistent Learner",
                description="Use ICARUS for 7 consecutive days",
                achievement_type=AchievementType.CONSISTENCY,
                icon="📅",
                points=30,
                requirements={"consecutive_days": 7},
            ),
            Achievement(
                id="explorer",
                title="Explorer",
                description="Try all major features of ICARUS",
                achievement_type=AchievementType.EXPLORATION,
                icon="🗺️",
                points=60,
                requirements={
                    "features_used": [
                        "airfoil",
                        "airplane",
                        "workflow",
                        "optimization",
                    ],
                },
            ),
            Achievement(
                id="workflow_architect",
                title="Workflow Architect",
                description="Create and successfully run a complex workflow",
                achievement_type=AchievementType.FEATURE_MASTERY,
                icon="🏗️",
                points=80,
                requirements={"complex_workflows_created": 1},
            ),
        ]

        for achievement in achievements:
            self.achievements[achievement.id] = achievement

    def get_user_progress(self, user_id: str) -> UserProgress:
        """Get or create user progress."""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = UserProgress(user_id=user_id)
        return self.user_progress[user_id]

    def complete_tutorial(self, user_id: str, tutorial_id: str) -> List[Achievement]:
        """Mark tutorial as completed and check for achievements."""
        progress = self.get_user_progress(user_id)
        progress.completed_tutorials.add(tutorial_id)

        # Log activity
        self._log_activity(user_id, "tutorial_completed", {"tutorial_id": tutorial_id})

        # Check for achievements
        new_achievements = self._check_achievements(user_id)

        # Award points
        progress.total_points += 10  # Base points for tutorial completion

        return new_achievements

    def complete_module(
        self,
        user_id: str,
        module_id: str,
        score: int = 100,
    ) -> List[Achievement]:
        """Mark learning module as completed."""
        progress = self.get_user_progress(user_id)
        progress.completed_modules.add(module_id)

        # Log activity
        self._log_activity(
            user_id,
            "module_completed",
            {"module_id": module_id, "score": score},
        )

        # Award points based on module difficulty and score
        module = self.learning_modules.get(module_id)
        if module:
            base_points = {
                SkillLevel.BEGINNER: 20,
                SkillLevel.INTERMEDIATE: 40,
                SkillLevel.ADVANCED: 60,
                SkillLevel.EXPERT: 80,
            }
            points = int(base_points[module.skill_level] * (score / 100))
            progress.total_points += points

        # Check for achievements
        new_achievements = self._check_achievements(user_id)

        return new_achievements

    def record_analysis(
        self,
        user_id: str,
        analysis_type: str,
        success: bool = True,
    ) -> List[Achievement]:
        """Record an analysis completion."""
        progress = self.get_user_progress(user_id)

        # Log activity
        self._log_activity(
            user_id,
            "analysis_completed",
            {"analysis_type": analysis_type, "success": success},
        )

        # Award points for successful analysis
        if success:
            progress.total_points += 5

        # Check for achievements
        new_achievements = self._check_achievements(user_id)

        return new_achievements

    def record_error_resolution(
        self,
        user_id: str,
        error_type: str,
    ) -> List[Achievement]:
        """Record successful error resolution."""
        progress = self.get_user_progress(user_id)

        # Log activity
        self._log_activity(user_id, "error_resolved", {"error_type": error_type})

        # Award points for problem solving
        progress.total_points += 8

        # Check for achievements
        new_achievements = self._check_achievements(user_id)

        return new_achievements

    def _log_activity(
        self,
        user_id: str,
        activity_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Log user activity."""
        progress = self.get_user_progress(user_id)

        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "data": data,
        }

        progress.activity_log.append(activity)

        # Keep only last 1000 activities to prevent unbounded growth
        if len(progress.activity_log) > 1000:
            progress.activity_log = progress.activity_log[-1000:]

    def _check_achievements(self, user_id: str) -> List[Achievement]:
        """Check and award new achievements."""
        progress = self.get_user_progress(user_id)
        new_achievements = []

        # Get current achievement IDs
        current_achievement_ids = {ach.id for ach in progress.achievements}

        for achievement_id, achievement in self.achievements.items():
            if achievement_id in current_achievement_ids:
                continue  # Already earned

            if self._check_achievement_requirements(user_id, achievement):
                # Award achievement
                earned_achievement = Achievement(
                    id=achievement.id,
                    title=achievement.title,
                    description=achievement.description,
                    achievement_type=achievement.achievement_type,
                    icon=achievement.icon,
                    points=achievement.points,
                    requirements=achievement.requirements,
                    unlocked_at=datetime.now().isoformat(),
                )

                progress.achievements.append(earned_achievement)
                progress.total_points += achievement.points
                new_achievements.append(earned_achievement)

                # Log achievement
                self._log_activity(
                    user_id,
                    "achievement_earned",
                    {"achievement_id": achievement_id, "points": achievement.points},
                )

        return new_achievements

    def _check_achievement_requirements(
        self,
        user_id: str,
        achievement: Achievement,
    ) -> bool:
        """Check if user meets achievement requirements."""
        progress = self.get_user_progress(user_id)

        for req_key, req_value in achievement.requirements.items():
            if req_key == "tutorials_completed":
                if len(progress.completed_tutorials) < req_value:
                    return False

            elif req_key == "beginner_tutorials_completed":
                if req_value == "all":
                    # Count beginner tutorials (would need tutorial system integration)
                    # For now, assume 3 beginner tutorials
                    if len(progress.completed_tutorials) < 3:
                        return False

            elif req_key == "analyses_completed":
                analysis_count = len(
                    [
                        a
                        for a in progress.activity_log
                        if a["type"] == "analysis_completed"
                        and a["data"].get("success", True)
                    ],
                )
                if analysis_count < req_value:
                    return False

            elif req_key == "errors_resolved":
                error_count = len(
                    [a for a in progress.activity_log if a["type"] == "error_resolved"],
                )
                if error_count < req_value:
                    return False

            elif req_key == "consecutive_days":
                if not self._check_consecutive_days(user_id, req_value):
                    return False

            elif req_key == "features_used":
                used_features = set()
                for activity in progress.activity_log:
                    if activity["type"] == "analysis_completed":
                        analysis_type = activity["data"].get("analysis_type", "")
                        if "airfoil" in analysis_type.lower():
                            used_features.add("airfoil")
                        elif "airplane" in analysis_type.lower():
                            used_features.add("airplane")
                        elif "workflow" in analysis_type.lower():
                            used_features.add("workflow")
                        elif "optimization" in analysis_type.lower():
                            used_features.add("optimization")

                if not all(feature in used_features for feature in req_value):
                    return False

            elif req_key == "complex_workflows_created":
                workflow_count = len(
                    [
                        a
                        for a in progress.activity_log
                        if a["type"] == "workflow_created"
                        and a["data"].get("complex", False)
                    ],
                )
                if workflow_count < req_value:
                    return False

        return True

    def _check_consecutive_days(self, user_id: str, required_days: int) -> bool:
        """Check if user has been active for consecutive days."""
        progress = self.get_user_progress(user_id)

        if not progress.activity_log:
            return False

        # Get unique activity dates
        activity_dates = set()
        for activity in progress.activity_log:
            try:
                date = datetime.fromisoformat(activity["timestamp"]).date()
                activity_dates.add(date)
            except (ValueError, KeyError):
                continue

        if len(activity_dates) < required_days:
            return False

        # Check for consecutive days
        sorted_dates = sorted(activity_dates, reverse=True)
        consecutive_count = 1

        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i - 1] - sorted_dates[i]).days == 1:
                consecutive_count += 1
                if consecutive_count >= required_days:
                    return True
            else:
                consecutive_count = 1

        return consecutive_count >= required_days

    def get_recommended_next_steps(self, user_id: str) -> List[Dict[str, Any]]:
        """Get recommended next learning steps for user."""
        progress = self.get_user_progress(user_id)
        recommendations = []

        # Find modules user can take based on prerequisites
        for module_id, module in self.learning_modules.items():
            if module_id in progress.completed_modules:
                continue  # Already completed

            # Check prerequisites
            if all(
                prereq in progress.completed_modules for prereq in module.prerequisites
            ):
                recommendations.append(
                    {
                        "type": "module",
                        "id": module_id,
                        "title": module.title,
                        "description": module.description,
                        "estimated_duration": module.estimated_duration,
                        "skill_level": module.skill_level.value,
                        "priority": self._calculate_priority(progress, module),
                    },
                )

        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)

        return recommendations[:5]  # Top 5 recommendations

    def _calculate_priority(
        self,
        progress: UserProgress,
        module: LearningModule,
    ) -> int:
        """Calculate priority score for a learning module."""
        priority = 0

        # Skill level match
        if module.skill_level == progress.skill_level:
            priority += 10
        elif (
            module.skill_level.value == "beginner"
            and progress.skill_level == SkillLevel.BEGINNER
        ):
            priority += 15

        # Category preferences
        category_preference = progress.preferences.get("preferred_categories", [])
        if module.category.lower() in [cat.lower() for cat in category_preference]:
            priority += 5

        # Recent activity in related area
        recent_activities = progress.activity_log[-20:]  # Last 20 activities
        for activity in recent_activities:
            if activity["type"] == "analysis_completed":
                analysis_type = activity["data"].get("analysis_type", "")
                if module.category.lower() in analysis_type.lower():
                    priority += 3

        return priority

    def get_learning_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user's learning statistics."""
        progress = self.get_user_progress(user_id)

        # Calculate statistics
        total_modules = len(self.learning_modules)
        completed_modules = len(progress.completed_modules)
        completion_rate = (
            (completed_modules / total_modules * 100) if total_modules > 0 else 0
        )

        # Activity statistics
        activity_counts = {}
        for activity in progress.activity_log:
            activity_type = activity["type"]
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1

        # Recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_activities = []
        for activity in progress.activity_log:
            try:
                activity_date = datetime.fromisoformat(activity["timestamp"])
                if activity_date >= week_ago:
                    recent_activities.append(activity)
            except (ValueError, KeyError):
                continue

        return {
            "skill_level": progress.skill_level.value,
            "total_points": progress.total_points,
            "modules_completed": completed_modules,
            "total_modules": total_modules,
            "completion_rate": round(completion_rate, 1),
            "achievements_earned": len(progress.achievements),
            "total_achievements": len(self.achievements),
            "activity_counts": activity_counts,
            "recent_activity_count": len(recent_activities),
            "learning_streak": self._calculate_learning_streak(user_id),
        }

    def _calculate_learning_streak(self, user_id: str) -> int:
        """Calculate current learning streak in days."""
        progress = self.get_user_progress(user_id)

        if not progress.activity_log:
            return 0

        # Get unique activity dates
        activity_dates = set()
        for activity in progress.activity_log:
            try:
                date = datetime.fromisoformat(activity["timestamp"]).date()
                activity_dates.add(date)
            except (ValueError, KeyError):
                continue

        if not activity_dates:
            return 0

        # Check streak from today backwards
        today = datetime.now().date()
        streak = 0
        current_date = today

        while current_date in activity_dates:
            streak += 1
            current_date -= timedelta(days=1)

        return streak

    def save_progress(self, filepath: Path = None) -> None:
        """Save all progress data."""
        if filepath is None:
            filepath = self.data_dir / "progress_data.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "learning_modules": {
                mid: module.to_dict() for mid, module in self.learning_modules.items()
            },
            "user_progress": {
                uid: progress.to_dict() for uid, progress in self.user_progress.items()
            },
            "achievements": {
                aid: achievement.to_dict()
                for aid, achievement in self.achievements.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_progress(self, filepath: Path = None) -> None:
        """Load progress data."""
        if filepath is None:
            filepath = self.data_dir / "progress_data.json"

        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)

            # Load learning modules
            for module_id, module_data in data.get("learning_modules", {}).items():
                module = LearningModule(
                    id=module_data["id"],
                    title=module_data["title"],
                    description=module_data["description"],
                    category=module_data["category"],
                    skill_level=SkillLevel(module_data["skill_level"]),
                    prerequisites=module_data.get("prerequisites", []),
                    learning_objectives=module_data.get("learning_objectives", []),
                    estimated_duration=module_data.get("estimated_duration", 0),
                    activities=module_data.get("activities", []),
                    assessments=module_data.get("assessments", []),
                )
                self.learning_modules[module_id] = module

            # Load user progress
            for user_id, progress_data in data.get("user_progress", {}).items():
                achievements = []
                for ach_data in progress_data.get("achievements", []):
                    achievement = Achievement(
                        id=ach_data["id"],
                        title=ach_data["title"],
                        description=ach_data["description"],
                        achievement_type=AchievementType(ach_data["achievement_type"]),
                        icon=ach_data["icon"],
                        points=ach_data["points"],
                        requirements=ach_data.get("requirements", {}),
                        unlocked_at=ach_data.get("unlocked_at"),
                    )
                    achievements.append(achievement)

                progress = UserProgress(
                    user_id=progress_data["user_id"],
                    skill_level=SkillLevel(
                        progress_data.get("skill_level", "beginner"),
                    ),
                    total_points=progress_data.get("total_points", 0),
                    completed_tutorials=set(
                        progress_data.get("completed_tutorials", []),
                    ),
                    completed_modules=set(progress_data.get("completed_modules", [])),
                    achievements=achievements,
                    activity_log=progress_data.get("activity_log", []),
                    learning_path=progress_data.get("learning_path", []),
                    preferences=progress_data.get("preferences", {}),
                )
                self.user_progress[user_id] = progress

            # Load achievements
            for achievement_id, ach_data in data.get("achievements", {}).items():
                achievement = Achievement(
                    id=ach_data["id"],
                    title=ach_data["title"],
                    description=ach_data["description"],
                    achievement_type=AchievementType(ach_data["achievement_type"]),
                    icon=ach_data["icon"],
                    points=ach_data["points"],
                    requirements=ach_data.get("requirements", {}),
                    unlocked_at=ach_data.get("unlocked_at"),
                )
                self.achievements[achievement_id] = achievement
