"""Error Analytics and Reporting System

This module provides advanced analytics, reporting, and insights
for error patterns and system health monitoring.
"""

import json
import sqlite3
import statistics
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .error_handler import ErrorRecord
from .error_handler import ErrorSeverity


@dataclass
class ErrorTrend:
    """Represents an error trend over time."""

    period: str  # 'hour', 'day', 'week', 'month'
    timestamp: datetime
    error_count: int
    severity_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    recovery_rate: float


@dataclass
class ErrorPattern:
    """Represents a detected error pattern."""

    pattern_id: str
    description: str
    frequency: int
    first_occurrence: datetime
    last_occurrence: datetime
    affected_components: List[str]
    common_context: Dict[str, Any]
    suggested_fixes: List[str]
    confidence_score: float


@dataclass
class SystemHealthMetrics:
    """System health metrics."""

    timestamp: datetime
    overall_health_score: float  # 0-100
    error_rate: float  # errors per hour
    recovery_success_rate: float
    mean_time_to_recovery: float  # seconds
    critical_error_count: int
    degraded_components: List[str]
    recommendations: List[str]


class ErrorAnalytics:
    """Advanced error analytics and reporting system."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("cli/logs/error_analytics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Pattern detection settings
        self.min_pattern_frequency = 3
        self.pattern_time_window = timedelta(hours=24)
        self.health_score_weights = {
            "error_rate": 0.3,
            "recovery_rate": 0.25,
            "critical_errors": 0.25,
            "component_health": 0.2,
        }

    def _init_database(self):
        """Initialize the analytics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_id TEXT UNIQUE,
                    timestamp TEXT,
                    severity TEXT,
                    category TEXT,
                    component TEXT,
                    operation TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    recovery_attempted BOOLEAN,
                    recovery_successful BOOLEAN,
                    recovery_strategy TEXT,
                    context_json TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE,
                    description TEXT,
                    frequency INTEGER,
                    first_occurrence TEXT,
                    last_occurrence TEXT,
                    affected_components TEXT,
                    common_context TEXT,
                    suggested_fixes TEXT,
                    confidence_score REAL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    overall_health_score REAL,
                    error_rate REAL,
                    recovery_success_rate REAL,
                    mean_time_to_recovery REAL,
                    critical_error_count INTEGER,
                    degraded_components TEXT,
                    recommendations TEXT
                )
            """)

            # Create indexes for better query performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON error_records(timestamp)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_component ON error_records(component)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_severity ON error_records(severity)",
            )

    def record_error(self, error_record: ErrorRecord):
        """Record an error in the analytics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO error_records (
                    error_id, timestamp, severity, category, component, operation,
                    error_type, error_message, recovery_attempted, recovery_successful,
                    recovery_strategy, context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    error_record.error_id,
                    error_record.timestamp.isoformat(),
                    error_record.severity.value,
                    error_record.category.value,
                    error_record.component,
                    error_record.operation,
                    error_record.error_type,
                    error_record.error_message,
                    error_record.recovery_attempted,
                    error_record.recovery_successful,
                    error_record.recovery_strategy.value
                    if error_record.recovery_strategy
                    else None,
                    json.dumps(asdict(error_record.context), default=str),
                ),
            )

    def analyze_error_trends(
        self,
        period: str = "day",
        days_back: int = 7,
    ) -> List[ErrorTrend]:
        """Analyze error trends over time."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, severity, category, recovery_attempted, recovery_successful
                FROM error_records
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """,
                (start_time.isoformat(), end_time.isoformat()),
            )

            records = cursor.fetchall()

        # Group records by time period
        trends = []
        if period == "hour":
            time_format = "%Y-%m-%d %H:00:00"
            delta = timedelta(hours=1)
        elif period == "day":
            time_format = "%Y-%m-%d"
            delta = timedelta(days=1)
        elif period == "week":
            time_format = "%Y-W%U"
            delta = timedelta(weeks=1)
        else:  # month
            time_format = "%Y-%m"
            delta = timedelta(days=30)

        # Group records by time period
        period_data = defaultdict(list)
        for record in records:
            timestamp = datetime.fromisoformat(record[0])
            if period == "week":
                period_key = timestamp.strftime(time_format)
            else:
                period_key = timestamp.strftime(time_format)
            period_data[period_key].append(record)

        # Create trend objects
        for period_key, period_records in period_data.items():
            if period == "week":
                # Parse week format back to datetime
                year, week = period_key.split("-W")
                period_time = datetime.strptime(f"{year}-W{week}-1", "%Y-W%U-%w")
            else:
                period_time = datetime.strptime(period_key, time_format)

            severity_dist = Counter(record[1] for record in period_records)
            category_dist = Counter(record[2] for record in period_records)

            # Calculate recovery rate
            recovery_attempts = sum(1 for record in period_records if record[3])
            successful_recoveries = sum(
                1 for record in period_records if record[3] and record[4]
            )
            recovery_rate = (
                successful_recoveries / recovery_attempts
                if recovery_attempts > 0
                else 0.0
            )

            trend = ErrorTrend(
                period=period,
                timestamp=period_time,
                error_count=len(period_records),
                severity_distribution=dict(severity_dist),
                category_distribution=dict(category_dist),
                recovery_rate=recovery_rate,
            )
            trends.append(trend)

        return sorted(trends, key=lambda x: x.timestamp)

    def detect_error_patterns(
        self,
        time_window: Optional[timedelta] = None,
    ) -> List[ErrorPattern]:
        """Detect recurring error patterns."""
        if time_window is None:
            time_window = self.pattern_time_window

        start_time = datetime.now() - time_window

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT error_type, error_message, component, operation,
                       timestamp, context_json
                FROM error_records
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """,
                (start_time.isoformat(),),
            )

            records = cursor.fetchall()

        # Group similar errors
        error_groups = defaultdict(list)
        for record in records:
            # Create a pattern key based on error type and simplified message
            error_message = record[1]
            simplified_message = self._simplify_error_message(error_message)
            pattern_key = f"{record[0]}:{simplified_message}"

            error_groups[pattern_key].append(
                {
                    "component": record[2],
                    "operation": record[3],
                    "timestamp": datetime.fromisoformat(record[4]),
                    "context": json.loads(record[5]) if record[5] else {},
                },
            )

        # Identify patterns
        patterns = []
        for pattern_key, occurrences in error_groups.items():
            if len(occurrences) >= self.min_pattern_frequency:
                error_type, simplified_message = pattern_key.split(":", 1)

                # Analyze pattern
                timestamps = [occ["timestamp"] for occ in occurrences]
                components = [occ["component"] for occ in occurrences]

                # Find common context elements
                common_context = self._find_common_context(
                    [occ["context"] for occ in occurrences],
                )

                # Generate suggested fixes
                suggested_fixes = self._generate_pattern_fixes(
                    error_type,
                    simplified_message,
                    common_context,
                )

                # Calculate confidence score
                confidence = self._calculate_pattern_confidence(occurrences)

                pattern = ErrorPattern(
                    pattern_id=f"pattern_{hash(pattern_key)}",
                    description=f"Recurring {error_type}: {simplified_message}",
                    frequency=len(occurrences),
                    first_occurrence=min(timestamps),
                    last_occurrence=max(timestamps),
                    affected_components=list(set(components)),
                    common_context=common_context,
                    suggested_fixes=suggested_fixes,
                    confidence_score=confidence,
                )
                patterns.append(pattern)

        # Store patterns in database
        self._store_patterns(patterns)

        return sorted(patterns, key=lambda x: x.frequency, reverse=True)

    def _simplify_error_message(self, message: str) -> str:
        """Simplify error message for pattern matching."""
        # Remove specific file paths, numbers, and other variable content
        import re

        # Remove file paths
        message = re.sub(r"/[^\s]+", "[PATH]", message)
        message = re.sub(r"[A-Z]:\\[^\s]+", "[PATH]", message)

        # Remove numbers
        message = re.sub(r"\b\d+\.?\d*\b", "[NUMBER]", message)

        # Remove quotes content
        message = re.sub(r'"[^"]*"', "[QUOTED]", message)
        message = re.sub(r"'[^']*'", "[QUOTED]", message)

        # Remove memory addresses
        message = re.sub(r"0x[0-9a-fA-F]+", "[ADDRESS]", message)

        return message.strip()

    def _find_common_context(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common elements across error contexts."""
        if not contexts:
            return {}

        common = {}

        # Find keys that appear in most contexts
        all_keys = set()
        for ctx in contexts:
            all_keys.update(ctx.keys())

        for key in all_keys:
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            if len(values) >= len(contexts) * 0.7:  # Present in 70% of cases
                # Find most common value
                value_counts = Counter(str(v) for v in values if v is not None)
                if value_counts:
                    most_common_value = value_counts.most_common(1)[0][0]
                    common[key] = most_common_value

        return common

    def _generate_pattern_fixes(
        self,
        error_type: str,
        message: str,
        context: Dict[str, Any],
    ) -> List[str]:
        """Generate suggested fixes for error patterns."""
        fixes = []

        if "FileNotFoundError" in error_type:
            fixes.extend(
                [
                    "Verify file paths are correct",
                    "Check file permissions",
                    "Ensure required files exist before operation",
                ],
            )
        elif "MemoryError" in error_type:
            fixes.extend(
                [
                    "Reduce problem size or batch size",
                    "Close unnecessary applications",
                    "Consider system memory upgrade",
                ],
            )
        elif "ConnectionError" in error_type or "TimeoutError" in error_type:
            fixes.extend(
                [
                    "Check network connectivity",
                    "Increase timeout values",
                    "Implement retry logic with exponential backoff",
                ],
            )
        elif "solver" in message.lower():
            fixes.extend(
                [
                    "Verify solver installation",
                    "Check solver configuration",
                    "Use alternative solver if available",
                ],
            )
        elif "config" in message.lower():
            fixes.extend(
                [
                    "Validate configuration file syntax",
                    "Reset to default configuration",
                    "Check configuration file permissions",
                ],
            )

        # Add context-specific fixes
        if "component" in context:
            component = context["component"]
            fixes.append(f"Review {component} component configuration")

        return fixes[:5]  # Limit to top 5 suggestions

    def _calculate_pattern_confidence(self, occurrences: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for a pattern."""
        # Base confidence on frequency and time distribution
        frequency_score = min(len(occurrences) / 10.0, 1.0)  # Max at 10 occurrences

        # Time distribution score (more spread out = higher confidence)
        timestamps = [occ["timestamp"] for occ in occurrences]
        if len(timestamps) > 1:
            time_spans = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            time_variance = (
                statistics.variance(time_spans) if len(time_spans) > 1 else 0
            )
            time_score = min(time_variance / 3600, 1.0)  # Normalize by hour
        else:
            time_score = 0.0

        # Component diversity score
        components = [occ["component"] for occ in occurrences]
        component_diversity = len(set(components)) / len(components)

        # Weighted average
        confidence = (
            frequency_score * 0.5 + time_score * 0.3 + component_diversity * 0.2
        )

        return min(confidence, 1.0)

    def _store_patterns(self, patterns: List[ErrorPattern]):
        """Store detected patterns in database."""
        with sqlite3.connect(self.db_path) as conn:
            for pattern in patterns:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO error_patterns (
                        pattern_id, description, frequency, first_occurrence,
                        last_occurrence, affected_components, common_context,
                        suggested_fixes, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern.pattern_id,
                        pattern.description,
                        pattern.frequency,
                        pattern.first_occurrence.isoformat(),
                        pattern.last_occurrence.isoformat(),
                        json.dumps(pattern.affected_components),
                        json.dumps(pattern.common_context),
                        json.dumps(pattern.suggested_fixes),
                        pattern.confidence_score,
                    ),
                )

    def calculate_system_health(self) -> SystemHealthMetrics:
        """Calculate comprehensive system health metrics."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_hour = now - timedelta(hours=1)

        with sqlite3.connect(self.db_path) as conn:
            # Get recent errors
            cursor = conn.execute(
                """
                SELECT severity, recovery_attempted, recovery_successful,
                       component, timestamp
                FROM error_records
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """,
                (last_24h.isoformat(),),
            )

            recent_errors = cursor.fetchall()

            # Get errors in last hour for rate calculation
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM error_records
                WHERE timestamp >= ?
            """,
                (last_hour.isoformat(),),
            )

            hourly_errors = cursor.fetchone()[0]

        # Calculate metrics
        total_errors = len(recent_errors)
        error_rate = hourly_errors  # errors per hour

        # Recovery metrics
        recovery_attempts = sum(1 for error in recent_errors if error[1])
        successful_recoveries = sum(
            1 for error in recent_errors if error[1] and error[2]
        )
        recovery_success_rate = (
            successful_recoveries / recovery_attempts if recovery_attempts > 0 else 1.0
        )

        # Critical errors
        critical_errors = sum(
            1 for error in recent_errors if error[0] == ErrorSeverity.CRITICAL.value
        )

        # Component health
        component_errors = defaultdict(int)
        for error in recent_errors:
            component_errors[error[3]] += 1

        degraded_components = [
            comp for comp, count in component_errors.items() if count > 5
        ]  # More than 5 errors in 24h

        # Calculate overall health score
        health_score = self._calculate_health_score(
            error_rate,
            recovery_success_rate,
            critical_errors,
            len(degraded_components),
        )

        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            error_rate,
            recovery_success_rate,
            critical_errors,
            degraded_components,
        )

        metrics = SystemHealthMetrics(
            timestamp=now,
            overall_health_score=health_score,
            error_rate=error_rate,
            recovery_success_rate=recovery_success_rate,
            mean_time_to_recovery=0.0,  # Would need more detailed timing data
            critical_error_count=critical_errors,
            degraded_components=degraded_components,
            recommendations=recommendations,
        )

        # Store metrics
        self._store_health_metrics(metrics)

        return metrics

    def _calculate_health_score(
        self,
        error_rate: float,
        recovery_rate: float,
        critical_errors: int,
        degraded_component_count: int,
    ) -> float:
        """Calculate overall system health score (0-100)."""

        # Error rate score (lower is better)
        error_rate_score = max(0, 100 - (error_rate * 10))  # -10 points per error/hour

        # Recovery rate score (higher is better)
        recovery_rate_score = recovery_rate * 100

        # Critical error score (lower is better)
        critical_error_score = max(
            0,
            100 - (critical_errors * 20),
        )  # -20 points per critical error

        # Component health score (lower degraded components is better)
        component_health_score = max(
            0,
            100 - (degraded_component_count * 15),
        )  # -15 points per degraded component

        # Weighted average
        weights = self.health_score_weights
        health_score = (
            error_rate_score * weights["error_rate"]
            + recovery_rate_score * weights["recovery_rate"]
            + critical_error_score * weights["critical_errors"]
            + component_health_score * weights["component_health"]
        )

        return max(0, min(100, health_score))

    def _generate_health_recommendations(
        self,
        error_rate: float,
        recovery_rate: float,
        critical_errors: int,
        degraded_components: List[str],
    ) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []

        if error_rate > 5:
            recommendations.append(
                "High error rate detected - review recent operations and system logs",
            )

        if recovery_rate < 0.7:
            recommendations.append(
                "Low recovery success rate - review and improve error recovery strategies",
            )

        if critical_errors > 0:
            recommendations.append(
                "Critical errors detected - immediate attention required",
            )

        if degraded_components:
            recommendations.append(
                f"Components showing high error rates: {', '.join(degraded_components)}",
            )

        if not recommendations:
            recommendations.append("System health is good - continue monitoring")

        return recommendations

    def _store_health_metrics(self, metrics: SystemHealthMetrics):
        """Store health metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO health_metrics (
                    timestamp, overall_health_score, error_rate,
                    recovery_success_rate, mean_time_to_recovery,
                    critical_error_count, degraded_components, recommendations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.timestamp.isoformat(),
                    metrics.overall_health_score,
                    metrics.error_rate,
                    metrics.recovery_success_rate,
                    metrics.mean_time_to_recovery,
                    metrics.critical_error_count,
                    json.dumps(metrics.degraded_components),
                    json.dumps(metrics.recommendations),
                ),
            )

    def get_error_report(
        self,
        days_back: int = 7,
        include_patterns: bool = True,
    ) -> Dict[str, Any]:
        """Generate comprehensive error report."""

        # Get basic statistics
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM error_records
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """,
                (start_time.isoformat(), end_time.isoformat()),
            )

            records = cursor.fetchall()

        # Compile report
        report = {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": days_back,
            },
            "summary": {
                "total_errors": len(records),
                "errors_per_day": len(records) / days_back,
                "recovery_attempts": sum(
                    1 for r in records if r[8]
                ),  # recovery_attempted
                "successful_recoveries": sum(
                    1 for r in records if r[8] and r[9]
                ),  # recovery_successful
            },
            "breakdown": {
                "by_severity": dict(Counter(r[2] for r in records)),  # severity
                "by_category": dict(Counter(r[3] for r in records)),  # category
                "by_component": dict(Counter(r[4] for r in records)),  # component
                "by_error_type": dict(Counter(r[6] for r in records)),  # error_type
            },
            "trends": self.analyze_error_trends("day", days_back),
            "health_metrics": self.calculate_system_health(),
        }

        # Add patterns if requested
        if include_patterns:
            report["patterns"] = self.detect_error_patterns()

        # Calculate additional insights
        if report["summary"]["recovery_attempts"] > 0:
            report["summary"]["recovery_success_rate"] = (
                report["summary"]["successful_recoveries"]
                / report["summary"]["recovery_attempts"]
            )
        else:
            report["summary"]["recovery_success_rate"] = 0.0

        return report

    def export_analytics_data(self, filepath: Path, format: str = "json"):
        """Export analytics data for external analysis."""
        report = self.get_error_report(days_back=30, include_patterns=True)

        if format.lower() == "json":
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
        elif format.lower() == "csv":
            import csv

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                # Write summary data
                writer.writerow(["Metric", "Value"])
                for key, value in report["summary"].items():
                    writer.writerow([key, value])
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global analytics instance
error_analytics = ErrorAnalytics()
