"""
Analysis plugin example for ICARUS CLI.

This plugin demonstrates how to create custom analysis modules.
"""

import os
import sys
from typing import Any
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import IcarusPlugin
from api import PluginAuthor
from api import PluginManifest
from api import PluginPermission
from api import PluginType
from api import PluginVersion
from api import SecurityLevel


class SimpleAirfoilAnalysis:
    """
    Simple airfoil analysis implementation.

    This is a mock analysis for demonstration purposes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """Run the analysis."""
        # Mock analysis - in reality this would call actual analysis code
        airfoil_name = self.config.get("airfoil", "NACA0012")
        reynolds = self.config.get("reynolds", 1e6)
        alpha_range = self.config.get("alpha_range", [-10, 10])

        # Generate mock data
        alphas = np.linspace(alpha_range[0], alpha_range[1], 21)
        cls = 2 * np.pi * np.sin(np.radians(alphas)) * 0.8  # Mock lift curve
        cds = 0.01 + 0.02 * (np.radians(alphas)) ** 2  # Mock drag curve

        self.results = {
            "airfoil": airfoil_name,
            "reynolds": reynolds,
            "alpha": alphas.tolist(),
            "cl": cls.tolist(),
            "cd": cds.tolist(),
            "cl_cd": (cls / cds).tolist(),
        }

        return self.results

    def get_results(self) -> Dict[str, Any]:
        """Get analysis results."""
        return self.results


class AnalysisExamplePlugin(IcarusPlugin):
    """
    Example plugin that adds a custom analysis type.
    """

    def get_manifest(self) -> PluginManifest:
        """Return plugin manifest."""
        return PluginManifest(
            name="analysis_example",
            version=PluginVersion(1, 0, 0),
            description="Example plugin demonstrating custom analysis integration",
            author=PluginAuthor(name="ICARUS Team", email="team@icarus.example.com"),
            plugin_type=PluginType.ANALYSIS,
            security_level=SecurityLevel.RESTRICTED,
            main_module="analysis_example",
            main_class="AnalysisExamplePlugin",
            permissions=[
                PluginPermission(
                    name="data_access",
                    description="Access to analysis data storage",
                    required=True,
                ),
                PluginPermission(
                    name="file_read",
                    description="Read airfoil geometry files",
                    required=True,
                ),
            ],
            keywords=["analysis", "airfoil", "example"],
            default_config={
                "default_reynolds": 1000000,
                "default_alpha_range": [-10, 10],
                "auto_save_results": True,
            },
        )

    def on_activate(self):
        """Called when plugin is activated."""
        self.api.log_info("Analysis Example plugin activated")

        # Register custom analysis type
        self.api.register_analysis_type("simple_airfoil", SimpleAirfoilAnalysis)

        # Add menu items
        self.api.add_menu_item(
            "Analysis/Custom/Simple Airfoil",
            "Run Simple Airfoil Analysis",
            self.run_simple_analysis,
            icon="✈️",
        )

        # Register commands
        self.api.register_command(
            "analysis.simple_airfoil",
            self.run_simple_analysis,
            "Run simple airfoil analysis",
            "analysis.simple_airfoil [airfoil_name]",
        )

        # Register event handlers
        self.api.register_event_handler(
            "analysis_requested",
            self.on_analysis_requested,
        )
        self.api.register_event_handler("analysis_complete", self.on_analysis_complete)

    def run_simple_analysis(self, airfoil_name: str = None):
        """Run simple airfoil analysis."""
        try:
            # Get configuration
            config = {
                "airfoil": airfoil_name
                or self.api.get_config("default_airfoil", "NACA0012"),
                "reynolds": self.api.get_config("default_reynolds", 1000000),
                "alpha_range": self.api.get_config("default_alpha_range", [-10, 10]),
            }

            self.api.log_info(
                f"Starting simple airfoil analysis for {config['airfoil']}",
            )

            # Create and run analysis
            analysis = SimpleAirfoilAnalysis(config)
            results = analysis.run()

            # Save results if configured
            if self.api.get_config("auto_save_results", True):
                analysis_id = f"simple_airfoil_{config['airfoil']}_{hash(str(config))}"
                self.api.save_analysis_data(analysis_id, results)
                self.api.log_info(f"Results saved with ID: {analysis_id}")

            # Show completion notification
            self.api.show_notification(
                f"Simple airfoil analysis complete for {config['airfoil']}",
                "success",
                3000,
            )

            # Emit completion event
            self.api.emit_event(
                "analysis_complete",
                {
                    "type": "simple_airfoil",
                    "airfoil": config["airfoil"],
                    "results": results,
                },
            )

        except Exception as e:
            self.api.log_error(f"Analysis failed: {e}", exc_info=True)
            self.api.show_notification(f"Analysis failed: {e}", "error", 5000)

    def on_analysis_requested(self, data):
        """Handle analysis requested event."""
        if data.get("type") == "simple_airfoil":
            self.api.log_info("Simple airfoil analysis requested via event")
            self.run_simple_analysis(data.get("airfoil"))

    def on_analysis_complete(self, data):
        """Handle analysis completion event."""
        if data.get("type") == "simple_airfoil":
            self.api.log_info(
                f"Simple airfoil analysis completed for {data.get('airfoil')}",
            )

            # Could trigger visualization or further processing here
            self.api.emit_event(
                "visualization_requested",
                {"type": "polar_plot", "data": data.get("results")},
            )


# Plugin manifest for directory-based loading
PLUGIN_MANIFEST = {
    "name": "analysis_example",
    "version": "1.0.0",
    "description": "Example plugin demonstrating custom analysis integration",
    "author": {"name": "ICARUS Team", "email": "team@icarus.example.com"},
    "type": "analysis",
    "security_level": "restricted",
    "main_module": "analysis_example",
    "main_class": "AnalysisExamplePlugin",
    "permissions": [
        {
            "name": "data_access",
            "description": "Access to analysis data storage",
            "required": True,
        },
        {
            "name": "file_read",
            "description": "Read airfoil geometry files",
            "required": True,
        },
    ],
    "keywords": ["analysis", "airfoil", "example"],
    "install_requires": ["numpy"],
    "default_config": {
        "default_reynolds": 1000000,
        "default_alpha_range": [-10, 10],
        "auto_save_results": True,
    },
}
