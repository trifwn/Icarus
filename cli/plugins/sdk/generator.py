"""
Plugin generator for creating new ICARUS CLI plugins.

This module provides tools to generate plugin scaffolding and boilerplate code.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..models import PluginType
from ..models import SecurityLevel


@dataclass
class PluginTemplate:
    """Plugin template configuration."""

    name: str
    description: str
    plugin_type: PluginType
    security_level: SecurityLevel
    files: List[str]
    dependencies: List[str] = None
    permissions: List[str] = None


class PluginGenerator:
    """
    Plugin generator that creates plugin scaffolding and boilerplate code.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)

        # Built-in templates
        self.templates = {
            "basic": PluginTemplate(
                name="Basic Plugin",
                description="A basic plugin with minimal functionality",
                plugin_type=PluginType.UTILITY,
                security_level=SecurityLevel.SAFE,
                files=["__init__.py", "plugin.py", "manifest.json", "README.md"],
            ),
            "analysis": PluginTemplate(
                name="Analysis Plugin",
                description="A plugin that adds custom analysis capabilities",
                plugin_type=PluginType.ANALYSIS,
                security_level=SecurityLevel.RESTRICTED,
                files=[
                    "__init__.py",
                    "plugin.py",
                    "analysis.py",
                    "manifest.json",
                    "README.md",
                    "tests/test_analysis.py",
                ],
                dependencies=["numpy", "scipy"],
                permissions=["data_access", "file_read"],
            ),
            "visualization": PluginTemplate(
                name="Visualization Plugin",
                description="A plugin that adds custom visualization capabilities",
                plugin_type=PluginType.VISUALIZATION,
                security_level=SecurityLevel.SAFE,
                files=[
                    "__init__.py",
                    "plugin.py",
                    "visualizer.py",
                    "manifest.json",
                    "README.md",
                    "assets/styles.css",
                ],
                dependencies=["matplotlib", "plotly"],
            ),
            "integration": PluginTemplate(
                name="Integration Plugin",
                description="A plugin that integrates with external tools",
                plugin_type=PluginType.INTEGRATION,
                security_level=SecurityLevel.ELEVATED,
                files=[
                    "__init__.py",
                    "plugin.py",
                    "connector.py",
                    "manifest.json",
                    "README.md",
                    "config/settings.json",
                ],
                dependencies=["requests"],
                permissions=["network_access", "file_write"],
            ),
            "workflow": PluginTemplate(
                name="Workflow Plugin",
                description="A plugin that adds custom workflow steps",
                plugin_type=PluginType.WORKFLOW,
                security_level=SecurityLevel.RESTRICTED,
                files=[
                    "__init__.py",
                    "plugin.py",
                    "workflow_steps.py",
                    "manifest.json",
                    "README.md",
                    "templates/workflow.json",
                ],
                permissions=["workflow_access"],
            ),
        }

    def list_templates(self) -> List[str]:
        """List available plugin templates."""
        return list(self.templates.keys())

    def get_template_info(self, template_name: str) -> Optional[PluginTemplate]:
        """Get information about a specific template."""
        return self.templates.get(template_name)

    def generate_plugin(
        self,
        plugin_name: str,
        template_name: str,
        output_dir: str,
        author_name: str,
        author_email: str = None,
        description: str = None,
        **kwargs,
    ) -> bool:
        """
        Generate a new plugin from a template.

        Args:
            plugin_name: Name of the plugin
            template_name: Template to use
            output_dir: Output directory
            author_name: Plugin author name
            author_email: Plugin author email
            description: Plugin description
            **kwargs: Additional template variables

        Returns:
            True if generation successful, False otherwise
        """
        try:
            template = self.templates.get(template_name)
            if not template:
                self.logger.error(f"Template not found: {template_name}")
                return False

            # Create output directory
            plugin_dir = Path(output_dir) / plugin_name
            plugin_dir.mkdir(parents=True, exist_ok=True)

            # Template variables
            template_vars = {
                "plugin_name": plugin_name,
                "plugin_class": self._to_class_name(plugin_name),
                "plugin_module": plugin_name.lower().replace("-", "_"),
                "description": description or f"A {template.name.lower()}",
                "author_name": author_name,
                "author_email": author_email
                or f"{author_name.lower().replace(' ', '.')}@example.com",
                "plugin_type": template.plugin_type.value,
                "security_level": template.security_level.value,
                "dependencies": template.dependencies or [],
                "permissions": template.permissions or [],
                "creation_date": datetime.now().isoformat(),
                **kwargs,
            }

            # Generate files
            for file_path in template.files:
                self._generate_file(plugin_dir, file_path, template_vars)

            self.logger.info(
                f"Plugin '{plugin_name}' generated successfully in {plugin_dir}",
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate plugin: {e}")
            return False

    def _generate_file(
        self,
        plugin_dir: Path,
        file_path: str,
        template_vars: Dict[str, Any],
    ) -> None:
        """Generate a single file from template."""
        file_full_path = plugin_dir / file_path
        file_full_path.parent.mkdir(parents=True, exist_ok=True)

        # Get template content
        template_content = self._get_template_content(file_path, template_vars)

        # Write file
        with open(file_full_path, "w") as f:
            f.write(template_content)

    def _get_template_content(
        self,
        file_path: str,
        template_vars: Dict[str, Any],
    ) -> str:
        """Get template content for a specific file."""
        file_name = Path(file_path).name

        if file_name == "__init__.py":
            return self._generate_init_py(template_vars)
        elif file_name == "plugin.py":
            return self._generate_plugin_py(template_vars)
        elif file_name == "manifest.json":
            return self._generate_manifest_json(template_vars)
        elif file_name == "README.md":
            return self._generate_readme_md(template_vars)
        elif file_name == "analysis.py":
            return self._generate_analysis_py(template_vars)
        elif file_name == "visualizer.py":
            return self._generate_visualizer_py(template_vars)
        elif file_name == "connector.py":
            return self._generate_connector_py(template_vars)
        elif file_name == "workflow_steps.py":
            return self._generate_workflow_steps_py(template_vars)
        elif file_path.startswith("tests/"):
            return self._generate_test_file(file_path, template_vars)
        elif file_name.endswith(".json"):
            return self._generate_json_file(file_path, template_vars)
        elif file_name.endswith(".css"):
            return self._generate_css_file(template_vars)
        else:
            return f"# Generated file: {file_path}\n# TODO: Add content\n"

    def _generate_init_py(self, vars: Dict[str, Any]) -> str:
        """Generate __init__.py file."""
        return f'''"""
{vars['plugin_name']} plugin for ICARUS CLI.

{vars['description']}

Author: {vars['author_name']}
Created: {vars['creation_date']}
"""

from .plugin import {vars['plugin_class']}

__version__ = "1.0.0"
__author__ = "{vars['author_name']}"
__email__ = "{vars['author_email']}"

# Plugin entry point
PLUGIN_CLASS = {vars['plugin_class']}
'''

    def _generate_plugin_py(self, vars: Dict[str, Any]) -> str:
        """Generate main plugin.py file."""
        imports = [
            "from typing import Dict, Any, Optional",
            "import logging",
            "",
            "from icarus_cli.plugins.api import (",
            "    IcarusPlugin, PluginManifest, PluginType, SecurityLevel,",
            "    PluginAuthor, PluginVersion",
        ]

        if vars.get("permissions"):
            imports.append("    PluginPermission")

        imports.extend([")", ""])

        # Add specific imports based on plugin type
        if vars["plugin_type"] == "analysis":
            imports.append("from .analysis import CustomAnalysis")
        elif vars["plugin_type"] == "visualization":
            imports.append("from .visualizer import CustomVisualizer")
        elif vars["plugin_type"] == "integration":
            imports.append("from .connector import ExternalConnector")
        elif vars["plugin_type"] == "workflow":
            imports.append("from .workflow_steps import CustomWorkflowStep")

        imports.append("")

        # Generate class
        class_content = f'''class {vars['plugin_class']}(IcarusPlugin):
    """
    {vars['description']}
    """

    def get_manifest(self) -> PluginManifest:
        """Return plugin manifest."""
        manifest = PluginManifest(
            name="{vars['plugin_module']}",
            version=PluginVersion(1, 0, 0),
            description="{vars['description']}",
            author=PluginAuthor(
                name="{vars['author_name']}",
                email="{vars['author_email']}"
            ),
            plugin_type=PluginType.{vars['plugin_type'].upper()},
            security_level=SecurityLevel.{vars['security_level'].upper()},
            main_module="{vars['plugin_module']}",
            main_class="{vars['plugin_class']}",
            keywords=["{vars['plugin_type']}", "custom"],
            license="MIT"
        )

        # Add permissions if required
        permissions = {vars.get('permissions', [])}
        if permissions:
            from icarus_cli.plugins.api import PluginPermission
            manifest.permissions = [
                PluginPermission(name=perm, description=f"Permission for {{perm}}", required=True)
                for perm in permissions
            ]

        return manifest

    def on_activate(self):
        """Called when plugin is activated."""
        self.api.log_info("{vars['plugin_class']} activated")

        # Add menu items
        self.api.add_menu_item(
            "Plugins/{vars['plugin_name']}",
            "Open {vars['plugin_name']}",
            self.main_action,
            icon="🔌"
        )

        # Register commands
        self.api.register_command(
            "{vars['plugin_module']}.main",
            self.main_action,
            "Execute main {vars['plugin_name']} action",
            "{vars['plugin_module']}.main"
        )

        # Plugin-specific activation
        self._setup_plugin_features()

    def on_deactivate(self):
        """Called when plugin is deactivated."""
        self.api.log_info("{vars['plugin_class']} deactivated")

    def main_action(self):
        """Main plugin action."""
        self.api.show_notification(
            "{vars['plugin_name']} action executed!",
            'info',
            3000
        )
        self.api.log_info("Main action executed")

    def _setup_plugin_features(self):
        """Setup plugin-specific features."""
        # TODO: Implement plugin-specific setup
        pass
'''

        return "\\n".join(imports) + "\\n\\n" + class_content

    def _generate_manifest_json(self, vars: Dict[str, Any]) -> str:
        """Generate manifest.json file."""
        manifest = {
            "name": vars["plugin_module"],
            "version": "1.0.0",
            "description": vars["description"],
            "author": {"name": vars["author_name"], "email": vars["author_email"]},
            "type": vars["plugin_type"],
            "security_level": vars["security_level"],
            "main_module": vars["plugin_module"],
            "main_class": vars["plugin_class"],
            "keywords": [vars["plugin_type"], "custom"],
            "license": "MIT",
            "python_version": ">=3.8",
            "icarus_version": ">=1.0.0",
        }

        if vars.get("dependencies"):
            manifest["install_requires"] = vars["dependencies"]

        if vars.get("permissions"):
            manifest["permissions"] = [
                {
                    "name": perm,
                    "description": f"Permission for {perm}",
                    "required": True,
                }
                for perm in vars["permissions"]
            ]

        return json.dumps(manifest, indent=2)

    def _generate_readme_md(self, vars: Dict[str, Any]) -> str:
        """Generate README.md file."""
        return f"""# {vars['plugin_name']}

{vars['description']}

## Installation

1. Copy this plugin directory to your ICARUS CLI plugins folder
2. Restart ICARUS CLI or reload plugins
3. Activate the plugin from the plugin manager

## Usage

After activation, you can access the plugin features through:

- Menu: Plugins → {vars['plugin_name']}
- Command: `{vars['plugin_module']}.main`

## Configuration

This plugin supports the following configuration options:

```json
{{
  "enabled": true,
  "setting1": "value1",
  "setting2": "value2"
}}
```

## Development

### Requirements

- Python >= 3.8
- ICARUS CLI >= 1.0.0
{f"- Dependencies: {', '.join(vars['dependencies'])}" if vars.get('dependencies') else ""}

### Testing

Run the plugin tests:

```bash
python -m pytest tests/
```

## License

MIT License

## Author

{vars['author_name']} ({vars['author_email']})

Created: {vars['creation_date']}
"""

    def _generate_analysis_py(self, vars: Dict[str, Any]) -> str:
        """Generate analysis.py file for analysis plugins."""
        return '''"""
Custom analysis implementation.
"""

from typing import Dict, Any, Optional
import numpy as np


class CustomAnalysis:
    """
    Custom analysis implementation.

    This class implements the actual analysis logic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """
        Run the analysis.

        Returns:
            Analysis results dictionary
        """
        # TODO: Implement actual analysis logic

        # Example mock analysis
        input_data = self.config.get('input_data', [])

        # Perform calculations
        if input_data:
            results = {
                'mean': np.mean(input_data),
                'std': np.std(input_data),
                'min': np.min(input_data),
                'max': np.max(input_data)
            }
        else:
            results = {
                'status': 'no_data',
                'message': 'No input data provided'
            }

        self.results = results
        return results

    def get_results(self) -> Dict[str, Any]:
        """Get analysis results."""
        return self.results

    def validate_config(self) -> bool:
        """Validate analysis configuration."""
        # TODO: Implement configuration validation
        return True
'''

    def _generate_visualizer_py(self, vars: Dict[str, Any]) -> str:
        """Generate visualizer.py file for visualization plugins."""
        return '''"""
Custom visualization implementation.
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt


class CustomVisualizer:
    """
    Custom visualization implementation.

    This class implements custom visualization capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def create_plot(self, data: Dict[str, Any]) -> Optional[plt.Figure]:
        """
        Create a custom plot.

        Args:
            data: Data to visualize

        Returns:
            Matplotlib figure or None
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # TODO: Implement actual plotting logic

            # Example plot
            if 'x' in data and 'y' in data:
                ax.plot(data['x'], data['y'], 'b-', linewidth=2)
                ax.set_xlabel('X Values')
                ax.set_ylabel('Y Values')
                ax.set_title('Custom Plot')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data to plot',
                       ha='center', va='center', transform=ax.transAxes)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error creating plot: {e}")
            return None

    def export_plot(self, fig: plt.Figure, filename: str, format: str = 'png') -> bool:
        """
        Export plot to file.

        Args:
            fig: Matplotlib figure
            filename: Output filename
            format: Output format

        Returns:
            True if successful, False otherwise
        """
        try:
            fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"Error exporting plot: {e}")
            return False
'''

    def _generate_connector_py(self, vars: Dict[str, Any]) -> str:
        """Generate connector.py file for integration plugins."""
        return '''"""
External tool connector implementation.
"""

from typing import Dict, Any, Optional
import requests
import json


class ExternalConnector:
    """
    Connector for external tool integration.

    This class handles communication with external tools and services.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:8080')
        self.api_key = config.get('api_key')
        self.timeout = config.get('timeout', 30)

    def connect(self) -> bool:
        """
        Test connection to external tool.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
                headers=self._get_headers()
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def send_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send data to external tool.

        Args:
            data: Data to send

        Returns:
            Response data or None if failed
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/data",
                json=data,
                timeout=self.timeout,
                headers=self._get_headers()
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Request failed with status {response.status_code}")
                return None

        except Exception as e:
            print(f"Error sending data: {e}")
            return None

    def get_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get results from external tool.

        Args:
            job_id: Job identifier

        Returns:
            Results data or None if failed
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/results/{job_id}",
                timeout=self.timeout,
                headers=self._get_headers()
            )

            if response.status_code == 200:
                return response.json()
            else:
                return None

        except Exception as e:
            print(f"Error getting results: {e}")
            return None

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {'Content-Type': 'application/json'}

        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        return headers
'''

    def _generate_workflow_steps_py(self, vars: Dict[str, Any]) -> str:
        """Generate workflow_steps.py file for workflow plugins."""
        return '''"""
Custom workflow steps implementation.
"""

from typing import Dict, Any, Optional


class CustomWorkflowStep:
    """
    Custom workflow step implementation.

    This class implements custom workflow step logic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'Custom Step')
        self.description = config.get('description', 'A custom workflow step')

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow step.

        Args:
            input_data: Input data from previous steps

        Returns:
            Output data for next steps
        """
        try:
            # TODO: Implement actual step logic

            # Example processing
            output_data = input_data.copy()
            output_data['processed_by'] = self.name
            output_data['step_config'] = self.config

            return {
                'success': True,
                'data': output_data,
                'message': f'Step {self.name} completed successfully'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Step {self.name} failed'
            }

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for this step.

        Args:
            input_data: Input data to validate

        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement validation logic
        return True

    def get_requirements(self) -> Dict[str, Any]:
        """
        Get step requirements.

        Returns:
            Dictionary describing step requirements
        """
        return {
            'inputs': ['data'],
            'outputs': ['processed_data'],
            'parameters': list(self.config.keys()),
            'dependencies': []
        }
'''

    def _generate_test_file(self, file_path: str, vars: Dict[str, Any]) -> str:
        """Generate test file."""
        test_name = Path(file_path).stem
        return f'''"""
Tests for {vars['plugin_name']} plugin.
"""

import unittest
from unittest.mock import Mock, patch

from {vars['plugin_module']}.plugin import {vars['plugin_class']}


class Test{vars['plugin_class']}(unittest.TestCase):
    """Test cases for {vars['plugin_class']}."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = {vars['plugin_class']}()
        self.mock_api = Mock()
        self.plugin.api = self.mock_api

    def test_get_manifest(self):
        """Test plugin manifest."""
        manifest = self.plugin.get_manifest()

        self.assertEqual(manifest.name, "{vars['plugin_module']}")
        self.assertEqual(manifest.plugin_type.value, "{vars['plugin_type']}")
        self.assertEqual(manifest.security_level.value, "{vars['security_level']}")

    def test_activation(self):
        """Test plugin activation."""
        self.plugin.on_activate()

        # Verify API calls were made
        self.mock_api.log_info.assert_called()
        self.mock_api.add_menu_item.assert_called()
        self.mock_api.register_command.assert_called()

    def test_main_action(self):
        """Test main plugin action."""
        self.plugin.main_action()

        # Verify notification was shown
        self.mock_api.show_notification.assert_called()
        self.mock_api.log_info.assert_called()

    def test_deactivation(self):
        """Test plugin deactivation."""
        self.plugin.on_deactivate()

        # Verify cleanup was performed
        self.mock_api.log_info.assert_called()


if __name__ == '__main__':
    unittest.main()
'''

    def _generate_json_file(self, file_path: str, vars: Dict[str, Any]) -> str:
        """Generate JSON configuration file."""
        if "settings" in file_path:
            return json.dumps(
                {
                    "default_setting": "value",
                    "numeric_setting": 42,
                    "boolean_setting": True,
                },
                indent=2,
            )
        elif "workflow" in file_path:
            return json.dumps(
                {
                    "name": "Custom Workflow",
                    "description": "A custom workflow template",
                    "steps": [{"name": "Step 1", "type": "custom_step", "config": {}}],
                },
                indent=2,
            )
        else:
            return json.dumps({"placeholder": "configuration"}, indent=2)

    def _generate_css_file(self, vars: Dict[str, Any]) -> str:
        """Generate CSS styles file."""
        return """/* Custom styles for visualization plugin */

.custom-plot {
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
}

.plot-title {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
}

.plot-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.control-button {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 3px;
    cursor: pointer;
}

.control-button:hover {
    background-color: #0056b3;
}
"""

    def _to_class_name(self, plugin_name: str) -> str:
        """Convert plugin name to class name."""
        # Remove special characters and convert to PascalCase
        words = plugin_name.replace("-", " ").replace("_", " ").split()
        return "".join(word.capitalize() for word in words) + "Plugin"

    def create_custom_template(
        self,
        template_name: str,
        template_config: Dict[str, Any],
    ) -> bool:
        """
        Create a custom plugin template.

        Args:
            template_name: Name of the template
            template_config: Template configuration

        Returns:
            True if created successfully, False otherwise
        """
        try:
            template = PluginTemplate(
                name=template_config["name"],
                description=template_config["description"],
                plugin_type=PluginType(template_config["plugin_type"]),
                security_level=SecurityLevel(template_config["security_level"]),
                files=template_config["files"],
                dependencies=template_config.get("dependencies"),
                permissions=template_config.get("permissions"),
            )

            self.templates[template_name] = template

            # Save template to file
            template_file = self.templates_dir / f"{template_name}.json"
            with open(template_file, "w") as f:
                json.dump(template_config, f, indent=2)

            self.logger.info(f"Custom template '{template_name}' created")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create custom template: {e}")
            return False
