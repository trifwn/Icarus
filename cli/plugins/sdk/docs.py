"""
Plugin documentation generator for ICARUS CLI plugins.

This module provides tools for automatically generating comprehensive
documentation for plugins.
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class DocSection:
    """Documentation section."""

    def __init__(self, title: str, content: str, level: int = 1):
        self.title = title
        self.content = content
        self.level = level
        self.subsections: List["DocSection"] = []

    def add_subsection(self, subsection: "DocSection") -> None:
        """Add a subsection."""
        self.subsections.append(subsection)

    def to_markdown(self) -> str:
        """Convert to markdown."""
        md = f"{'#' * self.level} {self.title}\n\n"
        md += f"{self.content}\n\n"

        for subsection in self.subsections:
            md += subsection.to_markdown()

        return md


class PluginDocGenerator:
    """
    Plugin documentation generator that creates comprehensive
    documentation from plugin code, manifests, and docstrings.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Documentation templates
        self.templates = {
            "readme": self._get_readme_template(),
            "api": self._get_api_template(),
            "user_guide": self._get_user_guide_template(),
            "developer_guide": self._get_developer_guide_template(),
        }

    def generate_documentation(
        self,
        plugin_path: str,
        output_dir: str = None,
        formats: List[str] = None,
        include_api: bool = True,
        include_examples: bool = True,
    ) -> Dict[str, str]:
        """
        Generate comprehensive documentation for a plugin.

        Args:
            plugin_path: Path to plugin directory or file
            output_dir: Output directory for documentation
            formats: Documentation formats to generate (markdown, html, rst)
            include_api: Whether to include API documentation
            include_examples: Whether to include code examples

        Returns:
            Dictionary mapping document types to file paths
        """
        try:
            plugin_path = Path(plugin_path)

            if not output_dir:
                output_dir = (
                    plugin_path.parent / "docs"
                    if plugin_path.is_file()
                    else plugin_path / "docs"
                )

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            formats = formats or ["markdown"]

            self.logger.info(f"Generating documentation for: {plugin_path}")

            # Analyze plugin
            plugin_info = self._analyze_plugin(plugin_path)

            # Generate documentation sections
            sections = self._generate_sections(
                plugin_info,
                include_api,
                include_examples,
            )

            # Generate documents
            generated_docs = {}

            for doc_type, section_list in sections.items():
                for format_type in formats:
                    doc_path = self._generate_document(
                        doc_type,
                        section_list,
                        output_dir,
                        format_type,
                    )
                    generated_docs[f"{doc_type}.{format_type}"] = str(doc_path)

            self.logger.info(f"Documentation generated in: {output_dir}")
            return generated_docs

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            return {}

    def _analyze_plugin(self, plugin_path: Path) -> Dict[str, Any]:
        """Analyze plugin structure and extract information."""
        info = {
            "path": plugin_path,
            "manifest": None,
            "plugin_class": None,
            "modules": [],
            "functions": [],
            "classes": [],
            "examples": [],
            "tests": [],
        }

        try:
            # Load manifest
            info["manifest"] = self._load_manifest(plugin_path)

            # Analyze Python files
            if plugin_path.is_file():
                python_files = [plugin_path]
            else:
                python_files = list(plugin_path.glob("**/*.py"))

            for py_file in python_files:
                file_info = self._analyze_python_file(py_file)

                # Find main plugin class
                if info["manifest"]:
                    main_class = info["manifest"].get("main_class")
                    for class_info in file_info["classes"]:
                        if class_info["name"] == main_class:
                            info["plugin_class"] = class_info

                info["modules"].append(
                    {
                        "file": py_file,
                        "relative_path": str(py_file.relative_to(plugin_path))
                        if plugin_path.is_dir()
                        else py_file.name,
                        **file_info,
                    },
                )

            # Find examples
            info["examples"] = self._find_examples(plugin_path)

            # Find tests
            info["tests"] = self._find_tests(plugin_path)

        except Exception as e:
            self.logger.error(f"Plugin analysis failed: {e}")

        return info

    def _load_manifest(self, plugin_path: Path) -> Optional[Dict[str, Any]]:
        """Load plugin manifest."""
        if plugin_path.is_file():
            # Look for inline manifest
            return self._extract_inline_manifest(plugin_path)
        else:
            # Look for manifest.json
            manifest_file = plugin_path / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    return json.load(f)

        return None

    def _extract_inline_manifest(self, plugin_file: Path) -> Optional[Dict[str, Any]]:
        """Extract inline manifest from Python file."""
        try:
            with open(plugin_file) as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "PLUGIN_MANIFEST"
                ):
                    if isinstance(node.value, ast.Dict):
                        return ast.literal_eval(node.value)

            return None

        except Exception:
            return None

    def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file and extract information."""
        info = {
            "docstring": None,
            "imports": [],
            "classes": [],
            "functions": [],
            "constants": [],
        }

        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content)

            # Module docstring
            info["docstring"] = ast.get_docstring(tree)

            # Analyze nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        info["imports"].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        info["imports"].append(f"{module}.{alias.name}")

                elif isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    info["classes"].append(class_info)

                elif isinstance(node, ast.FunctionDef):
                    if not self._is_method(node, tree):
                        func_info = self._analyze_function(node)
                        info["functions"].append(func_info)

                elif isinstance(node, ast.Assign):
                    # Look for constants (uppercase variables)
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            info["constants"].append(
                                {
                                    "name": target.id,
                                    "value": self._get_node_value(node.value),
                                },
                            )

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")

        return info

    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition."""
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "bases": [self._get_node_name(base) for base in node.bases],
            "methods": [],
            "properties": [],
        }

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item)
                method_info["is_method"] = True

                if item.name.startswith("__") and item.name.endswith("__"):
                    method_info["type"] = "magic"
                elif item.name.startswith("_"):
                    method_info["type"] = "private"
                else:
                    method_info["type"] = "public"

                class_info["methods"].append(method_info)

        return class_info

    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition."""
        func_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [],
            "returns": None,
            "decorators": [],
        }

        # Arguments
        for arg in node.args.args:
            arg_info = {"name": arg.arg}
            if arg.annotation:
                arg_info["type"] = self._get_node_name(arg.annotation)
            func_info["args"].append(arg_info)

        # Return type
        if node.returns:
            func_info["returns"] = self._get_node_name(node.returns)

        # Decorators
        for decorator in node.decorator_list:
            func_info["decorators"].append(self._get_node_name(decorator))

        return func_info

    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method (inside a class)."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False

    def _get_node_name(self, node: ast.AST) -> str:
        """Get the name of an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)

    def _get_node_value(self, node: ast.AST) -> Any:
        """Get the value of an AST node."""
        try:
            return ast.literal_eval(node)
        except:
            return str(node)

    def _find_examples(self, plugin_path: Path) -> List[Dict[str, Any]]:
        """Find example files and code."""
        examples = []

        if plugin_path.is_dir():
            # Look for examples directory
            examples_dir = plugin_path / "examples"
            if examples_dir.exists():
                for example_file in examples_dir.glob("**/*.py"):
                    examples.append(
                        {
                            "file": example_file,
                            "name": example_file.stem,
                            "content": example_file.read_text(),
                        },
                    )

            # Look for example files in root
            for example_file in plugin_path.glob("example*.py"):
                examples.append(
                    {
                        "file": example_file,
                        "name": example_file.stem,
                        "content": example_file.read_text(),
                    },
                )

        return examples

    def _find_tests(self, plugin_path: Path) -> List[Dict[str, Any]]:
        """Find test files."""
        tests = []

        if plugin_path.is_dir():
            # Look for tests directory
            tests_dir = plugin_path / "tests"
            if tests_dir.exists():
                for test_file in tests_dir.glob("**/*.py"):
                    tests.append(
                        {
                            "file": test_file,
                            "name": test_file.stem,
                            "type": "unittest"
                            if "unittest" in test_file.read_text()
                            else "pytest",
                        },
                    )

            # Look for test files in root
            for test_file in plugin_path.glob("test_*.py"):
                tests.append(
                    {
                        "file": test_file,
                        "name": test_file.stem,
                        "type": "unittest"
                        if "unittest" in test_file.read_text()
                        else "pytest",
                    },
                )

        return tests

    def _generate_sections(
        self,
        plugin_info: Dict[str, Any],
        include_api: bool,
        include_examples: bool,
    ) -> Dict[str, List[DocSection]]:
        """Generate documentation sections."""
        sections = {"readme": [], "user_guide": [], "developer_guide": []}

        manifest = plugin_info.get("manifest", {})

        # README sections
        sections["readme"].extend(
            [
                self._generate_overview_section(plugin_info),
                self._generate_installation_section(plugin_info),
                self._generate_usage_section(plugin_info),
                self._generate_configuration_section(plugin_info),
                self._generate_license_section(plugin_info),
            ],
        )

        # User guide sections
        sections["user_guide"].extend(
            [
                self._generate_getting_started_section(plugin_info),
                self._generate_features_section(plugin_info),
                self._generate_configuration_section(plugin_info),
                self._generate_troubleshooting_section(plugin_info),
            ],
        )

        # Developer guide sections
        sections["developer_guide"].extend(
            [
                self._generate_architecture_section(plugin_info),
                self._generate_development_setup_section(plugin_info),
            ],
        )

        if include_api:
            sections["developer_guide"].append(self._generate_api_section(plugin_info))

        if include_examples and plugin_info.get("examples"):
            sections["user_guide"].append(self._generate_examples_section(plugin_info))
            sections["developer_guide"].append(
                self._generate_examples_section(plugin_info),
            )

        return sections

    def _generate_overview_section(self, plugin_info: Dict[str, Any]) -> DocSection:
        """Generate overview section."""
        manifest = plugin_info.get("manifest", {})

        title = manifest.get("name", "Plugin")
        description = manifest.get("description", "An ICARUS CLI plugin.")

        content = f"{description}\n\n"

        if manifest.get("author"):
            author = manifest["author"]
            if isinstance(author, dict):
                author_name = author.get("name", "Unknown")
                author_email = author.get("email")
                if author_email:
                    content += f"**Author:** {author_name} ({author_email})\n"
                else:
                    content += f"**Author:** {author_name}\n"
            else:
                content += f"**Author:** {author}\n"

        if manifest.get("version"):
            content += f"**Version:** {manifest['version']}\n"

        if manifest.get("license"):
            content += f"**License:** {manifest['license']}\n"

        if manifest.get("type"):
            content += f"**Type:** {manifest['type'].title()} Plugin\n"

        return DocSection(title, content)

    def _generate_installation_section(self, plugin_info: Dict[str, Any]) -> DocSection:
        """Generate installation section."""
        manifest = plugin_info.get("manifest", {})

        content = "## Installation\n\n"
        content += "### Prerequisites\n\n"

        python_version = manifest.get("python_version", ">=3.8")
        icarus_version = manifest.get("icarus_version", ">=1.0.0")

        content += f"- Python {python_version}\n"
        content += f"- ICARUS CLI {icarus_version}\n"

        if manifest.get("install_requires"):
            content += "- Additional dependencies:\n"
            for dep in manifest["install_requires"]:
                content += f"  - {dep}\n"

        content += "\n### Installation Steps\n\n"
        content += "1. Download the plugin package\n"
        content += "2. Extract to your ICARUS plugins directory\n"
        content += "3. Restart ICARUS CLI or reload plugins\n"
        content += "4. Activate the plugin from the plugin manager\n"

        return DocSection("Installation", content, level=2)

    def _generate_usage_section(self, plugin_info: Dict[str, Any]) -> DocSection:
        """Generate usage section."""
        manifest = plugin_info.get("manifest", {})
        plugin_class = plugin_info.get("plugin_class")

        content = (
            "After installation and activation, you can use the plugin through:\n\n"
        )

        plugin_name = manifest.get("name", "plugin")

        # Menu access
        content += "### Menu Access\n\n"
        content += (
            f"Navigate to **Plugins → {plugin_name.title()}** in the main menu.\n\n"
        )

        # Command access
        content += "### Command Line\n\n"
        content += f"Use the command: `{plugin_name}.main`\n\n"

        # API usage if it's a development plugin
        if plugin_class and plugin_class.get("methods"):
            content += "### Programmatic Usage\n\n"
            content += "```python\n"
            content += "# Access the plugin through the ICARUS CLI API\n"
            content += f"plugin = app.get_plugin('{plugin_name}')\n"
            content += "plugin.main_action()\n"
            content += "```\n\n"

        return DocSection("Usage", content, level=2)

    def _generate_configuration_section(
        self,
        plugin_info: Dict[str, Any],
    ) -> DocSection:
        """Generate configuration section."""
        manifest = plugin_info.get("manifest", {})

        content = "The plugin can be configured through the plugin settings.\n\n"

        default_config = manifest.get("default_config")
        if default_config:
            content += "### Default Configuration\n\n"
            content += "```json\n"
            content += json.dumps(default_config, indent=2)
            content += "\n```\n\n"

        config_schema = manifest.get("config_schema")
        if config_schema:
            content += "### Configuration Schema\n\n"
            content += "```json\n"
            content += json.dumps(config_schema, indent=2)
            content += "\n```\n\n"

        return DocSection("Configuration", content, level=2)

    def _generate_license_section(self, plugin_info: Dict[str, Any]) -> DocSection:
        """Generate license section."""
        manifest = plugin_info.get("manifest", {})

        license_name = manifest.get("license", "MIT")

        content = f"This plugin is licensed under the {license_name} License.\n\n"

        # Look for LICENSE file
        plugin_path = plugin_info["path"]
        if plugin_path.is_dir():
            license_files = list(plugin_path.glob("LICENSE*"))
            if license_files:
                license_file = license_files[0]
                try:
                    license_content = license_file.read_text()
                    content += "```\n"
                    content += license_content
                    content += "\n```\n"
                except Exception:
                    pass

        return DocSection("License", content, level=2)

    def _generate_getting_started_section(
        self,
        plugin_info: Dict[str, Any],
    ) -> DocSection:
        """Generate getting started section."""
        content = "This section provides a quick start guide for using the plugin.\n\n"

        # Add basic usage steps
        content += "### Quick Start\n\n"
        content += "1. Ensure the plugin is installed and activated\n"
        content += "2. Access the plugin through the main menu\n"
        content += "3. Follow the on-screen instructions\n"
        content += "4. Configure settings as needed\n\n"

        return DocSection("Getting Started", content)

    def _generate_features_section(self, plugin_info: Dict[str, Any]) -> DocSection:
        """Generate features section."""
        plugin_class = plugin_info.get("plugin_class")

        content = "This plugin provides the following features:\n\n"

        if plugin_class and plugin_class.get("methods"):
            for method in plugin_class["methods"]:
                if method["type"] == "public" and not method["name"].startswith("on_"):
                    method_name = method["name"].replace("_", " ").title()
                    content += f"- **{method_name}**"

                    if method.get("docstring"):
                        # Extract first line of docstring
                        first_line = method["docstring"].split("\n")[0].strip()
                        content += f": {first_line}"

                    content += "\n"

        content += "\n"

        return DocSection("Features", content)

    def _generate_troubleshooting_section(
        self,
        plugin_info: Dict[str, Any],
    ) -> DocSection:
        """Generate troubleshooting section."""
        content = "Common issues and solutions:\n\n"

        content += "### Plugin Not Loading\n\n"
        content += "- Check that all dependencies are installed\n"
        content += "- Verify the plugin manifest is valid\n"
        content += "- Check the ICARUS CLI logs for error messages\n\n"

        content += "### Plugin Not Activating\n\n"
        content += "- Ensure the plugin is compatible with your ICARUS CLI version\n"
        content += "- Check for permission requirements\n"
        content += "- Verify the plugin class inherits from IcarusPlugin\n\n"

        content += "### Configuration Issues\n\n"
        content += "- Reset plugin configuration to defaults\n"
        content += "- Check configuration schema validation\n"
        content += "- Verify JSON syntax in configuration files\n\n"

        return DocSection("Troubleshooting", content)

    def _generate_architecture_section(self, plugin_info: Dict[str, Any]) -> DocSection:
        """Generate architecture section."""
        content = "This section describes the plugin's architecture and design.\n\n"

        # Plugin structure
        content += "### Plugin Structure\n\n"

        modules = plugin_info.get("modules", [])
        if modules:
            content += "```\n"
            for module in modules:
                content += f"{module['relative_path']}\n"
            content += "```\n\n"

        # Main components
        plugin_class = plugin_info.get("plugin_class")
        if plugin_class:
            content += f"### Main Plugin Class: {plugin_class['name']}\n\n"

            if plugin_class.get("docstring"):
                content += f"{plugin_class['docstring']}\n\n"

            # Methods
            if plugin_class.get("methods"):
                content += "#### Methods\n\n"
                for method in plugin_class["methods"]:
                    if method["type"] == "public":
                        content += f"- `{method['name']}()`"
                        if method.get("docstring"):
                            first_line = method["docstring"].split("\n")[0].strip()
                            content += f": {first_line}"
                        content += "\n"
                content += "\n"

        return DocSection("Architecture", content)

    def _generate_development_setup_section(
        self,
        plugin_info: Dict[str, Any],
    ) -> DocSection:
        """Generate development setup section."""
        content = "Instructions for setting up a development environment.\n\n"

        content += "### Prerequisites\n\n"
        content += "- Python development environment\n"
        content += "- ICARUS CLI development version\n"
        content += "- Git (for version control)\n\n"

        content += "### Setup Steps\n\n"
        content += "1. Clone the plugin repository\n"
        content += "2. Create a virtual environment\n"
        content += "3. Install development dependencies\n"
        content += "4. Run tests to verify setup\n\n"

        content += "### Development Workflow\n\n"
        content += "1. Make changes to plugin code\n"
        content += "2. Run tests: `python -m pytest tests/`\n"
        content += "3. Test in ICARUS CLI development environment\n"
        content += "4. Update documentation as needed\n\n"

        return DocSection("Development Setup", content)

    def _generate_api_section(self, plugin_info: Dict[str, Any]) -> DocSection:
        """Generate API documentation section."""
        content = "API documentation for the plugin.\n\n"

        plugin_class = plugin_info.get("plugin_class")
        if plugin_class:
            content += f"## {plugin_class['name']} Class\n\n"

            if plugin_class.get("docstring"):
                content += f"{plugin_class['docstring']}\n\n"

            # Methods documentation
            if plugin_class.get("methods"):
                content += "### Methods\n\n"

                for method in plugin_class["methods"]:
                    content += f"#### `{method['name']}(`"

                    # Arguments
                    if method.get("args"):
                        args_str = ", ".join(
                            [
                                f"{arg['name']}: {arg.get('type', 'Any')}"
                                for arg in method["args"]
                                if arg["name"] != "self"
                            ],
                        )
                        content += args_str

                    content += ")`\n\n"

                    if method.get("docstring"):
                        content += f"{method['docstring']}\n\n"

                    # Arguments details
                    if method.get("args") and len(method["args"]) > 1:
                        content += "**Arguments:**\n\n"
                        for arg in method["args"]:
                            if arg["name"] != "self":
                                content += f"- `{arg['name']}`"
                                if arg.get("type"):
                                    content += f" ({arg['type']})"
                                content += "\n"
                        content += "\n"

                    # Return value
                    if method.get("returns"):
                        content += f"**Returns:** {method['returns']}\n\n"

        return DocSection("API Documentation", content)

    def _generate_examples_section(self, plugin_info: Dict[str, Any]) -> DocSection:
        """Generate examples section."""
        content = "Code examples and usage patterns.\n\n"

        examples = plugin_info.get("examples", [])

        if examples:
            for example in examples:
                content += f"### {example['name'].replace('_', ' ').title()}\n\n"
                content += "```python\n"
                content += example["content"]
                content += "\n```\n\n"
        else:
            content += "No examples available yet. Check back later or contribute examples!\n\n"

        return DocSection("Examples", content)

    def _generate_document(
        self,
        doc_type: str,
        sections: List[DocSection],
        output_dir: Path,
        format_type: str,
    ) -> Path:
        """Generate a complete document."""
        if format_type == "markdown":
            return self._generate_markdown_document(doc_type, sections, output_dir)
        elif format_type == "html":
            return self._generate_html_document(doc_type, sections, output_dir)
        elif format_type == "rst":
            return self._generate_rst_document(doc_type, sections, output_dir)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _generate_markdown_document(
        self,
        doc_type: str,
        sections: List[DocSection],
        output_dir: Path,
    ) -> Path:
        """Generate markdown document."""
        output_file = output_dir / f"{doc_type}.md"

        content = ""
        for section in sections:
            content += section.to_markdown()

        with open(output_file, "w") as f:
            f.write(content)

        return output_file

    def _generate_html_document(
        self,
        doc_type: str,
        sections: List[DocSection],
        output_dir: Path,
    ) -> Path:
        """Generate HTML document."""
        output_file = output_dir / f"{doc_type}.html"

        # Convert markdown to HTML (simplified)
        markdown_content = ""
        for section in sections:
            markdown_content += section.to_markdown()

        # Basic HTML wrapper
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{doc_type.title()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
{self._markdown_to_html(markdown_content)}
</body>
</html>"""

        with open(output_file, "w") as f:
            f.write(html_content)

        return output_file

    def _generate_rst_document(
        self,
        doc_type: str,
        sections: List[DocSection],
        output_dir: Path,
    ) -> Path:
        """Generate reStructuredText document."""
        output_file = output_dir / f"{doc_type}.rst"

        # Convert to RST format (simplified)
        rst_content = ""
        for section in sections:
            rst_content += self._section_to_rst(section)

        with open(output_file, "w") as f:
            f.write(rst_content)

        return output_file

    def _markdown_to_html(self, markdown: str) -> str:
        """Simple markdown to HTML conversion."""
        html = markdown

        # Headers
        html = re.sub(r"^### (.*)", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.*)", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.*)", r"<h1>\1</h1>", html, flags=re.MULTILINE)

        # Code blocks
        html = re.sub(
            r"```(\w+)?\n(.*?)\n```",
            r"<pre><code>\2</code></pre>",
            html,
            flags=re.DOTALL,
        )

        # Inline code
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

        # Bold
        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)

        # Lists
        html = re.sub(r"^- (.*)", r"<li>\1</li>", html, flags=re.MULTILINE)

        # Paragraphs
        html = re.sub(r"\n\n", r"</p><p>", html)
        html = f"<p>{html}</p>"

        return html

    def _section_to_rst(self, section: DocSection) -> str:
        """Convert section to reStructuredText."""
        rst = f"{section.title}\n"
        rst += "=" * len(section.title) + "\n\n"
        rst += section.content + "\n"

        for subsection in section.subsections:
            rst += self._section_to_rst(subsection)

        return rst

    def _get_readme_template(self) -> str:
        """Get README template."""
        return """# {plugin_name}

{description}

## Installation

{installation_instructions}

## Usage

{usage_instructions}

## Configuration

{configuration_details}

## License

{license_info}
"""

    def _get_api_template(self) -> str:
        """Get API documentation template."""
        return """# API Documentation

{api_overview}

## Classes

{class_documentation}

## Functions

{function_documentation}
"""

    def _get_user_guide_template(self) -> str:
        """Get user guide template."""
        return """# User Guide

{getting_started}

## Features

{features_overview}

## Configuration

{configuration_guide}

## Troubleshooting

{troubleshooting_guide}
"""

    def _get_developer_guide_template(self) -> str:
        """Get developer guide template."""
        return """# Developer Guide

{architecture_overview}

## Development Setup

{setup_instructions}

## API Reference

{api_reference}

## Contributing

{contribution_guidelines}
"""
