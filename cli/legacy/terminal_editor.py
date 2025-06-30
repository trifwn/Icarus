#!/usr/bin/env python3
"""Terminal Python Editor Integration for ICARUS CLI

This module provides integration with terminal-based Python editors and REPLs
for enhanced code editing capabilities within the ICARUS TUI.
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any, List


class TerminalEditor:
    """Base class for terminal editor integration."""

    def __init__(self):
        self.editor_name = "default"
        self.supported_editors = ["nano", "vim", "emacs", "micro"]

    def get_available_editors(self) -> List[str]:
        """Get list of available editors on the system."""
        available = []
        for editor in self.supported_editors:
            if self._check_editor_available(editor):
                available.append(editor)
        return available

    def _check_editor_available(self, editor: str) -> bool:
        """Check if an editor is available on the system."""
        try:
            result = subprocess.run([editor, "--version"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def edit_file(self, content: str = "", editor: Optional[str] = None) -> Optional[str]:
        """Edit a file with the specified editor and return the content."""
        if editor is None:
            editor = self._get_default_editor()

        if not self._check_editor_available(editor):
            raise ValueError(f"Editor '{editor}' not available")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Open editor
            subprocess.run([editor, temp_file], check=True)

            # Read back the content
            with open(temp_file, "r") as f:
                return f.read()
        finally:
            # Clean up
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    def _get_default_editor(self) -> str:
        """Get the default editor based on environment variables."""
        # Check environment variables
        for var in ["EDITOR", "VISUAL"]:
            if var in os.environ:
                editor = os.environ[var].split()[0]  # Take first word
                if self._check_editor_available(editor):
                    return editor

        # Check available editors
        available = self.get_available_editors()
        if available:
            return available[0]

        raise RuntimeError("No suitable editor found")


class PythonREPL:
    """Python REPL integration for enhanced code execution."""

    def __init__(self):
        self.history_file = Path.home() / ".icarus" / "repl_history"
        self.history_file.parent.mkdir(exist_ok=True)

    def launch_interactive_repl(self, namespace: Dict[str, Any]) -> None:
        """Launch an interactive Python REPL with the given namespace."""
        try:
            # Create a temporary script to set up the REPL
            script_content = f"""
import sys
import readline
import rlcompleter

# Set up readline for history
readline.parse_and_bind("tab: complete")

# Add namespace to globals
for key, value in {namespace}.items():
    globals()[key] = value

print("ICARUS Python REPL")
print("Available objects:", list({namespace}.keys()))
print("Use Ctrl+D to exit")

# Start interactive session
"""

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script_content)
                temp_script = f.name

            try:
                subprocess.run([sys.executable, "-i", temp_script], check=True)
            finally:
                os.unlink(temp_script)

        except Exception as e:
            print(f"Error launching REPL: {e}")


class CodeSnippetManager:
    """Manager for code snippets and templates."""

    def __init__(self):
        self.snippets_dir = Path.home() / ".icarus" / "snippets"
        self.snippets_dir.mkdir(parents=True, exist_ok=True)

    def get_snippets(self) -> Dict[str, str]:
        """Get available code snippets."""
        snippets = {}
        for file in self.snippets_dir.glob("*.py"):
            with open(file, "r") as f:
                snippets[file.stem] = f.read()
        return snippets

    def save_snippet(self, name: str, code: str) -> None:
        """Save a code snippet."""
        snippet_file = self.snippets_dir / f"{name}.py"
        with open(snippet_file, "w") as f:
            f.write(code)

    def load_snippet(self, name: str) -> Optional[str]:
        """Load a code snippet."""
        snippet_file = self.snippets_dir / f"{name}.py"
        if snippet_file.exists():
            with open(snippet_file, "r") as f:
                return f.read()
        return None

    def delete_snippet(self, name: str) -> bool:
        """Delete a code snippet."""
        snippet_file = self.snippets_dir / f"{name}.py"
        if snippet_file.exists():
            snippet_file.unlink()
            return True
        return False


# Predefined code templates
CODE_TEMPLATES = {
    "airfoil_basic": """# Basic airfoil analysis
from ICARUS.airfoils import Airfoil

# Create NACA airfoil
airfoil = Airfoil.naca("2412")
namespace.add_object("airfoil", airfoil, "airfoil")

print(f"Created airfoil: {airfoil}")
""",
    "airfoil_analysis": """# Airfoil analysis with XFoil
from ICARUS.airfoils import Airfoil
from ICARUS.computation.solvers.Xfoil import xfoil

# Create airfoil
airfoil = Airfoil.naca("2412")
namespace.add_object("airfoil", airfoil, "airfoil")

# Run analysis
angles = list(range(0, 16, 2))
reynolds = 1e6

results = xfoil(airfoil, angles, reynolds)
namespace.add_object("results", results, "analysis_results")

print(f"Analysis completed for {len(angles)} angles")
""",
    "airplane_basic": """# Basic airplane creation
from ICARUS.vehicle import Airplane

# Create simple airplane
airplane = Airplane()
# Add components here...
namespace.add_object("airplane", airplane, "airplane")

print(f"Created airplane: {airplane}")
""",
    "database_query": """# Query ICARUS database
from ICARUS.database import Database

# Initialize database
db = Database()
namespace.add_object("db", db, "database")

# List available airfoils
airfoils = db.list_airfoils()
print(f"Available airfoils: {len(airfoils)}")
for airfoil in airfoils[:5]:  # Show first 5
    print(f"  - {airfoil}")
""",
}


def get_available_editors() -> List[str]:
    """Get list of available terminal editors."""
    editor = TerminalEditor()
    return editor.get_available_editors()


def edit_code_in_terminal(content: str = "", editor: Optional[str] = None) -> Optional[str]:
    """Edit code in a terminal editor."""
    editor_instance = TerminalEditor()
    return editor_instance.edit_file(content, editor)


def launch_python_repl(namespace: Dict[str, Any]) -> None:
    """Launch an interactive Python REPL."""
    repl = PythonREPL()
    repl.launch_interactive_repl(namespace)


def get_code_templates() -> Dict[str, str]:
    """Get available code templates."""
    return CODE_TEMPLATES.copy()


def get_snippet_manager() -> CodeSnippetManager:
    """Get the code snippet manager."""
    return CodeSnippetManager()
