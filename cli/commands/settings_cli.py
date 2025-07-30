"""Settings Command Line Interface

This module provides command-line tools for managing ICARUS CLI settings,
including import/export, backup/restore, and configuration management.
"""

import argparse
import json
import logging
import sys
from typing import Any
from typing import Dict
from typing import List

from ..core.settings import SettingsFormat
from ..core.settings import SettingsManager
from ..core.settings import SettingsScope


class SettingsCLI:
    """Command-line interface for settings management."""

    def __init__(self):
        self.settings_manager = SettingsManager()
        self.logger = logging.getLogger(__name__)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="ICARUS CLI Settings Management",
            prog="icarus-settings",
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Show command
        show_parser = subparsers.add_parser("show", help="Show current settings")
        show_parser.add_argument(
            "--scope",
            choices=["global", "workspace", "project", "all"],
            default="all",
            help="Settings scope to show",
        )
        show_parser.add_argument(
            "--format",
            choices=["json", "yaml", "table"],
            default="table",
            help="Output format",
        )

        # Set command
        set_parser = subparsers.add_parser("set", help="Set a setting value")
        set_parser.add_argument("key", help="Setting key (e.g., theme.theme_name)")
        set_parser.add_argument("value", help="Setting value")
        set_parser.add_argument(
            "--scope",
            choices=["global", "workspace", "project"],
            default="global",
            help="Settings scope",
        )

        # Get command
        get_parser = subparsers.add_parser("get", help="Get a setting value")
        get_parser.add_argument("key", help="Setting key")
        get_parser.add_argument(
            "--scope",
            choices=["global", "workspace", "project"],
            help="Settings scope",
        )

        # Export command
        export_parser = subparsers.add_parser("export", help="Export settings")
        export_parser.add_argument("filepath", help="Output file path")
        export_parser.add_argument(
            "--scope",
            choices=["global", "workspace", "project", "session"],
            default="global",
            help="Settings scope to export",
        )
        export_parser.add_argument(
            "--format",
            choices=["json", "yaml"],
            default="json",
            help="Export format",
        )

        # Import command
        import_parser = subparsers.add_parser("import", help="Import settings")
        import_parser.add_argument("filepath", help="Input file path")
        import_parser.add_argument(
            "--merge",
            action="store_true",
            help="Merge with existing settings instead of replacing",
        )

        # Reset command
        reset_parser = subparsers.add_parser("reset", help="Reset settings to defaults")
        reset_parser.add_argument(
            "--scope",
            choices=["global", "workspace", "project", "session"],
            default="global",
            help="Settings scope to reset",
        )
        reset_parser.add_argument(
            "--confirm",
            action="store_true",
            help="Skip confirmation prompt",
        )

        # Validate command
        validate_parser = subparsers.add_parser("validate", help="Validate settings")
        validate_parser.add_argument(
            "--fix",
            action="store_true",
            help="Attempt to fix validation issues",
        )

        # Backup commands
        backup_parser = subparsers.add_parser("backup", help="Backup management")
        backup_subparsers = backup_parser.add_subparsers(dest="backup_command")

        # Create backup
        create_backup_parser = backup_subparsers.add_parser(
            "create",
            help="Create backup",
        )
        create_backup_parser.add_argument(
            "--name",
            help="Backup name (auto-generated if not provided)",
        )

        # List backups
        backup_subparsers.add_parser("list", help="List available backups")

        # Restore backup
        restore_backup_parser = backup_subparsers.add_parser(
            "restore",
            help="Restore backup",
        )
        restore_backup_parser.add_argument("name", help="Backup name to restore")
        restore_backup_parser.add_argument(
            "--confirm",
            action="store_true",
            help="Skip confirmation prompt",
        )

        # Delete backup
        delete_backup_parser = backup_subparsers.add_parser(
            "delete",
            help="Delete backup",
        )
        delete_backup_parser.add_argument("name", help="Backup name to delete")
        delete_backup_parser.add_argument(
            "--confirm",
            action="store_true",
            help="Skip confirmation prompt",
        )

        # Cleanup backups
        cleanup_backup_parser = backup_subparsers.add_parser(
            "cleanup",
            help="Cleanup old backups",
        )
        cleanup_backup_parser.add_argument(
            "--keep",
            type=int,
            default=10,
            help="Number of backups to keep",
        )

        # Workspace commands
        workspace_parser = subparsers.add_parser(
            "workspace",
            help="Workspace management",
        )
        workspace_subparsers = workspace_parser.add_subparsers(dest="workspace_command")

        # List workspaces
        workspace_subparsers.add_parser("list", help="List workspaces")

        # Create workspace
        create_workspace_parser = workspace_subparsers.add_parser(
            "create",
            help="Create workspace",
        )
        create_workspace_parser.add_argument("name", help="Workspace name")
        create_workspace_parser.add_argument(
            "--description",
            help="Workspace description",
        )

        # Switch workspace
        switch_workspace_parser = workspace_subparsers.add_parser(
            "switch",
            help="Switch workspace",
        )
        switch_workspace_parser.add_argument("name", help="Workspace name")

        # Delete workspace
        delete_workspace_parser = workspace_subparsers.add_parser(
            "delete",
            help="Delete workspace",
        )
        delete_workspace_parser.add_argument("name", help="Workspace name")
        delete_workspace_parser.add_argument(
            "--confirm",
            action="store_true",
            help="Skip confirmation prompt",
        )

        # Theme commands
        theme_parser = subparsers.add_parser("theme", help="Theme management")
        theme_subparsers = theme_parser.add_subparsers(dest="theme_command")

        # List themes
        theme_subparsers.add_parser("list", help="List available themes")

        # Apply theme
        apply_theme_parser = theme_subparsers.add_parser(
            "apply",
            help="Apply theme preset",
        )
        apply_theme_parser.add_argument("preset", help="Theme preset name")

        return parser

    async def run(self, args: List[str] = None) -> int:
        """Run the settings CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            return 1

        try:
            # Load settings
            self.settings_manager.load_all_settings()

            # Execute command
            if parsed_args.command == "show":
                return await self._cmd_show(parsed_args)
            elif parsed_args.command == "set":
                return await self._cmd_set(parsed_args)
            elif parsed_args.command == "get":
                return await self._cmd_get(parsed_args)
            elif parsed_args.command == "export":
                return await self._cmd_export(parsed_args)
            elif parsed_args.command == "import":
                return await self._cmd_import(parsed_args)
            elif parsed_args.command == "reset":
                return await self._cmd_reset(parsed_args)
            elif parsed_args.command == "validate":
                return await self._cmd_validate(parsed_args)
            elif parsed_args.command == "backup":
                return await self._cmd_backup(parsed_args)
            elif parsed_args.command == "workspace":
                return await self._cmd_workspace(parsed_args)
            elif parsed_args.command == "theme":
                return await self._cmd_theme(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}")
                return 1

        except Exception as e:
            print(f"Error: {e}")
            self.logger.error(f"CLI error: {e}")
            return 1

    async def _cmd_show(self, args) -> int:
        """Show current settings."""
        settings = self.settings_manager.get_all_settings()

        if args.scope != "all":
            if args.scope in settings:
                settings = {args.scope: settings[args.scope]}
            else:
                print(f"No settings found for scope: {args.scope}")
                return 1

        if args.format == "json":
            print(json.dumps(settings, indent=2))
        elif args.format == "yaml":
            import yaml

            print(yaml.dump(settings, default_flow_style=False))
        else:
            self._print_settings_table(settings)

        return 0

    async def _cmd_set(self, args) -> int:
        """Set a setting value."""
        # Parse key (e.g., "theme.theme_name" -> scope="theme", key="theme_name")
        if "." in args.key:
            scope_name, key = args.key.split(".", 1)
        else:
            scope_name = args.scope
            key = args.key

        # Convert string value to appropriate type
        value = self._parse_value(args.value)

        # Set the value
        success = self.settings_manager.set_setting(key, value)

        if success:
            self.settings_manager.save_all_settings()
            print(f"Set {args.key} = {value}")
            return 0
        else:
            print(f"Failed to set {args.key}")
            return 1

    async def _cmd_get(self, args) -> int:
        """Get a setting value."""
        scope = SettingsScope(args.scope) if args.scope else None
        value = self.settings_manager.get_setting(args.key, scope)

        if value is not None:
            print(value)
            return 0
        else:
            print(f"Setting not found: {args.key}")
            return 1

    async def _cmd_export(self, args) -> int:
        """Export settings."""
        scope = SettingsScope(args.scope)
        format_type = SettingsFormat(args.format)

        success = self.settings_manager.export_settings(
            args.filepath,
            scope,
            format_type,
        )

        if success:
            print(f"Settings exported to {args.filepath}")
            return 0
        else:
            print(f"Failed to export settings to {args.filepath}")
            return 1

    async def _cmd_import(self, args) -> int:
        """Import settings."""
        success = self.settings_manager.import_settings(args.filepath, args.merge)

        if success:
            self.settings_manager.save_all_settings()
            print(f"Settings imported from {args.filepath}")
            return 0
        else:
            print(f"Failed to import settings from {args.filepath}")
            return 1

    async def _cmd_reset(self, args) -> int:
        """Reset settings to defaults."""
        if not args.confirm:
            response = input(f"Reset {args.scope} settings to defaults? (y/N): ")
            if response.lower() != "y":
                print("Reset cancelled")
                return 0

        scope = SettingsScope(args.scope)
        self.settings_manager.reset_to_defaults(scope)
        self.settings_manager.save_all_settings()

        print(f"Reset {args.scope} settings to defaults")
        return 0

    async def _cmd_validate(self, args) -> int:
        """Validate settings."""
        issues = self.settings_manager.validate_settings()

        if not issues:
            print("All settings are valid")
            return 0

        print("Settings validation issues found:")
        for category, category_issues in issues.items():
            print(f"\n{category.title()}:")
            for issue in category_issues:
                print(f"  - {issue}")

        if args.fix:
            print("\nAttempting to fix issues...")
            # This would implement automatic fixes
            print("Automatic fixes not yet implemented")

        return 1 if issues else 0

    async def _cmd_backup(self, args) -> int:
        """Handle backup commands."""
        if args.backup_command == "create":
            backup_name = self.settings_manager.create_backup(args.name)
            print(f"Backup created: {backup_name}")
            return 0

        elif args.backup_command == "list":
            backups = self.settings_manager.list_backups()
            if not backups:
                print("No backups found")
                return 0

            print("Available backups:")
            for backup in backups:
                print(f"  {backup['name']} - {backup['created_at']}")
            return 0

        elif args.backup_command == "restore":
            if not args.confirm:
                response = input(f"Restore backup '{args.name}'? (y/N): ")
                if response.lower() != "y":
                    print("Restore cancelled")
                    return 0

            success = self.settings_manager.restore_backup(args.name)
            if success:
                print(f"Backup restored: {args.name}")
                return 0
            else:
                print(f"Failed to restore backup: {args.name}")
                return 1

        elif args.backup_command == "delete":
            if not args.confirm:
                response = input(f"Delete backup '{args.name}'? (y/N): ")
                if response.lower() != "y":
                    print("Delete cancelled")
                    return 0

            success = self.settings_manager.delete_backup(args.name)
            if success:
                print(f"Backup deleted: {args.name}")
                return 0
            else:
                print(f"Failed to delete backup: {args.name}")
                return 1

        elif args.backup_command == "cleanup":
            deleted_count = self.settings_manager.cleanup_old_backups(args.keep)
            print(f"Cleaned up {deleted_count} old backups")
            return 0

        else:
            print(f"Unknown backup command: {args.backup_command}")
            return 1

    async def _cmd_workspace(self, args) -> int:
        """Handle workspace commands."""
        if args.workspace_command == "list":
            workspaces = self.settings_manager.list_workspaces()
            current = self.settings_manager.current_workspace

            if not workspaces:
                print("No workspaces found")
                return 0

            print("Available workspaces:")
            for workspace in workspaces:
                marker = " (current)" if workspace == current else ""
                print(f"  {workspace}{marker}")
            return 0

        elif args.workspace_command == "create":
            kwargs = {}
            if args.description:
                kwargs["description"] = args.description

            success = self.settings_manager.create_workspace(args.name, **kwargs)
            if success:
                print(f"Workspace created: {args.name}")
                return 0
            else:
                print(f"Failed to create workspace: {args.name}")
                return 1

        elif args.workspace_command == "switch":
            success = self.settings_manager.switch_workspace(args.name)
            if success:
                print(f"Switched to workspace: {args.name}")
                return 0
            else:
                print(f"Failed to switch to workspace: {args.name}")
                return 1

        elif args.workspace_command == "delete":
            if not args.confirm:
                response = input(f"Delete workspace '{args.name}'? (y/N): ")
                if response.lower() != "y":
                    print("Delete cancelled")
                    return 0

            success = self.settings_manager.delete_workspace(args.name)
            if success:
                print(f"Workspace deleted: {args.name}")
                return 0
            else:
                print(f"Failed to delete workspace: {args.name}")
                return 1

        else:
            print(f"Unknown workspace command: {args.workspace_command}")
            return 1

    async def _cmd_theme(self, args) -> int:
        """Handle theme commands."""
        if args.theme_command == "list":
            presets = ["aerospace_dark", "aerospace_light", "scientific", "classic"]
            print("Available theme presets:")
            for preset in presets:
                print(f"  {preset}")
            return 0

        elif args.theme_command == "apply":
            success = self.settings_manager.apply_theme_preset(args.preset)
            if success:
                self.settings_manager.save_all_settings()
                print(f"Applied theme preset: {args.preset}")
                return 0
            else:
                print(f"Failed to apply theme preset: {args.preset}")
                return 1

        else:
            print(f"Unknown theme command: {args.theme_command}")
            return 1

    def _print_settings_table(self, settings: Dict[str, Any]) -> None:
        """Print settings in table format."""

        def print_section(name: str, data: Dict[str, Any], indent: int = 0):
            prefix = "  " * indent
            print(f"{prefix}{name.upper()}:")

            for key, value in data.items():
                if isinstance(value, dict):
                    print_section(key, value, indent + 1)
                else:
                    print(f"{prefix}  {key}: {value}")
            print()

        for section_name, section_data in settings.items():
            if isinstance(section_data, dict):
                print_section(section_name, section_data)

    def _parse_value(self, value_str: str) -> Any:
        """Parse string value to appropriate type."""
        # Try to parse as JSON first
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass

        # Try boolean
        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"

        # Try integer
        try:
            return int(value_str)
        except ValueError:
            pass

        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Return as string
        return value_str


def main():
    """Main entry point for settings CLI."""
    import asyncio

    cli = SettingsCLI()
    exit_code = asyncio.run(cli.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
