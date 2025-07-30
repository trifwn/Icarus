"""
Command-line interface for the ICARUS CLI Plugin Development SDK.

This module provides a CLI tool for plugin developers to generate,
validate, test, package, and publish plugins.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .docs import PluginDocGenerator
from .generator import PluginGenerator
from .marketplace import PluginMarketplace
from .packager import PluginPackager
from .tester import PluginTester
from .validator import PluginValidator


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    return logging.getLogger(__name__)


def cmd_generate(args) -> int:
    """Generate a new plugin from template."""
    logger = setup_logging(args.verbose)
    generator = PluginGenerator(logger)

    # List templates if requested
    if args.list_templates:
        templates = generator.list_templates()
        print("Available templates:")
        for template_name in templates:
            template_info = generator.get_template_info(template_name)
            print(f"  {template_name}: {template_info.description}")
        return 0

    # Validate required arguments
    if not args.name or not args.template:
        print("Error: Plugin name and template are required")
        return 1

    # Generate plugin
    success = generator.generate_plugin(
        plugin_name=args.name,
        template_name=args.template,
        output_dir=args.output or ".",
        author_name=args.author,
        author_email=args.email,
        description=args.description,
    )

    if success:
        print(f"Plugin '{args.name}' generated successfully!")
        return 0
    else:
        print("Plugin generation failed")
        return 1


def cmd_validate(args) -> int:
    """Validate a plugin."""
    logger = setup_logging(args.verbose)
    validator = PluginValidator(logger)

    result = validator.validate_plugin(args.plugin_path)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        report = validator.get_validation_report(args.plugin_path)
        print(report)

    return 0 if result.is_valid else 1


def cmd_test(args) -> int:
    """Test a plugin."""
    logger = setup_logging(args.verbose)
    tester = PluginTester(logger)

    test_config = {}
    if args.performance:
        test_config["performance_tests"] = True
    if args.security:
        test_config["security_tests"] = True

    if args.generate_tests:
        # Generate test suite
        test_file = tester.create_test_file(args.plugin_path, args.output)
        print(f"Test file generated: {test_file}")
        return 0

    # Run tests
    results = tester.run_plugin_tests(args.plugin_path, test_config)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Test Results for: {results['plugin_path']}")
        print(f"Overall Status: {results['overall_status'].upper()}")
        print()

        for test_name, test_result in results.get("tests", {}).items():
            status = "PASS" if test_result.get("passed", False) else "FAIL"
            print(f"{test_name}: {status}")

            if args.verbose and test_result.get("details"):
                for detail in test_result["details"]:
                    print(f"  - {detail}")
                print()

    return 0 if results["overall_status"] == "passed" else 1


def cmd_package(args) -> int:
    """Package a plugin for distribution."""
    logger = setup_logging(args.verbose)
    packager = PluginPackager(logger)

    if args.extract:
        # Extract package
        success = packager.extract_package(args.plugin_path, args.output or ".")
        if success:
            print("Package extracted successfully!")
            return 0
        else:
            print("Package extraction failed")
            return 1

    if args.verify:
        # Verify package
        result = packager.verify_package(args.plugin_path)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Package Verification: {'VALID' if result['valid'] else 'INVALID'}")

            if result["errors"]:
                print("\nErrors:")
                for error in result["errors"]:
                    print(f"  - {error}")

            if result["warnings"]:
                print("\nWarnings:")
                for warning in result["warnings"]:
                    print(f"  - {warning}")

        return 0 if result["valid"] else 1

    if args.info:
        # Show package info
        info = packager.get_package_info(args.plugin_path)
        print(json.dumps(info, indent=2))
        return 0

    # Package plugin
    output_path = args.output or f"{Path(args.plugin_path).name}.zip"

    success = packager.package_plugin(
        plugin_path=args.plugin_path,
        output_path=output_path,
        format=args.format,
        include_tests=args.include_tests,
        include_docs=args.include_docs,
        validate=not args.no_validate,
    )

    if success:
        print(f"Plugin packaged successfully: {output_path}")
        return 0
    else:
        print("Plugin packaging failed")
        return 1


def cmd_marketplace(args) -> int:
    """Interact with plugin marketplaces."""
    logger = setup_logging(args.verbose)
    marketplace = PluginMarketplace(logger)

    if args.action == "list":
        marketplaces = marketplace.list_marketplaces()

        if args.json:
            print(json.dumps(marketplaces, indent=2))
        else:
            print("Configured Marketplaces:")
            for mp in marketplaces:
                print(f"  {mp['name']}: {mp['display_name']}")
                print(f"    URL: {mp['base_url']}")
                print(f"    Authenticated: {'Yes' if mp['has_api_key'] else 'No'}")
                print()

        return 0

    elif args.action == "search":
        if not args.query:
            print("Error: Search query is required")
            return 1

        results = marketplace.search_plugins(
            query=args.query,
            marketplace=args.marketplace,
            limit=args.limit or 20,
        )

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"Search Results for '{args.query}':")
            print()

            for result in results:
                print(f"Name: {result.get('name', 'Unknown')}")
                print(f"Description: {result.get('description', 'No description')}")
                print(f"Version: {result.get('version', 'Unknown')}")
                print(f"Author: {result.get('author', {}).get('name', 'Unknown')}")
                print(f"Downloads: {result.get('downloads', 0)}")
                print(f"Marketplace: {result.get('marketplace', 'Unknown')}")
                print("-" * 40)

        return 0

    elif args.action == "download":
        if not args.plugin_id or not args.marketplace:
            print("Error: Plugin ID and marketplace are required")
            return 1

        output_path = args.output or f"{args.plugin_id}.zip"

        success = marketplace.download_plugin(
            plugin_id=args.plugin_id,
            marketplace=args.marketplace,
            output_path=output_path,
            version=args.version,
        )

        if success:
            print(f"Plugin downloaded: {output_path}")
            return 0
        else:
            print("Plugin download failed")
            return 1

    elif args.action == "publish":
        if not args.plugin_path or not args.marketplace:
            print("Error: Plugin path and marketplace are required")
            return 1

        success = marketplace.publish_plugin(
            plugin_package_path=args.plugin_path,
            marketplace=args.marketplace,
        )

        if success:
            print("Plugin published successfully!")
            return 0
        else:
            print("Plugin publishing failed")
            return 1

    elif args.action == "test":
        if not args.marketplace:
            print("Error: Marketplace name is required")
            return 1

        result = marketplace.test_marketplace_connection(args.marketplace)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Marketplace: {result['marketplace']}")
            print(f"Connected: {'Yes' if result['connected'] else 'No'}")
            print(f"Authenticated: {'Yes' if result['authenticated'] else 'No'}")

            if result.get("response_time"):
                print(f"Response Time: {result['response_time']:.3f}s")

            if result.get("error"):
                print(f"Error: {result['error']}")

        return 0 if result["connected"] else 1

    else:
        print(f"Unknown marketplace action: {args.action}")
        return 1


def cmd_docs(args) -> int:
    """Generate plugin documentation."""
    logger = setup_logging(args.verbose)
    doc_generator = PluginDocGenerator(logger)

    formats = args.formats.split(",") if args.formats else ["markdown"]

    generated_docs = doc_generator.generate_documentation(
        plugin_path=args.plugin_path,
        output_dir=args.output,
        formats=formats,
        include_api=args.include_api,
        include_examples=args.include_examples,
    )

    if generated_docs:
        print("Documentation generated:")
        for doc_type, doc_path in generated_docs.items():
            print(f"  {doc_type}: {doc_path}")
        return 0
    else:
        print("Documentation generation failed")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ICARUS CLI Plugin Development SDK",
        prog="icarus-plugin-sdk",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a new plugin")
    gen_parser.add_argument("--name", "-n", help="Plugin name")
    gen_parser.add_argument("--template", "-t", help="Template to use")
    gen_parser.add_argument("--author", "-a", required=True, help="Author name")
    gen_parser.add_argument("--email", "-e", help="Author email")
    gen_parser.add_argument("--description", "-d", help="Plugin description")
    gen_parser.add_argument("--output", "-o", help="Output directory")
    gen_parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates",
    )
    gen_parser.set_defaults(func=cmd_generate)

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a plugin")
    val_parser.add_argument("plugin_path", help="Path to plugin directory or file")
    val_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    val_parser.set_defaults(func=cmd_validate)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test a plugin")
    test_parser.add_argument("plugin_path", help="Path to plugin directory or file")
    test_parser.add_argument(
        "--generate-tests",
        action="store_true",
        help="Generate test suite",
    )
    test_parser.add_argument(
        "--performance",
        action="store_true",
        help="Include performance tests",
    )
    test_parser.add_argument(
        "--security",
        action="store_true",
        help="Include security tests",
    )
    test_parser.add_argument("--output", "-o", help="Output path for generated tests")
    test_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    test_parser.set_defaults(func=cmd_test)

    # Package command
    pkg_parser = subparsers.add_parser("package", help="Package a plugin")
    pkg_parser.add_argument(
        "plugin_path",
        help="Path to plugin directory or package file",
    )
    pkg_parser.add_argument("--output", "-o", help="Output path")
    pkg_parser.add_argument(
        "--format",
        choices=["zip", "tar.gz", "tar.bz2"],
        default="zip",
        help="Package format",
    )
    pkg_parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files",
    )
    pkg_parser.add_argument(
        "--include-docs",
        action="store_true",
        default=True,
        help="Include documentation",
    )
    pkg_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation",
    )
    pkg_parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract package instead of creating",
    )
    pkg_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify package integrity",
    )
    pkg_parser.add_argument(
        "--info",
        action="store_true",
        help="Show package information",
    )
    pkg_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    pkg_parser.set_defaults(func=cmd_package)

    # Marketplace command
    mp_parser = subparsers.add_parser(
        "marketplace",
        help="Interact with plugin marketplaces",
    )
    mp_parser.add_argument(
        "action",
        choices=["list", "search", "download", "publish", "test"],
        help="Marketplace action",
    )
    mp_parser.add_argument("--marketplace", "-m", help="Marketplace name")
    mp_parser.add_argument("--query", "-q", help="Search query")
    mp_parser.add_argument("--plugin-id", help="Plugin identifier")
    mp_parser.add_argument("--plugin-path", help="Path to plugin package")
    mp_parser.add_argument("--version", help="Plugin version")
    mp_parser.add_argument("--output", "-o", help="Output path")
    mp_parser.add_argument("--limit", type=int, help="Maximum number of results")
    mp_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    mp_parser.set_defaults(func=cmd_marketplace)

    # Documentation command
    docs_parser = subparsers.add_parser("docs", help="Generate plugin documentation")
    docs_parser.add_argument("plugin_path", help="Path to plugin directory or file")
    docs_parser.add_argument("--output", "-o", help="Output directory")
    docs_parser.add_argument(
        "--formats",
        help="Documentation formats (comma-separated): markdown,html,rst",
    )
    docs_parser.add_argument(
        "--include-api",
        action="store_true",
        default=True,
        help="Include API documentation",
    )
    docs_parser.add_argument(
        "--include-examples",
        action="store_true",
        default=True,
        help="Include examples",
    )
    docs_parser.set_defaults(func=cmd_docs)

    # Parse arguments
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
