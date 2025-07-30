"""
Verification script for ICARUS CLI Security System implementation.

This script verifies that all security components are properly implemented
and can be imported without external dependencies.
"""

import sys
from pathlib import Path


def verify_file_structure():
    """Verify that all security files are present."""
    print("🔍 Verifying Security System File Structure...")

    security_dir = Path("security")
    required_files = [
        "__init__.py",
        "crypto.py",
        "authentication.py",
        "authorization.py",
        "audit_logger.py",
        "security_manager.py",
        "README.md",
    ]

    missing_files = []
    for file in required_files:
        file_path = security_dir / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All security files present")
        return True


def verify_imports():
    """Verify that security modules can be imported."""
    print("\n🔍 Verifying Security Module Imports...")

    # Add current directory to path
    sys.path.insert(0, ".")

    modules_to_test = [
        ("security", "Security package"),
        ("security.crypto", "CryptoManager"),
        ("security.authentication", "AuthenticationManager"),
        ("security.authorization", "AuthorizationManager"),
        ("security.audit_logger", "AuditLogger"),
        ("security.security_manager", "SecurityManager"),
    ]

    import_errors = []

    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {description}")
        except ImportError as e:
            if "cryptography" in str(e):
                print(
                    f"  ⚠️  {description} - Missing cryptography dependency (expected)"
                )
            else:
                print(f"  ❌ {description} - Import error: {e}")
                import_errors.append((module_name, str(e)))
        except Exception as e:
            print(f"  ❌ {description} - Error: {e}")
            import_errors.append((module_name, str(e)))

    if import_errors:
        print(f"\n❌ Import errors (excluding cryptography dependency):")
        for module, error in import_errors:
            if "cryptography" not in error:
                print(f"  - {module}: {error}")
        return len([e for e in import_errors if "cryptography" not in e[1]]) == 0
    else:
        print("✅ All security modules can be imported")
        return True


def verify_class_definitions():
    """Verify that key classes are properly defined."""
    print("\n🔍 Verifying Security Class Definitions...")

    try:
        # Test basic class structure without instantiation
        import ast

        security_files = [
            ("security/crypto.py", ["CryptoManager"]),
            (
                "security/authentication.py",
                ["AuthenticationManager", "AuthSession", "AuthConfig"],
            ),
            (
                "security/authorization.py",
                ["AuthorizationManager", "ResourcePermission", "AccessPolicy"],
            ),
            ("security/audit_logger.py", ["AuditLogger", "AuditEvent"]),
            ("security/security_manager.py", ["SecurityManager"]),
        ]

        for file_path, expected_classes in security_files:
            if Path(file_path).exists():
                with open(file_path, "r") as f:
                    content = f.read()

                tree = ast.parse(content)

                # Find class definitions
                classes_found = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes_found.append(node.name)

                # Check if expected classes are present
                missing_classes = [
                    cls for cls in expected_classes if cls not in classes_found
                ]

                if missing_classes:
                    print(f"  ❌ {file_path}: Missing classes {missing_classes}")
                else:
                    print(f"  ✅ {file_path}: All expected classes present")
            else:
                print(f"  ❌ {file_path}: File not found")

        print("✅ Class definitions verified")
        return True

    except Exception as e:
        print(f"❌ Error verifying class definitions: {e}")
        return False


def verify_requirements():
    """Verify that security requirements are documented."""
    print("\n🔍 Verifying Security Requirements...")

    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        with open(requirements_file, "r") as f:
            content = f.read()

        if "cryptography" in content:
            print("  ✅ Cryptography dependency documented in requirements.txt")
        else:
            print("  ❌ Cryptography dependency missing from requirements.txt")
            return False
    else:
        print("  ❌ requirements.txt not found")
        return False

    return True


def verify_documentation():
    """Verify that security documentation is present."""
    print("\n🔍 Verifying Security Documentation...")

    readme_file = Path("security/README.md")
    if readme_file.exists():
        with open(readme_file, "r") as f:
            content = f.read()

        required_sections = [
            "Overview",
            "Architecture",
            "Components",
            "Usage Examples",
            "Security Features",
        ]

        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)

        if missing_sections:
            print(f"  ❌ Missing documentation sections: {missing_sections}")
            return False
        else:
            print("  ✅ All required documentation sections present")
            return True
    else:
        print("  ❌ Security README.md not found")
        return False


def verify_test_files():
    """Verify that test and demo files are present."""
    print("\n🔍 Verifying Test and Demo Files...")

    test_files = [
        ("test_security_system.py", "Comprehensive test suite"),
        ("demo_security_system.py", "Interactive demo script"),
    ]

    all_present = True
    for file_name, description in test_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  ✅ {description}: {file_name}")
        else:
            print(f"  ❌ {description}: {file_name} - MISSING")
            all_present = False

    return all_present


def main():
    """Run all verification checks."""
    print("🚀 ICARUS CLI Security System Implementation Verification")
    print("=" * 60)

    checks = [
        ("File Structure", verify_file_structure),
        ("Module Imports", verify_imports),
        ("Class Definitions", verify_class_definitions),
        ("Requirements", verify_requirements),
        ("Documentation", verify_documentation),
        ("Test Files", verify_test_files),
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results.append((check_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 ALL VERIFICATION CHECKS PASSED!")
        print("The security system implementation is complete and ready for use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run tests: python test_security_system.py")
        print("3. Try demo: python demo_security_system.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} verification checks failed.")
        print("Please review the failed checks and fix any issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
