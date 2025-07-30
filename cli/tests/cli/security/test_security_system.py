"""
Comprehensive test suite for the ICARUS CLI Security System.

Tests all security components including authentication, authorization,
audit logging, encryption, and plugin security validation.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from datetime import timedelta

# Import related components
from collaboration.user_manager import User
from collaboration.user_manager import UserManager
from collaboration.user_manager import UserRole
from security.audit_logger import AuditEventType
from security.audit_logger import AuditLogger
from security.audit_logger import AuditSeverity
from security.authentication import AuthenticationManager
from security.authorization import AccessLevel
from security.authorization import AuthorizationManager
from security.authorization import ResourceType

# Import security components
from security.crypto import CryptoManager
from security.security_manager import SecurityManager


class TestCryptoManager(unittest.TestCase):
    """Test cryptographic operations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.crypto_manager = CryptoManager()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_symmetric_encryption(self):
        """Test symmetric encryption and decryption."""
        original_data = "This is sensitive test data"

        # Encrypt data
        encrypted = self.crypto_manager.encrypt_data(original_data)
        self.assertIsInstance(encrypted, str)
        self.assertNotEqual(encrypted, original_data)

        # Decrypt data
        decrypted = self.crypto_manager.decrypt_data(encrypted)
        self.assertEqual(decrypted, original_data)

    def test_asymmetric_encryption(self):
        """Test asymmetric encryption and decryption."""
        original_data = "Test message for RSA encryption"

        # Encrypt with public key
        encrypted = self.crypto_manager.encrypt_with_public_key(original_data)
        self.assertIsInstance(encrypted, str)
        self.assertNotEqual(encrypted, original_data)

        # Decrypt with private key
        decrypted = self.crypto_manager.decrypt_with_private_key(encrypted)
        self.assertEqual(decrypted, original_data)

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"

        # Hash password
        hashed, salt = self.crypto_manager.hash_password(password)
        self.assertIsInstance(hashed, str)
        self.assertIsInstance(salt, str)
        self.assertNotEqual(hashed, password)

        # Verify correct password
        self.assertTrue(self.crypto_manager.verify_password(password, hashed, salt))

        # Verify incorrect password
        self.assertFalse(
            self.crypto_manager.verify_password("wrong_password", hashed, salt),
        )

    def test_secure_token_generation(self):
        """Test secure token generation."""
        token1 = self.crypto_manager.generate_secure_token()
        token2 = self.crypto_manager.generate_secure_token()

        self.assertIsInstance(token1, str)
        self.assertIsInstance(token2, str)
        self.assertNotEqual(token1, token2)
        self.assertGreater(len(token1), 20)  # Should be reasonably long

    def test_api_key_generation(self):
        """Test API key generation."""
        api_key = self.crypto_manager.generate_api_key()

        self.assertIsInstance(api_key, str)
        self.assertTrue(api_key.startswith("icarus_"))

    def test_sensitive_config_encryption(self):
        """Test encryption of sensitive configuration fields."""
        config = {
            "database_url": "postgresql://user:pass@localhost/db",
            "api_key": "secret_key_123",
            "password": "user_password",
            "normal_setting": "not_sensitive",
            "token": "auth_token_456",
        }

        # Encrypt sensitive fields
        encrypted_config = self.crypto_manager.encrypt_sensitive_config(config)

        # Check that sensitive fields are encrypted
        self.assertNotEqual(encrypted_config["api_key"], config["api_key"])
        self.assertNotEqual(encrypted_config["password"], config["password"])
        self.assertNotEqual(encrypted_config["token"], config["token"])

        # Check that non-sensitive fields are unchanged
        self.assertEqual(encrypted_config["normal_setting"], config["normal_setting"])

        # Decrypt and verify
        decrypted_config = self.crypto_manager.decrypt_sensitive_config(
            encrypted_config,
        )
        self.assertEqual(decrypted_config, config)


class TestAuthenticationManager(unittest.TestCase):
    """Test authentication functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.crypto_manager = CryptoManager()
        self.user_manager = UserManager(data_dir=self.temp_dir)
        self.auth_manager = AuthenticationManager(
            user_manager=self.user_manager,
            crypto_manager=self.crypto_manager,
            data_dir=self.temp_dir,
        )

        # Create test user
        self.test_user = self.user_manager.create_user(
            username="testuser",
            email="test@example.com",
            display_name="Test User",
            password="TestPassword123!",
            role=UserRole.COLLABORATOR,
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_user_authentication_success(self):
        """Test successful user authentication."""
        token = self.auth_manager.authenticate_user("testuser", "TestPassword123!")

        self.assertIsNotNone(token)
        self.assertIsInstance(token, str)

        # Verify token authentication
        user = self.auth_manager.authenticate_token(token)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "testuser")

    def test_user_authentication_failure(self):
        """Test failed user authentication."""
        # Wrong password
        token = self.auth_manager.authenticate_user("testuser", "WrongPassword")
        self.assertIsNone(token)

        # Non-existent user
        token = self.auth_manager.authenticate_user("nonexistent", "password")
        self.assertIsNone(token)

    def test_session_management(self):
        """Test session creation and management."""
        # Authenticate user
        token = self.auth_manager.authenticate_user("testuser", "TestPassword123!")
        self.assertIsNotNone(token)

        # Get session info
        session_info = self.auth_manager.get_session_info(token)
        self.assertIsNotNone(session_info)
        self.assertEqual(session_info["username"], "testuser")

        # Revoke session
        success = self.auth_manager.revoke_session(token)
        self.assertTrue(success)

        # Verify session is revoked
        user = self.auth_manager.authenticate_token(token)
        self.assertIsNone(user)

    def test_api_key_management(self):
        """Test API key generation and authentication."""
        # Generate API key
        api_key = self.auth_manager.generate_api_key(self.test_user.id)
        self.assertIsNotNone(api_key)
        self.assertTrue(api_key.startswith("icarus_"))

        # Authenticate with API key
        user = self.auth_manager.authenticate_api_key(api_key)
        self.assertIsNotNone(user)
        self.assertEqual(user.id, self.test_user.id)

        # Revoke API key
        success = self.auth_manager.revoke_api_key(api_key)
        self.assertTrue(success)

        # Verify API key is revoked
        user = self.auth_manager.authenticate_api_key(api_key)
        self.assertIsNone(user)

    def test_password_validation(self):
        """Test password strength validation."""
        # Valid password
        is_valid, issues = self.auth_manager.validate_password_strength(
            "StrongPass123!",
        )
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

        # Too short
        is_valid, issues = self.auth_manager.validate_password_strength("Short1!")
        self.assertFalse(is_valid)
        self.assertIn("at least", issues[0])

        # No uppercase
        is_valid, issues = self.auth_manager.validate_password_strength("lowercase123!")
        self.assertFalse(is_valid)
        self.assertIn("uppercase", issues[0])

        # No numbers
        is_valid, issues = self.auth_manager.validate_password_strength("NoNumbers!")
        self.assertFalse(is_valid)
        self.assertIn("number", issues[0])

        # No special characters
        is_valid, issues = self.auth_manager.validate_password_strength("NoSpecial123")
        self.assertFalse(is_valid)
        self.assertIn("special", issues[0])


class TestAuthorizationManager(unittest.TestCase):
    """Test authorization and access control."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.authz_manager = AuthorizationManager(data_dir=self.temp_dir)

        # Create test users
        self.owner_user = User(
            id="owner_id",
            username="owner",
            email="owner@example.com",
            display_name="Owner User",
            role=UserRole.OWNER,
            permissions=set(),
            created_at=datetime.now(),
            last_active=datetime.now(),
        )

        self.collaborator_user = User(
            id="collab_id",
            username="collaborator",
            email="collab@example.com",
            display_name="Collaborator User",
            role=UserRole.COLLABORATOR,
            permissions=set(),
            created_at=datetime.now(),
            last_active=datetime.now(),
        )

        self.viewer_user = User(
            id="viewer_id",
            username="viewer",
            email="viewer@example.com",
            display_name="Viewer User",
            role=UserRole.VIEWER,
            permissions=set(),
            created_at=datetime.now(),
            last_active=datetime.now(),
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_role_based_access_control(self):
        """Test role-based access control."""
        resource_id = "test_analysis_1"

        # Owner should have full access
        self.assertTrue(
            self.authz_manager.check_access(
                self.owner_user,
                ResourceType.ANALYSIS,
                resource_id,
                AccessLevel.OWNER,
            ),
        )

        # Collaborator should have write access but not owner
        self.assertTrue(
            self.authz_manager.check_access(
                self.collaborator_user,
                ResourceType.ANALYSIS,
                resource_id,
                AccessLevel.WRITE,
            ),
        )
        self.assertFalse(
            self.authz_manager.check_access(
                self.collaborator_user,
                ResourceType.ANALYSIS,
                resource_id,
                AccessLevel.OWNER,
            ),
        )

        # Viewer should have read access but not write
        self.assertTrue(
            self.authz_manager.check_access(
                self.viewer_user,
                ResourceType.ANALYSIS,
                resource_id,
                AccessLevel.READ,
            ),
        )
        self.assertFalse(
            self.authz_manager.check_access(
                self.viewer_user,
                ResourceType.ANALYSIS,
                resource_id,
                AccessLevel.WRITE,
            ),
        )

    def test_explicit_permissions(self):
        """Test explicit permission granting and revoking."""
        resource_id = "test_workflow_1"

        # Initially viewer should not have write access
        self.assertFalse(
            self.authz_manager.check_access(
                self.viewer_user,
                ResourceType.WORKFLOW,
                resource_id,
                AccessLevel.WRITE,
            ),
        )

        # Grant explicit write permission
        success = self.authz_manager.grant_permission(
            self.viewer_user.id,
            ResourceType.WORKFLOW,
            resource_id,
            AccessLevel.WRITE,
            self.owner_user.id,
        )
        self.assertTrue(success)

        # Now viewer should have write access
        self.assertTrue(
            self.authz_manager.check_access(
                self.viewer_user,
                ResourceType.WORKFLOW,
                resource_id,
                AccessLevel.WRITE,
            ),
        )

        # Revoke permission
        success = self.authz_manager.revoke_permission(
            self.viewer_user.id,
            ResourceType.WORKFLOW,
            resource_id,
        )
        self.assertTrue(success)

        # Access should be revoked
        self.assertFalse(
            self.authz_manager.check_access(
                self.viewer_user,
                ResourceType.WORKFLOW,
                resource_id,
                AccessLevel.WRITE,
            ),
        )

    def test_resource_ownership(self):
        """Test resource ownership functionality."""
        resource_id = "owned_resource_1"

        # Set resource owner
        self.authz_manager.set_resource_owner(resource_id, self.collaborator_user.id)

        # Owner should have full access regardless of role
        self.assertTrue(
            self.authz_manager.check_access(
                self.collaborator_user,
                ResourceType.DATA,
                resource_id,
                AccessLevel.OWNER,
            ),
        )

    def test_permission_expiration(self):
        """Test permission expiration."""
        resource_id = "temp_resource_1"
        expires_at = datetime.now() + timedelta(seconds=1)

        # Grant temporary permission
        success = self.authz_manager.grant_permission(
            self.viewer_user.id,
            ResourceType.DATA,
            resource_id,
            AccessLevel.WRITE,
            self.owner_user.id,
            expires_at,
        )
        self.assertTrue(success)

        # Should have access initially
        self.assertTrue(
            self.authz_manager.check_access(
                self.viewer_user,
                ResourceType.DATA,
                resource_id,
                AccessLevel.WRITE,
            ),
        )

        # Wait for expiration and cleanup
        import time

        time.sleep(2)
        self.authz_manager.cleanup_expired_permissions()

        # Should no longer have access
        self.assertFalse(
            self.authz_manager.check_access(
                self.viewer_user,
                ResourceType.DATA,
                resource_id,
                AccessLevel.WRITE,
            ),
        )


class TestAuditLogger(unittest.TestCase):
    """Test audit logging functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.crypto_manager = CryptoManager()
        self.audit_logger = AuditLogger(
            data_dir=self.temp_dir,
            crypto_manager=self.crypto_manager,
            enable_encryption=True,
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_event_logging(self):
        """Test basic event logging."""
        event_id = self.audit_logger.log_event(
            AuditEventType.LOGIN_SUCCESS,
            "User logged in successfully",
            user_id="test_user_id",
            username="testuser",
            session_id="test_session",
            severity=AuditSeverity.LOW,
        )

        self.assertIsNotNone(event_id)
        self.assertIsInstance(event_id, str)

    def test_event_querying(self):
        """Test event querying with filters."""
        # Log multiple events
        self.audit_logger.log_event(
            AuditEventType.LOGIN_SUCCESS,
            "User 1 logged in",
            user_id="user1",
            username="user1",
        )

        self.audit_logger.log_event(
            AuditEventType.LOGIN_FAILURE,
            "User 2 failed login",
            user_id="user2",
            username="user2",
            success=False,
        )

        self.audit_logger.log_event(
            AuditEventType.ACCESS_DENIED,
            "Access denied to resource",
            user_id="user1",
            username="user1",
            success=False,
        )

        # Query all events
        all_events = self.audit_logger.query_events()
        self.assertGreaterEqual(len(all_events), 3)

        # Query by event type
        login_events = self.audit_logger.query_events(
            event_types=[AuditEventType.LOGIN_SUCCESS, AuditEventType.LOGIN_FAILURE],
        )
        self.assertGreaterEqual(len(login_events), 2)

        # Query by user
        user1_events = self.audit_logger.query_events(username="user1")
        self.assertGreaterEqual(len(user1_events), 2)

        # Query by success status
        failed_events = self.audit_logger.query_events(success=False)
        self.assertGreaterEqual(len(failed_events), 2)

    def test_security_summary(self):
        """Test security summary generation."""
        # Log various security events
        self.audit_logger.log_event(
            AuditEventType.LOGIN_FAILURE,
            "Failed login",
            success=False,
            username="attacker",
        )

        self.audit_logger.log_event(
            AuditEventType.ACCESS_DENIED,
            "Access denied",
            success=False,
            username="user1",
        )

        self.audit_logger.log_event(
            AuditEventType.SECURITY_VIOLATION,
            "Security violation detected",
            severity=AuditSeverity.HIGH,
            username="user2",
        )

        # Generate summary
        summary = self.audit_logger.get_security_summary(days=1)

        self.assertIsInstance(summary, dict)
        self.assertIn("total_events", summary)
        self.assertIn("failed_logins", summary)
        self.assertIn("access_denied_count", summary)
        self.assertIn("security_violations", summary)
        self.assertGreater(summary["total_events"], 0)

    def test_audit_log_export(self):
        """Test audit log export functionality."""
        # Log some events
        self.audit_logger.log_event(
            AuditEventType.LOGIN_SUCCESS,
            "Test login",
            username="testuser",
        )

        # Export to JSON
        export_file = os.path.join(self.temp_dir, "audit_export.json")
        success = self.audit_logger.export_audit_log(export_file, format="json")

        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_file))

        # Verify export content
        with open(export_file) as f:
            exported_data = json.load(f)

        self.assertIsInstance(exported_data, list)
        self.assertGreater(len(exported_data), 0)


class TestSecurityManager(unittest.TestCase):
    """Test the main security manager integration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.user_manager = UserManager(data_dir=self.temp_dir)
        self.security_manager = SecurityManager(
            user_manager=self.user_manager,
            data_dir=self.temp_dir,
            enable_encryption=True,
        )

        # Create test user
        self.test_user = self.user_manager.create_user(
            username="testuser",
            email="test@example.com",
            display_name="Test User",
            password="TestPassword123!",
            role=UserRole.COLLABORATOR,
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_integrated_authentication(self):
        """Test integrated authentication through security manager."""
        # Authenticate user
        token = self.security_manager.authenticate_user("testuser", "TestPassword123!")
        self.assertIsNotNone(token)

        # Verify token
        user = self.security_manager.authenticate_token(token)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "testuser")

        # Logout
        success = self.security_manager.logout_user(token)
        self.assertTrue(success)

    def test_integrated_authorization(self):
        """Test integrated authorization through security manager."""
        resource_id = "test_resource_1"

        # Check access
        has_access = self.security_manager.check_access(
            self.test_user,
            "analysis",
            resource_id,
            "write",
        )
        self.assertTrue(has_access)  # Collaborator should have write access

        # Check higher access
        has_admin_access = self.security_manager.check_access(
            self.test_user,
            "user_management",
            "permissions",
            "admin",
        )
        self.assertFalse(has_admin_access)  # Collaborator should not have admin access

    def test_security_event_logging(self):
        """Test security event logging through security manager."""
        event_id = self.security_manager.log_security_event(
            "login_success",
            "Test login event",
            user=self.test_user,
            severity="low",
        )

        self.assertIsNotNone(event_id)
        self.assertIsInstance(event_id, str)

    def test_encryption_integration(self):
        """Test encryption functionality through security manager."""
        sensitive_data = "This is sensitive information"

        # Encrypt data
        encrypted = self.security_manager.encrypt_sensitive_data(sensitive_data)
        self.assertIsNotNone(encrypted)
        self.assertNotEqual(encrypted, sensitive_data)

        # Decrypt data
        decrypted = self.security_manager.decrypt_sensitive_data(encrypted)
        self.assertEqual(decrypted, sensitive_data)

    def test_configuration_encryption(self):
        """Test configuration encryption through security manager."""
        config = {
            "database_password": "secret_db_pass",
            "api_key": "secret_api_key",
            "normal_setting": "not_secret",
        }

        # Encrypt configuration
        encrypted_config = self.security_manager.encrypt_configuration(config)

        # Verify sensitive fields are encrypted
        self.assertNotEqual(
            encrypted_config["database_password"],
            config["database_password"],
        )
        self.assertNotEqual(encrypted_config["api_key"], config["api_key"])
        self.assertEqual(encrypted_config["normal_setting"], config["normal_setting"])

        # Decrypt configuration
        decrypted_config = self.security_manager.decrypt_configuration(encrypted_config)
        self.assertEqual(decrypted_config, config)

    def test_security_summary(self):
        """Test comprehensive security summary."""
        # Generate some activity
        self.security_manager.authenticate_user("testuser", "TestPassword123!")
        self.security_manager.log_security_event(
            "access_granted",
            "Access granted to resource",
            user=self.test_user,
        )

        # Get security summary
        summary = self.security_manager.get_security_summary(days=1)

        self.assertIsInstance(summary, dict)
        self.assertIn("audit_summary", summary)
        self.assertIn("authentication", summary)
        self.assertIn("authorization", summary)
        self.assertIn("system_health", summary)

    def test_security_cleanup(self):
        """Test security data cleanup."""
        # This should run without errors
        self.security_manager.cleanup_security_data()

    def test_security_report_export(self):
        """Test security report export."""
        export_file = os.path.join(self.temp_dir, "security_report.json")
        success = self.security_manager.export_security_report(export_file)

        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_file))


def run_security_tests():
    """Run all security system tests."""
    print("Running ICARUS CLI Security System Tests...")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestCryptoManager),
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestAuthenticationManager),
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestAuthorizationManager),
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestAuditLogger),
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSecurityManager),
    )

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    run_security_tests()
