# ICARUS CLI Security System

A comprehensive security framework providing data encryption, authentication, authorization, audit logging, and plugin security for the ICARUS CLI application.

## Overview

The security system implements enterprise-grade security features including:

- **Data Encryption**: AES-256 encryption for sensitive data and configuration
- **Authentication**: Multi-method authentication with session management
- **Authorization**: Role-based access control with explicit permissions
- **Audit Logging**: Comprehensive security event logging and monitoring
- **Plugin Security**: Static analysis and sandboxing for plugin validation

## Architecture

```
Security System
├── SecurityManager (Main coordinator)
├── CryptoManager (Encryption/Decryption)
├── AuthenticationManager (User authentication)
├── AuthorizationManager (Access control)
├── AuditLogger (Security event logging)
└── PluginSecurity (Plugin validation)
```

## Components

### 1. CryptoManager (`crypto.py`)

Handles all cryptographic operations:

- **Symmetric Encryption**: AES-256 via Fernet for data encryption
- **Asymmetric Encryption**: RSA-2048 for key exchange and secure communication
- **Password Hashing**: PBKDF2-SHA256 with salt for secure password storage
- **Token Generation**: Cryptographically secure random tokens
- **Configuration Encryption**: Automatic encryption of sensitive config fields

**Key Features:**
- Automatic key generation and secure storage
- Support for both symmetric and asymmetric encryption
- Password strength validation and secure hashing
- Sensitive configuration field detection and encryption

### 2. AuthenticationManager (`authentication.py`)

Manages user authentication and session lifecycle:

- **Multi-Method Authentication**: Password, token, API key, certificate support
- **Session Management**: Secure session creation, validation, and cleanup
- **Failed Attempt Tracking**: Automatic lockout after failed login attempts
- **Token Management**: Session tokens and API key generation/validation

**Key Features:**
- Configurable session timeout and security policies
- Automatic session cleanup and user lockout
- Support for multiple concurrent sessions per user
- Encrypted session storage with audit trail

### 3. AuthorizationManager (`authorization.py`)

Implements role-based access control:

- **Role-Based Permissions**: Default permissions based on user roles
- **Explicit Permissions**: Grant/revoke specific resource access
- **Resource Ownership**: Automatic owner-level access for resource creators
- **Policy Management**: Configurable access policies per resource type

**Key Features:**
- Hierarchical access levels (None → Read → Write → Execute → Admin → Owner)
- Time-based permission expiration
- Resource-specific access control
- Policy-driven authorization decisions

### 4. AuditLogger (`audit_logger.py`)

Comprehensive security event logging:

- **Event Classification**: Categorized security events with severity levels
- **Encrypted Storage**: Optional encryption of audit logs
- **Query Interface**: Flexible event querying with filters
- **Anomaly Detection**: Automatic detection of suspicious activity patterns

**Key Features:**
- SQLite database for structured event storage
- Real-time anomaly detection and alerting
- Comprehensive security reporting and analytics
- Configurable log retention and cleanup policies

### 5. SecurityManager (`security_manager.py`)

Main coordinator integrating all security components:

- **Unified Interface**: Single entry point for all security operations
- **Event Correlation**: Automatic audit logging for security events
- **Policy Enforcement**: Consistent security policy application
- **Health Monitoring**: System-wide security health monitoring

## Usage Examples

### Basic Setup

```python
from security.security_manager import SecurityManager
from collaboration.user_manager import UserManager

# Initialize security system
user_manager = UserManager()
security_manager = SecurityManager(
    user_manager=user_manager,
    enable_encryption=True
)
```

### Authentication

```python
# Authenticate user
token = security_manager.authenticate_user(
    username="user@example.com",
    password="secure_password",
    ip_address="192.168.1.100"
)

# Validate token
user = security_manager.authenticate_token(token)

# Generate API key
api_key = security_manager.auth_manager.generate_api_key(user.id)
```

### Authorization

```python
# Check access
has_access = security_manager.check_access(
    user=user,
    resource_type="analysis",
    resource_id="wind_tunnel_test_1",
    required_access="write"
)

# Grant explicit permission
security_manager.grant_permission(
    admin_user=admin_user,
    target_user_id=user.id,
    resource_type="workflow",
    resource_id="optimization_flow",
    access_level="execute",
    expires_at=datetime.now() + timedelta(hours=24)
)
```

### Data Encryption

```python
# Encrypt sensitive data
encrypted = security_manager.encrypt_sensitive_data("sensitive_info")
decrypted = security_manager.decrypt_sensitive_data(encrypted)

# Encrypt configuration
config = {"api_key": "secret", "password": "pass123"}
encrypted_config = security_manager.encrypt_configuration(config)
```

### Audit Logging

```python
# Log security event
event_id = security_manager.log_security_event(
    event_type="resource_accessed",
    action="User accessed analysis results",
    user=user,
    resource_type="analysis",
    resource_id="test_results_1",
    severity="medium"
)

# Get security summary
summary = security_manager.get_security_summary(days=7)
```

### Plugin Security

```python
# Validate plugin
is_safe = security_manager.validate_plugin_security(plugin_info, user)

# Trust/block plugins
security_manager.trust_plugin(plugin_id, admin_user)
security_manager.block_plugin(plugin_id, admin_user)
```

## Configuration

### Authentication Configuration

```python
auth_config = AuthConfig(
    session_timeout_hours=24,
    max_sessions_per_user=5,
    password_min_length=8,
    password_require_special=True,
    password_require_numbers=True,
    password_require_uppercase=True,
    enable_2fa=False,
    max_login_attempts=5,
    lockout_duration_minutes=30,
    enable_session_encryption=True
)
```

### Access Policies

```python
# Define custom access policy
policy = AccessPolicy(
    resource_type=ResourceType.ANALYSIS,
    default_permissions={
        UserRole.OWNER: AccessLevel.OWNER,
        UserRole.ADMIN: AccessLevel.ADMIN,
        UserRole.COLLABORATOR: AccessLevel.WRITE,
        UserRole.VIEWER: AccessLevel.READ,
        UserRole.GUEST: AccessLevel.NONE
    },
    required_permissions=[Permission.RUN_ANALYSIS],
    allow_owner_override=True,
    require_explicit_grant=False
)
```

## Security Features

### Data Protection

- **Encryption at Rest**: All sensitive data encrypted using AES-256
- **Encryption in Transit**: TLS for network communications
- **Key Management**: Automatic key generation and secure storage
- **Data Classification**: Automatic detection of sensitive data fields

### Access Control

- **Role-Based Access**: Default permissions based on user roles
- **Attribute-Based Access**: Fine-grained permissions based on resource attributes
- **Time-Based Access**: Temporary permissions with automatic expiration
- **Context-Aware Access**: IP-based and device-based access controls

### Monitoring and Compliance

- **Comprehensive Auditing**: All security events logged with full context
- **Real-Time Monitoring**: Automatic detection of security anomalies
- **Compliance Reporting**: Built-in reports for security compliance
- **Forensic Analysis**: Detailed event correlation and investigation tools

### Plugin Security

- **Static Analysis**: Automatic code analysis for security vulnerabilities
- **Sandboxing**: Isolated execution environment for untrusted plugins
- **Permission Model**: Granular permissions for plugin capabilities
- **Trust Management**: Admin controls for plugin trust levels

## Security Best Practices

### For Administrators

1. **Regular Security Reviews**: Monitor security summaries and audit logs
2. **Access Control Hygiene**: Regularly review and clean up permissions
3. **Plugin Management**: Carefully review plugins before trusting
4. **Incident Response**: Establish procedures for security incidents

### For Developers

1. **Secure Coding**: Follow secure coding practices for plugins
2. **Minimal Permissions**: Request only necessary permissions
3. **Input Validation**: Validate all user inputs and external data
4. **Error Handling**: Avoid exposing sensitive information in errors

### For Users

1. **Strong Passwords**: Use complex passwords with special characters
2. **Session Management**: Log out when finished, especially on shared systems
3. **Suspicious Activity**: Report unusual system behavior
4. **Plugin Caution**: Only install plugins from trusted sources

## Testing

Run the comprehensive test suite:

```bash
python cli/test_security_system.py
```

Run the interactive demo:

```bash
python cli/demo_security_system.py
```

## File Structure

```
cli/security/
├── __init__.py              # Package initialization
├── README.md               # This documentation
├── crypto.py               # Cryptographic operations
├── authentication.py       # Authentication management
├── authorization.py        # Authorization and access control
├── audit_logger.py         # Security event logging
└── security_manager.py     # Main security coordinator
```

## Dependencies

- `cryptography>=41.0.0` - Cryptographic operations
- `sqlite3` - Audit log storage (built-in)
- `json` - Configuration and data serialization
- `pathlib` - File system operations
- `datetime` - Time-based operations

## Integration Points

The security system integrates with:

- **User Management**: Authentication and user role management
- **Plugin System**: Security validation and sandboxing
- **Configuration System**: Encryption of sensitive settings
- **API Layer**: Authentication and authorization for web endpoints
- **Collaboration System**: Session sharing and access control

## Performance Considerations

- **Encryption Overhead**: Minimal impact on system performance
- **Database Optimization**: Indexed audit log queries for fast retrieval
- **Memory Management**: Efficient session and permission caching
- **Cleanup Processes**: Automatic cleanup of expired data

## Security Considerations

- **Key Storage**: Encryption keys stored with restricted file permissions
- **Session Security**: Sessions encrypted and automatically expired
- **Audit Integrity**: Audit logs protected against tampering
- **Plugin Isolation**: Plugins executed in sandboxed environments

## Future Enhancements

- **Multi-Factor Authentication**: TOTP and hardware token support
- **Certificate-Based Authentication**: X.509 certificate support
- **Advanced Threat Detection**: Machine learning-based anomaly detection
- **Integration APIs**: REST APIs for external security tools
- **Compliance Frameworks**: Built-in support for SOC2, ISO27001, etc.

## Support

For security-related issues or questions:

1. Check the audit logs for relevant events
2. Review the security summary for system health
3. Consult the demo script for usage examples
4. Run the test suite to verify system integrity

## License

This security system is part of the ICARUS CLI project and follows the same licensing terms.
