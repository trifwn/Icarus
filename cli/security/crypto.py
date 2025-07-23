"""
Cryptographic utilities for data encryption and security.

Provides encryption/decryption for sensitive data including:
- User credentials and session tokens
- Analysis parameters and results
- Configuration data
- Plugin data
"""

import base64
import hashlib
import os
import secrets
from typing import Optional
from typing import Tuple
from typing import Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class CryptoManager:
    """Manages encryption and decryption operations."""

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize crypto manager.

        Args:
            master_key: Optional master key for encryption. If not provided,
                       will be generated or loaded from secure storage.
        """
        self._master_key = master_key
        self._fernet = None
        self._private_key = None
        self._public_key = None

        # Initialize encryption
        self._initialize_encryption()

    def _initialize_encryption(self):
        """Initialize encryption keys and ciphers."""
        if self._master_key:
            # Use provided master key
            key = self._derive_key_from_password(self._master_key)
        else:
            # Generate or load key
            key = self._get_or_create_key()

        self._fernet = Fernet(key)

        # Initialize RSA keys for asymmetric encryption
        self._initialize_rsa_keys()

    def _derive_key_from_password(
        self,
        password: str,
        salt: Optional[bytes] = None,
    ) -> bytes:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: Password to derive key from
            salt: Optional salt bytes. If not provided, generates new salt.

        Returns:
            Base64-encoded key suitable for Fernet
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def _get_or_create_key(self) -> bytes:
        """Get existing key or create new one."""
        key_file = os.path.expanduser("~/.icarus/security/master.key")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(key_file), exist_ok=True)

        if os.path.exists(key_file):
            # Load existing key
            try:
                with open(key_file, "rb") as f:
                    return f.read()
            except Exception:
                # If key file is corrupted, generate new one
                pass

        # Generate new key
        key = Fernet.generate_key()

        # Save key securely
        try:
            with open(key_file, "wb") as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
        except Exception:
            # If we can't save the key, continue with in-memory key
            pass

        return key

    def _initialize_rsa_keys(self):
        """Initialize RSA key pair for asymmetric encryption."""
        private_key_file = os.path.expanduser("~/.icarus/security/private.pem")
        public_key_file = os.path.expanduser("~/.icarus/security/public.pem")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(private_key_file), exist_ok=True)

        if os.path.exists(private_key_file) and os.path.exists(public_key_file):
            # Load existing keys
            try:
                with open(private_key_file, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                    )
                with open(public_key_file, "rb") as f:
                    self._public_key = serialization.load_pem_public_key(f.read())
                return
            except Exception:
                # If keys are corrupted, generate new ones
                pass

        # Generate new RSA key pair
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self._public_key = self._private_key.public_key()

        # Save keys
        try:
            # Save private key
            private_pem = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            with open(private_key_file, "wb") as f:
                f.write(private_pem)
            os.chmod(private_key_file, 0o600)

            # Save public key
            public_pem = self._public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            with open(public_key_file, "wb") as f:
                f.write(public_pem)
            os.chmod(public_key_file, 0o644)

        except Exception:
            # If we can't save keys, continue with in-memory keys
            pass

    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data using symmetric encryption.

        Args:
            data: Data to encrypt (string or bytes)

        Returns:
            Base64-encoded encrypted data
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        encrypted = self._fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode("utf-8")

    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt data using symmetric encryption.

        Args:
            encrypted_data: Base64-encoded encrypted data

        Returns:
            Decrypted data as string
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode("utf-8"))
        decrypted = self._fernet.decrypt(encrypted_bytes)
        return decrypted.decode("utf-8")

    def encrypt_with_public_key(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data using RSA public key.

        Args:
            data: Data to encrypt

        Returns:
            Base64-encoded encrypted data
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        encrypted = self._public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return base64.urlsafe_b64encode(encrypted).decode("utf-8")

    def decrypt_with_private_key(self, encrypted_data: str) -> str:
        """
        Decrypt data using RSA private key.

        Args:
            encrypted_data: Base64-encoded encrypted data

        Returns:
            Decrypted data as string
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode("utf-8"))
        decrypted = self._private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return decrypted.decode("utf-8")

    def hash_password(
        self,
        password: str,
        salt: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Hash password with salt using PBKDF2.

        Args:
            password: Password to hash
            salt: Optional salt. If not provided, generates new salt.

        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,
        )

        return password_hash.hex(), salt

    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify password against hash.

        Args:
            password: Password to verify
            hashed_password: Stored password hash
            salt: Salt used for hashing

        Returns:
            True if password matches, False otherwise
        """
        computed_hash, _ = self.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, hashed_password)

    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate cryptographically secure random token.

        Args:
            length: Token length in bytes

        Returns:
            URL-safe base64-encoded token
        """
        return secrets.token_urlsafe(length)

    def generate_api_key(self) -> str:
        """
        Generate API key for external integrations.

        Returns:
            Secure API key
        """
        return f"icarus_{secrets.token_urlsafe(32)}"

    def encrypt_sensitive_config(self, config_data: dict) -> dict:
        """
        Encrypt sensitive configuration data.

        Args:
            config_data: Configuration dictionary

        Returns:
            Configuration with sensitive fields encrypted
        """
        sensitive_fields = {
            "password",
            "token",
            "key",
            "secret",
            "credential",
            "api_key",
            "auth_token",
            "session_token",
        }

        encrypted_config = {}
        for key, value in config_data.items():
            if any(field in key.lower() for field in sensitive_fields):
                if isinstance(value, str):
                    encrypted_config[key] = self.encrypt_data(value)
                else:
                    encrypted_config[key] = value
            else:
                encrypted_config[key] = value

        return encrypted_config

    def decrypt_sensitive_config(self, encrypted_config: dict) -> dict:
        """
        Decrypt sensitive configuration data.

        Args:
            encrypted_config: Configuration with encrypted fields

        Returns:
            Configuration with sensitive fields decrypted
        """
        sensitive_fields = {
            "password",
            "token",
            "key",
            "secret",
            "credential",
            "api_key",
            "auth_token",
            "session_token",
        }

        decrypted_config = {}
        for key, value in encrypted_config.items():
            if any(field in key.lower() for field in sensitive_fields):
                if isinstance(value, str):
                    try:
                        decrypted_config[key] = self.decrypt_data(value)
                    except Exception:
                        # If decryption fails, assume it's not encrypted
                        decrypted_config[key] = value
                else:
                    decrypted_config[key] = value
            else:
                decrypted_config[key] = value

        return decrypted_config

    def get_public_key_pem(self) -> str:
        """
        Get public key in PEM format for sharing.

        Returns:
            Public key as PEM string
        """
        public_pem = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return public_pem.decode("utf-8")

    def load_public_key_from_pem(self, pem_data: str):
        """
        Load public key from PEM data for encrypting data for others.

        Args:
            pem_data: Public key in PEM format

        Returns:
            Public key object
        """
        return serialization.load_pem_public_key(pem_data.encode("utf-8"))
