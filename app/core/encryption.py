"""
Document Encryption Module

This module provides encryption and decryption functionality for sensitive document content.
Uses AES-256-GCM for authenticated encryption, providing both confidentiality and integrity.

Features:
- AES-256-GCM encryption for document content
- Key derivation using PBKDF2-SHA256
- Base64 encoding for encrypted data storage
- Secure random nonce/IV generation
- Authentication tag verification
"""

import base64
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.exceptions import InvalidTag
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from app.core.logging_config import get_logger


logger = get_logger(__name__)


# Constants
KEY_LENGTH = 32  # 256 bits for AES-256
NONCE_LENGTH = 12  # 96 bits for GCM
SALT_LENGTH = 16  # 128 bits
ITERATIONS = 100000  # PBKDF2 iterations


@dataclass
class EncryptionResult:
    """Result of encryption operation"""
    encrypted_data: str  # Base64 encoded
    nonce: str  # Base64 encoded
    salt: str  # Base64 encoded
    tag: str  # Base64 encoded (auth tag)


class EncryptionError(Exception):
    """Base exception for encryption errors"""
    pass


class KeyDerivationError(EncryptionError):
    """Exception raised when key derivation fails"""
    pass


class DecryptionError(EncryptionError):
    """Exception raised when decryption fails"""
    pass


class DocumentEncryption:
    """
    Document encryption and decryption using AES-256-GCM.
    
    This class provides methods to encrypt and decrypt document content
    using industry-standard cryptographic primitives. Each encryption
    operation uses a unique nonce and salt for security.
    
    Example:
        >>> encryptor = DocumentEncryption(password="secure_key")
        >>> result = encryptor.encrypt("sensitive content")
        >>> decrypted = encryptor.decrypt(result.encrypted_data, result.nonce, result.salt)
        >>> assert decrypted == "sensitive content"
    """
    
    def __init__(self, password: Optional[str] = None):
        """
        Initialize encryption with optional password.
        
        Args:
            password: Encryption password. If None, uses environment variable
                     or raises error.
        
        Raises:
            ImportError: If cryptography library is not installed
            ValueError: If no password is provided and none in environment
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError(
                "cryptography library is required for encryption. "
                "Install it with: pip install cryptography"
            )
        
        self.password = password or self._get_default_password()
        
        if not self.password:
            raise ValueError(
                "Encryption password must be provided either as parameter "
                "or through DOCUMENT_ENCRYPTION_KEY environment variable"
            )
        
        logger.debug("DocumentEncryption initialized")
    
    @staticmethod
    def _get_default_password() -> Optional[str]:
        """Get default password from environment variable."""
        return os.environ.get("DOCUMENT_ENCRYPTION_KEY")
    
    def _derive_key(self, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            salt: Salt for key derivation
        
        Returns:
            Derived key of KEY_LENGTH bytes
        
        Raises:
            KeyDerivationError: If key derivation fails
        """
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=KEY_LENGTH,
                salt=salt,
                iterations=ITERATIONS,
            )
            key = kdf.derive(self.password.encode('utf-8'))
            return key
        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            raise KeyDerivationError(f"Failed to derive key: {e}")
    
    def encrypt(self, plaintext: str) -> EncryptionResult:
        """
        Encrypt plaintext using AES-256-GCM.
        
        Args:
            plaintext: Text to encrypt
        
        Returns:
            EncryptionResult containing encrypted data, nonce, salt, and tag
        
        Raises:
            EncryptionError: If encryption fails
        """
        if not plaintext:
            logger.warning("Attempted to encrypt empty string")
            return EncryptionResult(
                encrypted_data="",
                nonce="",
                salt="",
                tag=""
            )
        
        try:
            # Generate random salt and nonce
            salt = os.urandom(SALT_LENGTH)
            nonce = os.urandom(NONCE_LENGTH)
            
            # Derive key from password and salt
            key = self._derive_key(salt)
            
            # Encrypt using AES-256-GCM
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
            
            # Split ciphertext and tag (last 16 bytes is the tag)
            encrypted_data = ciphertext[:-16]
            tag = ciphertext[-16:]
            
            # Encode to base64 for storage
            result = EncryptionResult(
                encrypted_data=base64.b64encode(encrypted_data).decode('utf-8'),
                nonce=base64.b64encode(nonce).decode('utf-8'),
                salt=base64.b64encode(salt).decode('utf-8'),
                tag=base64.b64encode(tag).decode('utf-8')
            )
            
            logger.debug(f"Successfully encrypted {len(plaintext)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt data: {e}")
    
    def decrypt(
        self,
        encrypted_data: str,
        nonce: str,
        salt: str,
        tag: str
    ) -> str:
        """
        Decrypt ciphertext using AES-256-GCM.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            nonce: Base64 encoded nonce
            salt: Base64 encoded salt
            tag: Base64 encoded authentication tag
        
        Returns:
            Decrypted plaintext
        
        Raises:
            DecryptionError: If decryption fails or authentication fails
        """
        if not encrypted_data:
            logger.warning("Attempted to decrypt empty string")
            return ""
        
        try:
            # Decode from base64
            ciphertext = base64.b64decode(encrypted_data)
            nonce_bytes = base64.b64decode(nonce)
            salt_bytes = base64.b64decode(salt)
            tag_bytes = base64.b64decode(tag)
            
            # Derive key from password and salt
            key = self._derive_key(salt_bytes)
            
            # Combine ciphertext and tag for decryption
            ciphertext_with_tag = ciphertext + tag_bytes
            
            # Decrypt using AES-256-GCM
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce_bytes, ciphertext_with_tag, None)
            
            result = plaintext.decode('utf-8')
            logger.debug(f"Successfully decrypted {len(result)} characters")
            return result
            
        except InvalidTag:
            logger.error("Decryption failed: Invalid authentication tag")
            raise DecryptionError(
                "Decryption failed: Authentication tag invalid. "
                "Data may have been tampered with or wrong password."
            )
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt data: {e}")
    
    def encrypt_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt document content and update metadata.
        
        Convenience method that encrypts content and adds encryption
        metadata to track encryption status.
        
        Args:
            content: Document content to encrypt
            metadata: Document metadata dictionary
        
        Returns:
            Updated metadata with encryption information
        """
        result = self.encrypt(content)
        
        # Update metadata with encryption info
        updated_metadata = metadata.copy()
        updated_metadata['encrypted'] = True
        updated_metadata['encryption_version'] = '1.0'
        updated_metadata['encryption_algorithm'] = 'AES-256-GCM'
        
        return updated_metadata
    
    def is_encrypted(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if document metadata indicates encryption.
        
        Args:
            metadata: Document metadata dictionary
        
        Returns:
            True if document is marked as encrypted
        """
        return metadata.get('encrypted', False)
