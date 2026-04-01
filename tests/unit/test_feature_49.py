"""
Unit tests for Document Encryption (Feature 49)

Tests encryption and decryption functionality for sensitive document content.
Covers success cases, edge cases, error conditions, and integration with document loader.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from app.core.encryption import (
    DocumentEncryption,
    EncryptionResult,
    EncryptionError,
    DecryptionError,
    KeyDerivationError
)
from app.services.document_loader import Document, DocumentLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def encryption_password():
    """Test encryption password."""
    return "test_secure_password_123"


@pytest.fixture
def encryptor(encryption_password):
    """Create DocumentEncryption instance for testing."""
    return DocumentEncryption(password=encryption_password)


@pytest.fixture
def sample_content():
    """Sample document content for testing."""
    return "This is sensitive document content that needs encryption."


@pytest.fixture
def temp_text_file(tmp_path):
    """Create a temporary text file."""
    p = tmp_path / "secret.txt"
    p.write_text("Confidential information here.")
    return str(p)


# ---------------------------------------------------------------------------
# DocumentEncryption Initialization Tests
# ---------------------------------------------------------------------------


class TestDocumentEncryptionInit:
    """Tests for DocumentEncryption initialization."""

    def test_init_with_password(self, encryption_password):
        """Test initialization with password parameter."""
        encryptor = DocumentEncryption(password=encryption_password)
        assert encryptor.password == encryption_password

    def test_init_without_password_raises_error(self):
        """Test initialization without password raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="password must be provided"):
                DocumentEncryption()

    def test_init_from_environment_variable(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {'DOCUMENT_ENCRYPTION_KEY': 'env_password'}):
            encryptor = DocumentEncryption()
            assert encryptor.password == 'env_password'

    def test_init_without_cryptography_raises_import_error(self):
        """Test that missing cryptography library raises ImportError."""
        with patch('app.core.encryption.CRYPTOGRAPHY_AVAILABLE', False):
            with pytest.raises(ImportError, match="cryptography library"):
                DocumentEncryption(password="test")


# ---------------------------------------------------------------------------
# Encryption Tests
# ---------------------------------------------------------------------------


class TestEncryption:
    """Tests for encryption functionality."""

    def test_encrypt_returns_valid_result(self, encryptor, sample_content):
        """Test encryption returns valid EncryptionResult."""
        result = encryptor.encrypt(sample_content)
        
        assert isinstance(result, EncryptionResult)
        assert result.encrypted_data
        assert result.nonce
        assert result.salt
        assert result.tag

    def test_encrypt_empty_string(self, encryptor):
        """Test encryption of empty string."""
        result = encryptor.encrypt("")
        
        assert result.encrypted_data == ""
        assert result.nonce == ""
        assert result.salt == ""
        assert result.tag == ""

    def test_encrypt_generates_unique_values(self, encryptor, sample_content):
        """Test that each encryption generates unique nonce and salt."""
        result1 = encryptor.encrypt(sample_content)
        result2 = encryptor.encrypt(sample_content)
        
        # Each encryption should have unique nonce and salt
        assert result1.nonce != result2.nonce
        assert result1.salt != result2.salt
        # But encrypted data might be different due to different nonce/salt

    def test_encrypt_special_characters(self, encryptor):
        """Test encryption of content with special characters."""
        special_content = "Test with émojis 🎉 and sp€cial charact&rs!"
        result = encryptor.encrypt(special_content)
        
        assert result.encrypted_data
        # Should be able to decrypt back
        decrypted = encryptor.decrypt(
            result.encrypted_data,
            result.nonce,
            result.salt,
            result.tag
        )
        assert decrypted == special_content

    def test_encrypt_large_content(self, encryptor):
        """Test encryption of large content (1MB+)."""
        large_content = "A" * (1024 * 1024)  # 1MB
        result = encryptor.encrypt(large_content)
        
        assert result.encrypted_data
        decrypted = encryptor.decrypt(
            result.encrypted_data,
            result.nonce,
            result.salt,
            result.tag
        )
        assert decrypted == large_content


# ---------------------------------------------------------------------------
# Decryption Tests
# ---------------------------------------------------------------------------


class TestDecryption:
    """Tests for decryption functionality."""

    def test_decrypt_success(self, encryptor, sample_content):
        """Test successful decryption."""
        result = encryptor.encrypt(sample_content)
        decrypted = encryptor.decrypt(
            result.encrypted_data,
            result.nonce,
            result.salt,
            result.tag
        )
        assert decrypted == sample_content

    def test_decrypt_empty_string(self, encryptor):
        """Test decryption of empty string."""
        result = encryptor.encrypt("")
        decrypted = encryptor.decrypt(
            result.encrypted_data,
            result.nonce,
            result.salt,
            result.tag
        )
        assert decrypted == ""

    def test_decrypt_with_wrong_password_fails(self):
        """Test decryption fails with wrong password."""
        encryptor1 = DocumentEncryption(password="password1")
        encryptor2 = DocumentEncryption(password="password2")
        
        result = encryptor1.encrypt("secret content")
        
        with pytest.raises(DecryptionError, match="Authentication tag invalid"):
            encryptor2.decrypt(
                result.encrypted_data,
                result.nonce,
                result.salt,
                result.tag
            )

    def test_decrypt_tampered_data_fails(self, encryptor, sample_content):
        """Test decryption fails with tampered data."""
        result = encryptor.encrypt(sample_content)
        
        # Tamper with encrypted data
        tampered_data = result.encrypted_data[:-10] + "TAMPERED"
        
        with pytest.raises(DecryptionError):
            encryptor.decrypt(
                tampered_data,
                result.nonce,
                result.salt,
                result.tag
            )

    def test_decrypt_invalid_base64_fails(self, encryptor):
        """Test decryption fails with invalid base64."""
        with pytest.raises(DecryptionError):
            encryptor.decrypt(
                "invalid_base64!!!",
                "invalid_nonce",
                "invalid_salt",
                "invalid_tag"
            )


# ---------------------------------------------------------------------------
# Document Integration Tests
# ---------------------------------------------------------------------------


class TestDocumentEncryptionIntegration:
    """Tests for encryption integration with Document class."""

    def test_document_encrypt_content(self, encryptor):
        """Test encrypting document content."""
        doc = Document(
            content="Sensitive data",
            metadata={"source": "test.txt"}
        )
        
        doc.encrypt_content(encryptor)
        
        assert doc.is_encrypted()
        assert doc.encrypted_content
        assert doc.metadata['encrypted']
        assert 'encryption_nonce' in doc.metadata
        assert 'encryption_salt' in doc.metadata
        assert 'encryption_tag' in doc.metadata

    def test_document_decrypt_content(self, encryptor):
        """Test decrypting document content."""
        original_content = "Sensitive data"
        doc = Document(
            content=original_content,
            metadata={"source": "test.txt"}
        )
        
        doc.encrypt_content(encryptor)
        decrypted = doc.decrypt_content(encryptor)
        
        assert decrypted == original_content

    def test_document_decrypt_without_encryption_fails(self, encryptor):
        """Test decryption fails when document is not encrypted."""
        doc = Document(
            content="Not encrypted",
            metadata={"source": "test.txt"}
        )
        
        with pytest.raises(ValueError, match="no encrypted content"):
            doc.decrypt_content(encryptor)

    def test_document_is_encrypted(self, encryptor):
        """Test is_encrypted method."""
        doc = Document(
            content="Data",
            metadata={"source": "test.txt"}
        )
        
        assert not doc.is_encrypted()
        
        doc.encrypt_content(encryptor)
        assert doc.is_encrypted()


# ---------------------------------------------------------------------------
# DocumentLoader Integration Tests
# ---------------------------------------------------------------------------


class TestDocumentLoaderEncryption:
    """Tests for encryption integration with DocumentLoader."""

    def test_load_text_file_with_encryption(self, encryptor, temp_text_file):
        """Test loading text file with encryption."""
        doc = DocumentLoader.load_text_file(temp_text_file, encryptor=encryptor)
        
        assert doc.is_encrypted()
        assert doc.encrypted_content
        assert doc.metadata['encrypted']

    def test_load_text_file_without_encryption(self, temp_text_file):
        """Test loading text file without encryption."""
        doc = DocumentLoader.load_text_file(temp_text_file)
        
        assert not doc.is_encrypted()
        assert doc.content
        assert not doc.encrypted_content

    def test_load_directory_with_encryption(self, encryptor, tmp_path):
        """Test loading directory with encryption enabled."""
        # Create test files
        (tmp_path / "file1.txt").write_text("Secret 1")
        (tmp_path / "file2.txt").write_text("Secret 2")
        
        docs = DocumentLoader.load_directory(str(tmp_path), encryptor=encryptor)
        
        assert len(docs) == 2
        for doc in docs:
            assert doc.is_encrypted()

    def test_roundtrip_encrypted_document(self, encryptor, temp_text_file):
        """Test complete roundtrip: load, encrypt, decrypt, verify."""
        # Load with encryption
        doc = DocumentLoader.load_text_file(temp_text_file, encryptor=encryptor)
        
        # Decrypt
        decrypted_content = doc.decrypt_content(encryptor)
        
        # Verify original content
        assert "Confidential information" in decrypted_content

    def test_multiple_documents_unique_encryption(self, encryptor, tmp_path):
        """Test that multiple documents get unique encryption parameters."""
        (tmp_path / "doc1.txt").write_text("Content 1")
        (tmp_path / "doc2.txt").write_text("Content 2")
        
        docs = DocumentLoader.load_directory(str(tmp_path), encryptor=encryptor)
        
        # Each document should have unique encryption parameters
        nonces = [doc.metadata['encryption_nonce'] for doc in docs]
        salts = [doc.metadata['encryption_salt'] for doc in docs]
        
        assert len(set(nonces)) == len(nonces)  # All unique
        assert len(set(salts)) == len(salts)  # All unique


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestEncryptionErrorHandling:
    """Tests for error handling in encryption operations."""

    def test_encrypt_none_value(self, encryptor):
        """Test that encrypting None value is handled gracefully."""
        # None is treated as falsy and handled by empty string check
        result = encryptor.encrypt(None)  # type: ignore
        # Should return empty result for None input
        assert result.encrypted_data == ""

    def test_key_derivation_failure_propagates(self, encryption_password):
        """Test that key derivation failures are properly reported."""
        encryptor = DocumentEncryption(password=encryption_password)
        
        # Mock _derive_key to raise an exception
        with patch.object(encryptor, '_derive_key', side_effect=Exception("KDF failed")):
            with pytest.raises(EncryptionError):
                encryptor.encrypt("test content")

    def test_encryption_failure_logs_error(self, encryptor):
        """Test that encryption failures are logged."""
        with patch('app.core.encryption.logger') as mock_logger:
            # Mock AESGCM to raise an exception
            with patch('app.core.encryption.AESGCM', side_effect=Exception("Crypto failed")):
                with pytest.raises(EncryptionError):
                    encryptor.encrypt("test")
            
            # Verify error was logged
            mock_logger.error.assert_called()

    def test_decryption_failure_logs_error(self, encryptor):
        """Test that decryption failures are logged."""
        with patch('app.core.encryption.logger') as mock_logger:
            with pytest.raises(DecryptionError):
                encryptor.decrypt("invalid", "invalid", "invalid", "invalid")
            
            # Verify error was logged
            mock_logger.error.assert_called()
