"""
Regression tests for DocumentEncryption.encrypt_document (Feature 49 / MS-003)

encrypt_document previously computed the EncryptionResult and discarded it,
returning metadata that advertised ``encrypted=True`` while carrying no
ciphertext, nonce, salt or tag. Any document "encrypted" through that path was
silently unrecoverable.

These tests pin the fixed contract:
- the encrypted payload is persisted in the returned metadata,
- the payload round-trips back to the original plaintext via decrypt(),
- the metadata key names match Document.encrypt_content / decrypt_content
  (app/services/document_loader.py) so both encryption paths share one schema,
- the caller's metadata dict is never mutated.
"""

import copy

import pytest

from app.core.encryption import DocumentEncryption, EncryptionResult


@pytest.fixture
def encryptor():
    """Create a real DocumentEncryption instance (AES-256-GCM)."""
    return DocumentEncryption(password="test_secure_password_123")


@pytest.fixture
def stub_encryptor():
    """DocumentEncryption whose encrypt() is stubbed with a fixed result.

    __init__ is bypassed so the metadata-propagation logic can be exercised
    without depending on the cryptography backend.
    """
    enc = DocumentEncryption.__new__(DocumentEncryption)
    enc.password = "test_secure_password_123"
    enc.encrypt = lambda content: EncryptionResult(
        encrypted_data="BASE64_CIPHERTEXT",
        nonce="BASE64_NONCE",
        salt="BASE64_SALT",
        tag="BASE64_TAG",
    )
    return enc


class TestEncryptDocumentPayloadPersistence:
    """The encrypted payload must survive in the returned metadata."""

    def test_payload_is_persisted_in_metadata(self, stub_encryptor):
        """Regression: the EncryptionResult must not be discarded."""
        metadata = {"source": "policy.pdf", "collection": "hr"}

        result_meta = stub_encryptor.encrypt_document("sensitive", metadata)

        assert result_meta["encrypted_content"] == "BASE64_CIPHERTEXT"
        assert result_meta["encryption_nonce"] == "BASE64_NONCE"
        assert result_meta["encryption_salt"] == "BASE64_SALT"
        assert result_meta["encryption_tag"] == "BASE64_TAG"

    def test_encryption_status_flags_are_preserved(self, stub_encryptor):
        """Existing status metadata must keep working."""
        result_meta = stub_encryptor.encrypt_document("sensitive", {})

        assert result_meta["encrypted"] is True
        assert result_meta["encryption_version"] == "1.0"
        assert result_meta["encryption_algorithm"] == "AES-256-GCM"
        assert stub_encryptor.is_encrypted(result_meta) is True

    def test_original_metadata_is_preserved(self, stub_encryptor):
        """Caller-supplied metadata keys must survive encryption."""
        result_meta = stub_encryptor.encrypt_document(
            "sensitive", {"source": "policy.pdf", "page": 3}
        )

        assert result_meta["source"] == "policy.pdf"
        assert result_meta["page"] == 3

    def test_caller_metadata_is_not_mutated(self, stub_encryptor):
        """encrypt_document must not modify the dict it was given."""
        original = {"source": "policy.pdf"}
        snapshot = copy.deepcopy(original)

        stub_encryptor.encrypt_document("sensitive", original)

        assert original == snapshot

    def test_content_is_forwarded_to_encrypt(self):
        """The plaintext must reach encrypt() unchanged."""
        enc = DocumentEncryption.__new__(DocumentEncryption)
        seen = {}

        def spy(content):
            seen["content"] = content
            return EncryptionResult("ct", "n", "s", "t")

        enc.encrypt = spy
        enc.encrypt_document("the real plaintext", {})

        assert seen["content"] == "the real plaintext"


class TestEncryptDocumentRoundTrip:
    """The persisted payload must actually decrypt back to the plaintext."""

    def test_metadata_round_trips_to_original_content(self, encryptor):
        """End-to-end proof that documents stay recoverable."""
        content = "Confidential quarterly figures."

        meta = encryptor.encrypt_document(content, {"source": "q3.txt"})
        decrypted = encryptor.decrypt(
            meta["encrypted_content"],
            meta["encryption_nonce"],
            meta["encryption_salt"],
            meta["encryption_tag"],
        )

        assert decrypted == content
        assert meta["encrypted_content"] != content

    def test_wrong_password_cannot_decrypt(self, encryptor):
        """The payload stays protected by the derived key."""
        from app.core.encryption import DecryptionError

        meta = encryptor.encrypt_document("secret", {})
        other = DocumentEncryption(password="a_completely_different_password")

        with pytest.raises(DecryptionError):
            other.decrypt(
                meta["encrypted_content"],
                meta["encryption_nonce"],
                meta["encryption_salt"],
                meta["encryption_tag"],
            )

    def test_each_call_uses_fresh_nonce_and_salt(self, encryptor):
        """Encrypting the same content twice must not repeat nonce/salt."""
        first = encryptor.encrypt_document("same content", {})
        second = encryptor.encrypt_document("same content", {})

        assert first["encryption_nonce"] != second["encryption_nonce"]
        assert first["encryption_salt"] != second["encryption_salt"]
        assert first["encrypted_content"] != second["encrypted_content"]


class TestEncryptDocumentSchemaConsistency:
    """Both encryption paths must expose the same metadata schema."""

    def test_keys_match_document_encrypt_content(self, encryptor):
        """Document.encrypt_content and encrypt_document must agree.

        Document.decrypt_content reads encryption_nonce/salt/tag from
        metadata; encrypt_document must use those same names so a document
        encrypted through either path is decryptable by the same reader.
        """
        from app.services.document_loader import Document

        doc = Document(content="Shared schema check", metadata={"source": "s.txt"})
        doc.encrypt_content(encryptor)

        meta = encryptor.encrypt_document("Shared schema check", {"source": "s.txt"})

        payload_keys = {"encryption_nonce", "encryption_salt", "encryption_tag"}
        assert payload_keys <= set(doc.metadata)
        assert payload_keys <= set(meta)
        assert meta["encrypted"] == doc.metadata["encrypted"]
