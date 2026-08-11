"""Coverage-focused tests for app/core/encryption.py.

The existing tests (test_feature_49.py) cover the
CRYPTOGRAPHY_AVAILABLE guard at construction and the *caller's*
handling of a _derive_key failure by MOCKING _derive_key itself
(patch.object(encryptor, '_derive_key', side_effect=...)). Mocking
the method means its own body never runs, so its `except Exception
-> KeyDerivationError` wrap (encryption.py L136-138) stays uncovered.

To cover that branch the REAL _derive_key body must execute and fail:
patching the module-level PBKDF2HMAC makes `kdf = PBKDF2HMAC(...)`
raise inside the try, exercising the wrap.
"""
from unittest.mock import Mock

import pytest

from app.core.encryption import DocumentEncryption, KeyDerivationError


class TestDeriveKeyFailure:
    """Cover _derive_key's except branch (L136-138)."""

    def test_derive_key_failure_wraps_as_key_derivation_error(
        self, monkeypatch
    ):
        # PBKDF2HMAC(...) is constructed inside _derive_key's try block;
        # making it raise forces the real method into its except branch.
        monkeypatch.setattr(
            "app.core.encryption.PBKDF2HMAC",
            Mock(side_effect=Exception("kdf construction boom")),
        )
        encryptor = DocumentEncryption(password="any-password")

        # Call _derive_key directly: encrypt() would re-wrap the
        # KeyDerivationError as EncryptionError (its own except), so
        # direct invocation asserts this method's own wrap faithfully.
        with pytest.raises(KeyDerivationError, match="Failed to derive key"):
            encryptor._derive_key(b"some-salt")
