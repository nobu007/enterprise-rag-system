"""
Tests for API key authentication
"""

import pytest
from datetime import datetime, timedelta
from app.models.api_key import APIKey, APIKeyRole
from app.core.auth import APIKeyAuth


class TestAPIKeyModel:
    """Test APIKey model"""

    def test_create_api_key(self):
        """Test creating a new API key"""
        api_key = APIKey(
            key_id="test_key_123",
            key_value="sk_test_secret_hash",
            name="Test Key",
            role=APIKeyRole.USER,
            created_by="admin"
        )

        assert api_key.key_id == "test_key_123"
        assert api_key.name == "Test Key"
        assert api_key.role == APIKeyRole.USER
        assert api_key.is_active is True
        assert api_key.created_at is not None

    def test_api_key_is_expired(self):
        """Test API key expiration"""
        # Expired key
        expired_key = APIKey(
            key_id="expired_key",
            key_value="secret",
            name="Expired Key",
            role=APIKeyRole.USER,
            created_by="admin",
            expires_at=datetime.utcnow() - timedelta(days=1)
        )

        assert expired_key.is_expired() is True

        # Non-expired key
        valid_key = APIKey(
            key_id="valid_key",
            key_value="secret",
            name="Valid Key",
            role=APIKeyRole.USER,
            created_by="admin",
            expires_at=datetime.utcnow() + timedelta(days=30)
        )

        assert valid_key.is_expired() is False

        # Key without expiration
        no_expire_key = APIKey(
            key_id="no_expire_key",
            key_value="secret",
            name="No Expire Key",
            role=APIKeyRole.USER,
            created_by="admin"
        )

        assert no_expire_key.is_expired() is False

    def test_api_key_is_revoked(self):
        """Test API key revocation"""
        api_key = APIKey(
            key_id="test_key",
            key_value="secret",
            name="Test Key",
            role=APIKeyRole.USER,
            created_by="admin"
        )

        assert api_key.is_revoked() is False

        api_key.revoke()
        assert api_key.is_revoked() is True
        assert api_key.is_active is False


class TestAPIKeyAuth:
    """Test API key authentication"""

    @pytest.fixture
    def auth_manager(self):
        """Create an APIKeyAuth instance for testing"""
        return APIKeyAuth()

    def test_create_api_key(self, auth_manager):
        """Test creating a new API key"""
        api_key = auth_manager.create_api_key(
            name="Test Key",
            role=APIKeyRole.USER,
            created_by="admin",
            expires_in_days=30
        )

        assert api_key.key_id is not None
        assert api_key.key_value is not None
        assert api_key.key_value.startswith("sk_")
        assert api_key.name == "Test Key"
        assert api_key.role == APIKeyRole.USER
        assert api_key.is_active is True
        assert api_key.expires_at is not None

    def test_verify_api_key_success(self, auth_manager):
        """Test successful API key verification"""
        # Create and store an API key
        api_key = auth_manager.create_api_key(
            name="Valid Key",
            role=APIKeyRole.USER,
            created_by="admin"
        )

        # Verify the key
        verified_key = auth_manager.verify_api_key(api_key.key_value)

        assert verified_key is not None
        assert verified_key.key_id == api_key.key_id
        assert verified_key.name == "Valid Key"

    def test_verify_api_key_invalid(self, auth_manager):
        """Test verification of invalid API key"""
        verified_key = auth_manager.verify_api_key("invalid_key_value")

        assert verified_key is None

    def test_verify_api_key_revoked(self, auth_manager):
        """Test verification of revoked API key"""
        # Create and revoke an API key
        api_key = auth_manager.create_api_key(
            name="Revoked Key",
            role=APIKeyRole.USER,
            created_by="admin"
        )

        api_key.revoke()

        # Try to verify the revoked key
        verified_key = auth_manager.verify_api_key(api_key.key_value)

        assert verified_key is None

    def test_verify_api_key_expired(self, auth_manager):
        """Test verification of expired API key"""
        # Create an API key and manually set it as expired
        api_key = auth_manager.create_api_key(
            name="Expired Key",
            role=APIKeyRole.USER,
            created_by="admin"
        )
        # Manually expire the key
        api_key.expires_at = datetime.utcnow() - timedelta(days=1)

        # Try to verify the expired key
        verified_key = auth_manager.verify_api_key(api_key.key_value)

        assert verified_key is None

    def test_revoke_api_key(self, auth_manager):
        """Test revoking an API key"""
        api_key = auth_manager.create_api_key(
            name="Revoke Test Key",
            role=APIKeyRole.USER,
            created_by="admin"
        )

        # Revoke the key
        result = auth_manager.revoke_api_key(api_key.key_id)

        assert result is True

        # Verify it's revoked
        verified_key = auth_manager.verify_api_key(api_key.key_value)
        assert verified_key is None

    def test_list_api_keys(self, auth_manager):
        """Test listing API keys"""
        # Create multiple keys
        auth_manager.create_api_key(
            name="Key 1",
            role=APIKeyRole.USER,
            created_by="admin"
        )
        auth_manager.create_api_key(
            name="Key 2",
            role=APIKeyRole.ADMIN,
            created_by="admin"
        )

        # List all keys
        keys = auth_manager.list_api_keys()

        assert len(keys) >= 2

    def test_get_api_key_by_id(self, auth_manager):
        """Test getting API key by ID"""
        api_key = auth_manager.create_api_key(
            name="Get Test Key",
            role=APIKeyRole.USER,
            created_by="admin"
        )

        # Get the key by ID
        retrieved_key = auth_manager.get_api_key(api_key.key_id)

        assert retrieved_key is not None
        assert retrieved_key.key_id == api_key.key_id
        assert retrieved_key.name == "Get Test Key"
