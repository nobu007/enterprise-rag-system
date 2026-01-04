"""
API Key Authentication

This module provides API key authentication functionality for FastAPI applications.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
import secrets

from app.models.api_key import APIKey, APIKeyRole
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class APIKeyAuth:
    """
    API Key Authentication Manager

    Manages API key creation, verification, and revocation.
    Stores keys in memory (can be extended to use a database).
    """

    def __init__(self):
        """Initialize the API key authentication manager"""
        # In-memory storage (use a database in production)
        self._keys: Dict[str, APIKey] = {}  # key_id -> APIKey
        self._key_hash_index: Dict[str, str] = {}  # hashed_value -> key_id

        logger.info("API Key Authentication initialized")

    def create_api_key(
        self,
        name: str,
        role: APIKeyRole,
        created_by: str,
        expires_in_days: Optional[int] = None,
    ) -> APIKey:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key
            role: Access role/permissions
            created_by: User who is creating the key
            expires_in_days: Optional expiration in days (None = no expiration)

        Returns:
            Created APIKey instance

        Raises:
            ValueError: If parameters are invalid
        """
        if not name or not name.strip():
            raise ValueError("API key name cannot be empty")

        if not created_by or not created_by.strip():
            raise ValueError("Creator cannot be empty")

        # Generate unique key ID
        key_id = secrets.token_hex(8)

        # Generate and hash the key value
        key_value = APIKey.generate_key_value()
        hashed_value = APIKey.hash_key(key_value)

        # Calculate expiration
        expires_at = None
        if expires_in_days is not None:
            if expires_in_days <= 0:
                raise ValueError("Expiration days must be positive")
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create API key instance
        api_key = APIKey(
            key_id=key_id,
            key_value=hashed_value,  # Store hashed value
            name=name.strip(),
            role=role,
            created_by=created_by.strip(),
            is_active=True,
            expires_at=expires_at,
        )

        # Store the key
        self._keys[key_id] = api_key
        self._key_hash_index[hashed_value] = key_id

        logger.info(
            f"Created API key: {key_id} (name={name}, role={role.value}, "
            f"expires={expires_at})"
        )

        # Return the key with the raw value (only time it's visible)
        api_key.key_value = key_value
        return api_key

    def verify_api_key(self, key_value: str) -> Optional[APIKey]:
        """
        Verify an API key.

        Args:
            key_value: The API key value to verify

        Returns:
            APIKey instance if valid, None otherwise
        """
        if not key_value:
            return None

        # Hash the provided key value
        hashed_value = APIKey.hash_key(key_value)

        # Look up the key
        key_id = self._key_hash_index.get(hashed_value)
        if not key_id:
            logger.warning(f"API key not found: {hashed_value[:10]}...")
            return None

        # Get the key instance
        api_key = self._keys.get(key_id)
        if not api_key:
            logger.warning(f"API key instance not found: {key_id}")
            return None

        # Check if key is valid
        if not api_key.is_valid():
            logger.warning(
                f"API key is not valid: {key_id} "
                f"(active={api_key.is_active}, "
                f"expired={api_key.is_expired()}, "
                f"revoked={api_key.is_revoked()})"
            )
            return None

        # Update last used timestamp
        api_key.update_last_used()

        logger.debug(f"API key verified successfully: {key_id}")
        return api_key

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: The ID of the key to revoke

        Returns:
            True if revoked successfully, False otherwise
        """
        api_key = self._keys.get(key_id)
        if not api_key:
            logger.warning(f"Cannot revoke non-existent key: {key_id}")
            return False

        api_key.revoke()
        logger.info(f"Revoked API key: {key_id}")
        return True

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """
        Get an API key by ID.

        Args:
            key_id: The ID of the key

        Returns:
            APIKey instance if found, None otherwise
        """
        return self._keys.get(key_id)

    def list_api_keys(self, include_revoked: bool = False) -> list[APIKey]:
        """
        List all API keys.

        Args:
            include_revoked: Whether to include revoked keys

        Returns:
            List of APIKey instances
        """
        keys = list(self._keys.values())

        if not include_revoked:
            keys = [key for key in keys if not key.is_revoked()]

        return keys

    def delete_api_key(self, key_id: str) -> bool:
        """
        Delete an API key permanently.

        Args:
            key_id: The ID of the key to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        api_key = self._keys.get(key_id)
        if not api_key:
            logger.warning(f"Cannot delete non-existent key: {key_id}")
            return False

        # Remove from hash index
        hashed_value = api_key.key_value
        if hashed_value in self._key_hash_index:
            del self._key_hash_index[hashed_value]

        # Remove from storage
        del self._keys[key_id]

        logger.info(f"Deleted API key: {key_id}")
        return True


# Global instance for use in dependency injection
_auth_manager: Optional[APIKeyAuth] = None


def get_auth_manager() -> APIKeyAuth:
    """
    Get the global API key authentication manager.

    Returns:
        APIKeyAuth instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = APIKeyAuth()
    return _auth_manager
