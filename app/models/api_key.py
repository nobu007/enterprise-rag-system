"""
API Key Model

This module defines the APIKey model and related enums for API key management.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from secrets import token_hex
import hashlib


class APIKeyRole(str, Enum):
    """API Key Roles"""

    ADMIN = "admin"
    USER = "user"
    READ_ONLY = "read_only"


class APIKey:
    """
    API Key Model

    Represents an API key with metadata for authentication and authorization.
    """

    def __init__(
        self,
        key_id: str,
        key_value: str,
        name: str,
        role: APIKeyRole,
        created_by: str,
        is_active: bool = True,
        expires_at: Optional[datetime] = None,
        last_used_at: Optional[datetime] = None,
        revoked_at: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
    ):
        """
        Initialize an APIKey instance.

        Args:
            key_id: Unique identifier for the key
            key_value: The actual API key value (hashed)
            name: Human-readable name for the key
            role: Access role/permissions
            created_by: User who created the key
            is_active: Whether the key is active
            expires_at: Optional expiration timestamp
            last_used_at: Last time the key was used
            revoked_at: When the key was revoked
            created_at: When the key was created
        """
        self.key_id = key_id
        self.key_value = key_value
        self.name = name
        self.role = role
        self.created_by = created_by
        self.is_active = is_active
        self.expires_at = expires_at
        self.last_used_at = last_used_at
        self.revoked_at = revoked_at
        self.created_at = created_at or datetime.utcnow()

    def is_expired(self) -> bool:
        """
        Check if the API key is expired.

        Returns:
            True if expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_revoked(self) -> bool:
        """
        Check if the API key is revoked.

        Returns:
            True if revoked, False otherwise
        """
        return self.revoked_at is not None

    def is_valid(self) -> bool:
        """
        Check if the API key is valid (active, not expired, not revoked).

        Returns:
            True if valid, False otherwise
        """
        return self.is_active and not self.is_expired() and not self.is_revoked()

    def revoke(self):
        """Revoke the API key"""
        self.is_active = False
        self.revoked_at = datetime.utcnow()

    def update_last_used(self):
        """Update the last used timestamp"""
        self.last_used_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """
        Convert API key to dictionary (excluding sensitive data).

        Returns:
            Dictionary representation of the key
        """
        return {
            "key_id": self.key_id,
            "name": self.name,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
        }

    @staticmethod
    def generate_key_value() -> str:
        """
        Generate a new API key value.

        Returns:
            New API key value starting with 'sk_'
        """
        random_part = token_hex(16)
        return f"sk_{random_part}"

    @staticmethod
    def hash_key(key_value: str) -> str:
        """
        Hash an API key value for storage.

        Args:
            key_value: The raw API key value

        Returns:
            Hashed key value
        """
        return hashlib.sha256(key_value.encode()).hexdigest()
