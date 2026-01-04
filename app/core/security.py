"""
Security validation utilities for detecting and preventing common attacks.

This module provides validators for:
- SQL injection patterns
- XSS (Cross-Site Scripting) patterns
- Path traversal attempts
- Input sanitization
"""

import re
from typing import Optional


class SecurityValidator:
    """
    Security validator for detecting malicious input patterns.

    Provides static methods to detect common attack patterns including
    SQL injection, XSS, and path traversal attacks.
    """

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bor\b.*1\s*=\s*1)",
        r"(\band\b.*1\s*=\s*1)",
        r"('\s+or\s+')",  # ' OR ' pattern
        r"('\s+and\s+')",  # ' AND ' pattern
        r"(\bdrop\b.*\btable\b)",
        r"(;.*\bexec\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(--)",  # SQL comment
        r"(/\*.*\*/)",  # SQL comment
        r"(\bor\b\s+\d+\s*=\s*\d+)",
        r"(\band\b\s+\d+\s*=\s*\d+)",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onerror\s*=",
        r"onload\s*=",
        r"onclick\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
        r"onmouseover\s*=",
        r"onfocus\s*=",
        r"onblur\s*=",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\" ,
        r"%2e%2e",
        r"..%2f",
        r"..%5c",
        r"%252e",
    ]

    @staticmethod
    def detect_sql_injection(text: str) -> bool:
        """
        Detect SQL injection patterns in input text.

        Args:
            text: Input string to validate

        Returns:
            True if SQL injection pattern is detected, False otherwise
        """
        if not isinstance(text, str):
            return False

        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in SecurityValidator.SQL_INJECTION_PATTERNS)

    @staticmethod
    def detect_xss(text: str) -> bool:
        """
        Detect XSS (Cross-Site Scripting) patterns in input text.

        Args:
            text: Input string to validate

        Returns:
            True if XSS pattern is detected, False otherwise
        """
        if not isinstance(text, str):
            return False

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in SecurityValidator.XSS_PATTERNS)

    @staticmethod
    def detect_path_traversal(text: str) -> bool:
        """
        Detect path traversal patterns in input text.

        Args:
            text: Input string to validate

        Returns:
            True if path traversal pattern is detected, False otherwise
        """
        if not isinstance(text, str):
            return False

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in SecurityValidator.PATH_TRAVERSAL_PATTERNS)

    @staticmethod
    def sanitize_input(text: str, max_length: int = 10000) -> str:
        """
        Sanitize input by checking length and stripping whitespace.

        Args:
            text: Input string to sanitize
            max_length: Maximum allowed length (default: 10000)

        Returns:
            Sanitized string

        Raises:
            ValueError: If input exceeds maximum length
        """
        if not isinstance(text, str):
            return text

        if len(text) > max_length:
            raise ValueError(f"Input too long (max {max_length} chars, got {len(text)})")

        return text.strip()

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format.

        Args:
            email: Email address to validate

        Returns:
            True if email format is valid, False otherwise
        """
        if not isinstance(email, str):
            return False

        # Basic email regex pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None

    @staticmethod
    def detect_command_injection(text: str) -> bool:
        """
        Detect command injection patterns in input text.

        Args:
            text: Input string to validate

        Returns:
            True if command injection pattern is detected, False otherwise
        """
        if not isinstance(text, str):
            return False

        command_injection_patterns = [
            r";\s*\w+",  # semicolon followed by command
            r"\|",  # pipe
            r"`",  # backtick
            r"\$\(",  # command substitution
            r">\s*/",  # output redirection
            r"<\s*/",  # input redirection
        ]

        return any(re.search(pattern, text) for pattern in command_injection_patterns)
