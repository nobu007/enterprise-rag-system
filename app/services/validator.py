"""
Document Validation Service

This module provides comprehensive document validation before ingestion,
including content quality checks, security validation, PII detection,
and format-specific validation.

Key features:
- Content quality validation (length, encoding, emptiness)
- Security checks (XSS, SQL injection, path traversal, command injection)
- PII detection (email, phone, SSN, credit card)
- Format-specific validation (PDF, Markdown, text)
- Metadata completeness checks
- Batch validation support
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from app.core.logging_config import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Classes and Enums
# ---------------------------------------------------------------------------


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """
    Represents a validation error or warning.

    Attributes:
        code: Unique error code (e.g., "EMPTY_CONTENT", "SECURITY_XSS")
        message: Human-readable error message
        severity: Error severity level (error/warning/info)
        field: Optional field name that caused the error
    """
    code: str
    message: str
    severity: str = "error"
    field: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


@dataclass
class ValidationResult:
    """
    Result of document validation.

    Attributes:
        is_valid: Whether the document passed validation
        errors: List of validation errors (severity="error")
        warnings: List of validation warnings (severity="warning")
        metadata: Additional metadata about the validation
    """
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, code: str, message: str, field: Optional[str] = None):
        """Add an error to the result."""
        self.errors.append(ValidationError(code, message, "error", field))
        self.is_valid = False

    def add_warning(self, message: str):
        """Add a warning to the result."""
        self.warnings.append(message)


# ---------------------------------------------------------------------------
# Main Validator Class
# ---------------------------------------------------------------------------


class DocumentValidator:
    """
    Comprehensive document validator for RAG ingestion pipeline.

    Features:
    - Content quality checks (length, encoding, emptiness)
    - Security validation (XSS, SQL injection, etc.)
    - PII detection (email, phone, SSN)
    - Format-specific validation
    - Metadata completeness checks
    - Batch validation support
    """

    # Security patterns
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
    ]

    SQL_INJECTION_PATTERNS = [
        r"(';.*--)|(\bor\b\s+\d+\s*=\s*\d+)|(\bunion\b.*\bselect\b)",
        r"(;.*\bdrop\b)|(;\bdelete\b)|(;\btruncate\b)",
        r"(\bexec\b)|(execute\s*\()",
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./',
        r'\.\.\\',
        r'%2e%2e',
    ]

    COMMAND_INJECTION_PATTERNS = [
        r';\s*\w+\s+',  # Command followed by semicolon
        r'\|\s*\w+',    # Pipe to command
        r'`.*`',        # Backtick execution
        r'\$\(.*\)',    # Command substitution
    ]

    # PII patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
    CREDIT_CARD_PATTERN = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'

    # Supported formats
    SUPPORTED_FORMATS = ['txt', 'md', 'markdown', 'pdf', 'html']

    def __init__(
        self,
        min_content_length: int = 50,
        max_content_length: int = 10 * 1024 * 1024,  # 10MB
        enable_security_check: bool = True,
        enable_pii_detection: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize DocumentValidator.

        Args:
            min_content_length: Minimum content length in characters
            max_content_length: Maximum content length in characters
            enable_security_check: Enable security pattern detection
            enable_pii_detection: Enable PII detection
            strict_mode: If True, warnings become errors
        """
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.enable_security_check = enable_security_check
        self.enable_pii_detection = enable_pii_detection
        self.strict_mode = strict_mode

        # Compile regex patterns for performance
        self._compile_patterns()

        logger.info(
            f"DocumentValidator initialized: "
            f"min_length={min_content_length}, max_length={max_content_length}, "
            f"security_check={enable_security_check}, pii_detection={enable_pii_detection}"
        )

    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.xss_regex = re.compile(
            '|'.join(self.XSS_PATTERNS),
            re.IGNORECASE | re.DOTALL
        )
        self.sql_regex = re.compile(
            '|'.join(self.SQL_INJECTION_PATTERNS),
            re.IGNORECASE | re.DOTALL
        )
        self.path_regex = re.compile(
            '|'.join(self.PATH_TRAVERSAL_PATTERNS),
            re.IGNORECASE
        )
        self.command_regex = re.compile(
            '|'.join(self.COMMAND_INJECTION_PATTERNS),
            re.IGNORECASE
        )
        self.email_regex = re.compile(self.EMAIL_PATTERN)
        self.phone_regex = re.compile(self.PHONE_PATTERN)
        self.ssn_regex = re.compile(self.SSN_PATTERN)
        self.credit_card_regex = re.compile(self.CREDIT_CARD_PATTERN)

    # -----------------------------------------------------------------------
    # Main Validation Methods
    # -----------------------------------------------------------------------

    def validate(self, document) -> ValidationResult:
        """
        Perform complete validation of a document.

        This runs all validation checks in the following order:
        1. Content validation
        2. Security validation
        3. PII detection
        4. Format validation
        5. Metadata validation

        Args:
            document: Document object to validate

        Returns:
            ValidationResult with all errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # Run all validation steps
        self._validate_content(document, result)
        self._validate_security(document, result)
        self._validate_pii(document, result)
        self._validate_format(document, result)
        self._validate_metadata(document, result)

        # In strict mode, warnings become errors
        if self.strict_mode and result.warnings:
            for warning in result.warnings:
                result.add_error("STRICT_MODE_WARNING", warning)

        logger.debug(
            f"Validation complete: is_valid={result.is_valid}, "
            f"errors={len(result.errors)}, warnings={len(result.warnings)}"
        )

        return result

    def validate_batch(self, documents: List) -> List[ValidationResult]:
        """
        Validate multiple documents in batch.

        Args:
            documents: List of Document objects

        Returns:
            List of ValidationResult objects (same order as input)
        """
        results = []
        for doc in documents:
            result = self.validate(doc)
            results.append(result)

        logger.info(
            f"Batch validation complete: {len(results)} documents, "
            f"{sum(1 for r in results if r.is_valid)} valid"
        )

        return results

    # -----------------------------------------------------------------------
    # Individual Validation Methods (Public API)
    # -----------------------------------------------------------------------

    def validate_content(self, document) -> ValidationResult:
        """Validate document content only."""
        result = ValidationResult(is_valid=True)
        self._validate_content(document, result)
        return result

    def validate_security(self, document) -> ValidationResult:
        """Validate security patterns only."""
        result = ValidationResult(is_valid=True)
        self._validate_security(document, result)
        return result

    def validate_pii(self, document) -> ValidationResult:
        """Validate PII patterns only."""
        result = ValidationResult(is_valid=True)
        self._validate_pii(document, result)
        return result

    def validate_format(self, document) -> ValidationResult:
        """Validate document format only."""
        result = ValidationResult(is_valid=True)
        self._validate_format(document, result)
        return result

    def validate_metadata(self, document) -> ValidationResult:
        """Validate metadata only."""
        result = ValidationResult(is_valid=True)
        self._validate_metadata(document, result)
        return result

    # -----------------------------------------------------------------------
    # Private Validation Methods
    # -----------------------------------------------------------------------

    def _validate_content(self, document, result: ValidationResult):
        """Validate document content quality."""
        content = document.content

        # Check for None or non-string content
        if content is None:
            result.add_error("EMPTY_CONTENT", "Document content is None")
            return

        if not isinstance(content, str):
            result.add_error(
                "INVALID_CONTENT_TYPE",
                f"Content must be string, got {type(content).__name__}"
            )
            return

        # Check for empty content
        stripped_content = content.strip()
        if not stripped_content:
            result.add_error("EMPTY_CONTENT", "Document content is empty")
            return

        # Check minimum length
        content_length = len(stripped_content)
        if content_length < self.min_content_length:
            result.add_error(
                "CONTENT_TOO_SHORT",
                f"Content length ({content_length}) is below minimum ({self.min_content_length})"
            )

        # Check maximum length
        if content_length > self.max_content_length:
            result.add_error(
                "CONTENT_TOO_LONG",
                f"Content length ({content_length}) exceeds maximum ({self.max_content_length})"
            )

        # Check for encoding issues (non-UTF-8 characters)
        try:
            content.encode('utf-8')
        except UnicodeEncodeError as e:
            result.add_error(
                "ENCODING_ERROR",
                f"Content contains invalid characters: {e}"
            )

    def _validate_security(self, document, result: ValidationResult):
        """Validate security patterns."""
        if not self.enable_security_check:
            return

        content = document.content
        if not content:
            return

        # Check XSS patterns
        xss_matches = self.xss_regex.findall(content)
        if xss_matches:
            result.add_error(
                "SECURITY_XSS",
                f"Potential XSS attack pattern detected: {xss_matches[0][:50]}..."
            )

        # Check SQL injection patterns
        sql_matches = self.sql_regex.findall(content)
        if sql_matches:
            result.add_error(
                "SECURITY_SQL_INJECTION",
                f"SQL injection pattern detected"
            )

        # Check path traversal patterns
        path_matches = self.path_regex.findall(content)
        if path_matches:
            result.add_error(
                "SECURITY_PATH_TRAVERSAL",
                f"Path traversal pattern detected"
            )

        # Check command injection patterns
        command_matches = self.command_regex.findall(content)
        if command_matches:
            result.add_error(
                "SECURITY_COMMAND_INJECTION",
                f"Command injection pattern detected"
            )

    def _validate_pii(self, document, result: ValidationResult):
        """Validate PII (Personally Identifiable Information) patterns."""
        if not self.enable_pii_detection:
            return

        content = document.content
        if not content:
            return

        # Check email addresses
        emails = self.email_regex.findall(content)
        if emails:
            result.add_warning(f"PII_DETECTION: Found {len(emails)} email address(es)")
            # In non-strict mode, PII is a warning
            # Optionally add as error in strict mode
            if self.strict_mode:
                result.add_error(
                    "PII_EMAIL",
                    f"Email addresses detected: {emails[0]}"
                )

        # Check phone numbers
        phones = self.phone_regex.findall(content)
        if phones:
            result.add_warning(f"PII_DETECTION: Found {len(phones)} phone number(s)")
            if self.strict_mode:
                result.add_error(
                    "PII_PHONE",
                    f"Phone numbers detected: {phones[0]}"
                )

        # Check SSN
        ssns = self.ssn_regex.findall(content)
        if ssns:
            result.add_warning("PII_DETECTION: Found SSN pattern(s)")
            if self.strict_mode:
                result.add_error(
                    "PII_SSN",
                    f"SSN patterns detected"
                )

        # Check credit cards
        cards = self.credit_card_regex.findall(content)
        if cards:
            result.add_warning("PII_DETECTION: Found credit card pattern(s)")
            if self.strict_mode:
                result.add_error(
                    "PII_CREDIT_CARD",
                    f"Credit card patterns detected"
                )

    def _validate_format(self, document, result: ValidationResult):
        """Validate document format."""
        metadata = document.metadata
        file_type = metadata.get('file_type', '').lower()

        # Check if format is supported
        if file_type and file_type not in self.SUPPORTED_FORMATS:
            result.add_error(
                "UNSUPPORTED_FORMAT",
                f"Unsupported file format: {file_type}. Supported: {self.SUPPORTED_FORMATS}"
            )

        # Format-specific validation
        if file_type == 'markdown' or file_type == 'md':
            self._validate_markdown(document, result)
        elif file_type == 'pdf':
            self._validate_pdf(document, result)

    def _validate_markdown(self, document, result: ValidationResult):
        """Validate markdown-specific content."""
        content = document.content

        # Check for dangerous links (javascript:, data:, etc.)
        dangerous_links = re.findall(
            r'\[.*?\]\((javascript:|data:|vbscript:)',
            content,
            re.IGNORECASE
        )
        if dangerous_links:
            result.add_error(
                "MARKDOWN_DANGEROUS_LINK",
                f"Markdown contains dangerous link: {dangerous_links[0]}"
            )

    def _validate_pdf(self, document, result: ValidationResult):
        """Validate PDF-specific metadata."""
        metadata = document.metadata

        # Check if page information is present
        if 'page' not in metadata:
            result.add_warning("PDF metadata missing page number")

        # Validate page number
        if 'page' in metadata:
            page = metadata['page']
            if not isinstance(page, int) or page < 1:
                result.add_error(
                    "INVALID_METADATA",
                    f"Invalid page number: {page}"
                )

    def _validate_metadata(self, document, result: ValidationResult):
        """Validate metadata completeness."""
        metadata = document.metadata

        # Check if metadata exists
        if not metadata:
            result.add_error(
                "MISSING_METADATA",
                "Document has no metadata"
            )
            return

        # Check for required fields
        if 'source' not in metadata:
            result.add_error(
                "MISSING_METADATA",
                "Metadata missing required field: 'source'"
            )

        # Warn about optional but recommended fields
        if 'file_type' not in metadata:
            result.add_warning("Metadata missing recommended field: 'file_type'")

        if 'filename' not in metadata:
            result.add_warning("Metadata missing recommended field: 'filename'")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def create_default_validator() -> DocumentValidator:
    """Create a validator with default settings."""
    return DocumentValidator()


def create_strict_validator() -> DocumentValidator:
    """Create a validator with strict settings (warnings become errors)."""
    return DocumentValidator(
        min_content_length=100,
        enable_security_check=True,
        enable_pii_detection=True,
        strict_mode=True
    )


def create_permissive_validator() -> DocumentValidator:
    """Create a permissive validator (minimal checks)."""
    return DocumentValidator(
        min_content_length=10,
        enable_security_check=False,
        enable_pii_detection=False,
        strict_mode=False
    )
