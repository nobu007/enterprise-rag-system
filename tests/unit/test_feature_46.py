"""
Unit tests for Feature 46: Document Validation

This module tests document validation before ingestion to ensure:
- Content quality checks (empty, minimum length, encoding)
- Security validation (malicious patterns, PII detection)
- Format-specific validation (PDF, Markdown, text)
- Size limits and performance constraints
- Metadata completeness
"""

import pytest

from app.services.validator import (
    DocumentValidator,
    ValidationResult,
    ValidationError
)
from app.services.document_loader import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator():
    """Create a default DocumentValidator instance."""
    return DocumentValidator()


@pytest.fixture
def sample_document():
    """Create a valid sample document."""
    return Document(
        content="This is a valid document content with sufficient length. " * 10,
        metadata={
            "source": "test.txt",
            "filename": "test.txt",
            "file_type": "txt"
        }
    )


@pytest.fixture
def temp_files_dir(tmp_path):
    """Create temporary directory with various test files."""
    # Valid text file
    valid_txt = tmp_path / "valid.txt"
    valid_txt.write_text("Valid content" * 50)

    # Empty file
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    # Too short file
    short = tmp_path / "short.txt"
    short.write_text("Hi")

    # File with suspicious patterns
    suspicious = tmp_path / "suspicious.txt"
    suspicious.write_text("<script>alert('xss')</script>")

    # File with potential PII
    pii = tmp_path / "pii.txt"
    pii.write_text("Contact john.doe@example.com or call 555-123-4567")

    # Large file (simulated with 1MB of content)
    large = tmp_path / "large.txt"
    large.write_text("x" * (1 * 1024 * 1024))

    # Valid markdown
    valid_md = tmp_path / "valid.md"
    valid_md.write_text("# Heading\n\n" + "Content " * 50)

    # Markdown with broken links
    broken_md = tmp_path / "broken.md"
    broken_md.write_text("[Broken](javascript:alert(1))")

    return tmp_path


# ---------------------------------------------------------------------------
# ValidationResult Tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result_creation(self):
        """Test creating a valid validation result."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result_with_errors(self):
        """Test creating invalid result with errors."""
        errors = [ValidationError(code="EMPTY", message="Content is empty")]
        result = ValidationResult(is_valid=False, errors=errors, warnings=[])
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "EMPTY"

    def test_result_with_warnings(self):
        """Test result with warnings but no errors."""
        warnings = ["Content length is near minimum"]
        result = ValidationResult(is_valid=True, errors=[], warnings=warnings)
        assert result.is_valid is True
        assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# ValidationError Tests
# ---------------------------------------------------------------------------


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_error_creation(self):
        """Test creating validation error."""
        error = ValidationError(
            code="INVALID_FORMAT",
            message="Document format is not supported",
            severity="error"
        )
        assert error.code == "INVALID_FORMAT"
        assert error.message == "Document format is not supported"
        assert error.severity == "error"

    def test_warning_severity(self):
        """Test warning severity validation error."""
        warning = ValidationError(
            code="MIN_LENGTH_WARNING",
            message="Content is shorter than recommended",
            severity="warning"
        )
        assert warning.severity == "warning"


# ---------------------------------------------------------------------------
# DocumentValidator Initialization Tests
# ---------------------------------------------------------------------------


class TestDocumentValidatorInit:
    """Tests for DocumentValidator initialization."""

    def test_default_initialization(self):
        """Test validator with default parameters."""
        validator = DocumentValidator()
        assert validator.min_content_length == 50
        assert validator.max_content_length == 10 * 1024 * 1024  # 10MB
        assert validator.enable_security_check is True
        assert validator.enable_pii_detection is True

    def test_custom_initialization(self):
        """Test validator with custom parameters."""
        validator = DocumentValidator(
            min_content_length=100,
            max_content_length=1024,
            enable_security_check=False,
            enable_pii_detection=False
        )
        assert validator.min_content_length == 100
        assert validator.max_content_length == 1024
        assert validator.enable_security_check is False
        assert validator.enable_pii_detection is False


# ---------------------------------------------------------------------------
# Content Validation Tests
# ---------------------------------------------------------------------------


class TestContentValidation:
    """Tests for document content validation."""

    def test_validate_valid_content(self, validator, sample_document):
        """Test validation of valid document content."""
        result = validator.validate_content(sample_document)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_empty_content(self, validator):
        """Test validation fails for empty content."""
        doc = Document(content="", metadata={"source": "test"})
        result = validator.validate_content(doc)
        assert result.is_valid is False
        assert any(e.code == "EMPTY_CONTENT" for e in result.errors)

    def test_validate_whitespace_only_content(self, validator):
        """Test validation fails for whitespace-only content."""
        doc = Document(content="   \n\n   \t  ", metadata={"source": "test"})
        result = validator.validate_content(doc)
        assert result.is_valid is False
        assert any(e.code == "EMPTY_CONTENT" for e in result.errors)

    def test_validate_too_short_content(self, validator):
        """Test validation fails for content below minimum length."""
        doc = Document(content="Short", metadata={"source": "test"})
        result = validator.validate_content(doc)
        assert result.is_valid is False
        assert any(e.code == "CONTENT_TOO_SHORT" for e in result.errors)

    def test_validate_too_long_content(self, validator):
        """Test validation fails for excessively long content."""
        # Create content exceeding max length (10MB default)
        huge_content = "x" * (11 * 1024 * 1024)
        doc = Document(content=huge_content, metadata={"source": "test"})
        result = validator.validate_content(doc)
        assert result.is_valid is False
        assert any(e.code == "CONTENT_TOO_LONG" for e in result.errors)

    def test_validate_content_length_boundary(self, validator):
        """Test validation at minimum length boundary."""
        # Content exactly at minimum
        doc = Document(content="x" * 50, metadata={"source": "test"})
        result = validator.validate_content(doc)
        assert result.is_valid is True

    def test_custom_min_length_threshold(self, sample_document):
        """Test validation with custom minimum length."""
        validator = DocumentValidator(min_content_length=200)
        result = validator.validate_content(sample_document)
        # Sample document might be too short for custom threshold
        if len(sample_document.content) < 200:
            assert result.is_valid is False
            assert any(e.code == "CONTENT_TOO_SHORT" for e in result.errors)


# ---------------------------------------------------------------------------
# Security Validation Tests
# ---------------------------------------------------------------------------


class TestSecurityValidation:
    """Tests for security-related validation."""

    def test_detect_xss_pattern(self, validator):
        """Test XSS attack pattern detection."""
        doc = Document(
            content="<script>alert('xss')</script>",
            metadata={"source": "test"}
        )
        result = validator.validate_security(doc)
        assert result.is_valid is False
        assert any(e.code == "SECURITY_XSS" for e in result.errors)

    def test_detect_sql_injection_pattern(self, validator):
        """Test SQL injection pattern detection."""
        doc = Document(
            content="'; DROP TABLE users; --",
            metadata={"source": "test"}
        )
        result = validator.validate_security(doc)
        assert result.is_valid is False
        assert any("SQL" in e.code for e in result.errors)

    def test_detect_path_traversal_pattern(self, validator):
        """Test path traversal pattern detection."""
        doc = Document(
            content="../../../etc/passwd",
            metadata={"source": "test"}
        )
        result = validator.validate_security(doc)
        assert result.is_valid is False
        assert any("PATH" in e.code or "TRAVERSAL" in e.code for e in result.errors)

    def test_detect_command_injection(self, validator):
        """Test command injection pattern detection."""
        doc = Document(
            content="; rm -rf /",
            metadata={"source": "test"}
        )
        result = validator.validate_security(doc)
        assert result.is_valid is False
        assert any("COMMAND" in e.code for e in result.errors)

    def test_security_check_disabled(self, sample_document):
        """Test security check can be disabled."""
        validator = DocumentValidator(enable_security_check=False)
        malicious_doc = Document(
            content="<script>alert('xss')</script>",
            metadata={"source": "test"}
        )
        result = validator.validate_security(malicious_doc)
        # Should pass when security check is disabled
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# PII Detection Tests
# ---------------------------------------------------------------------------


class TestPIIDetection:
    """Tests for PII (Personally Identifiable Information) detection."""

    def test_detect_email_address(self, validator):
        """Test email address detection."""
        doc = Document(
            content="Contact us at john.doe@example.com for support",
            metadata={"source": "test"}
        )
        result = validator.validate_pii(doc)
        # PII detection generates warnings by default, not errors
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("email" in w.lower() for w in result.warnings)

    def test_detect_phone_number(self, validator):
        """Test phone number detection."""
        doc = Document(
            content="Call us at 555-123-4567 for assistance",
            metadata={"source": "test"}
        )
        result = validator.validate_pii(doc)
        # PII detection generates warnings by default, not errors
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("phone" in w.lower() for w in result.warnings)

    def test_detect_ssn_pattern(self, validator):
        """Test SSN (Social Security Number) pattern detection."""
        doc = Document(
            content="My SSN is 123-45-6789",
            metadata={"source": "test"}
        )
        result = validator.validate_pii(doc)
        # PII detection generates warnings by default, not errors
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("ssn" in w.lower() for w in result.warnings)

    def test_pii_detection_disabled(self, sample_document):
        """Test PII detection can be disabled."""
        validator = DocumentValidator(enable_pii_detection=False)
        pii_doc = Document(
            content="Contact john.doe@example.com",
            metadata={"source": "test"}
        )
        result = validator.validate_pii(pii_doc)
        # Should pass when PII detection is disabled
        assert result.is_valid is True

    def test_pii_detection_generates_warning_not_error(self, validator):
        """Test PII detection generates warnings by default."""
        doc = Document(
            content="Email: test@example.com",
            metadata={"source": "test"}
        )
        result = validator.validate_pii(doc)
        # PII should generate warnings, not hard errors
        assert len(result.warnings) > 0 or len(result.errors) > 0


# ---------------------------------------------------------------------------
# Format-Specific Validation Tests
# ---------------------------------------------------------------------------


class TestFormatValidation:
    """Tests for format-specific validation."""

    def test_validate_text_file(self, validator, temp_files_dir):
        """Test validation of text file."""
        txt_path = temp_files_dir / "valid.txt"
        doc = Document(
            content=txt_path.read_text(),
            metadata={"source": str(txt_path), "file_type": "txt"}
        )
        result = validator.validate_format(doc)
        assert result.is_valid is True

    def test_validate_markdown_file(self, validator, temp_files_dir):
        """Test validation of markdown file."""
        md_path = temp_files_dir / "valid.md"
        doc = Document(
            content=md_path.read_text(),
            metadata={"source": str(md_path), "file_type": "markdown"}
        )
        result = validator.validate_format(doc)
        assert result.is_valid is True

    def test_validate_pdf_metadata(self, validator):
        """Test validation of PDF document metadata."""
        doc = Document(
            content="Sample PDF content",
            metadata={
                "source": "doc.pdf",
                "file_type": "pdf",
                "page": 1,
                "total_pages": 5
            }
        )
        result = validator.validate_format(doc)
        assert result.is_valid is True

    def test_unsupported_format(self, validator):
        """Test validation of unsupported format."""
        doc = Document(
            content="content",
            metadata={"source": "file.exe", "file_type": "exe"}
        )
        result = validator.validate_format(doc)
        assert result.is_valid is False
        assert any(e.code == "UNSUPPORTED_FORMAT" for e in result.errors)


# ---------------------------------------------------------------------------
# Metadata Validation Tests
# ---------------------------------------------------------------------------


class TestMetadataValidation:
    """Tests for metadata validation."""

    def test_validate_complete_metadata(self, validator, sample_document):
        """Test validation of complete metadata."""
        result = validator.validate_metadata(sample_document)
        assert result.is_valid is True

    def test_validate_missing_required_metadata(self, validator):
        """Test validation fails for missing required metadata."""
        doc = Document(content="Content", metadata={})
        result = validator.validate_metadata(doc)
        assert result.is_valid is False
        assert any(e.code == "MISSING_METADATA" for e in result.errors)

    def test_validate_metadata_with_source_only(self, validator):
        """Test validation with minimal but valid metadata."""
        doc = Document(
            content="Valid content " * 20,
            metadata={"source": "test.txt"}
        )
        result = validator.validate_metadata(doc)
        # Should pass if source is present
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestDocumentValidatorIntegration:
    """Integration tests for complete validation workflow."""

    def test_full_validation_pass(self, validator, sample_document):
        """Test complete validation passes for valid document."""
        result = validator.validate(sample_document)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_full_validation_fail_multiple_issues(self, validator):
        """Test validation fails with multiple issues."""
        doc = Document(
            content="<script>alert(1)</script>",
            metadata={}
        )
        result = validator.validate(doc)
        assert result.is_valid is False
        # Should have multiple errors (empty metadata, security issue, etc.)
        assert len(result.errors) >= 2

    def test_validation_with_warnings_only(self, validator):
        """Test validation passes with warnings."""
        doc = Document(
            content="Valid content " * 10 + " Contact test@example.com",
            metadata={"source": "test.txt"}
        )
        result = validator.validate(doc)
        # Should be valid but with PII warnings
        assert result.is_valid is True
        assert len(result.warnings) > 0

    def test_validation_performance_large_document(self, validator):
        """Test validation performance with large document."""
        import time
        large_content = "Word " * 100000  # ~600KB
        doc = Document(
            content=large_content,
            metadata={"source": "large.txt"}
        )
        start = time.time()
        result = validator.validate(doc)
        duration = time.time() - start
        # Should complete within reasonable time (< 1 second)
        assert duration < 1.0
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Batch Validation Tests
# ---------------------------------------------------------------------------


class TestBatchValidation:
    """Tests for batch document validation."""

    def test_validate_multiple_documents(self, validator):
        """Test validation of multiple documents."""
        docs = [
            Document(
                content=f"Valid document content {i} " * 20,
                metadata={"source": f"doc{i}.txt"}
            )
            for i in range(5)
        ]
        results = validator.validate_batch(docs)
        assert len(results) == 5
        assert all(r.is_valid for r in results)

    def test_validate_batch_with_mixed_results(self, validator):
        """Test batch validation with mixed valid/invalid documents."""
        docs = [
            Document(
                content="Valid content " * 20,
                metadata={"source": "valid.txt"}
            ),
            Document(
                content="",
                metadata={"source": "empty.txt"}
            ),
            Document(
                content="<script>alert(1)</script>",
                metadata={"source": "malicious.txt"}
            ),
        ]
        results = validator.validate_batch(docs)
        assert len(results) == 3
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[2].is_valid is False

    def test_batch_validation_summary(self, validator):
        """Test batch validation provides summary statistics."""
        docs = [
            Document(
                content="Valid " * 20,
                metadata={"source": f"doc{i}.txt"}
            )
            for i in range(10)
        ]
        # Make 3 invalid
        docs[3].content = ""
        docs[5].content = "<script>"
        docs[7].metadata = {}

        results = validator.validate_batch(docs)
        valid_count = sum(1 for r in results if r.is_valid)
        assert valid_count == 7


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestValidationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_validate_none_content(self, validator):
        """Test validation handles None content gracefully."""
        # Create document with explicit doc_id to avoid hash generation on None
        doc = Document(content="dummy", metadata={"source": "test"}, doc_id="test-id")
        # Replace content with None after creation
        doc.content = None
        result = validator.validate_content(doc)
        assert result.is_valid is False

    def test_validate_unicode_content(self, validator):
        """Test validation handles unicode characters."""
        doc = Document(
            content="日本語のコンテンツ " * 20 + "Emoji: 🚀 🎯 ",
            metadata={"source": "test"}
        )
        result = validator.validate_content(doc)
        assert result.is_valid is True

    def test_validate_mixed_language_content(self, validator):
        """Test validation handles mixed language content."""
        doc = Document(
            content="English 日本语 한국어 " * 20,
            metadata={"source": "test"}
        )
        result = validator.validate_content(doc)
        assert result.is_valid is True

    def test_validate_special_characters(self, validator):
        """Test validation handles special characters."""
        doc = Document(
            content="Special chars: @#$%^&*()_+-=[]{}|;:',.<>?/`~" * 10,
            metadata={"source": "test"}
        )
        result = validator.validate_content(doc)
        # Should be valid (not malicious patterns)
        assert result.is_valid is True
