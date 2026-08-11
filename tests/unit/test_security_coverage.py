"""
Coverage tests for the uncovered branches of app/core/security.py.

The existing test_validation_middleware.py exercises the happy paths of every
SecurityValidator method but never passes non-string inputs (so the isinstance
guards are unreached) and never calls validate_email at all. These tests close
that gap with exact-output assertions on the validation boundaries.
"""

import pytest

from app.core.security import SecurityValidator


class TestValidateEmail:
    """validate_email is entirely uncovered -> assert exact True/False."""

    @pytest.mark.parametrize(
        "email",
        [
            "user@example.com",
            "User.Name+tag@sub.example.co",
            "a.b-c_d@e.f-g.org",
            "x@y.io",            # minimum-length 2-char TLD
            "123@numbers123.dev",
        ],
    )
    def test_valid_emails(self, email):
        assert SecurityValidator.validate_email(email) is True

    @pytest.mark.parametrize(
        "email",
        [
            "notanemail",          # no @
            "@example.com",        # empty local part
            "user@",               # empty domain
            "user@.com",           # domain starts with a dot
            "user@example",        # no TLD (no dot before TLD)
            "user@example.c",      # 1-char TLD (< {2,})
            "user @example.com",   # whitespace
            "user@exa mple.com",   # whitespace in domain
            "",                    # empty string
        ],
    )
    def test_invalid_emails(self, email):
        assert SecurityValidator.validate_email(email) is False

    @pytest.mark.parametrize("non_string", [None, 123, 4.5, [], {}, object()])
    def test_non_string_returns_false(self, non_string):
        # Defensive guard: never raise on non-string input.
        assert SecurityValidator.validate_email(non_string) is False


class TestNonStringInputGuards:
    """Non-string: detectors -> False; sanitize_input returns it as-is."""

    @pytest.mark.parametrize("non_string", [None, 123, [], object()])
    def test_detect_sql_injection_non_string(self, non_string):
        assert SecurityValidator.detect_sql_injection(non_string) is False

    @pytest.mark.parametrize("non_string", [None, 123, [], object()])
    def test_detect_xss_non_string(self, non_string):
        assert SecurityValidator.detect_xss(non_string) is False

    @pytest.mark.parametrize("non_string", [None, 123, [], object()])
    def test_detect_path_traversal_non_string(self, non_string):
        assert SecurityValidator.detect_path_traversal(non_string) is False

    @pytest.mark.parametrize("non_string", [None, 123, [], object()])
    def test_detect_command_injection_non_string(self, non_string):
        assert SecurityValidator.detect_command_injection(non_string) is False

    @pytest.mark.parametrize("non_string", [None, 123, 4.5])
    def test_sanitize_input_non_string_passes_through(self, non_string):
        # sanitize_input returns non-string unchanged (no length check/strip).
        assert SecurityValidator.sanitize_input(non_string) is non_string
