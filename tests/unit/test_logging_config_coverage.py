"""Tests for app.core.logging_config — focused on sanitize_for_log (CWE-117)."""

import logging

from app.core.logging_config import sanitize_for_log, RequestIDFilter


class TestSanitizeForLog:
    """sanitize_for_log must neutralise CR/LF so client text can't forge log lines."""

    def test_newline_is_escaped_not_preserved(self):
        """The core CWE-117 regression: a newline in a client-controlled query
        value must not survive into the logged string, or it would start a fake
        subsequent log line. Before the fix rag_pipeline/streaming interpolated
        the raw query (e.g. f"Cache hit for query: {question[:50]}"), so a body
        like {"query": "q\\n2000-01-01 INFO admin login ok"} forged an extra line.
        """
        malicious = "q\n2000-01-01 INFO admin login ok"
        sanitized = sanitize_for_log(malicious)
        # No literal newline survives -> cannot terminate the real log line.
        assert "\n" not in sanitized
        assert "\r" not in sanitized
        # The content is still visible (escaped), not silently dropped.
        assert "2000-01-01 INFO admin login ok" in sanitized
        assert sanitized == "q\\n2000-01-01 INFO admin login ok"

    def test_carriage_return_and_crlf_escaped(self):
        assert sanitize_for_log("a\rb") == "a\\rb"
        assert sanitize_for_log("a\r\nb") == "a\\r\\nb"

    def test_tab_escaped(self):
        assert sanitize_for_log("col1\tcol2") == "col1\\tcol2"

    def test_other_control_chars_escaped_as_hex(self):
        # NUL, ESC, DEL (and the rest of the C0 range) -> visible \\xNN.
        assert sanitize_for_log("x\x00y") == "x\\x00y"
        assert sanitize_for_log("x\x1by") == "x\\x1by"
        assert sanitize_for_log("x\x7fy") == "x\\x7fy"

    def test_simulated_log_message_is_single_line(self):
        """End-to-end at the message-construction level: the exact pattern used
        by rag_pipeline/streaming (f-string interpolation of a truncated query)
        must yield a string with no embedded line break."""
        question = "legit\nFAKE 2000-01-01 INFO forged entry"
        message = f"Cache hit for query: {sanitize_for_log(question[:50])}..."
        assert message.count("\n") == 0
        assert message.count("\r") == 0
        assert message.endswith("...")
        # The escape is visible in the rendered message.
        assert "\\n" in message

    def test_clean_text_preserved_unchanged(self):
        assert sanitize_for_log("normal query") == "normal query"
        assert sanitize_for_log("") == ""

    def test_non_ascii_preserved(self):
        """CJK / accented Latin must pass through untouched — the sanitizer
        targets control characters only, not Unicode (distinct from the
        cache-key ASCII-stripping bug)."""
        assert sanitize_for_log("東京タワー") == "東京タワー"
        assert sanitize_for_log("café — naïve") == "café — naïve"

    def test_non_string_coerced(self):
        assert sanitize_for_log(123) == "123"
        assert sanitize_for_log(3.14) == "3.14"
        assert sanitize_for_log(None) == "None"
        assert sanitize_for_log(True) == "True"

    def test_printable_special_chars_preserved(self):
        """Printable specials (the kind legitimate request IDs / queries
        contain) are NOT escaped — only control characters are."""
        specials = "req-with-special_chars-123!@#$%^&*()"
        assert sanitize_for_log(specials) == specials

    def test_control_char_only_at_boundary(self):
        assert sanitize_for_log("\n") == "\\n"
        assert sanitize_for_log("hello\n") == "hello\\n"
        assert sanitize_for_log("\nhello") == "\\nhello"


class TestRequestIDFilter:
    """Light coverage of the existing RequestIDFilter (unchanged behavior)."""

    def _record(self):
        return logging.LogRecord(
            name="t", level=logging.INFO, pathname="t.py", lineno=1,
            msg="m", args=(), exc_info=None,
        )

    def test_filter_defaults_request_id_when_absent(self):
        record = self._record()
        assert not hasattr(record, "request_id")
        assert RequestIDFilter().filter(record) is True
        assert record.request_id == "N/A"

    def test_filter_preserves_existing_request_id(self):
        record = self._record()
        record.request_id = "req-xyz"
        RequestIDFilter().filter(record)
        assert record.request_id == "req-xyz"


class TestRequestIDContextVarPropagation:
    """The per-request id must reach records emitted on CHILD loggers.

    Regression: RequestIDMiddleware used to add a per-request filter to the
    ROOT logger, but CPython applies a logger's own filters only to records
    emitted directly on that logger -- child-logger records propagate to the
    root HANDLER via callHandlers (running handler filters, never the parent
    logger's). So every app.* log line showed request_id=N/A and request
    tracing was silently non-functional. The fix binds the id to a ContextVar
    (set_request_id) and the handler-level RequestIDFilter reads it.
    """

    def test_child_record_carries_contextvar_id(self):
        import io

        from app.core.logging_config import set_request_id, reset_request_id

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("[%(request_id)s] %(message)s"))
        handler.addFilter(RequestIDFilter())

        root = logging.getLogger()
        root.addHandler(handler)
        try:
            root.setLevel(logging.INFO)
            child = logging.getLogger("app.services.regression_child")
            # No handler on the child -- the record propagates to root, whose
            # handler filter (RequestIDFilter) must read the ContextVar.
            token = set_request_id("req-real-ABC")
            try:
                child.info("serving request")
            finally:
                reset_request_id(token)

            out = stream.getvalue()
            assert "[req-real-ABC] serving request" in out
            assert "N/A" not in out
        finally:
            root.removeHandler(handler)

    def test_outside_request_falls_back_to_default(self):
        """With no id bound, child records still get the 'N/A' default."""
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("[%(request_id)s] %(message)s"))
        handler.addFilter(RequestIDFilter())

        root = logging.getLogger()
        root.addHandler(handler)
        try:
            root.setLevel(logging.INFO)
            child = logging.getLogger("app.services.regression_child2")
            child.info("startup message")
            assert "[N/A] startup message" in stream.getvalue()
        finally:
            root.removeHandler(handler)
