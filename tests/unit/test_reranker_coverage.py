"""Coverage-focused tests for app/services/reranker.py.

The existing test_reranker.py covers the missing-package
(`except ImportError`) branch of Reranker.__init__. This covers the
remaining branch: a non-ImportError failure while loading the
cross-encoder model (e.g. corrupted weights, unloadable checkpoint),
which `__init__` wraps as RuntimeError (reranker.py L47-49).

sentence_transformers is not installed in this environment, so a fake
module is injected via monkeypatch; its CrossEncoder constructor raises
OSError to simulate a model-load failure that is NOT an ImportError.
"""
import sys
from unittest.mock import Mock

import pytest

from app.services.reranker import Reranker


class TestRerankerInitModelLoadError:
    """Cover the model-load failure branch of __init__ (L47-49)."""

    def test_model_load_failure_raises_runtime_error(self, monkeypatch):
        fake = Mock()
        # CrossEncoder(model_name) raises a non-ImportError -> the
        # `except Exception` branch wraps it as RuntimeError.
        fake.CrossEncoder = Mock(side_effect=OSError("weights missing"))
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

        with pytest.raises(RuntimeError, match="Failed to initialize"):
            Reranker(model_name="bad-model")
