"""
DEPRECATED: Legacy configuration module.

This module is superseded by app.core.config which is the canonical
configuration source for the application. This file is kept only for
backward compatibility reference and will be removed in a future version.

Use instead:
    from app.core.config import get_settings
"""

import warnings

warnings.warn(
    "app.config is deprecated. Use app.core.config instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location for any lingering imports
from app.core.config import Settings, get_settings, settings  # noqa: F401
