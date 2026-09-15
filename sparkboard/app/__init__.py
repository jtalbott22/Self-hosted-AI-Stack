"""
Sparkboard.

config is imported first, and deliberately: it sets psutil.PROCFS_PATH when a
host /proc has been mounted, and that has to happen before any other module
makes its first psutil call.
"""

from . import config  # noqa: F401  (imported for its side effect)

__all__ = ["config"]
