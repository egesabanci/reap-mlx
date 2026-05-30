"""Backend namespace for optional REAP runtimes.

This package is intentionally import-light. Backend implementations should keep
heavy or optional runtime imports inside explicit implementation modules or
functions, not at namespace import time.
"""

__all__: tuple[str, ...] = ()
