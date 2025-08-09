"""Aggregation helpers for failure analysis results.

Utilities in this module group and summarize outputs produced by
`FailureManager` runs. Functions are factored here to keep `manager.py`
focused on orchestration. This module intentionally avoids importing heavy
dependencies to keep import cost low in the common path.
"""

from __future__ import annotations
