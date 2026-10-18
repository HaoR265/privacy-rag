"""Deployment trust root distributed as reviewed code, never supplied by a caller.

No deployment has been materialized for this isolated integration. A future
value must name the canonical deployment record for a reviewed build and its
BFV applicability evidence. This file must not enter that record's build hash:
doing so would create a circular identity. Setting it does not enable NUM-004.
"""

TRUSTED_DEPLOYMENT_SHA256: str | None = None
