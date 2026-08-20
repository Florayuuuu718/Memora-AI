"""External system adapters."""

from memora.integrations.immich import ImmichClient, ImmichError, ImmichSyncResult

__all__ = ["ImmichClient", "ImmichError", "ImmichSyncResult"]
