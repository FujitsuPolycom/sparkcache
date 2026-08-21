"""Stable public import for the manifest-store interface."""

from __future__ import annotations

from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    CapacityPolicy,
    CommitConflict,
    ContextChunk,
    EntryKey,
    IncompleteEntry,
    LookupResult,
    MaintenanceReport,
    ManifestStore,
    StateRecord,
)

__all__ = [
    "CacheIdentity",
    "CapacityPolicy",
    "CommitConflict",
    "ContextChunk",
    "EntryKey",
    "IncompleteEntry",
    "LookupResult",
    "MaintenanceReport",
    "ManifestStore",
    "StateRecord",
]
