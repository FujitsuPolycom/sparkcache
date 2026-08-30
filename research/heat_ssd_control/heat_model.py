"""Bounded 8-bit hit counters and chunk reference accounting.

Process-local, in-memory modeling of per-context heat (verified-restore
frequency) and of the byte cost a stored root's removal would reclaim. No
disk state, no locks, and no imports from the serving package.
"""


import binascii
import hashlib
import json
from collections import Counter
from dataclasses import dataclass

RING_SCHEMA = "sparkcache-research-heat-ring/v1"
_DIGEST = 64
_KEY_SEPARATOR = 0x0A  # domain separator between storage_key and context_digest bytes


class ResearchFormatError(ValueError):
    """A modeling input violates its schema or bounded-domain rule.

    Callers must stop on this error: nothing here contradicts serving state,
    so a disagreement means the caller's data or assumptions are wrong.
    """


def require_digest(value: str, field: str) -> None:
    """Reject any value that is not a 64-character lowercase hex digest."""
    if not isinstance(value, str) or len(value) != _DIGEST or not all(
        character in "0123456789abcdef" for character in value
    ):
        raise ResearchFormatError(f"{field} must be a 64-character lowercase SHA-256 hex digest")


_require_digest = require_digest


def recomputation_tokens_avoided(
    committed_tokens: int,
    num_computed_tokens: int,
) -> int:
    """Prefill tokens replaced by one completed, verified restore."""
    if (
        type(committed_tokens) is not int
        or type(num_computed_tokens) is not int
        or committed_tokens < 0
        or num_computed_tokens < 0
        or num_computed_tokens > committed_tokens
    ):
        raise ResearchFormatError(
            "token counts must be non-negative integers with "
            "num_computed_tokens <= committed_tokens"
        )
    return committed_tokens - num_computed_tokens


@dataclass(frozen=True, order=True)
class HeatKey:
    """Identity of one stored root: cache identity namespace plus context digest."""

    storage_key: str
    context_digest: str

    def __post_init__(self) -> None:
        _require_digest(self.storage_key, "storage_key")
        _require_digest(self.context_digest, "context_digest")

    def slot(self, capacity: int) -> int:
        """Ring index for this key in one ring of ``capacity`` slots."""
        if type(capacity) is not int or capacity <= 0 or capacity & (capacity - 1):
            raise ResearchFormatError("capacity must be a positive power of two")
        digest = hashlib.blake2b(
            self.storage_key.encode("ascii")
            + bytes((_KEY_SEPARATOR,))
            + self.context_digest.encode("ascii"),
            digest_size=8,
        ).digest()
        mask = capacity - 1
        return int.from_bytes(digest, "little") & mask


@dataclass(frozen=True)
class HitRingConfig:
    capacity: int = 131072
    decay_window: int = 8192
    decay_shift: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.capacity) is not int
            or self.capacity <= 0
            or self.capacity & (self.capacity - 1)
        ):
            raise ResearchFormatError("capacity must be a positive power of two")
        if type(self.decay_window) is not int or self.decay_window <= 0:
            raise ResearchFormatError("decay_window must be at least 1")
        if type(self.decay_shift) is not int or not 0 <= self.decay_shift <= 7:
            raise ResearchFormatError("decay_shift must be in [0, 7]")
        if self.decay_shift > self.decay_window // 2:
            raise ResearchFormatError(
                "decay_shift must be much smaller than decay_window or the sketch collapses"
            )


class HitRing:
    """Fixed-size ring of saturating 8-bit access counters with epoch decay.

    One slot per ring index; distinct contexts may share a slot (false
    sharing) and their estimates add. Estimates are comparative inputs for
    admission experiments only; per the heat-isolation contract no
    correctness decision may consume them.
    """

    def __init__(self, config: HitRingConfig | None = None) -> None:
        self._config = config or HitRingConfig()
        self._counts = bytearray(self._config.capacity)
        self._since_decay = 0

    @property
    def config(self) -> HitRingConfig:
        return self._config

    def record_hit(self, key: HeatKey) -> int:
        """Increment one key's counter and run one decay sweep when due."""
        if not isinstance(key, HeatKey):
            raise ResearchFormatError("key must be a HeatKey")
        index = key.slot(self._config.capacity)
        self._since_decay += 1
        self._maybe_decay()
        current = self._counts[index]
        if current < 0xFF:
            self._counts[index] = current + 1
        return self._counts[index]

    def estimate(self, key: HeatKey) -> int:
        if not isinstance(key, HeatKey):
            raise ResearchFormatError("key must be a HeatKey")
        return self._counts[key.slot(self._config.capacity)]

    def _maybe_decay(self) -> None:
        if self._since_decay < self._config.decay_window:
            return
        shift = self._config.decay_shift
        for index in range(self._config.capacity):
            self._counts[index] >>= shift
        self._since_decay = 0

    def snapshot(self) -> str:
        """Serialize all counters as one ``heat-ring/v1`` JSON document.

        Key identities are not serialized; estimates for keys that re-derive
        the same slot survive a reload and all others restart at zero.
        """
        document = {
            "schema": RING_SCHEMA,
            "capacity": self._config.capacity,
            "decay_window": self._config.decay_window,
            "decay_shift": self._config.decay_shift,
            "increments_since_decay": self._since_decay,
            "counts_hex": self._counts.hex(),
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> "HitRing":
        try:
            document = json.loads(payload)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ResearchFormatError(f"heat ring snapshot is not JSON: {error}") from error
        if not isinstance(document, dict) or set(document) != {
            "schema",
            "capacity",
            "decay_window",
            "decay_shift",
            "increments_since_decay",
            "counts_hex",
        }:
            raise ResearchFormatError(f"heat ring snapshot must match schema {RING_SCHEMA}")
        if document["schema"] != RING_SCHEMA:
            raise ResearchFormatError(f"unknown heat ring schema {document['schema']!r}")
        config = HitRingConfig(
            capacity=document["capacity"],
            decay_window=document["decay_window"],
            decay_shift=document["decay_shift"],
        )
        counts_hex = document["counts_hex"]
        expected = config.capacity * 2
        if not isinstance(counts_hex, str) or len(counts_hex) != expected:
            raise ResearchFormatError(
                f"counts_hex must be {expected} hex characters for capacity {config.capacity}"
            )
        try:
            counts = binascii.unhexlify(counts_hex)
        except (ValueError, binascii.Error) as error:
            raise ResearchFormatError(f"counts_hex is not hex: {error}") from error
        since_decay = document["increments_since_decay"]
        if not isinstance(since_decay, int) or not 0 <= since_decay < config.decay_window:
            raise ResearchFormatError("increments_since_decay is out of range")
        ring = object.__new__(cls)
        ring._config = config
        ring._counts = bytearray(counts)
        ring._since_decay = since_decay
        return ring


@dataclass(frozen=True)
class PublishedContext:
    """One stored root as the ledger sees it: ordered chunks plus metadata files."""

    key: HeatKey
    chunk_digests: tuple[str, ...]
    chunk_bytes: tuple[int, ...]
    chunk_token_counts: tuple[int, ...]
    manifest_bytes: int
    segment_digests: tuple[str, ...] = ()
    segment_bytes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, HeatKey):
            raise ResearchFormatError("key must be a HeatKey")
        if len(self.chunk_digests) != len(self.chunk_bytes):
            raise ResearchFormatError("chunk digest and byte counts must align")
        if len(self.chunk_digests) != len(self.chunk_token_counts):
            raise ResearchFormatError("chunk digest and token counts must align")
        if len(self.segment_digests) != len(self.segment_bytes):
            raise ResearchFormatError("segment digest and byte counts must align")
        byte_counts = (self.manifest_bytes, *self.chunk_bytes, *self.segment_bytes)
        if any(type(size) is not int or size < 0 for size in byte_counts):
            raise ResearchFormatError("byte counts must be non-negative integers")
        if any(
            type(token_count) is not int or token_count <= 0
            for token_count in self.chunk_token_counts
        ):
            raise ResearchFormatError("chunk token counts must be positive integers")
        if len(set(self.chunk_digests)) != len(self.chunk_digests):
            raise ResearchFormatError("chunk digests must be unique within one root")
        if len(set(self.segment_digests)) != len(self.segment_digests):
            raise ResearchFormatError("segment digests must be unique within one root")
        for digest in self.chunk_digests:
            _require_digest(digest, "chunk digest")
        for digest in self.segment_digests:
            _require_digest(digest, "segment digest")


@dataclass(frozen=True)
class ContextHeatReport:
    """Per-root cost and share facts derived from publication state."""

    chunk_count: int
    shared_chunk_count: int
    shared_tokens: int
    retained_shared_bytes: int
    marginal_bytes: int
    encoded_bytes: int


class ChunkLedger:
    """Reference counts over content-addressed chunks and descriptor segments.

    Mirrors the dedup structure of the production store: one published chunk
    is shared by every root whose manifest or alias chain references it.
    ``marginal_bytes`` reproduces what ``MaintenanceReport.bytes_reclaimed``
    would measure for removing one root; it never counts a shared object
    toward more than one removal.
    """

    def __init__(self, chunk_tokens: int = 256) -> None:
        if type(chunk_tokens) is not int or chunk_tokens <= 0:
            raise ResearchFormatError("chunk_tokens must be a positive integer")
        self._chunk_tokens = chunk_tokens
        self._chunk_refs: Counter[str] = Counter()
        self._chunk_bytes: dict[str, int] = {}
        self._chunk_token_counts: dict[str, int] = {}
        self._segment_refs: Counter[tuple[str, str]] = Counter()
        self._segment_bytes: dict[tuple[str, str], int] = {}
        self._contexts: dict[HeatKey, PublishedContext] = {}

    @property
    def chunk_tokens(self) -> int:
        return self._chunk_tokens

    def publish(self, context: PublishedContext) -> None:
        if not isinstance(context, PublishedContext):
            raise ResearchFormatError("context must be a PublishedContext")
        if context.key in self._contexts:
            raise ResearchFormatError(f"context {context.key} is already published")
        # Validate the complete publication before changing reference counts.
        # A rejected research input must not leave the ledger half-updated.
        for digest, size, token_count in zip(
            context.chunk_digests,
            context.chunk_bytes,
            context.chunk_token_counts,
        ):
            if token_count > self._chunk_tokens:
                raise ResearchFormatError(
                    f"chunk {digest} exceeds the configured token geometry"
                )
            recorded = self._chunk_bytes.get(digest)
            if recorded is not None and recorded != size:
                raise ResearchFormatError(
                    f"chunk {digest} byte count changed across publications"
                )
            recorded_tokens = self._chunk_token_counts.get(digest)
            if recorded_tokens is not None and recorded_tokens != token_count:
                raise ResearchFormatError(
                    f"chunk {digest} token count changed across publications"
                )
        namespace = context.key.storage_key
        for digest, size in zip(context.segment_digests, context.segment_bytes):
            reference = (namespace, digest)
            recorded = self._segment_bytes.get(reference)
            if recorded is not None and recorded != size:
                raise ResearchFormatError(
                    f"segment {digest} byte count changed across publications"
                )

        for digest, size, token_count in zip(
            context.chunk_digests,
            context.chunk_bytes,
            context.chunk_token_counts,
        ):
            self._chunk_bytes.setdefault(digest, size)
            self._chunk_token_counts.setdefault(digest, token_count)
            self._chunk_refs[digest] += 1
        for digest, size in zip(context.segment_digests, context.segment_bytes):
            reference = (namespace, digest)
            self._segment_bytes.setdefault(reference, size)
            self._segment_refs[reference] += 1
        self._contexts[context.key] = context

    def remove(self, key: HeatKey) -> None:
        if not isinstance(key, HeatKey):
            raise ResearchFormatError("key must be a HeatKey")
        context = self._contexts.pop(key, None)
        if context is None:
            raise ResearchFormatError(f"context {key} is not published")
        for digest in context.chunk_digests:
            self._chunk_refs[digest] -= 1
            if self._chunk_refs[digest] == 0:
                del self._chunk_bytes[digest]
                del self._chunk_token_counts[digest]
                del self._chunk_refs[digest]
        namespace = key.storage_key
        for digest in context.segment_digests:
            reference = (namespace, digest)
            self._segment_refs[reference] -= 1
            if self._segment_refs[reference] == 0:
                del self._segment_bytes[reference]
                del self._segment_refs[reference]

    def contexts(self) -> tuple["PublishedContext", ...]:
        """Every published context in publication order."""
        return tuple(self._contexts.values())

    def report(self, key: HeatKey) -> ContextHeatReport:
        if not isinstance(key, HeatKey):
            raise ResearchFormatError("key must be a HeatKey")
        context = self._contexts.get(key)
        if context is None:
            raise ResearchFormatError(f"context {key} is not published")
        shared_bytes = 0
        marginal_bytes = context.manifest_bytes
        shared_count = 0
        shared_tokens = 0
        for digest, size, token_count in zip(
            context.chunk_digests,
            context.chunk_bytes,
            context.chunk_token_counts,
        ):
            if self._chunk_refs[digest] >= 2:
                shared_count += 1
                shared_bytes += size
                shared_tokens += token_count
            else:
                marginal_bytes += size
        for digest, size in zip(context.segment_digests, context.segment_bytes):
            if self._segment_refs[(context.key.storage_key, digest)] == 1:
                marginal_bytes += size
        encoded = context.manifest_bytes + sum(context.chunk_bytes) + sum(context.segment_bytes)
        return ContextHeatReport(
            chunk_count=len(context.chunk_digests),
            shared_chunk_count=shared_count,
            shared_tokens=shared_tokens,
            retained_shared_bytes=shared_bytes,
            marginal_bytes=marginal_bytes,
            encoded_bytes=encoded,
        )


def context_reports(ledger: ChunkLedger) -> dict[HeatKey, ContextHeatReport]:
    """Heat reports keyed by complete storage namespace and context identity."""
    if not isinstance(ledger, ChunkLedger):
        raise ResearchFormatError("ledger must be a ChunkLedger")
    return {
        context.key: ledger.report(context.key)
        for context in ledger.contexts()
    }
