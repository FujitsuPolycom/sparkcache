"""Pure codec for SparkCache per-token DCP shards.

DCP shard math and byte-record packing shared by the vLLM connector and its
tests. This module must not import vllm or torch: everything operates on
plain integers and bytes so the contract is testable model-free.

Per-token-row layout invariants:
- DCP token interleave size 1: global position ``p`` is owned by DCP rank
  ``p % dcp_degree``.
- Dense local prefix: owned position ``p`` sits at local ordinal
  ``p // dcp_degree``; the local KV slot is
  ``block_ids[ordinal // block_size] * block_size + ordinal % block_size``.
- Record byte order within a chunk is the sorted layer-name order captured
  at registration time; each layer contributes ``rows_per_chunk`` rows of
  its fixed per-token record width.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from array import array
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

CHUNK_TOKENS = 256
_POSITION_STRUCT = struct.Struct("<I")


class CodecError(ValueError):
    """Deterministic packing/unpacking failure. Callers convert this to a
    cache miss and recomputation; it must never crash a serving process."""


def _u32_array(values: Iterable[int], label: str) -> array:
    """Collect unsigned 32-bit integers in the frozen little-endian ABI.

    ``array`` performs the conversion in C, avoiding one Python object per
    token or position on large contexts.
    """
    try:
        packed = array("I", values)
    except (OverflowError, TypeError, ValueError) as error:
        raise CodecError(f"{label} must contain unsigned 32-bit integers") from error
    if packed.itemsize != _POSITION_STRUCT.size:
        raise CodecError("platform unsigned-int width does not match cache ABI")
    if sys.byteorder != "little":
        packed.byteswap()
    return packed


def _pack_u32(values: Iterable[int], label: str) -> bytes:
    """Pack unsigned 32-bit integers as immutable bytes."""

    return _u32_array(values, label).tobytes()


def owned_positions(
    span_tokens: int, dcp_degree: int, dcp_rank: int
) -> tuple[int, ...]:
    """Global positions in [0, span) owned by this rank (interleave 1)."""
    if span_tokens <= 0:
        raise CodecError("span must be positive")
    if not 0 <= dcp_rank < dcp_degree:
        raise CodecError("dcp_rank out of range")
    return tuple(range(dcp_rank, span_tokens, dcp_degree))


def local_slots_for_positions(
    positions: Sequence[int],
    block_ids: Sequence[int],
    block_size: int,
    dcp_degree: int,
) -> tuple[int, ...]:
    """Map owned global positions to this rank's local KV slots."""
    if block_size <= 0:
        raise CodecError("block_size must be positive")
    capacity = len(block_ids) * block_size
    slots = []
    for p in positions:
        ordinal = p // dcp_degree
        if ordinal >= capacity:
            raise CodecError(
                f"position {p} ordinal {ordinal} exceeds allocated capacity {capacity}"
            )
        block = block_ids[ordinal // block_size]
        slots.append(block * block_size + ordinal % block_size)
    return tuple(slots)


def context_digest(token_ids: Iterable[int], identity_salt: str) -> str:
    """Content digest for a block-aligned prompt span. The identity salt
    binds the digest to the cache identity so distinct configurations can
    never alias to the same key even inside one store root."""
    return context_prefix_digest(token_ids, identity_salt, token_count=None)


def context_prefix_digest(
    token_ids: Iterable[int],
    identity_salt: str,
    *,
    token_count: int | None,
) -> str:
    """Digest a packed token prefix without allocating a Python list slice."""

    packed = _u32_array(token_ids, "token_ids")
    if token_count is None:
        token_count = len(packed)
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or not 0 <= token_count <= len(packed)
    ):
        raise CodecError("token_count must identify a prefix of token_ids")
    digest = hashlib.sha256()
    digest.update(identity_salt.encode("ascii"))
    digest.update(b"\x00")
    digest.update(memoryview(packed).cast("B")[: token_count * packed.itemsize])
    return digest.hexdigest()


def chunk_prefix_digests(
    token_ids: Iterable[int],
    identity_salt: str,
    *,
    boundaries: Iterable[int],
) -> tuple[tuple[int, str], ...]:
    """Return existing exact-context digests in one incremental hash pass."""

    packed = _u32_array(token_ids, "token_ids")
    try:
        requested = tuple(boundaries)
    except TypeError as error:
        raise CodecError("boundaries must be an iterable of token counts") from error
    previous = 0
    for boundary in requested:
        if (
            isinstance(boundary, bool)
            or not isinstance(boundary, int)
            or boundary <= 0
            or boundary % CHUNK_TOKENS
        ):
            raise CodecError(
                f"boundaries must be positive multiples of {CHUNK_TOKENS} tokens"
            )
        if boundary <= previous:
            raise CodecError("boundaries must be strictly increasing")
        if boundary > len(packed):
            raise CodecError("boundary must identify a prefix of token_ids")
        previous = boundary
    if not requested:
        return ()
    digest = hashlib.sha256()
    digest.update(identity_salt.encode("ascii"))
    digest.update(b"\x00")
    raw = memoryview(packed).cast("B")
    result = []
    previous = 0
    for boundary in requested:
        digest.update(raw[previous * packed.itemsize : boundary * packed.itemsize])
        result.append((boundary, digest.copy().hexdigest()))
        previous = boundary
    return tuple(result)


def chunk_count(span_tokens: int, chunk_tokens: int = CHUNK_TOKENS) -> int:
    if chunk_tokens <= 0:
        raise CodecError("chunk_tokens must be positive")
    if span_tokens <= 0 or span_tokens % chunk_tokens:
        raise CodecError("span must be a positive multiple of chunk_tokens")
    return span_tokens // chunk_tokens


@dataclass(frozen=True)
class LayerPlan:
    """Byte layout of one registered cache layer inside a record kind."""

    name: str
    record_kind: str
    bytes_per_token: int

    def __post_init__(self) -> None:
        if self.bytes_per_token <= 0:
            raise CodecError(f"layer {self.name} has no per-token bytes")


# Generic callers classify an unrecognized layer as target state. Deployment
# profiles supply any model-specific naming rules and required record families.
DEFAULT_CLASSIFICATION_RULES: tuple[tuple[str, str], ...] = ()
DEFAULT_REQUIRED_FAMILIES = frozenset({"target_ckv"})


def classify_layer(
    name: str,
    rules: Sequence[tuple[str, str]] = DEFAULT_CLASSIFICATION_RULES,
    default_family: str = "target_ckv",
) -> str:
    """Map a registered kv-cache layer name onto a persistent record kind.

    ``rules`` is an ordered sequence of case-insensitive substring
    patterns; the first matching pattern's family wins, and a name that
    matches nothing falls to ``default_family``.
    """
    lowered = name.lower()
    for pattern, family in rules:
        if pattern in lowered:
            return family
    return default_family


def build_layer_plans(
    layer_bytes_per_token: Mapping[str, int],
    *,
    allow_missing: frozenset[str] = frozenset(),
    required_families: frozenset[str] = DEFAULT_REQUIRED_FAMILIES,
    classification_rules: Sequence[tuple[str, str]] = DEFAULT_CLASSIFICATION_RULES,
    default_family: str = "target_ckv",
) -> tuple[LayerPlan, ...]:
    """Stable, sorted plan over every registered layer. Raises if any
    required record kind has no contributing layer, because storing a chunk
    that silently omits state is exactly the failure mode this cache exists
    to prevent. A kind may be exempted only through an explicit declared
    policy (``allow_missing``), e.g. draft KV colocated in the target pool
    when the runtime registers drafter layers without a naming marker."""
    plans = tuple(
        LayerPlan(
            name,
            classify_layer(name, classification_rules, default_family),
            layer_bytes_per_token[name],
        )
        for name in sorted(layer_bytes_per_token)
    )
    kinds = {plan.record_kind for plan in plans}
    missing = set(required_families) - kinds - set(allow_missing)
    if missing:
        raise CodecError(
            "no registered cache layer for record kinds: " + ", ".join(sorted(missing))
        )
    return plans


def pack_positions(positions: Sequence[int]) -> bytes:
    if not positions:
        raise CodecError("a chunk shard must own at least one position")
    return _pack_u32(positions, "positions")


def unpack_positions(payload: bytes) -> tuple[int, ...]:
    if not payload or len(payload) % _POSITION_STRUCT.size:
        raise CodecError("malformed positions payload")
    unpacked = array("I")
    unpacked.frombytes(payload)
    if unpacked.itemsize != _POSITION_STRUCT.size:
        raise CodecError("platform unsigned-int width does not match cache ABI")
    if sys.byteorder != "little":
        unpacked.byteswap()
    return tuple(unpacked)


def pack_record(
    plans: Sequence[LayerPlan],
    record_kind: str,
    layer_rows: Mapping[str, bytes],
    rows: int,
) -> bytes:
    """Concatenate per-layer row bytes for one record kind, sorted order.

    ``layer_rows`` maps layer name to the raw bytes of exactly ``rows``
    per-token records for that layer.
    """
    parts = []
    for plan in plans:
        if plan.record_kind != record_kind:
            continue
        payload = layer_rows.get(plan.name)
        if payload is None:
            raise CodecError(f"missing rows for layer {plan.name}")
        expected = plan.bytes_per_token * rows
        if len(payload) != expected:
            raise CodecError(
                f"layer {plan.name} rows are {len(payload)} bytes, expected {expected}"
            )
        parts.append(payload)
    if not parts:
        raise CodecError(f"no layers contribute to {record_kind}")
    return b"".join(parts)


def unpack_record(
    plans: Sequence[LayerPlan],
    record_kind: str,
    payload: bytes,
    rows: int,
) -> dict[str, bytes]:
    """Split a packed record back into per-layer row bytes, verifying the
    total length exactly; trailing bytes are a hard error."""
    out: dict[str, bytes] = {}
    offset = 0
    for plan in plans:
        if plan.record_kind != record_kind:
            continue
        size = plan.bytes_per_token * rows
        if offset + size > len(payload):
            raise CodecError(f"record {record_kind} truncated at layer {plan.name}")
        out[plan.name] = payload[offset : offset + size]
        offset += size
    if not out:
        raise CodecError(f"no layers contribute to {record_kind}")
    if offset != len(payload):
        raise CodecError(
            f"record {record_kind} carries {len(payload) - offset}"
            " unclaimed trailing bytes"
        )
    return out
