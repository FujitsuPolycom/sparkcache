"""Pure block-page codec for heterogeneous KV-cache groups.

The codec treats every layer page as opaque bytes. Group block tables define
logical ordering; layer names, dtypes, page shapes, bytes-per-page, and the
semantic reuse window define the layout. This preserves compressed attention
and recurrent compressor state without assigning misleading per-token
geometry to either one.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

_MAGIC = b"SPHP1\x00"
_HEADER_LENGTH = struct.Struct("<I")


class HybridCodecError(ValueError):
    """A block-page snapshot does not match its declared layout."""


@dataclass(frozen=True)
class PageLayer:
    name: str
    dtype: str
    page_shape: tuple[int, ...]
    bytes_per_page: int

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.page_shape:
            raise HybridCodecError("page layer metadata is incomplete")
        if self.bytes_per_page <= 0 or any(size <= 0 for size in self.page_shape):
            raise HybridCodecError("page layer geometry must be positive")


@dataclass(frozen=True)
class PageGroup:
    block_size: int
    layers: tuple[PageLayer, ...]
    reuse_policy: str = "full"
    reuse_window_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.block_size <= 0 or not self.layers:
            raise HybridCodecError("page group geometry is incomplete")
        if self.reuse_policy not in {"full", "sliding", "recurrent_align"}:
            raise HybridCodecError("page group reuse policy is unsupported")
        if self.reuse_policy in {"full", "recurrent_align"} and (
            self.reuse_window_tokens is not None
        ):
            raise HybridCodecError(
                f"{self.reuse_policy} page group cannot declare a reuse window"
            )
        if self.reuse_policy == "sliding" and (
            self.reuse_window_tokens is None or self.reuse_window_tokens <= 1
        ):
            raise HybridCodecError("sliding page group reuse window is invalid")
        names = [layer.name for layer in self.layers]
        if names != sorted(names) or len(names) != len(set(names)):
            raise HybridCodecError("page group layers must be sorted and unique")


@dataclass(frozen=True)
class PageLayout:
    groups: tuple[PageGroup, ...]

    def __post_init__(self) -> None:
        if not self.groups:
            raise HybridCodecError("page layout has no groups")
        names = [layer.name for group in self.groups for layer in group.layers]
        if len(names) != len(set(names)):
            raise HybridCodecError("a layer belongs to multiple page groups")

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def encode_page_snapshot(
    layout: PageLayout,
    block_counts: Sequence[int],
    layer_payloads: Mapping[str, bytes],
) -> bytes:
    if len(block_counts) != len(layout.groups):
        raise HybridCodecError("block-count vector disagrees with page groups")
    counts = tuple(int(count) for count in block_counts)
    if any(count <= 0 for count in counts):
        raise HybridCodecError("every page group must contribute at least one block")
    expected_names = {layer.name for group in layout.groups for layer in group.layers}
    if set(layer_payloads) != expected_names:
        raise HybridCodecError("page payload names disagree with layout")

    parts: list[bytes] = []
    for group, count in zip(layout.groups, counts):
        for layer in group.layers:
            payload = layer_payloads[layer.name]
            expected = count * layer.bytes_per_page
            if len(payload) != expected:
                raise HybridCodecError(
                    f"layer {layer.name} carries {len(payload)} bytes, expected {expected}"
                )
            parts.append(payload)
    header = json.dumps(
        {
            "schema": "sparkcache-hybrid-pages/v1",
            "layout_sha256": layout.digest,
            "block_counts": counts,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _MAGIC + _HEADER_LENGTH.pack(len(header)) + header + b"".join(parts)


def decode_page_snapshot(
    layout: PageLayout,
    encoded: bytes,
    expected_block_counts: Sequence[int],
) -> dict[str, bytes]:
    prefix = len(_MAGIC) + _HEADER_LENGTH.size
    if len(encoded) < prefix or not encoded.startswith(_MAGIC):
        raise HybridCodecError("hybrid page snapshot has an invalid prefix")
    (header_length,) = _HEADER_LENGTH.unpack_from(encoded, len(_MAGIC))
    header_end = prefix + header_length
    if header_length <= 0 or header_end > len(encoded):
        raise HybridCodecError("hybrid page snapshot header is truncated")
    try:
        header = json.loads(encoded[prefix:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HybridCodecError("hybrid page snapshot header is invalid") from error
    counts = tuple(int(count) for count in expected_block_counts)
    if (
        header.get("schema") != "sparkcache-hybrid-pages/v1"
        or header.get("layout_sha256") != layout.digest
        or tuple(header.get("block_counts", ())) != counts
        or len(counts) != len(layout.groups)
    ):
        raise HybridCodecError("hybrid page snapshot identity or block counts differ")

    payloads: dict[str, bytes] = {}
    offset = header_end
    for group, count in zip(layout.groups, counts):
        for layer in group.layers:
            size = count * layer.bytes_per_page
            end = offset + size
            if end > len(encoded):
                raise HybridCodecError(
                    f"hybrid page snapshot is truncated at {layer.name}"
                )
            payloads[layer.name] = encoded[offset:end]
            offset = end
    if offset != len(encoded):
        raise HybridCodecError("hybrid page snapshot has trailing bytes")
    return payloads


def split_snapshot(encoded: bytes, part_count: int) -> tuple[bytes, ...]:
    if part_count <= 0 or len(encoded) < part_count:
        raise HybridCodecError("hybrid snapshot cannot cover the declared span")
    width, remainder = divmod(len(encoded), part_count)
    parts = []
    offset = 0
    for index in range(part_count):
        size = width + (1 if index < remainder else 0)
        parts.append(encoded[offset : offset + size])
        offset += size
    return tuple(parts)
