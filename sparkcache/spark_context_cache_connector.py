"""SparkCache persistent NVMe context-cache connector.

KVConnectorBase_V1 implementation that persists each DCP rank's token shard
of every registered cache layer to rank-local NVMe through the fail-closed
ManifestStore, and restores it after a runtime restart. Supported degrees:
any tensor-parallel size and any decode-context-parallel size that divides
the profile's chunk length (DCP1 stores each rank's full span). The mapping
from a model's cache layers to persistent record families comes from a named
ModelProfile (``sparkcache.spark_context_cache_profiles``). Implemented profiles cover
GLM-5.2 per-token rows and DeepSeek-V4 opaque hybrid-memory-allocator (HMA)
pages.

Restore uses KV-Connector-V1's asynchronous-load contract. A cache hit parks
only that request, allocates private blocks, and queues verification,
assembly, H2D copy, and scatter on a background loader. The request is
reported finished only after its writes have completed; unrelated requests
remain schedulable while it restores.

Fail-closed contract:
- Storage uses content-addressed chunks with per-record SHA-256 and an
  identity pinning model/quant/TP/DCP/shard-rank/chunk geometry.
- A failed asynchronous load publishes its invalid blocks and finished
  request id together. vLLM can then unpark it through the supported clean
  recompute path instead of exposing partially restored state.
- Store failures only log and count; they never fail a request.

Enabled explicitly via --kv-transfer-config naming this module. Without that
argument, vLLM does not instantiate SparkCache.
"""

from __future__ import annotations

import contextlib
import importlib
import math

import queue
import threading
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from vllm.logger import init_logger

from sparkcache.spark_context_cache_codec import (
    CHUNK_TOKENS,
    CodecError,
    build_layer_plans,
    chunk_count,
    classify_layer,
    context_digest,
    local_slots_for_positions,
    owned_positions,
    pack_positions,
    pack_record,
    unpack_positions,
    unpack_record,
)
from sparkcache.spark_context_cache_config import (
    SHA256_RE,
    parse_connector_config,
)
from sparkcache.spark_context_cache_hybrid import (
    HybridCodecError,
    PageGroup,
    PageLayer,
    PageLayout,
    decode_page_snapshot,
    encode_page_snapshot,
    split_snapshot,
)
from sparkcache.spark_context_cache_store import (
    CacheIdentity,
    ContextChunk,
    MaintenanceReport,
    ManifestStore,
    StateRecord,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger("vllm.spark_context_cache")

_CAPACITY_RETRY_SECONDS = 5.0

# The runtime is deliberately supplied by the embedding process instead of
# importing or constructing the optional streaming-snapshot runtime here.
# A scheduler and each worker have distinct ownership, so factories are keyed
# by connector role.  Installing a factory is an explicit deployment action;
# setting the feature flag without one remains a startup error.
_STREAMING_RUNTIME_FACTORIES: dict[
    KVConnectorRole, Callable[["SparkContextCacheConnector"], Any]
] = {}


def configure_streaming_snapshot_runtime(
    role: KVConnectorRole,
    factory: Callable[["SparkContextCacheConnector"], Any] | None,
) -> None:
    """Install or remove the explicitly injected streaming runtime factory.

    This is a narrow integration seam for an attested embedding runtime.  It
    does not create, import, or otherwise enable a native runtime by itself.
    ``factory`` receives the connector after its basic vLLM configuration is
    available and must return a role-appropriate adapter.
    """

    if not isinstance(role, KVConnectorRole):
        raise TypeError("streaming runtime role must be a KVConnectorRole")
    if factory is None:
        _STREAMING_RUNTIME_FACTORIES.pop(role, None)
        return
    if not callable(factory):
        raise TypeError("streaming runtime factory must be callable")
    _STREAMING_RUNTIME_FACTORIES[role] = factory


def _load_native_components() -> SimpleNamespace:
    """Lazy optional import: the Python restore path needs no native bundle."""

    placement = importlib.import_module(
        "sparkcache.spark_context_cache_native_placement"
    )
    restore = importlib.import_module("sparkcache.spark_context_cache_native_restore")
    return SimpleNamespace(
        NativePlacementLibrary=placement.NativePlacementLibrary,
        NativePlacementAdapter=placement.NativePlacementAdapter,
        ArenaMode=placement.ArenaMode,
        RecordKind=placement.RecordKind,
        execute_native_restore=restore.execute_native_restore,
    )

@dataclass
class _ReqPlan:
    request_id: str
    digest: str
    span_tokens: int
    block_ids: tuple[int, ...]
    is_store: bool
    block_ids_by_group: tuple[tuple[int, ...], ...] = ()

    @property
    def group_block_ids(self) -> tuple[tuple[int, ...], ...]:
        return self.block_ids_by_group or (self.block_ids,)


@dataclass(frozen=True)
class _StreamingSnapshotOffer:
    """Scheduler promise that becomes a worker completion after forward.

    ``completed_tokens`` is intentionally only a scheduler-side watermark.
    The worker must not hand it to the streaming runtime until
    :meth:`wait_for_save`, which vLLM calls after the corresponding forward.
    ``block_ids`` is the complete table known for that watermark, not merely
    the blocks allocated by the current cached-request step.
    """

    request_id: str
    digest: str
    span_tokens: int
    completed_tokens: int
    block_ids: tuple[int, ...]

    @property
    def span(self) -> int:
        return self.span_tokens

    @property
    def promised_completed_tokens(self) -> int:
        return self.completed_tokens

    @property
    def full_block_table(self) -> tuple[int, ...]:
        return self.block_ids


@dataclass
class _PendingStreamingCommit:
    """Durable streaming manifest awaiting capacity-safe advertisement."""

    receipt: Any
    accounted: bool = False


@dataclass(frozen=True)
class _StoreSnapshot:
    """CPU-owned immutable state required to commit one cache entry.

    No live KV tensor is retained after construction, so vLLM may reuse the
    source blocks as soon as ``wait_for_save`` returns.
    """

    plan: _ReqPlan
    rank: int
    identity: CacheIdentity
    positions: tuple[int, ...]
    layer_bytes: dict[str, bytes]
    layer_plans: tuple[Any, ...]
    record_kinds: tuple[str, ...]


@dataclass(frozen=True)
class _HybridStoreSnapshot:
    plan: _ReqPlan
    rank: int
    identity: CacheIdentity
    positions: tuple[int, ...]
    encoded_pages: bytes


class _HybridSnapshotChunks(Sequence[ContextChunk]):
    def __init__(self, snapshot: _HybridStoreSnapshot, chunk_tokens: int) -> None:
        self._snapshot = snapshot
        self._count = chunk_count(snapshot.plan.span_tokens, chunk_tokens)
        self._chunk_tokens = chunk_tokens
        self._parts = split_snapshot(snapshot.encoded_pages, self._count)

    def __len__(self) -> int:
        return self._count

    def __getitem__(
        self, index: int | slice
    ) -> ContextChunk | tuple[ContextChunk, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(self._count)))
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        start = index * self._chunk_tokens
        end = (index + 1) * self._chunk_tokens
        return ContextChunk(
            logical_start=start,
            logical_end=end,
            records={
                StateRecord.LOGICAL_POSITIONS: pack_positions(range(start, end)),
                StateRecord.TARGET_CKV: self._parts[index],
            },
        )


class _SnapshotChunks(Sequence[ContextChunk]):
    """Build one encoded chunk at a time from an immutable CPU snapshot."""

    def __init__(
        self,
        snapshot: _StoreSnapshot,
        dcp_degree: int,
        chunk_tokens: int = CHUNK_TOKENS,
    ) -> None:
        if chunk_tokens <= 0 or chunk_tokens % dcp_degree:
            raise CodecError(
                f"chunk_tokens {chunk_tokens} is not divisible by"
                f" dcp_degree {dcp_degree}"
            )
        self._snapshot = snapshot
        self._dcp_degree = dcp_degree
        self._chunk_tokens = chunk_tokens
        self._count = chunk_count(snapshot.plan.span_tokens, chunk_tokens)
        self._rows_per_chunk = chunk_tokens // dcp_degree

    def __len__(self) -> int:
        return self._count

    def __getitem__(
        self, index: int | slice
    ) -> ContextChunk | tuple[ContextChunk, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(self._count)))
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        row_slice = slice(
            index * self._rows_per_chunk,
            (index + 1) * self._rows_per_chunk,
        )
        records = {
            StateRecord.LOGICAL_POSITIONS: pack_positions(
                self._snapshot.positions[row_slice]
            )
        }
        for kind in self._snapshot.record_kinds:
            rows_map = {}
            for plan_entry in self._snapshot.layer_plans:
                if plan_entry.record_kind != kind:
                    continue
                width = plan_entry.bytes_per_token
                full = self._snapshot.layer_bytes[plan_entry.name]
                rows_map[plan_entry.name] = full[
                    row_slice.start * width : row_slice.stop * width
                ]
            records[StateRecord(kind)] = pack_record(
                self._snapshot.layer_plans,
                kind,
                rows_map,
                self._rows_per_chunk,
            )
        return ContextChunk(
            logical_start=index * self._chunk_tokens,
            logical_end=(index + 1) * self._chunk_tokens,
            records=records,
        )


@dataclass
class SparkCacheConnectorMetadata(KVConnectorMetadata):
    plans: list[_ReqPlan] = field(default_factory=list)
    streaming_snapshot_offers: list[_StreamingSnapshotOffer] = field(
        default_factory=list
    )
    preempted_request_ids: tuple[str, ...] = ()

    # Keep the transport field descriptive while making inspection by an
    # embedding runtime pleasantly direct.
    @property
    def offers(self) -> list[_StreamingSnapshotOffer]:
        return self.streaming_snapshot_offers


@dataclass
class SparkCacheStats(KVConnectorStats):
    """Per-rank "digests I can serve" reports, merged across workers.

    MUST be defined at module scope: these objects are pickled across the
    worker->engine shared-memory queue, and a class created inside a
    function is not picklable.
    """

    def reset(self):
        self.data = {"reports": []}

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        mine = list(self.data.get("reports", []))
        theirs = list(getattr(other, "data", {}).get("reports", []))
        merged: dict[int, dict[str, Any]] = {}
        for report in mine + theirs:
            if isinstance(report, dict) and isinstance(report.get("rank"), int):
                normalized: dict[str, Any] = {
                    "rank": report["rank"],
                    "held": list(report.get("held", [])),
                }
                generation = report.get("generation")
                if isinstance(generation, str) and generation:
                    normalized["generation"] = generation
                streaming = report.get("streaming")
                if isinstance(streaming, dict):
                    normalized["streaming"] = dict(streaming)
                capacity = report.get("capacity")
                if isinstance(capacity, dict):
                    normalized["capacity"] = dict(capacity)
                merged[report["rank"]] = normalized
        self.data = {"reports": [merged[rank] for rank in sorted(merged)]}
        return self

    def reduce(self) -> dict[str, int | float]:
        reports = self.data.get("reports", [])
        reduced: dict[str, int | float] = {
            "spark_cache_ranks_reporting": len(reports),
            "spark_cache_digests_held": sum(len(r.get("held", [])) for r in reports),
        }
        streaming = [
            report.get("streaming")
            for report in reports
            if isinstance(report.get("streaming"), dict)
        ]
        if streaming:
            reduced.update(
                spark_cache_streaming_ranks_reporting=len(streaming),
                spark_cache_streaming_active_contexts=sum(
                    int(status.get("active_contexts", 0)) for status in streaming
                ),
                spark_cache_streaming_active_leases=sum(
                    int(status.get("active_leases", 0)) for status in streaming
                ),
                spark_cache_streaming_active_tickets=sum(
                    int(status.get("active_tickets", 0)) for status in streaming
                ),
            )
        capacity = [
            report.get("capacity")
            for report in reports
            if isinstance(report.get("capacity"), dict)
        ]
        if capacity:
            reduced.update(
                spark_cache_capacity_ranks_reporting=len(capacity),
                spark_cache_capacity_bytes=sum(
                    int(status.get("bytes", 0)) for status in capacity
                ),
                spark_cache_capacity_max_bytes=sum(
                    int(status.get("max_bytes", 0)) for status in capacity
                ),
                spark_cache_capacity_manifests_evicted=sum(
                    int(status.get("manifests_evicted", 0))
                    for status in capacity
                ),
                spark_cache_capacity_bytes_reclaimed=sum(
                    int(status.get("bytes_reclaimed", 0)) for status in capacity
                ),
                spark_cache_capacity_pending_streaming_commits=sum(
                    int(status.get("pending_streaming_commits", 0))
                    for status in capacity
                ),
                spark_cache_streaming_store_evicted=sum(
                    int(status.get("streaming_store_evicted", 0))
                    for status in capacity
                ),
                spark_cache_capacity_invalid_streaming_receipts=sum(
                    int(status.get("invalid_streaming_receipts", 0))
                    for status in capacity
                ),
                spark_cache_capacity_shutdown_dropped_streaming_commits=sum(
                    int(status.get("shutdown_dropped_streaming_commits", 0))
                    for status in capacity
                ),
                spark_cache_capacity_satisfied=int(
                    all(
                        bool(status.get("capacity_satisfied", False))
                        for status in capacity
                    )
                ),
            )
        return reduced

    def is_empty(self) -> bool:
        return not self.data.get("reports")


class SparkContextCacheConnector(KVConnectorBase_V1, SupportsHMA):
    """Store/restore each rank's DCP shard on rank-local NVMe."""

    configure_streaming_snapshot_runtime = staticmethod(
        configure_streaming_snapshot_runtime
    )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._kv_cache_config = kv_cache_config
        config = parse_connector_config(
            vllm_config, self._kv_transfer_config, kv_cache_config
        )
        self._config = config
        self._block_size = config.block_size
        self._tp_degree = config.tp_degree
        self._dcp_degree = config.dcp_degree
        self._profile = config.profile
        self._storage_mode = config.storage_mode
        self._group_topology = config.group_topology
        self._chunk_tokens = config.chunk_tokens
        self._root = config.root
        self._capacity_policy = config.capacity_policy
        self._min_span = config.min_span
        self._max_span = config.max_span
        self._store_enabled = config.store_enabled
        self._restore_enabled = config.restore_enabled
        self._streaming_snapshots_enabled = config.streaming_snapshots_enabled
        self._streaming_runtime: Any = None
        if self._streaming_snapshots_enabled:
            # An explicit opt-in never falls back to end-of-prefill snapshots.
            # Preserve an explicitly injected test/deployment adapter. When
            # none exists, import the builtin factory only on this opt-in
            # path; default-off scheduler and worker processes never import
            # factory/native-ring modules.
            if role not in _STREAMING_RUNTIME_FACTORIES:
                from sparkcache.streaming.factory import (
                    make_model_serving_runtime_factory,
                )

                _STREAMING_RUNTIME_FACTORIES[role] = (
                    make_model_serving_runtime_factory(role.name)
                )
            # Resolve the adapter before native import/allocation so a
            # partially configured deployment fails closed at startup.
            self._install_streaming_runtime(role)
        self._native_restore_enabled = config.native_restore_enabled
        self._native_library_path = config.native_library_path
        self._native_library_sha256 = config.native_library_sha256
        self._native_arena_bytes = config.native_arena_bytes
        self._native_io_workers = config.native_io_workers
        self._identity_base = config.identity_base
        self._scheduler_probe = config.scheduler_probe
        self._shard_rank = 0
        self._store = ManifestStore(self._root)
        self._capacity_estimated_bytes = 0
        self._capacity_status: dict[str, int | bool] = {
            "max_bytes": self._capacity_policy.max_bytes,
            "low_watermark_bytes": self._capacity_policy.low_watermark_bytes,
            "ttl_seconds": self._capacity_policy.ttl_seconds,
            "bytes": 0,
            "bytes_exact": False,
            "capacity_satisfied": True,
        }
        self._capacity_wakeup = threading.Event()
        self._capacity_stop = threading.Event()
        self._capacity_thread: threading.Thread | None = None
        # Filesystem maintenance and its survivor publication form one serial
        # capacity operation. This lock is never taken by inference callbacks;
        # streaming callbacks enqueue a receipt and wake the janitor instead.
        self._capacity_lock = threading.RLock()
        self._capacity_commit_queue: "queue.SimpleQueue[tuple[str, Any]]" = (
            queue.SimpleQueue()
        )
        self._capacity_handoff_cv = threading.Condition()
        self._streaming_capacity_pending: set[str] = set()
        # Worker state.
        self._layer_tensors: dict[str, torch.Tensor] = {}
        self._plans = None
        self._page_layout: PageLayout | None = None
        self._record_kinds: tuple[str, ...] = ()
        # Digests this rank can offer: either discovered from a structurally
        # valid manifest at startup or published by a durable ManifestStore
        # commit. Every restore remains the byte/hash integrity boundary.
        # Reported to the scheduler each step so admission can require
        # unanimity.
        self._held: set[str] = set()
        self._pending_saves: dict[str, dict[str, torch.Tensor]] = {}
        self._load_errors: set[int] = set()
        self._load_lock = threading.Lock()
        self._load_cv = threading.Condition(self._load_lock)
        self._store_cv = threading.Condition(self._load_lock)
        self._store_queue: "queue.SimpleQueue[_StoreSnapshot | _HybridStoreSnapshot | None]" = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._store_inflight = 0
        self._store_accepting = True
        self._load_queue: "queue.SimpleQueue[_ReqPlan | None]" = queue.SimpleQueue()
        self._load_threads: list[threading.Thread] = []
        self._load_thread_limit = config.load_thread_limit
        if self._native_restore_enabled:
            # One native adapter owns one transaction and two arenas. Keep
            # outer restores serialized; its bounded inner preadv/hash pool
            # supplies the desired storage parallelism.
            self._load_thread_limit = 1
        self._inflight_load_reqs: set[str] = set()
        self._finished_load_reqs: set[str] = set()
        self._load_stream: Any = None
        self._native_adapter: Any = None
        self._native_execute_restore: Any = None
        self._native_required_record_mask = 0
        # Scheduler state.
        self._need_load: dict[str, tuple[str, int]] = {}
        # Bound on concurrently promised async restores. Enforced at
        # admission in get_num_new_matched_tokens; entries are consumed by
        # update_state_after_alloc / build_connector_meta and cleared by
        # request_finished, so the map cannot grow past in-flight requests.
        self._max_pending_restores = config.max_pending_restores
        # Async loads are parked after block allocation, so they do not
        # appear in the next SchedulerOutput. Carry their allocated blocks
        # across that boundary explicitly.
        self._pending_async_loads: dict[
            str, tuple[str, int, tuple[tuple[int, ...], ...]]
        ] = {}
        # Digests this scheduler has admitted a restore for, so a reported
        # load failure on ANY rank can retire the entry cluster-wide.
        # request_id -> (digest, restored block ids). The block set lets a
        # load-error report retire exactly the failing entry. Entries are
        # removed on rescheduling, on retirement, or by request_finished.
        self._admitted: dict[str, tuple[str, frozenset[int]]] = {}
        # Quorum map: digest -> ranks that can offer a compatible manifest.
        # A restore is only offered at full quorum; every participating worker
        # then verifies its chunk bytes before writing private request blocks.
        self._quorum: dict[str, set[int]] = {}
        # A worker process gets a fresh generation on every connector
        # construction. The scheduler withdraws that physical rank's old
        # confirmations before accepting a report from a different generation,
        # so an isolated worker restart cannot inherit stale quorum state.
        self._stats_generation = uuid.uuid4().hex
        self._worker_generations: dict[int, str] = {}
        self._store_progress: dict[str, tuple[str, int, int, list[list[int]]]] = {}
        self.counters: dict[str, int] = {
            "store_committed": 0,
            "store_failed": 0,
            "store_evicted": 0,
            "store_skipped_busy": 0,
            "store_skipped_present": 0,
            "store_skipped_quorum": 0,
            "restore_hit": 0,
            "restore_miss_absent": 0,
            "restore_miss_corrupt": 0,
            "restore_miss_incompatible": 0,
            "quorum_generation_resets": 0,
            "load_verified": 0,
            "load_failed": 0,
            "capacity_runs": 0,
            "capacity_skipped_busy": 0,
            "capacity_failed": 0,
            "capacity_manifests_evicted": 0,
            "capacity_chunks_deleted": 0,
            "capacity_bytes_reclaimed": 0,
            "capacity_retries": 0,
            "streaming_store_committed": 0,
            "streaming_store_evicted": 0,
            "streaming_capacity_queued": 0,
            "streaming_capacity_retries": 0,
            "streaming_capacity_invalid_receipts": 0,
            "streaming_capacity_shutdown_dropped": 0,
        }
        logger.info(
            "spark-context-cache: role=%s root=%s dcp=%d store=%s restore=%s"
            " native_restore=%s max_span=%d load_threads=%d max_bytes=%d"
            " low_bytes=%d ttl_seconds=%d",
            role.name,
            self._root,
            self._dcp_degree,
            self._store_enabled,
            self._restore_enabled,
            self._native_restore_enabled,
            self._max_span,
            self._load_thread_limit,
            self._capacity_policy.max_bytes,
            self._capacity_policy.low_watermark_bytes,
            self._capacity_policy.ttl_seconds,
        )

    def _install_streaming_runtime(self, role: KVConnectorRole) -> None:
        """Resolve the opt-in runtime without importing a native backend."""

        factory = _STREAMING_RUNTIME_FACTORIES.get(role)
        if factory is None:
            raise RuntimeError(
                "spark-context-cache: streaming snapshots requested but no "
                f"runtime is installed for {role.name.lower()} role"
            )
        try:
            runtime = factory(self)
        except Exception as error:
            raise RuntimeError(
                "spark-context-cache: streaming runtime installation failed closed"
            ) from error
        if runtime is None:
            raise RuntimeError(
                "spark-context-cache: streaming runtime installation failed closed"
            )
        if role is KVConnectorRole.SCHEDULER:
            required = (
                "observe_metadata",
                "request_finished",
                "take_finished",
                "shutdown",
            )
        else:
            required = (
                "bind_kv_caches",
                "offer_completed",
                "poll",
                "handle_preemptions",
                "request_finished",
                "take_finished",
                "shutdown",
            )
        missing = [
            name for name in required if not callable(getattr(runtime, name, None))
        ]
        if missing:
            raise RuntimeError(
                "spark-context-cache: streaming "
                f"{role.name.lower()} runtime is incomplete: " + ", ".join(missing)
            )
        self._streaming_runtime = runtime

    # ------------------------------------------------------------------
    # identity helpers
    # ------------------------------------------------------------------

    def _identity(
        self, shard_rank: int, tp_shard_rank: int | None = None
    ) -> CacheIdentity:
        """Build a CacheIdentity for the given DCP shard rank.

        tp_shard_rank defaults to the scheduler's shard rank (0) when
        None and the connector is on the scheduler side, or to the
        worker's physical TP rank when on the worker side.  This
        ensures persistent storage keys distinguish physical workers
        that share a DCP-local rank (e.g. TP0 and TP2 under DCP2).
        """
        if tp_shard_rank is None:
            tp_shard_rank = (
                self._shard_rank
                if self._role is KVConnectorRole.SCHEDULER
                else self._physical_rank()
            )
        return self._config.build_identity(shard_rank, tp_shard_rank)

    def _digest(self, token_ids: list[int], span: int) -> str:
        # The salt must be identical on every role and rank: it names the
        # shared context, not this process's shard. Pin both shard fields to
        # zero rather than letting the worker role substitute its physical
        # rank, which would fork the digest namespace per worker.
        salt = self._identity(0, tp_shard_rank=0).storage_key
        return context_digest(token_ids[:span], salt)

    def _aligned_span(self, prompt_len: int) -> int:
        span = (prompt_len - 1) // self._block_size * self._block_size
        return span // self._chunk_tokens * self._chunk_tokens

    @staticmethod
    def _normalize_group_blocks(
        block_ids: Sequence[Sequence[int]],
    ) -> tuple[tuple[int, ...], ...]:
        groups = tuple(tuple(int(block) for block in group) for group in block_ids)
        if not groups or any(not group for group in groups):
            raise RuntimeError("spark-context-cache: KV-cache block table is empty")
        return groups

    def _select_group_blocks_for_span(
        self,
        groups: tuple[tuple[int, ...], ...],
        span_tokens: int,
    ) -> tuple[tuple[int, ...], ...]:
        if len(groups) != len(self._group_topology):
            raise HybridCodecError("request block tables disagree with page groups")
        trimmed = []
        for group, topology in zip(groups, self._group_topology):
            block_size = int(topology["block_size"])
            required = (span_tokens + block_size - 1) // block_size
            if len(group) < required:
                raise HybridCodecError(
                    "request block table is shorter than the aligned cache span"
                )
            policy = topology["reuse_policy"]
            window = topology["reuse_window_tokens"]
            selected = required
            if policy == "sliding":
                if window is None:
                    raise HybridCodecError("sliding page group has no reuse window")
                # The window includes the token about to be computed. Only its
                # preceding window-1 KV positions must be resident.
                selected = min(
                    required,
                    (int(window) - 1 + block_size - 1) // block_size,
                )
            elif policy != "full":
                raise HybridCodecError(f"unsupported page reuse policy {policy!r}")
            # The table can include a live tail beyond the persistent span.
            # Sliding/recurrent groups can also contain recycled or null IDs
            # before their semantic window. Select relative to the persistent
            # boundary so neither class of unrelated page is copied. A
            # SparkCache manifest names one exact full span; its constituent
            # chunks are never matched as independent prefixes, so only the
            # reuse window at that span's final boundary is required.
            chosen = group[required - selected : required]
            if any(block <= 0 for block in chosen):
                raise HybridCodecError(
                    "selected page window contains vLLM's null block"
                )
            if len(set(chosen)) != len(chosen):
                raise HybridCodecError(
                    "selected page window contains duplicate physical blocks"
                )
            trimmed.append(chosen)
        return tuple(trimmed)

    def _has_full_quorum(self, digest: str) -> bool:
        """Return whether every physical TP worker already offers this cache entry.

        Quorum requires all physical TP workers in range(tp_degree),
        not just DCP-local ranks.  Under TP4/DCP2, this means all four
        physical workers (TP0..TP3) must confirm, preventing a single
        DCP group from falsely satisfying quorum while the other
        physical workers lack data.
        """

        confirmed = self._quorum.get(digest, ())
        return all(rank in confirmed for rank in range(self._tp_degree))

    # ------------------------------------------------------------------
    # scheduler side
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if not self._restore_enabled:
            return 0, False
        token_ids = list(request.prompt_token_ids or [])
        span = self._aligned_span(len(token_ids))
        if span < self._min_span or span <= num_computed_tokens:
            return 0, False
        if span > self._max_span:
            self.counters["restore_skip_oversize"] = (
                self.counters.get("restore_skip_oversize", 0) + 1
            )
            return 0, False
        digest = self._digest(token_ids, span)
        # Manifest-only probe: chunk payloads are re-read and re-hashed by
        # every worker at load time, and any worker failure degrades to a
        # rank-synchronous recompute, so hashing ~200 MB here would only
        # add scheduler latency without adding safety.
        if not self._has_full_quorum(digest):
            # Not every rank can offer a compatible manifest (or none has
            # reported yet). Treat as a plain miss: the request re-prefills
            # and republishes, which is also how a corrupted entry retires.
            self.counters["quorum_incomplete"] = (
                self.counters.get("quorum_incomplete", 0) + 1
            )
            return 0, False
        if self._scheduler_probe == "tp0":
            lookup = self._store.lookup(
                self._identity(self._shard_rank), digest, verify_chunks=False
            )
            if not lookup.is_hit:
                self.counters[f"restore_miss_{lookup.reason}"] = (
                    self.counters.get(f"restore_miss_{lookup.reason}", 0) + 1
                )
                return 0, False
        # probe mode "none": quorum alone gates admission; every worker
        # validated its own manifest at discovery or commit time, and any
        # rank's load failure still degrades to a clean recompute.
        self.counters["restore_hit"] += 1
        logger.info(
            "spark-context-cache: scheduler hit digest=%s span=%d (async)",
            digest[:12],
            span,
        )
        if (
            request.request_id not in self._need_load
            and len(self._need_load) >= self._max_pending_restores
        ):
            # Returning a hit commits vLLM to an asynchronous load, so the
            # bound must act here, before that promise is made. Evicting an
            # already-promised entry instead would leave its request parked
            # with nothing scheduled to load or finish it.
            self.counters["restore_skip_backlog"] = (
                self.counters.get("restore_skip_backlog", 0) + 1
            )
            return 0, False
        self._need_load[request.request_id] = (digest, span)
        return span - num_computed_tokens, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        request_id = request.request_id
        if num_external_tokens <= 0:
            self._need_load.pop(request_id, None)
            return
        entry = self._need_load.pop(request_id, None)
        if entry is None:
            return
        digest, span = entry
        group_blocks = self._normalize_group_blocks(blocks.get_block_ids())
        self._pending_async_loads[request_id] = (
            digest,
            span,
            group_blocks,
        )

    @staticmethod
    def _append_streaming_snapshot_offer(
        meta: SparkCacheConnectorMetadata,
        *,
        request_id: str,
        digest: str,
        span: int,
        promised_completed_tokens: int,
        block_ids: Sequence[int],
    ) -> None:
        """Record a scheduler promise without claiming forward completed it."""

        completed_tokens = min(span, promised_completed_tokens)
        if completed_tokens <= 0:
            return
        meta.streaming_snapshot_offers.append(
            _StreamingSnapshotOffer(
                request_id=request_id,
                digest=digest,
                span_tokens=span,
                completed_tokens=completed_tokens,
                block_ids=tuple(block_ids),
            )
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = SparkCacheConnectorMetadata(
            preempted_request_ids=tuple(
                sorted(getattr(scheduler_output, "preempted_req_ids", None) or ())
            )
        )
        for request_id, (
            digest,
            span,
            group_blocks,
        ) in self._pending_async_loads.items():
            all_blocks = frozenset(block for group in group_blocks for block in group)
            self._admitted[request_id] = (digest, all_blocks)
            meta.plans.append(
                _ReqPlan(
                    request_id,
                    digest,
                    span,
                    group_blocks[0],
                    is_store=False,
                    block_ids_by_group=group_blocks,
                )
            )
        self._pending_async_loads.clear()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            req_id = new_req.req_id
            self._need_load.pop(req_id, None)
            scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            group_blocks = self._normalize_group_blocks(new_req.block_ids)
            block_ids = group_blocks[0]
            span = self._aligned_span(len(token_ids))
            if self._store_enabled and self._min_span <= span <= self._max_span:
                digest = self._digest(token_ids, span)
                admitted = self._admitted.get(req_id)
                if admitted is not None and admitted[0] == digest:
                    # The restored entry already exists on every rank. A
                    # failed load retires this admission before recompute,
                    # so only verified restores take this fast exit.
                    self._admitted.pop(req_id, None)
                    continue
                if self._has_full_quorum(digest):
                    self.counters["store_skipped_quorum"] += 1
                    continue
                already = new_req.num_computed_tokens + scheduled
                if self._streaming_snapshots_enabled:
                    self._append_streaming_snapshot_offer(
                        meta,
                        request_id=req_id,
                        digest=digest,
                        span=span,
                        promised_completed_tokens=already,
                        block_ids=block_ids,
                    )
                    if already < span:
                        # CachedRequestData.new_block_ids only carries the
                        # current append. Keep the complete table required by
                        # each following offer, including after resume.
                        self._store_progress[req_id] = (
                            digest,
                            span,
                            already,
                            [list(group) for group in group_blocks],
                        )
                elif already >= span:
                    meta.plans.append(
                        _ReqPlan(
                            req_id,
                            digest,
                            span,
                            block_ids,
                            True,
                            block_ids_by_group=group_blocks,
                        )
                    )
                else:
                    # Chunked prefill: CachedRequestData.new_block_ids only
                    # carries each step's appended blocks, so accumulate the
                    # full table here until the span completes.
                    self._store_progress[req_id] = (
                        digest,
                        span,
                        already,
                        [list(group) for group in group_blocks],
                    )
        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            if req_id not in self._store_progress:
                continue
            digest, span, done, blocks_by_group = self._store_progress[req_id]
            new_block_ids = cached.new_block_ids[index]
            appended = (
                [list(group) for group in self._normalize_group_blocks(new_block_ids)]
                if new_block_ids is not None
                else [[] for _ in blocks_by_group]
            )
            if len(appended) != len(blocks_by_group):
                raise RuntimeError(
                    "spark-context-cache: KV-cache group count changed while"
                    " accumulating a store"
                )
            if req_id in cached.resumed_req_ids:
                blocks_by_group = appended
            else:
                blocks_by_group = [
                    existing + added
                    for existing, added in zip(blocks_by_group, appended)
                ]
            blocks = blocks_by_group[0]
            done = cached.num_computed_tokens[index] + (
                scheduler_output.num_scheduled_tokens.get(req_id, 0)
            )
            if self._streaming_snapshots_enabled:
                if done >= span:
                    del self._store_progress[req_id]
                    if self._has_full_quorum(digest):
                        self.counters["store_skipped_quorum"] += 1
                        continue
                else:
                    self._store_progress[req_id] = (
                        digest,
                        span,
                        done,
                        blocks_by_group,
                    )
                self._append_streaming_snapshot_offer(
                    meta,
                    request_id=req_id,
                    digest=digest,
                    span=span,
                    promised_completed_tokens=done,
                    block_ids=blocks,
                )
            elif done >= span:
                del self._store_progress[req_id]
                if self._has_full_quorum(digest):
                    self.counters["store_skipped_quorum"] += 1
                    continue
                normalized = tuple(tuple(group) for group in blocks_by_group)
                meta.plans.append(
                    _ReqPlan(
                        req_id,
                        digest,
                        span,
                        normalized[0],
                        True,
                        block_ids_by_group=normalized,
                    )
                )
            else:
                self._store_progress[req_id] = (
                    digest,
                    span,
                    done,
                    blocks_by_group,
                )
        runtime = self._streaming_runtime
        if runtime is not None and self._role is KVConnectorRole.SCHEDULER:
            # request_finished() runs on the scheduler side. Give its adapter
            # the exact offers sent to workers so it delays block reuse only
            # for requests whose final gather may still be in flight.
            runtime.observe_metadata(meta)
        return meta

    # ------------------------------------------------------------------
    # worker side
    # ------------------------------------------------------------------

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        runtime = self._streaming_runtime
        if runtime is not None:
            # vLLM calls this before it may overwrite preempted blocks.
            # This must remain synchronous: the runtime drains armed CUDA
            # fences or raises without releasing blocks of unknown status.
            runtime.handle_preemptions(kv_connector_metadata)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._layer_tensors = dict(kv_caches)
        if self._storage_mode == "block_pages_v1":
            page_groups = []
            assigned: set[str] = set()
            groups = tuple(getattr(self._kv_cache_config, "kv_cache_groups", ()) or ())
            for group in groups:
                spec = getattr(group, "kv_cache_spec", None)
                layers = []
                for name in sorted(getattr(group, "layer_names", ()) or ()):
                    if name not in kv_caches:
                        raise RuntimeError(
                            f"spark-context-cache: group layer {name} was not"
                            " registered"
                        )
                    tensor = kv_caches[name]
                    page_shape = tuple(int(size) for size in tensor.shape[1:])
                    layers.append(
                        PageLayer(
                            name=name,
                            dtype=str(tensor.dtype),
                            page_shape=page_shape,
                            bytes_per_page=(
                                math.prod(page_shape) * tensor.element_size()
                            ),
                        )
                    )
                    assigned.add(name)
                page_groups.append(
                    PageGroup(
                        block_size=int(getattr(spec, "block_size", 0) or 0),
                        layers=tuple(layers),
                        reuse_window_tokens=self._group_topology[len(page_groups)][
                            "reuse_window_tokens"
                        ],
                        reuse_policy=self._group_topology[len(page_groups)][
                            "reuse_policy"
                        ],
                    )
                )
            unassigned = set(kv_caches) - assigned
            if unassigned:
                raise RuntimeError(
                    "spark-context-cache: registered layers lack a KV-cache"
                    " group: " + ", ".join(sorted(unassigned))
                )
            self._page_layout = PageLayout(tuple(page_groups))
        widths = {}
        for name, tensor in sorted(kv_caches.items()):
            if tensor.dim() < 3:
                raise RuntimeError(
                    f"spark-context-cache: layer {name} has unexpected"
                    f" shape {tuple(tensor.shape)}"
                )
            row = tensor[0, 0]
            widths[name] = int(row.numel() * row.element_size())
        logger.info(
            "spark-context-cache: layer inventory (%d layers): %s",
            len(widths),
            {name: widths[name] for name in sorted(widths)},
        )
        profile = self._profile
        draft_named = any(
            classify_layer(name, profile.classification_rules, profile.default_family)
            == "mtp_draft_kv"
            for name in widths
        )
        policy = self._identity_base["draft_kv_policy"]
        expects_named = profile.expects_draft_named_layers(policy)
        if draft_named != expects_named:
            raise RuntimeError(
                "spark-context-cache: draft-layer registration disagrees with"
                f" the draft policy: profile {profile.name} under"
                f" draft_kv_policy={policy} expects draft-classified layer"
                f" names: {expects_named}; observed: {draft_named}. Set"
                " spark_cache_draft_policy to match how this runtime"
                " registers its drafter layers"
            )
        self._plans = build_layer_plans(
            widths,
            allow_missing=profile.allow_missing(policy),
            required_families=profile.required_families,
            classification_rules=profile.classification_rules,
            default_family=profile.default_family,
        )
        self._record_kinds = tuple(sorted({plan.record_kind for plan in self._plans}))
        kinds: dict[str, int] = {}
        for plan in self._plans:
            kinds[plan.record_kind] = kinds.get(plan.record_kind, 0) + 1
        logger.info(
            "spark-context-cache: registered %d layers %s policy=%s",
            len(self._plans),
            kinds,
            self._identity_base.get("draft_kv_policy", "separate"),
        )
        if self._native_restore_enabled and self._role is KVConnectorRole.WORKER:
            self._configure_native_restore()
        if self._streaming_snapshots_enabled and self._role is KVConnectorRole.WORKER:
            runtime = self._streaming_runtime
            if runtime is None:
                raise RuntimeError(
                    "spark-context-cache: streaming worker runtime vanished"
                )
            # The adapter closes over this connector. Bind only after the
            # complete tensor inventory and canonical layer plans exist.
            runtime.bind_kv_caches()
        if self._capacity_policy.enabled and self._role is KVConnectorRole.WORKER:
            report = self._maintain_capacity(force=True)
            self._ensure_capacity_thread()
            if (
                (report is not None and report.skipped_busy)
                or not bool(self._capacity_status["capacity_satisfied"])
            ):
                self._capacity_wakeup.set()
        if self._restore_enabled:
            self.discover_manifests()

    def _configure_native_restore(self) -> None:
        """Attest and bind native placement only after final CUDA inventory."""

        if self._native_adapter is not None:
            raise RuntimeError(
                "spark-context-cache: native placement is already configured"
            )
        assert self._plans is not None
        if not self._plans:
            raise RuntimeError("spark-context-cache: native restore has no layer plans")
        first_tensor = self._layer_tensors[self._plans[0].name]
        device = first_tensor.device
        if device.type != "cuda":
            raise RuntimeError(
                "spark-context-cache: native restore requires CUDA cache tensors"
            )
        device_ordinal = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        capacities = {
            int(tensor.shape[0]) * int(tensor.shape[1])
            for tensor in self._layer_tensors.values()
        }
        if len(capacities) != 1:
            # Restore addresses every layer through one shared slot-index
            # space; heterogeneous pool geometries would scatter into wrong
            # slots on the larger pools.
            raise RuntimeError(
                "spark-context-cache: registered layers disagree on pool"
                f" geometry (slot capacities {sorted(capacities)})"
            )
        slot_capacity = capacities.pop()
        local_rows_per_chunk = self._chunk_tokens // self._dcp_degree
        payload_floor = local_rows_per_chunk * (
            4 + sum(plan.bytes_per_token for plan in self._plans)
        )
        if payload_floor >= self._native_arena_bytes:
            # Per-rank chunk bytes scale inversely with dcp_degree (DCP1
            # stores the full chunk on every rank); an arena that cannot
            # hold one encoded chunk would fail every restore plan later.
            raise RuntimeError(
                "spark-context-cache: native arena"
                f" ({self._native_arena_bytes} bytes) cannot hold one"
                f" encoded chunk (floor {payload_floor} bytes) at"
                f" dcp_degree={self._dcp_degree}"
            )
        max_chunks_per_slab = min(
            4096,
            max(1, self._native_arena_bytes // max(payload_floor, 1) + 1),
        )
        adapter = None
        try:
            components = _load_native_components()
            library = components.NativePlacementLibrary.load(
                self._native_library_path,
                expected_sha256=self._native_library_sha256,
            )
            adapter = components.NativePlacementAdapter.create(
                library,
                arena_mode=components.ArenaMode.MAPPED_HOST,
                arena_bytes=self._native_arena_bytes,
                max_destinations=len(self._plans),
                max_slots=slot_capacity,
                max_chunks_per_slab=max_chunks_per_slab,
                device_ordinal=device_ordinal,
            )
            adapter.configure(self._plans, self._layer_tensors)
            required = 0
            ordinal_names = {
                "target_ckv": "TARGET_CKV",
                "sparse_indexer": "SPARSE_INDEXER",
                "mtp_draft_kv": "MTP_DRAFT_KV",
                "boundary_hidden": "BOUNDARY_HIDDEN",
            }
            for kind in self._record_kinds:
                ordinal = getattr(
                    components.RecordKind, ordinal_names.get(kind, ""), None
                )
                if ordinal is None:
                    raise RuntimeError(
                        f"spark-context-cache: record kind {kind} has no"
                        " native placement ordinal"
                    )
                required |= 1 << int(ordinal)
            native_execute = components.execute_native_restore
            if not callable(native_execute):
                raise TypeError("native restore orchestrator is not callable")
        except Exception as error:
            if adapter is not None:
                with contextlib.suppress(Exception):
                    adapter.close()
            raise RuntimeError(
                f"spark-context-cache: native restore configuration"
                f" failed closed: {error}"
            ) from error
        self._native_adapter = adapter
        self._native_execute_restore = native_execute
        self._native_required_record_mask = required
        self.counters["native_configured"] = 1
        logger.info(
            "spark-context-cache: native restore configured library=%s"
            " sha256=%s arena_mib=%d destinations=%d slots=%d"
            " max_chunks_per_slab=%d",
            self._native_library_path,
            self._native_library_sha256,
            self._native_arena_bytes // (1024 * 1024),
            len(self._plans),
            slot_capacity,
            max_chunks_per_slab,
        )

    def _maintain_capacity(
        self,
        *,
        force: bool = False,
        wake_worker_on_unsatisfied: bool = True,
    ) -> MaintenanceReport | None:
        with self._capacity_lock:
            return self._maintain_capacity_locked(
                force=force,
                wake_worker_on_unsatisfied=wake_worker_on_unsatisfied,
            )

    def _maintain_capacity_locked(
        self,
        *,
        force: bool = False,
        wake_worker_on_unsatisfied: bool = False,
    ) -> MaintenanceReport | None:
        """Run one capacity pass while ``_capacity_lock`` is held."""

        policy = self._capacity_policy
        if not policy.enabled:
            return None
        if not force and (
            policy.max_bytes == 0
            or self._capacity_estimated_bytes <= policy.max_bytes
        ):
            return None
        try:
            report = self._store.maintain(policy)
        except Exception as error:  # noqa: BLE001 - maintenance is nonfatal
            self.counters["capacity_failed"] += 1
            self._capacity_status.update(
                bytes=self._capacity_estimated_bytes,
                bytes_exact=False,
                capacity_satisfied=False,
            )
            self._reconcile_held_capacity()
            if wake_worker_on_unsatisfied:
                self._capacity_wakeup.set()
            logger.warning("spark-context-cache: capacity maintenance failed: %s", error)
            return None
        if report.skipped_busy:
            self.counters["capacity_skipped_busy"] += 1
            self._capacity_status.update(
                bytes=self._capacity_estimated_bytes,
                bytes_exact=False,
                capacity_satisfied=False,
            )
            self._reconcile_held_capacity()
            if wake_worker_on_unsatisfied:
                self._capacity_wakeup.set()
            return report
        self._capacity_estimated_bytes = report.bytes_after
        self._capacity_status.update(
            bytes=report.bytes_after,
            bytes_exact=True,
            capacity_satisfied=report.capacity_satisfied,
        )
        self.counters["capacity_runs"] += 1
        self.counters["capacity_manifests_evicted"] += report.manifests_evicted
        self.counters["capacity_chunks_deleted"] += report.chunks_deleted
        self.counters["capacity_bytes_reclaimed"] += report.bytes_reclaimed
        if report.evicted_entries:
            rank = self._worker_rank()
            storage_key = self._identity(rank).storage_key
            evicted = {
                entry.context_digest
                for entry in report.evicted_entries
                if entry.storage_key == storage_key
            }
            with self._store_cv:
                self._held.difference_update(evicted)
        self._reconcile_held_capacity()
        if not report.capacity_satisfied and wake_worker_on_unsatisfied:
            self._capacity_wakeup.set()
        if force or report.bytes_reclaimed:
            logger.info(
                "spark-context-cache: capacity bytes=%d max=%d reclaimed=%d"
                " manifests=%d chunks=%d orphans=%d satisfied=%s",
                report.bytes_after,
                policy.max_bytes,
                report.bytes_reclaimed,
                report.manifests_evicted,
                report.chunks_deleted,
                report.orphan_chunks_deleted,
                report.capacity_satisfied,
            )
        return report

    def _note_capacity_commit(self, encoded_bytes: int) -> None:
        with self._capacity_lock:
            self._note_capacity_commit_locked(encoded_bytes)

    def _note_capacity_commit_locked(self, encoded_bytes: int) -> None:
        if type(encoded_bytes) is not int or encoded_bytes < 0:
            raise ValueError("capacity commit bytes must be a non-negative integer")
        self._capacity_estimated_bytes += encoded_bytes
        max_bytes = self._capacity_policy.max_bytes
        previously_satisfied = bool(
            self._capacity_status["capacity_satisfied"]
        )
        self._capacity_status.update(
            bytes=self._capacity_estimated_bytes,
            bytes_exact=False,
            capacity_satisfied=(
                previously_satisfied
                and (
                    max_bytes == 0
                    or self._capacity_estimated_bytes <= max_bytes
                )
            ),
        )

    def _streaming_commit_allocated_bytes(
        self,
        context_digest: str,
        receipt: Any,
    ) -> int:
        """Validate the stable commit-receipt fields without class identity.

        Staged deployments load the storage engine through a compatibility
        shim, while the publisher imports its package path directly. The two
        receipts are structurally identical but may not share Python class
        identity, so this boundary deliberately validates the wire fields.
        """

        if SHA256_RE.fullmatch(context_digest) is None:
            raise RuntimeError("streaming commit context digest is invalid")
        manifest_digest = getattr(receipt, "manifest_digest", None)
        committed_tokens = getattr(receipt, "committed_tokens", None)
        encoded_bytes = getattr(receipt, "encoded_bytes", None)
        allocated_bytes = getattr(
            receipt,
            "allocated_bytes_upper_bound",
            None,
        )
        if (
            not isinstance(manifest_digest, str)
            or SHA256_RE.fullmatch(manifest_digest) is None
            or type(committed_tokens) is not int
            or committed_tokens <= 0
            or committed_tokens % self._chunk_tokens != 0
            or committed_tokens > self._max_span
            or type(encoded_bytes) is not int
            or encoded_bytes <= 0
            or type(allocated_bytes) is not int
            or allocated_bytes < encoded_bytes
        ):
            raise RuntimeError("streaming commit receipt is invalid")
        return allocated_bytes

    def _reject_streaming_capacity_receipts(self, count: int) -> None:
        """Drop invalid handoffs and request an exact background scan."""

        if count <= 0:
            return
        with self._capacity_handoff_cv:
            self.counters["streaming_capacity_invalid_receipts"] += count
            self._capacity_status.update(
                bytes_exact=False,
                capacity_satisfied=False,
            )
        logger.warning(
            "spark-context-cache: dropped %d invalid streaming commit"
            " receipt(s); scheduling exact capacity maintenance",
            count,
        )
        self._capacity_wakeup.set()

    def _handoff_streaming_commits(
        self,
        committed: Mapping[str, Any],
    ) -> set[str]:
        """Accept durable receipts without running maintenance on callbacks.

        Unbounded deployments advertise each durable receipt immediately.
        A bounded or TTL-managed deployment queues the receipt for the
        capacity worker; it returns no advertised digest until that worker has
        serialized accounting, maintenance, survivor verification, and the
        ``_held`` update.
        """

        if not self._capacity_policy.enabled:
            # Capacity accounting is disabled, so receipt byte fields have no
            # bearing on the established post-manifest advertisement path.
            advertised = {str(raw_digest) for raw_digest in committed}
            with self._store_cv:
                self._held.update(advertised)
                self.counters["streaming_store_committed"] += len(advertised)
            return advertised

        staged: list[tuple[str, Any, int]] = []
        invalid_receipts = 0
        for raw_digest, receipt in committed.items():
            try:
                digest = str(raw_digest)
                allocated_bytes = self._streaming_commit_allocated_bytes(
                    digest,
                    receipt,
                )
            except Exception:  # noqa: BLE001 - cache metadata fails closed
                invalid_receipts += 1
                continue
            staged.append((digest, receipt, allocated_bytes))
        self._reject_streaming_capacity_receipts(invalid_receipts)
        if not staged:
            return set()

        with self._store_cv:
            already_advertised = {
                digest
                for digest, _receipt, _bytes in staged
                if digest in self._held
            }
        queued = 0
        with self._capacity_handoff_cv:
            if self._capacity_stop.is_set():
                self.counters["streaming_capacity_shutdown_dropped"] += len(
                    {digest for digest, _receipt, _bytes in staged}
                    - already_advertised
                )
                return already_advertised
            for digest, receipt, _allocated_bytes in staged:
                if (
                    digest in already_advertised
                    or digest in self._streaming_capacity_pending
                ):
                    continue
                self._streaming_capacity_pending.add(digest)
                self._capacity_commit_queue.put((digest, receipt))
                queued += 1
        if queued:
            self.counters["streaming_capacity_queued"] += queued
            self._capacity_wakeup.set()
        return already_advertised

    def wait_for_pending_capacity_commits(
        self,
        timeout: float | None = None,
    ) -> bool:
        """Wait for tests/offline callers; serving callbacks never call this."""

        with self._capacity_handoff_cv:
            return self._capacity_handoff_cv.wait_for(
                lambda: not self._streaming_capacity_pending,
                timeout,
            )

    def _finalize_streaming_capacity_commits(
        self,
        pending: Mapping[str, _PendingStreamingCommit],
    ) -> bool:
        """Resolve one serialized batch, or retain all entries for retry."""

        with self._capacity_lock:
            for digest, entry in pending.items():
                if entry.accounted:
                    continue
                allocated_bytes = self._streaming_commit_allocated_bytes(
                    digest,
                    entry.receipt,
                )
                self._note_capacity_commit_locked(allocated_bytes)
                entry.accounted = True

            policy = self._capacity_policy
            maintenance_required = (
                not bool(self._capacity_status["capacity_satisfied"])
                or (
                    policy.max_bytes > 0
                    and self._capacity_estimated_bytes > policy.max_bytes
                )
            )
            if maintenance_required:
                report = self._maintain_capacity_locked(force=True)
                if (
                    report is None
                    or report.skipped_busy
                    or not report.capacity_satisfied
                ):
                    return False

            identity = self._identity(self._worker_rank())
            try:
                surviving = {
                    digest
                    for digest in pending
                    if self._store.lookup(
                        identity,
                        digest,
                        verify_chunks=False,
                        verify_chunk_metadata=True,
                    ).is_hit
                }
            except Exception as error:  # noqa: BLE001 - fail closed/retry
                self.counters["capacity_failed"] += 1
                self._capacity_status.update(
                    bytes=self._capacity_estimated_bytes,
                    bytes_exact=False,
                    capacity_satisfied=False,
                )
                logger.warning(
                    "spark-context-cache: capacity survivor check failed: %s",
                    error,
                )
                return False

            evicted = set(pending) - surviving
            # This is the only point at which a bounded streamed commit may
            # become visible to worker stats. The capacity lock stays held
            # across the survivor check and this short in-memory update, so a
            # competing maintenance pass cannot evict then resurrect it.
            with self._store_cv:
                if self._capacity_stop.is_set():
                    return False
                self._held.update(surviving)
                self._held.difference_update(evicted)
                self.counters["streaming_store_committed"] += len(surviving)
                self.counters["streaming_store_evicted"] += len(evicted)
            return True

    def _reconcile_held_capacity(self) -> None:
        if not self._held:
            return
        rank = self._worker_rank()
        identity = self._identity(rank)
        with self._store_cv:
            held = set(self._held)
        surviving = {
            digest
            for digest in held
            if self._store.lookup(
                identity,
                digest,
                verify_chunks=False,
                verify_chunk_metadata=True,
            ).is_hit
        }
        with self._store_cv:
            self._held.intersection_update(surviving)

    def _ensure_capacity_thread(self) -> None:
        if self._capacity_thread is not None:
            return
        thread = threading.Thread(
            target=self._capacity_worker_main,
            name="spark-cache-capacity",
            daemon=True,
        )
        self._capacity_thread = thread
        try:
            thread.start()
        except BaseException:
            self._capacity_thread = None
            raise

    def _capacity_worker_main(self) -> None:
        ttl_seconds = self._capacity_policy.ttl_seconds
        interval = (
            min(60, max(1, ttl_seconds // 2))
            if ttl_seconds
            else 300
        )
        pending: dict[str, _PendingStreamingCommit] = {}
        retry_unsatisfied = False
        while not self._capacity_stop.is_set():
            timeout = (
                _CAPACITY_RETRY_SECONDS
                if pending or retry_unsatisfied
                else interval
            )
            self._capacity_wakeup.wait(timeout=timeout)
            self._capacity_wakeup.clear()
            if self._capacity_stop.is_set():
                return
            while True:
                try:
                    digest, receipt = self._capacity_commit_queue.get_nowait()
                except queue.Empty:
                    break
                pending.setdefault(
                    digest,
                    _PendingStreamingCommit(receipt=receipt),
                )

            if pending:
                if self._finalize_streaming_capacity_commits(pending):
                    resolved = set(pending)
                    pending.clear()
                    with self._capacity_handoff_cv:
                        self._streaming_capacity_pending.difference_update(
                            resolved
                        )
                        self._capacity_handoff_cv.notify_all()
                    retry_unsatisfied = False
                else:
                    self.counters["streaming_capacity_retries"] += 1
                    retry_unsatisfied = True
                continue

            report = self._maintain_capacity(
                force=True,
                wake_worker_on_unsatisfied=False,
            )
            retry_unsatisfied = bool(
                report is None
                or not report.capacity_satisfied
                or not bool(self._capacity_status["capacity_satisfied"])
            )
            if retry_unsatisfied:
                self.counters["capacity_retries"] += 1

    def discover_manifests(self) -> dict[str, int]:
        """Offer structurally valid manifests without reading chunk payloads.

        Referenced chunks must exist at their declared sizes; that metadata
        check does not fault payload pages into memory. Restore remains the
        byte/hash integrity boundary. Same-size corruption discovered there
        revokes the offer and degrades the request to clean recompute.
        """

        checked = rejected = 0
        discovered: set[str] = set()
        # Serialize discovery's snapshot replacement with async store
        # admission/completion. A commit may publish its manifest while this
        # scan is in progress; its completion then adds the durable digest
        # after discovery releases this lock instead of having that offer
        # erased by the replacement below.
        with self._store_cv:
            try:
                rank = self._worker_rank()
                identity = self._identity(rank)
                root = Path(self._root) / "manifests" / identity.storage_key
                if root.is_dir():
                    for manifest_path in root.glob("*.json"):
                        digest = manifest_path.stem
                        checked += 1
                        lookup = self._store.lookup(
                            identity,
                            digest,
                            verify_chunks=False,
                            verify_chunk_metadata=True,
                        )
                        if lookup.is_hit:
                            discovered.add(digest)
                        else:
                            rejected += 1
                            # A rejected manifest is not an authority for
                            # deleting content-addressed chunks. Its
                            # descriptors may be corrupt, malicious, or name
                            # chunks shared by a healthy manifest.
                            removed = self._store.invalidate(
                                identity,
                                digest,
                                verify_chunk_payloads=False,
                            )
                            if not removed:
                                try:
                                    manifest_path.unlink()
                                except OSError:
                                    pass
            except (OSError, ValueError, RuntimeError) as error:
                logger.warning(
                    "spark-context-cache: manifest discovery aborted: %s",
                    error,
                )
            self._held = discovered
        if checked:
            logger.info(
                "spark-context-cache: manifest discovery checked=%d"
                " offered=%d rejected=%d",
                checked,
                len(discovered),
                rejected,
            )
        self.counters["discovery_checked"] = (
            self.counters.get("discovery_checked", 0) + checked
        )
        self.counters["discovery_rejected"] = (
            self.counters.get("discovery_rejected", 0) + rejected
        )
        return {
            "checked": checked,
            "offered": len(discovered),
            "rejected": rejected,
        }

    def sweep_integrity(self) -> dict[str, int]:
        """Verify every entry this rank owns and invalidate damaged ones.

        This explicit diagnostic is not part of startup or request completion.
        A manifest enters the offered set directly after ManifestStore's
        durable commit. Startup performs manifest-only discovery, and every
        load independently re-verifies bytes. Returns
        {"checked": n, "invalidated": m}.
        """
        checked = invalidated = 0
        verified: set[str] = set()
        with self._load_lock:
            held_at_start = set(self._held)
        try:
            rank = self._worker_rank()
            identity = self._identity(rank)
            root = Path(self._root) / "manifests" / identity.storage_key
            if not root.is_dir():
                with self._load_lock:
                    self._held.difference_update(held_at_start)
                return {"checked": 0, "invalidated": 0}
            for manifest_path in sorted(root.glob("*.json")):
                digest = manifest_path.stem
                checked += 1
                # restore() is the single parallel read/hash/decode pass.
                lookup = self._store.lookup(identity, digest, verify_chunks=False)
                if lookup.is_hit and self._store.restore(lookup) is not None:
                    verified.add(digest)
                    continue
                with self._load_lock:
                    self._held.discard(digest)
                if self._store.invalidate(identity, digest):
                    invalidated += 1
                    logger.warning(
                        "spark-context-cache: sweep invalidated %s (%s)",
                        digest[:12],
                        lookup.reason,
                    )
            with self._load_lock:
                # Remove only offers that existed when the sweep began and
                # failed to verify. Preserve durable entries published while
                # the diagnostic was reading older manifests.
                self._held.difference_update(held_at_start - verified)
                self._held.update(verified)
        except (OSError, ValueError, RuntimeError) as error:
            logger.warning("spark-context-cache: sweep aborted: %s", error)
        if checked:
            logger.info(
                "spark-context-cache: sweep checked=%d invalidated=%d",
                checked,
                invalidated,
            )
        self.counters["sweep_checked"] = self.counters.get("sweep_checked", 0) + checked
        self.counters["sweep_invalidated"] = (
            self.counters.get("sweep_invalidated", 0) + invalidated
        )
        return {"checked": checked, "invalidated": invalidated}

    def _physical_rank(self) -> int:
        """Return this worker's physical tensor-parallel rank.

        This is unique across all TP ranks (0..tp_degree-1), unlike
        _worker_rank() which returns the DCP-local rank (0..dcp_degree-1).
        Under TP4/DCP2, TP0 and TP2 both have DCP-local rank 0 but
        physical ranks 0 and 2 respectively.

        Used for quorum admission/withdrawal and persistent namespace
        identity (tp_shard_rank).  Token-position ownership and DCP
        slicing still use _worker_rank() (DCP-local rank).
        """
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())

    def _worker_rank(self) -> int:
        from vllm.distributed import get_dcp_group

        if self._dcp_degree <= 1:
            return 0
        return get_dcp_group().rank_in_group

    def _rows_view(self, tensor: torch.Tensor) -> torch.Tensor:
        # view, not reshape: restore scatters through this result, so it must
        # alias the registered KV storage. reshape silently returns a copy for
        # non-viewable layouts, and the scatter would write into a discarded
        # temporary while checksums and completion reporting still succeed.
        # view raises RuntimeError instead, which the load path converts into
        # a clean miss and recompute.
        num_blocks = tensor.shape[0]
        page = tensor.shape[1]
        return tensor.view(num_blocks * page, -1)

    def _ensure_load_threads(self) -> None:
        with self._load_lock:
            while len(self._load_threads) < self._load_thread_limit:
                thread = threading.Thread(
                    target=self._load_worker_main,
                    name=f"spark-cache-load-{len(self._load_threads)}",
                    daemon=True,
                )
                thread.start()
                self._load_threads.append(thread)

    def _load_worker_main(self) -> None:
        while True:
            plan = self._load_queue.get()
            if plan is None:
                return
            started = time.perf_counter()
            try:
                verified = self._load_one(plan)
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "spark-context-cache: load crashed fail-closed: %s", error
                )
                verified = False
            with self._load_cv:
                if verified:
                    self.counters["load_verified"] += 1
                else:
                    # Publish invalid blocks before the finished id while
                    # holding the same lock. vLLM collects finished ids then
                    # load errors in one connector-output pass.
                    self._load_errors.update(
                        block for group in plan.group_block_ids for block in group
                    )
                    self.counters["load_failed"] += 1
                self._finished_load_reqs.add(plan.request_id)
                self._inflight_load_reqs.discard(plan.request_id)
                self._load_cv.notify_all()
            if verified:
                with contextlib.suppress(Exception):
                    ttl_seconds = self._capacity_policy.ttl_seconds
                    self._store.touch(
                        self._identity(self._worker_rank()),
                        plan.digest,
                        minimum_interval_seconds=(
                            min(60, ttl_seconds // 2) if ttl_seconds else 60
                        ),
                    )
                logger.info(
                    "spark-context-cache: restored %d tokens async in %.1f ms",
                    plan.span_tokens,
                    1e3 * (time.perf_counter() - started),
                )

    def _load_write_context(self) -> Any:
        assert self._plans is not None
        device = self._layer_tensors[self._plans[0].name].device
        if device.type != "cuda":
            return contextlib.nullcontext(None)
        with self._load_lock:
            if self._load_stream is None:
                self._load_stream = torch.cuda.Stream(device=device)
        return torch.cuda.stream(self._load_stream)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SparkCacheConnectorMetadata):
            return
        load_plans = [plan for plan in metadata.plans if not plan.is_store]
        if not load_plans:
            return
        self._ensure_load_threads()
        with self._load_lock:
            self._inflight_load_reqs.update(plan.request_id for plan in load_plans)
        for plan in load_plans:
            self._load_queue.put(plan)

    def _load_one(self, plan: _ReqPlan) -> bool:
        try:
            rank = self._worker_rank()
            identity = self._identity(rank)
            if self._store.expired(
                identity,
                plan.digest,
                self._capacity_policy.ttl_seconds,
            ):
                with self._load_lock:
                    self._held.discard(plan.digest)
                self._capacity_wakeup.set()
                logger.info(
                    "spark-context-cache: worker rank %d TTL miss for %s",
                    rank,
                    plan.digest[:12],
                )
                return False
            # Restore is the integrity boundary and re-hashes every chunk in
            # parallel. A full lookup verification here would read/hash/decode
            # the complete context twice before installation.
            lookup = self._store.lookup(identity, plan.digest, verify_chunks=False)
            if not lookup.is_hit:
                logger.warning(
                    "spark-context-cache: worker rank %d miss (%s) for %s",
                    rank,
                    lookup.reason,
                    plan.digest[:12],
                )
                if lookup.reason == "corrupt":
                    self._invalidate_after_failure(plan.digest)
                return False
            if self._storage_mode == "block_pages_v1":
                return self._load_hybrid_pages(lookup, plan)
            if self._native_restore_enabled:
                if (
                    self._native_adapter is None
                    or not callable(self._native_execute_restore)
                    or self._native_required_record_mask == 0
                ):
                    raise RuntimeError(
                        "native restore selected without a configured adapter"
                    )
                positions = owned_positions(plan.span_tokens, self._dcp_degree, rank)
                slots = local_slots_for_positions(
                    positions,
                    plan.block_ids,
                    self._block_size,
                    self._dcp_degree,
                )
                try:
                    result = self._native_execute_restore(
                        adapter=self._native_adapter,
                        request_id=plan.request_id,
                        lookup=lookup,
                        cache_root=self._root,
                        slots=slots,
                        expected_span_tokens=plan.span_tokens,
                        dcp_degree=self._dcp_degree,
                        dcp_rank=rank,
                        arena_bytes=self._native_arena_bytes,
                        required_data_record_mask=(self._native_required_record_mask),
                        io_workers=self._native_io_workers,
                    )
                except Exception as error:  # noqa: BLE001
                    # The native transaction may already have written some
                    # private restore blocks. Never enter Python assembly or
                    # retry those blocks: retire the entry and publish all of
                    # this request's blocks as invalid for clean recompute.
                    logger.warning(
                        "spark-context-cache: native load failed closed: %s",
                        error,
                    )
                    self._invalidate_after_failure(plan.digest)
                    return False
                self.counters["native_load_verified"] = (
                    self.counters.get("native_load_verified", 0) + 1
                )
                self.counters["native_chunks_verified"] = self.counters.get(
                    "native_chunks_verified", 0
                ) + int(result.verified_chunks)
                logger.info(
                    "spark-context-cache: native restored %d chunks"
                    " (%d encoded bytes, %d slabs) read_hash=%.1f ms"
                    " parse_submit=%.1f ms finish=%.1f ms",
                    result.verified_chunks,
                    result.verified_encoded_bytes,
                    result.slabs,
                    result.read_and_hash_ms,
                    result.parse_and_submit_ms,
                    result.finish_ms,
                )
                return True
            chunks = self._store.restore(lookup)
            if chunks is None or len(chunks) != chunk_count(
                plan.span_tokens, self._chunk_tokens
            ):
                self._invalidate_after_failure(plan.digest)
                return False
            positions = owned_positions(plan.span_tokens, self._dcp_degree, rank)
            slots = local_slots_for_positions(
                positions, plan.block_ids, self._block_size, self._dcp_degree
            )
            per_chunk = self._chunk_tokens // self._dcp_degree
            assert self._plans is not None
            layer_rows: dict[str, list[bytes]] = {p.name: [] for p in self._plans}
            for index, chunk in enumerate(chunks):
                expected = positions[index * per_chunk : (index + 1) * per_chunk]
                stored = unpack_positions(chunk.records[StateRecord.LOGICAL_POSITIONS])
                if stored != tuple(expected):
                    raise CodecError("stored positions disagree with shard")
                for kind in self._record_kinds:
                    split = unpack_record(
                        self._plans,
                        kind,
                        chunk.records[StateRecord(kind)],
                        per_chunk,
                    )
                    for name, payload in split.items():
                        layer_rows[name].append(payload)
            slot_tensor = torch.tensor(slots, dtype=torch.long)
            with self._load_write_context():
                for plan_entry in self._plans:
                    tensor = self._layer_tensors[plan_entry.name]
                    rows = self._rows_view(tensor)
                    payload = b"".join(layer_rows[plan_entry.name])
                    flat = torch.frombuffer(
                        bytearray(payload), dtype=torch.uint8
                    ).reshape(len(slots), plan_entry.bytes_per_token)
                    staged = flat.to(rows.device)
                    rows_u8 = rows.view(torch.uint8)
                    rows_u8[slot_tensor.to(rows.device)] = staged
            if self._load_stream is not None:
                self._load_stream.synchronize()
            return True
        except (CodecError, KeyError, RuntimeError, ValueError) as error:
            logger.warning("spark-context-cache: load failed fail-closed: %s", error)
            self._invalidate_after_failure(plan.digest)
            return False

    def _load_hybrid_pages(self, lookup: Any, plan: _ReqPlan) -> bool:
        layout = self._page_layout
        if layout is None:
            raise RuntimeError("block-page layout was not registered")
        groups = self._select_group_blocks_for_span(
            plan.group_block_ids, plan.span_tokens
        )
        if len(groups) != len(layout.groups):
            raise HybridCodecError("request block tables disagree with page groups")
        chunks = self._store.restore(lookup)
        expected_chunks = chunk_count(plan.span_tokens, self._chunk_tokens)
        if chunks is None or len(chunks) != expected_chunks:
            raise HybridCodecError("hybrid snapshot chunk count differs")
        encoded_parts = []
        for index, chunk in enumerate(chunks):
            start = index * self._chunk_tokens
            end = (index + 1) * self._chunk_tokens
            stored = unpack_positions(chunk.records[StateRecord.LOGICAL_POSITIONS])
            if stored != tuple(range(start, end)):
                raise HybridCodecError("hybrid snapshot positions differ")
            encoded_parts.append(chunk.records[StateRecord.TARGET_CKV])
        payloads = decode_page_snapshot(
            layout,
            b"".join(encoded_parts),
            tuple(len(group) for group in groups),
        )
        with self._load_write_context():
            for page_group, block_ids in zip(layout.groups, groups):
                for layer in page_group.layers:
                    tensor = self._layer_tensors[layer.name]
                    source = torch.frombuffer(
                        bytearray(payloads[layer.name]), dtype=tensor.dtype
                    ).reshape((len(block_ids), *layer.page_shape))
                    index_tensor = torch.tensor(
                        block_ids, dtype=torch.long, device=tensor.device
                    )
                    tensor[index_tensor] = source.to(tensor.device)
        if self._load_stream is not None:
            self._load_stream.synchronize()
        return True

    def _invalidate_after_failure(self, digest: str) -> None:
        """Withdraw a failed entry without re-reading payloads inline.

        The current request already fell back to recompute. A later publisher
        can atomically repair a corrupt content-addressed chunk while
        republishing the manifest; full payload inspection remains the
        explicit integrity sweep's responsibility.
        """
        with self._load_lock:
            self._held.discard(digest)
        removed = self._store.invalidate(
            self._identity(self._worker_rank()),
            digest,
            verify_chunk_payloads=False,
        )
        if removed:
            with self._load_lock:
                self.counters["invalidated"] = self.counters.get("invalidated", 0) + 1
            logger.warning(
                "spark-context-cache: invalidated damaged entry %s",
                digest[:12],
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        # A finished request id never recurs, so its scheduler-side tracking
        # state is dropped here. This is what keeps _need_load, _admitted,
        # and _store_progress bounded without evicting live entries.
        request_id = request.request_id
        self._need_load.pop(request_id, None)
        self._pending_async_loads.pop(request_id, None)
        self._admitted.pop(request_id, None)
        self._store_progress.pop(request_id, None)
        runtime = self._streaming_runtime
        if runtime is None:
            return False, None
        delay_free = bool(runtime.request_finished(request_id, tuple(block_ids)))
        return delay_free, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Release scheduler bookkeeping for a hybrid-memory-allocator request.

        End-of-prefill stores do not retain request blocks after the CPU
        snapshot, so the multi-group path has no delayed-free work. Streaming
        snapshots require group-qualified leases and are refused until that
        contract is implemented.
        """
        if self._streaming_runtime is not None:
            raise RuntimeError(
                "spark-context-cache: streaming snapshots do not support"
                " multiple KV-cache groups"
            )
        return self.request_finished(request, [])

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] = set()
        runtime = self._streaming_runtime
        if runtime is not None:
            finished_sending.update(runtime.take_finished(finished_req_ids))
        with self._load_cv:
            finished_recving = set(self._finished_load_reqs)
            self._finished_load_reqs.clear()
        return finished_sending or None, finished_recving or None

    def wait_for_pending_loads(self, timeout: float | None = None) -> bool:
        with self._load_cv:
            return self._load_cv.wait_for(lambda: not self._inflight_load_reqs, timeout)

    def shutdown(self):
        runtime = self._streaming_runtime
        if runtime is not None:
            # Drain/cancel snapshot leases before any native cache or staging
            # resource can be destroyed. A failure remains fatal and prevents
            # unsafe teardown.
            runtime.shutdown()
            self._streaming_runtime = None
        capacity_thread = self._capacity_thread
        if capacity_thread is not None:
            # Give already handed-off durable commits one final background
            # capacity pass. A persistent busy/failure remains fail closed:
            # shutdown drops only the in-memory handoff, never advertises it.
            self._capacity_wakeup.set()
            self.wait_for_pending_capacity_commits(timeout=5.0)
        # Serialize the terminal stop edge with the final survivor check and
        # _held publication. Once this section completes, a capacity worker
        # can only leave an unresolved handoff unadvertised.
        with self._store_cv:
            self._capacity_stop.set()
        self._capacity_wakeup.set()
        if capacity_thread is not None:
            capacity_thread.join(timeout=5.0)
            if not capacity_thread.is_alive():
                self._capacity_thread = None
        with self._capacity_handoff_cv:
            dropped = len(self._streaming_capacity_pending)
            self._streaming_capacity_pending.clear()
            self._capacity_handoff_cv.notify_all()
        if dropped:
            self.counters["streaming_capacity_shutdown_dropped"] += dropped
        with self._store_cv:
            self._store_accepting = False
        store_idle = self.wait_for_pending_stores(timeout=5.0)
        store_thread = self._store_thread
        if store_thread is not None:
            self._store_queue.put(None)
            store_thread.join(timeout=5.0 if store_idle else 0.0)
            if store_thread.is_alive():
                self.counters["store_shutdown_thread_live"] = (
                    self.counters.get("store_shutdown_thread_live", 0) + 1
                )
                logger.warning(
                    "spark-context-cache: shutdown left saver alive;"
                    " process teardown will reclaim its CPU snapshot"
                )
            else:
                self._store_thread = None
        self.wait_for_pending_loads(timeout=5.0)
        for _ in self._load_threads:
            self._load_queue.put(None)
        deadline = time.monotonic() + 5.0
        for thread in self._load_threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        live_threads = sum(thread.is_alive() for thread in self._load_threads)
        if self._native_adapter is not None and live_threads:
            # A native loader can still own the handle or mapped arenas.
            # Keep the adapter strongly reachable and let process teardown
            # reclaim it; destroying it here would be a use-after-close race.
            self.counters["native_shutdown_handle_leaked"] = (
                self.counters.get("native_shutdown_handle_leaked", 0) + 1
            )
            logger.warning(
                "spark-context-cache: shutdown left %d loader(s) alive;"
                " retaining native placement handle until process exit",
                live_threads,
            )
            return None
        if self._native_adapter is not None:
            self._native_adapter.close()
            self._native_adapter = None
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        if self._streaming_snapshots_enabled:
            # Scheduler offers are promises for the step vLLM has just
            # forwarded.  Convert them to completed watermarks only here;
            # doing it in build_connector_meta would let a writer read KV
            # rows before the producer stream has populated them.
            if self._role is KVConnectorRole.WORKER:
                runtime = self._streaming_runtime
                if runtime is None:
                    raise RuntimeError(
                        "spark-context-cache: streaming worker runtime vanished"
                    )
                if isinstance(metadata, SparkCacheConnectorMetadata):
                    offers = metadata.streaming_snapshot_offers
                    if offers:
                        producer_stream = int(torch.cuda.current_stream().cuda_stream)
                    for offer in offers:
                        runtime.offer_completed(
                            offer,
                            producer_stream=producer_stream,
                        )
                # Poll even for an empty metadata object so background
                # completion, backpressure, and failure state advance on
                # every post-forward callback.
                runtime.poll()
            # Never invoke the synchronous CPU snapshot/ManifestStore path under
            # this flag: an incomplete injected runtime must fail closed, not
            # silently publish a synchronous end-of-prefill entry.
            return
        if not isinstance(metadata, SparkCacheConnectorMetadata):
            return
        for plan in metadata.plans:
            if not plan.is_store:
                continue
            # Admission is atomic and precedes the multi-GB CPU snapshot.
            # At most one snapshot may be queued or committing per rank; a
            # busy store is a cache miss opportunity, never a serving stall.
            with self._store_cv:
                if plan.digest in self._held:
                    self.counters["store_skipped_present"] += 1
                    logger.info(
                        "spark-context-cache: store skipped; entry already"
                        " present digest=%s",
                        plan.digest[:12],
                    )
                    continue
                if not self._store_accepting or self._store_inflight:
                    self.counters["store_skipped_busy"] += 1
                    logger.warning(
                        "spark-context-cache: store skipped while saver busy digest=%s",
                        plan.digest[:12],
                    )
                    continue
                self._store_inflight = 1
            snapshot_started = time.perf_counter()
            try:
                snapshot = self._snapshot_store(plan)
            except Exception as error:  # noqa: BLE001 - never kill serving
                self._finish_store(
                    plan.digest,
                    committed=False,
                    error=error,
                )
                continue
            try:
                self._ensure_store_thread()
                self._store_queue.put(snapshot)
                logger.info(
                    "spark-context-cache: rank %d snapshotted %d tokens"
                    " digest=%s in %.1f ms; queued background commit",
                    snapshot.rank,
                    plan.span_tokens,
                    plan.digest[:12],
                    1e3 * (time.perf_counter() - snapshot_started),
                )
            except Exception as error:  # noqa: BLE001 - enqueue must fail closed
                self._finish_store(
                    plan.digest,
                    committed=False,
                    error=error,
                )

    def _snapshot_store(self, plan: _ReqPlan) -> _StoreSnapshot | _HybridStoreSnapshot:
        """Synchronously detach source KV blocks into owned CPU bytes."""

        if self._storage_mode == "block_pages_v1":
            return self._snapshot_hybrid_store(plan)

        rank = self._worker_rank()
        identity = self._identity(rank)
        positions = owned_positions(plan.span_tokens, self._dcp_degree, rank)
        slots = local_slots_for_positions(
            positions, plan.block_ids, self._block_size, self._dcp_degree
        )
        assert self._plans is not None
        slot_tensor = torch.tensor(slots, dtype=torch.long)
        layer_bytes: dict[str, bytes] = {}
        for plan_entry in self._plans:
            tensor = self._layer_tensors[plan_entry.name]
            rows = self._rows_view(tensor).view(torch.uint8)
            gathered = rows[slot_tensor.to(rows.device)].cpu().contiguous()
            layer_bytes[plan_entry.name] = gathered.numpy().tobytes()
        return _StoreSnapshot(
            plan=plan,
            rank=rank,
            identity=identity,
            positions=positions,
            layer_bytes=layer_bytes,
            layer_plans=tuple(self._plans),
            record_kinds=self._record_kinds,
        )

    def _snapshot_hybrid_store(self, plan: _ReqPlan) -> _HybridStoreSnapshot:
        layout = self._page_layout
        if layout is None:
            raise RuntimeError("block-page layout was not registered")
        groups = self._select_group_blocks_for_span(
            plan.group_block_ids, plan.span_tokens
        )
        if len(groups) != len(layout.groups):
            raise HybridCodecError("request block tables disagree with page groups")
        layer_payloads: dict[str, bytes] = {}
        for page_group, block_ids in zip(layout.groups, groups):
            index = torch.tensor(block_ids, dtype=torch.long)
            for layer in page_group.layers:
                tensor = self._layer_tensors[layer.name]
                gathered = tensor[index.to(tensor.device)].cpu().contiguous()
                layer_payloads[layer.name] = (
                    gathered.view(torch.uint8).numpy().tobytes()
                )
        encoded = encode_page_snapshot(
            layout,
            tuple(len(group) for group in groups),
            layer_payloads,
        )
        rank = self._worker_rank()
        return _HybridStoreSnapshot(
            plan=plan,
            rank=rank,
            identity=self._identity(rank),
            positions=tuple(range(plan.span_tokens)),
            encoded_pages=encoded,
        )

    def _ensure_store_thread(self) -> None:
        with self._store_cv:
            if self._store_thread is not None:
                if not self._store_thread.is_alive():
                    raise RuntimeError("background saver exited unexpectedly")
                return
            if not self._store_accepting:
                raise RuntimeError("background saver is shutting down")
            thread = threading.Thread(
                target=self._store_worker_main,
                name="spark-cache-store",
                daemon=True,
            )
            thread.start()
            self._store_thread = thread

    def _store_worker_main(self) -> None:
        while True:
            snapshot = self._store_queue.get()
            if snapshot is None:
                return
            commit_started = time.perf_counter()
            try:
                chunks: Sequence[ContextChunk]
                if isinstance(snapshot, _HybridStoreSnapshot):
                    chunks = _HybridSnapshotChunks(snapshot, self._chunk_tokens)
                else:
                    chunks = _SnapshotChunks(
                        snapshot, self._dcp_degree, self._chunk_tokens
                    )
                receipt = self._store.commit(
                    identity=snapshot.identity,
                    context_digest=snapshot.plan.digest,
                    chunks=chunks,
                    span_tokens=snapshot.plan.span_tokens,
                )
                with self._capacity_lock:
                    self._note_capacity_commit_locked(
                        receipt.allocated_bytes_upper_bound
                    )
                    self._maintain_capacity_locked(
                        wake_worker_on_unsatisfied=True
                    )
                    evicted = not self._store.lookup(
                        snapshot.identity,
                        snapshot.plan.digest,
                        verify_chunks=False,
                        verify_chunk_metadata=True,
                    ).is_hit
                    logger.info(
                        "spark-context-cache: rank %d committed %d tokens"
                        " digest=%s manifest=%s in %.1f ms",
                        snapshot.rank,
                        receipt.committed_tokens,
                        snapshot.plan.digest[:12],
                        receipt.manifest_digest[:12],
                        1e3 * (time.perf_counter() - commit_started),
                    )
                    self._finish_store(
                        snapshot.plan.digest,
                        committed=not evicted,
                        evicted=evicted,
                    )
            except Exception as error:  # noqa: BLE001 - never kill serving
                self._finish_store(
                    snapshot.plan.digest,
                    committed=False,
                    error=error,
                )
                continue

    def _finish_store(
        self,
        digest: str,
        *,
        committed: bool,
        evicted: bool = False,
        error: BaseException | None = None,
    ) -> None:
        with self._store_cv:
            if committed:
                # ManifestStore publishes each fsynced immutable chunk before
                # atomically publishing the fsynced manifest. Load re-verifies
                # every byte, so no completion-time readback or global sweep
                # is needed to make this digest eligible for quorum.
                self._held.add(digest)
                self.counters["store_committed"] += 1
            else:
                self._held.discard(digest)
                self.counters["store_evicted" if evicted else "store_failed"] += 1
            self._store_inflight = 0
            self._store_cv.notify_all()
        if error is not None:
            logger.warning(
                "spark-context-cache: store failed (entry skipped): %s",
                error,
            )

    def wait_for_pending_stores(self, timeout: float | None = None) -> bool:
        with self._store_cv:
            return self._store_cv.wait_for(
                lambda: self._store_inflight == 0,
                timeout,
            )

    def _store_one(self, plan: _ReqPlan) -> None:
        """Compatibility helper for offline callers; not the request path."""

        snapshot = self._snapshot_store(plan)
        chunks: Sequence[ContextChunk]
        if isinstance(snapshot, _HybridStoreSnapshot):
            chunks = _HybridSnapshotChunks(snapshot, self._chunk_tokens)
        else:
            chunks = _SnapshotChunks(snapshot, self._dcp_degree, self._chunk_tokens)
        receipt = self._store.commit(
            identity=snapshot.identity,
            context_digest=plan.digest,
            chunks=chunks,
            span_tokens=plan.span_tokens,
        )
        with self._capacity_lock:
            self._note_capacity_commit_locked(
                receipt.allocated_bytes_upper_bound
            )
            self._maintain_capacity_locked(
                wake_worker_on_unsatisfied=True
            )
            evicted = not self._store.lookup(
                snapshot.identity,
                plan.digest,
                verify_chunks=False,
                verify_chunk_metadata=True,
            ).is_hit
            if evicted:
                with self._store_cv:
                    self._held.discard(plan.digest)
        logger.info(
            "spark-context-cache: rank %d committed %d tokens digest=%s manifest=%s",
            snapshot.rank,
            receipt.committed_tokens,
            plan.digest[:12],
            receipt.manifest_digest[:12],
        )

    def _absorb_quorum(self, connector_output: Any) -> None:
        stats = getattr(connector_output, "kv_connector_stats", None)
        data = getattr(stats, "data", None)
        if not isinstance(data, dict):
            return
        payload = data.get("reports", data.get("spark_context_cache"))
        reports = payload if isinstance(payload, list) else [payload]
        for report in reports:
            if not isinstance(report, dict):
                continue
            rank = report.get("rank")
            held = report.get("held")
            generation = report.get("generation")
            if (
                type(rank) is not int
                or not 0 <= rank < self._tp_degree
                or not isinstance(held, list)
            ):
                continue
            if not isinstance(generation, str) or not generation:
                # Reports without a generation field share one inert sentinel.
                # Connectors from this source always send a UUID generation, so
                # an unversioned report cannot reset a worker's UUID state.
                generation = "missing-generation-field"
            previous = self._worker_generations.get(rank)
            if previous is not None and previous != generation:
                for digest, ranks in list(self._quorum.items()):
                    ranks.discard(rank)
                    if not ranks:
                        self._quorum.pop(digest, None)
                self.counters["quorum_generation_resets"] += 1
            self._worker_generations[rank] = generation
            held_set = {d for d in held if isinstance(d, str)}
            for digest in held_set:
                self._quorum.setdefault(digest, set()).add(rank)
            # a digest this rank no longer holds loses its confirmation
            for digest, ranks in list(self._quorum.items()):
                if digest not in held_set and rank in ranks:
                    ranks.discard(rank)
                    if not ranks:
                        self._quorum.pop(digest, None)

    def update_connector_output(self, connector_output: Any) -> None:
        self._absorb_quorum(connector_output)
        invalid = getattr(connector_output, "invalid_block_ids", None)
        if not invalid or not self._admitted:
            return
        # Async failure output reaches this callback before the parked
        # request is rescheduled, so retire the admission before its clean
        # recompute can republish the entry. Only admissions whose restored
        # blocks intersect the reported invalid blocks are retired: the
        # report carries no request id, and retiring every admission would
        # destroy unrelated healthy entries.
        invalid_blocks = set(invalid)
        for request_id, (digest, block_ids) in list(self._admitted.items()):
            if not (block_ids & invalid_blocks):
                continue
            # Under probe mode "none" the scheduler owns no cache root to
            # unlink; each damaged rank retires its own entry on load
            # failure, and dropping quorum below suppresses re-admission.
            if self._scheduler_probe == "tp0" and self._store.invalidate(
                self._identity(self._shard_rank), digest
            ):
                self.counters["scheduler_retired"] = (
                    self.counters.get("scheduler_retired", 0) + 1
                )
                logger.warning(
                    "spark-context-cache: retired entry %s after a rank"
                    " reported load errors (request %s)",
                    digest[:12],
                    request_id,
                )
            self._quorum.pop(digest, None)
            self._admitted.pop(request_id, None)

    @classmethod
    def build_kv_connector_stats(cls, data=None):
        return SparkCacheStats(data=data if data is not None else {})

    def get_kv_connector_stats(self):
        """Report this rank's manifest-offered digests to the scheduler.

        This is the admission quorum's transport: the scheduler will only
        offer a restore that every physical TP worker has confirmed.
        Load-time corruption withdraws the failing worker's offer and
        publishes invalid blocks so the request cleanly re-prefills
        instead of exposing restored state.

        The report uses the physical TP rank (unique across all workers),
        not the DCP-local rank, so TP0 and TP2 are not deduplicated
        despite sharing DCP-local rank 0 under TP4/DCP2.
        """
        if self._role is not KVConnectorRole.WORKER:
            return None
        with self._load_lock:
            held = sorted(self._held)
        report: dict[str, Any] = {
            "rank": self._physical_rank(),
            "held": held,
            "generation": self._stats_generation,
        }
        runtime = self._streaming_runtime
        status = getattr(runtime, "status", None)
        if self._streaming_snapshots_enabled and callable(status):
            report["streaming"] = status()
        if self._capacity_policy.enabled:
            capacity = dict(self._capacity_status)
            with self._capacity_handoff_cv:
                pending_streaming_commits = len(
                    self._streaming_capacity_pending
                )
            capacity.update(
                manifests_evicted=self.counters["capacity_manifests_evicted"],
                chunks_deleted=self.counters["capacity_chunks_deleted"],
                bytes_reclaimed=self.counters["capacity_bytes_reclaimed"],
                pending_streaming_commits=pending_streaming_commits,
                streaming_store_committed=self.counters[
                    "streaming_store_committed"
                ],
                streaming_store_evicted=self.counters[
                    "streaming_store_evicted"
                ],
                invalid_streaming_receipts=self.counters[
                    "streaming_capacity_invalid_receipts"
                ],
                shutdown_dropped_streaming_commits=self.counters[
                    "streaming_capacity_shutdown_dropped"
                ],
                maintenance_retries=self.counters["capacity_retries"],
            )
            report["capacity"] = capacity
        return SparkCacheStats(data={"reports": [report]})

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._load_lock:
            errors, self._load_errors = self._load_errors, set()
        return errors
