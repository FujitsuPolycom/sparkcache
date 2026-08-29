#ifndef SPARK_CACHE_PLACEMENT_H_
#define SPARK_CACHE_PLACEMENT_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(SPARK_CACHE_PLACEMENT_BUILD)
#define SPARK_CACHE_API __declspec(dllexport)
#else
#define SPARK_CACHE_API __declspec(dllimport)
#endif
#else
#define SPARK_CACHE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_CACHE_PLACEMENT_ABI_VERSION 1u
#define SPARK_CACHE_PAGE_PLACEMENT_ABI_VERSION 1u
#define SPARK_CACHE_PLACEMENT_ARENA_COUNT 2u
#define SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS 4u

typedef enum SparkCachePlacementStatus {
  SPARK_CACHE_PLACEMENT_OK = 0,
  SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT = 1,
  SPARK_CACHE_PLACEMENT_INVALID_STATE = 2,
  SPARK_CACHE_PLACEMENT_FORMAT_ERROR = 3,
  SPARK_CACHE_PLACEMENT_CUDA_ERROR = 4,
  SPARK_CACHE_PLACEMENT_DEVICE_ERROR = 5
} SparkCachePlacementStatus;

typedef enum SparkCacheArenaMode {
  /*
   * cudaHostAllocMapped(): CPU pread/hash and the GPU scatter kernel touch the
   * same physical bytes. This is the preferred DGX Spark UMA experiment.
   */
  SPARK_CACHE_ARENA_MAPPED_HOST = 1,
  /*
   * cudaMallocManaged(): useful as an A/B, but page-fault/prefetch behavior
   * must be measured rather than assumed to beat mapped host memory.
   */
  SPARK_CACHE_ARENA_MANAGED = 2,
  /*
   * Pinned host producer arena plus a device arena and one H2D per slab.
   * This is the conservative discrete-GPU fallback.
   */
  SPARK_CACHE_ARENA_STAGED_DEVICE = 3
} SparkCacheArenaMode;

typedef enum SparkCacheRecordKind {
  SPARK_CACHE_RECORD_TARGET_CKV = 0,
  SPARK_CACHE_RECORD_SPARSE_INDEXER = 1,
  SPARK_CACHE_RECORD_MTP_DRAFT_KV = 2,
  SPARK_CACHE_RECORD_BOUNDARY_HIDDEN = 3
} SparkCacheRecordKind;

enum {
  SPARK_CACHE_CONFIG_PREFETCH_MANAGED = 1u << 0
};

enum {
  SPARK_CACHE_CAP_MAPPED_HOST = 1u << 0,
  SPARK_CACHE_CAP_MANAGED = 1u << 1,
  SPARK_CACHE_CAP_STAGED_DEVICE = 1u << 2,
  SPARK_CACHE_CAP_DIRECT_ENCODED = 1u << 3,
  SPARK_CACHE_CAP_TRANSPOSED = 1u << 4,
  SPARK_CACHE_CAP_LOW_PRIORITY_STREAMS = 1u << 5,
  /* The optional page ABI and its CPU byte-exact reference are present. */
  SPARK_CACHE_CAP_HYBRID_PAGE_REFERENCE = 1u << 6,
  SPARK_CACHE_CAP_HYBRID_PAGE_CUDA = 1u << 7
};

enum {
  SPARK_CACHE_PAGE_CAP_REFERENCE_SCATTER = 1u << 0,
  SPARK_CACHE_PAGE_CAP_CUDA_SCATTER = 1u << 1
};

enum {
  SPARK_CACHE_ARENA_VIEW_ACQUIRED = 1u << 0,
  SPARK_CACHE_ARENA_VIEW_MAPPED_HOST = 1u << 1,
  SPARK_CACHE_ARENA_VIEW_MANAGED = 1u << 2,
  SPARK_CACHE_ARENA_VIEW_STAGED_DEVICE = 1u << 3
};

/*
 * One stable destination entry per registered cache tensor. The table is
 * uploaded once after register_kv_caches(), not once per restored context.
 *
 * The source record is layer-major:
 *   [layer_ordinal][row_in_chunk][bytes_per_token].
 */
typedef struct SparkCacheDestinationDescriptor {
  uint64_t destination_base;
  uint64_t destination_rows;
  uint32_t destination_row_stride_bytes;
  uint32_t bytes_per_token;
  uint32_t record_kind;
  uint32_t source_layer_ordinal;
} SparkCacheDestinationDescriptor;

/*
 * One descriptor for an integrity-verified encoded .spcc chunk residing in
 * an arena. Offsets are relative to arena_base or the encoded payload.
 *
 * The descriptor deliberately contains no persisted physical KV slot. Slots
 * are supplied separately from the current request's allocator.
 */
typedef struct SparkCacheChunkDescriptor {
  uint64_t arena_offset_bytes;
  uint32_t encoded_bytes;
  uint32_t payload_offset_bytes;
  uint32_t first_slot_index;
  uint32_t row_count;
  uint32_t record_mask;
  uint32_t flags;
  uint32_t record_offset_bytes[SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS];
  uint32_t record_length_bytes[SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS];
} SparkCacheChunkDescriptor;

/*
 * Fallback input for the existing vectorized Python/NumPy slab assembler.
 * Each source is a layer-major [slot_count, bytes_per_token] byte matrix.
 */
typedef struct SparkCacheTransposedSource {
  uint64_t source_offset_bytes;
  uint32_t destination_index;
  uint32_t flags;
} SparkCacheTransposedSource;

/*
 * One opaque page tensor. `group_index` selects the current request's
 * flattened physical-slot subvector. Page bytes are copied without assigning
 * token-row semantics to attention, recurrent, or compressor state.
 */
typedef struct SparkCachePageDestinationDescriptor {
  uint64_t destination_base;
  uint64_t destination_pages;
  uint32_t destination_page_stride_bytes;
  uint32_t bytes_per_page;
  uint32_t group_index;
  uint32_t flags;
} SparkCachePageDestinationDescriptor;

/*
 * A page group owns one contiguous subvector of the flattened physical-page
 * slots. Slot values must be unique within a group; different groups may use
 * the same numeric physical slot because they address different allocators.
 */
typedef struct SparkCachePageGroupDescriptor {
  uint32_t first_slot_index;
  uint32_t slot_count;
  uint32_t flags;
  uint32_t reserved;
} SparkCachePageGroupDescriptor;

/*
 * One verified source extent. Spans must cover the logical snapshot in
 * ascending `snapshot_offset_bytes` order and cover every destination from
 * byte zero through all logical pages. Arena offsets may be discontiguous so
 * callers can describe payload extents split by encoded chunk framing.
 */
typedef struct SparkCachePageCopySpan {
  uint64_t arena_offset_bytes;
  uint64_t snapshot_offset_bytes;
  uint64_t destination_byte_offset;
  uint64_t byte_count;
  uint32_t destination_index;
  uint32_t flags;
} SparkCachePageCopySpan;

typedef struct SparkCachePlacementConfig {
  uint32_t abi_version;
  uint32_t arena_mode;
  uint64_t arena_bytes;
  uint32_t max_destinations;
  uint32_t max_slots;
  uint32_t max_chunks_per_slab;
  int32_t device_ordinal;
  uint32_t flags;
  uint32_t reserved[3];
} SparkCachePlacementConfig;

typedef struct SparkCachePlacementStats {
  uint64_t source_bytes;
  uint64_t staged_h2d_bytes;
  uint64_t restored_rows;
  uint32_t slot_uploads;
  uint32_t destination_table_uploads;
  uint32_t slabs_submitted;
  uint32_t scatter_kernel_launches;
  uint32_t device_error;
  uint32_t reserved[3];
} SparkCachePlacementStats;

/*
 * Runtime-owned ABI fingerprint. A ctypes caller must compare every reported
 * sizeof value with its local Structure before passing any descriptor.
 */
typedef struct SparkCachePlacementAbiInfo {
  uint32_t abi_version;
  uint32_t cudart_version;
  uint32_t arena_count;
  uint32_t max_record_kinds;
  uint32_t sizeof_config;
  uint32_t sizeof_destination;
  uint32_t sizeof_chunk;
  uint32_t sizeof_transposed_source;
  uint32_t sizeof_stats;
  uint32_t sizeof_arena_view;
  uint32_t capability_flags;
  uint32_t reserved[5];
} SparkCachePlacementAbiInfo;

/* Separate fingerprint keeps every placement ABI v1 structure unchanged. */
typedef struct SparkCachePagePlacementAbiInfo {
  uint32_t abi_version;
  uint32_t sizeof_destination;
  uint32_t sizeof_group;
  uint32_t sizeof_copy_span;
  uint32_t capability_flags;
  uint32_t reserved[3];
} SparkCachePagePlacementAbiInfo;

/*
 * Integer addresses make the arena ABI unambiguous for ctypes. `host_address`
 * is suitable for `(ctypes.c_ubyte * capacity).from_address(...)` followed by
 * a writable `memoryview(...).cast("B")` passed to os.preadv().
 */
typedef struct SparkCacheArenaView {
  uint64_t host_address;
  uint64_t device_address;
  uint64_t capacity_bytes;
  uint32_t arena_index;
  uint32_t arena_mode;
  uint32_t flags;
  uint32_t reserved;
} SparkCacheArenaView;

typedef struct SparkCachePlacement SparkCachePlacement;

SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_query_abi(
    SparkCachePlacementAbiInfo* output);

/*
 * Optional hybrid-page capability query. Callers must resolve this symbol
 * dynamically and retain the row path when it is absent.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_query_page_abi(
    SparkCachePagePlacementAbiInfo* output);

/*
 * Parse only after the caller has verified the manifest's outer SHA-256 over
 * the complete encoded chunk. This parser validates the v1 prefix, canonical
 * header, record bounds/order, required records, and every logical position.
 * It never allocates or copies record payloads.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_parse_verified_v1_chunk(
    const void* arena_base,
    uint64_t arena_used_bytes,
    uint64_t arena_offset_bytes,
    uint32_t encoded_bytes,
    uint32_t expected_logical_start,
    uint32_t dcp_degree,
    uint32_t dcp_rank,
    uint32_t first_slot_index,
    uint32_t required_data_record_mask,
    SparkCacheChunkDescriptor* output,
    char* error,
    size_t error_capacity);

/*
 * CPU byte-for-byte oracle for tests. Destination bases must be host pointers
 * when calling this function. GPU device destinations must use the CUDA
 * placement API below.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_reference_scatter_direct(
    const void* arena_base,
    uint64_t arena_used_bytes,
    const SparkCacheChunkDescriptor* chunks,
    uint32_t chunk_count,
    const SparkCacheDestinationDescriptor* destinations,
    uint32_t destination_count,
    const uint32_t* slots,
    uint32_t slot_count,
    char* error,
    size_t error_capacity);

/*
 * GPU-free byte-for-byte oracle for opaque hybrid pages. Destination bases
 * must be writable host pointers. All validation completes before the first
 * destination byte is changed.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_reference_scatter_pages(
    const void* arena_base,
    uint64_t arena_used_bytes,
    uint64_t snapshot_bytes,
    const SparkCachePageCopySpan* spans,
    uint32_t span_count,
    const SparkCachePageDestinationDescriptor* destinations,
    uint32_t destination_count,
    const SparkCachePageGroupDescriptor* groups,
    uint32_t group_count,
    const uint32_t* slots,
    uint32_t slot_count,
    char* error,
    size_t error_capacity);

SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_create(
    const SparkCachePlacementConfig* config,
    SparkCachePlacement** output);

SPARK_CACHE_API void spark_cache_placement_destroy(
    SparkCachePlacement* placement);

SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_configure_destinations(
    SparkCachePlacement* placement,
    const SparkCacheDestinationDescriptor* destinations,
    uint32_t destination_count);

SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_configure_page_destinations(
    SparkCachePlacement* placement,
    const SparkCachePageDestinationDescriptor* destinations,
    uint32_t destination_count);

/*
 * Begins one restore transaction and uploads the remapped physical slot vector
 * exactly once. All slots must be unique and in range for every destination.
 */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_begin_restore(
    SparkCachePlacement* placement,
    const uint32_t* slots,
    uint32_t slot_count);

SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_begin_page_restore(
    SparkCachePlacement* placement,
    const SparkCachePageGroupDescriptor* groups,
    uint32_t group_count,
    const uint32_t* slots,
    uint32_t slot_count,
    uint64_t snapshot_bytes);

/*
 * Waits until an arena is no longer read by a prior kernel, then returns the
 * CPU producer address. The caller may pread directly into this allocation.
 */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_acquire_arena(
    SparkCachePlacement* placement,
    uint32_t arena_index,
    void** host_pointer,
    uint64_t* capacity_bytes);

/*
 * ctypes-preferred equivalent of acquire_arena(). Both calls acquire the
 * arena; call exactly one of them for a given fill/submit cycle.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_acquire_arena_view(
    SparkCachePlacement* placement,
    uint32_t arena_index,
    SparkCacheArenaView* output);

/*
 * Direct path: scatter all configured destinations from verified encoded
 * chunks. Chunk row ranges must be contiguous, non-overlapping, and cover the
 * slot vector exactly across all submitted slabs.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_submit_direct_slab(
    SparkCachePlacement* placement,
    uint32_t arena_index,
    uint64_t arena_used_bytes,
    const SparkCacheChunkDescriptor* chunks,
    uint32_t chunk_count);

/*
 * Fallback path: scatter a subset of already-transposed layer slabs. Every
 * destination must appear exactly once across all submitted slabs.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_submit_transposed_slab(
    SparkCachePlacement* placement,
    uint32_t arena_index,
    uint64_t arena_used_bytes,
    const SparkCacheTransposedSource* sources,
    uint32_t source_count);

SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_submit_page_slab(
    SparkCachePlacement* placement,
    uint32_t arena_index,
    uint64_t arena_used_bytes,
    const SparkCachePageCopySpan* spans,
    uint32_t span_count);

/*
 * Synchronizes both arena streams, verifies complete coverage and the device
 * error word, and only then makes the restored request eligible to resume.
 */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_finish_restore(
    SparkCachePlacement* placement,
    SparkCachePlacementStats* stats);

/*
 * A CUDA kernel cannot be safely cancelled. Abort waits for submitted work,
 * discards the transaction, and leaves the request's parked KV blocks for the
 * caller to free/recompute. No consumer may observe them before finish().
 */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_abort_restore(
    SparkCachePlacement* placement);

/* Snapshot is available before, during, or after a restore. */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_get_stats(
    const SparkCachePlacement* placement,
    SparkCachePlacementStats* output);

SPARK_CACHE_API const char* spark_cache_placement_last_error(
    const SparkCachePlacement* placement);

/*
 * The runtime error is thread-local and remains useful when create() failed
 * before returning a handle. Prefer copy_last_error for exception-safe ctypes
 * wrappers that do not retain a borrowed C string.
 */
SPARK_CACHE_API const char* spark_cache_placement_runtime_last_error(void);

SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_copy_last_error(
    const SparkCachePlacement* placement,
    char* output,
    size_t output_capacity);

SPARK_CACHE_API const char* spark_cache_placement_status_string(
    SparkCachePlacementStatus status);

#ifdef __cplusplus
}
#endif

#endif  // SPARK_CACHE_PLACEMENT_H_
