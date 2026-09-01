#ifndef SPARK_CACHE_PAGE_CAPTURE_H_
#define SPARK_CACHE_PAGE_CAPTURE_H_

#include <stdint.h>

/*
 * CPU-visible descriptors for research-only CUDA manager-page capture.
 *
 * These structures do not enable capture. They define the immutable model
 * inventory, one request's group-qualified page table, and the raw payload
 * spans that the C++/CUDA implementation must produce. The output is the
 * exact opaque page body expected by SparkCache's block-page codec; physical
 * page IDs are transient and never enter persistent data.
 */
#define SPARK_CACHE_PAGE_CAPTURE_CONTRACT_VERSION 1u
#define SPARK_CACHE_PAGE_CAPTURE_MAX_GROUPS 16u
#define SPARK_CACHE_PAGE_CAPTURE_MAX_SOURCES 256u

typedef struct SparkCachePageCaptureSource {
  uint64_t source_base;
  uint64_t source_pages;
  uint64_t source_page_stride_bytes;
  uint32_t bytes_per_page;
  uint32_t group_index;
  uint32_t layer_ordinal;
  uint32_t flags;
} SparkCachePageCaptureSource;

/*
 * Each group points into one flattened array of request-owned physical page
 * IDs. Groups must cover that array exactly and in increasing group order.
 */
typedef struct SparkCachePageCaptureGroup {
  uint32_t physical_page_offset;
  uint32_t page_count;
  uint32_t reserved[2];
} SparkCachePageCaptureGroup;

/*
 * One span copies the selected pages for one layer. Spans are ordered by
 * group, then layer, and cover [0, used_bytes) without gaps or overlap.
 */
typedef struct SparkCachePageCaptureSpan {
  uint64_t destination_offset_bytes;
  uint64_t length_bytes;
  uint32_t source_index;
  uint32_t physical_page_offset;
  uint32_t page_count;
  uint32_t reserved;
} SparkCachePageCaptureSpan;

typedef struct SparkCachePageCapturePlan {
  uint64_t used_bytes;
  uint32_t span_count;
  uint32_t group_count;
  uint32_t source_count;
  uint32_t reserved;
} SparkCachePageCapturePlan;

typedef struct SparkCachePageCaptureSubmission {
  uint64_t context_sequence;
  uint64_t logical_start;
  uint32_t physical_page_count;
  uint32_t group_count;
  uint32_t flags;
  uint32_t reserved;
} SparkCachePageCaptureSubmission;

typedef struct SparkCachePageCaptureAbiInfo {
  uint32_t contract_version;
  uint32_t max_groups;
  uint32_t max_sources;
  uint32_t sizeof_source;
  uint32_t sizeof_group;
  uint32_t sizeof_span;
  uint32_t sizeof_plan;
  uint32_t sizeof_submission;
  uint32_t capability_flags;
  uint32_t reserved[3];
} SparkCachePageCaptureAbiInfo;

#endif  // SPARK_CACHE_PAGE_CAPTURE_H_
