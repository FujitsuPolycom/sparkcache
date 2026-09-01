#ifndef SPARK_CACHE_PAGE_CAPTURE_LAYOUT_HPP_
#define SPARK_CACHE_PAGE_CAPTURE_LAYOUT_HPP_

#include "spark_cache_page_capture.h"

#include <cstdint>
#include <string>

namespace spark_cache::page_capture {

bool validate_sources(
    const SparkCachePageCaptureSource* sources,
    std::uint32_t source_count,
    std::uint32_t group_count,
    std::string* detail);

/*
 * Build the exact raw page-body layout without touching CUDA memory.
 *
 * On failure, `output` and `spans` are unchanged. A CUDA submit edge can
 * therefore validate into temporary CPU storage before reserving a ring slot
 * or recording any event.
 */
bool plan_capture(
    const SparkCachePageCaptureSource* sources,
    std::uint32_t source_count,
    const SparkCachePageCaptureGroup* groups,
    std::uint32_t group_count,
    const std::uint32_t* physical_pages,
    std::uint32_t physical_page_count,
    std::uint64_t slot_bytes,
    SparkCachePageCaptureSpan* spans,
    std::uint32_t span_capacity,
    SparkCachePageCapturePlan* output,
    std::string* detail);

}  // namespace spark_cache::page_capture

#endif  // SPARK_CACHE_PAGE_CAPTURE_LAYOUT_HPP_
