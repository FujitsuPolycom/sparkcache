#include "spark_cache_page_capture_layout.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>

namespace spark_cache::page_capture {
namespace {

bool fail(std::string* detail, const char* message) {
  if (detail != nullptr) {
    *detail = message;
  }
  return false;
}

bool checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t* output) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  *output = left + right;
  return true;
}

bool checked_multiply(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t* output) {
  if (left != 0 &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *output = left * right;
  return true;
}

}  // namespace

bool validate_sources(
    const SparkCachePageCaptureSource* sources,
    std::uint32_t source_count,
    std::uint32_t group_count,
    std::string* detail) {
  if (sources == nullptr || source_count == 0 ||
      source_count > SPARK_CACHE_PAGE_CAPTURE_MAX_SOURCES ||
      group_count == 0 || group_count > SPARK_CACHE_PAGE_CAPTURE_MAX_GROUPS) {
    return fail(detail, "invalid manager-page source inventory size");
  }

  std::array<std::uint32_t, SPARK_CACHE_PAGE_CAPTURE_MAX_GROUPS>
      next_layer{};
  std::array<std::uint64_t, SPARK_CACHE_PAGE_CAPTURE_MAX_GROUPS>
      source_pages{};
  std::array<bool, SPARK_CACHE_PAGE_CAPTURE_MAX_GROUPS> seen{};
  std::uint32_t previous_group = 0;

  for (std::uint32_t index = 0; index < source_count; ++index) {
    const auto& source = sources[index];
    if (source.source_base == 0 || source.source_pages == 0 ||
        source.source_page_stride_bytes == 0 || source.bytes_per_page == 0 ||
        source.source_page_stride_bytes < source.bytes_per_page ||
        source.group_index >= group_count || source.flags != 0) {
      return fail(detail, "invalid manager-page source descriptor");
    }
    if (index != 0 && source.group_index < previous_group) {
      return fail(detail, "manager-page sources must be ordered by group");
    }
    previous_group = source.group_index;
    if (source.layer_ordinal != next_layer[source.group_index]) {
      return fail(
          detail,
          "manager-page layer ordinals must be dense within each group");
    }
    next_layer[source.group_index] += 1;
    if (!seen[source.group_index]) {
      seen[source.group_index] = true;
      source_pages[source.group_index] = source.source_pages;
    } else if (source_pages[source.group_index] != source.source_pages) {
      return fail(
          detail,
          "manager-page layers in one group must share page capacity");
    }

    std::uint64_t final_page_offset = 0;
    std::uint64_t required_bytes = 0;
    std::uint64_t source_end = 0;
    if (!checked_multiply(
            source.source_pages - 1,
            source.source_page_stride_bytes,
            &final_page_offset) ||
        !checked_add(
            final_page_offset,
            source.bytes_per_page,
            &required_bytes) ||
        !checked_add(source.source_base, required_bytes, &source_end)) {
      return fail(detail, "manager-page source address range overflows");
    }
  }

  for (std::uint32_t group = 0; group < group_count; ++group) {
    if (!seen[group]) {
      return fail(detail, "every manager-page group must contain a layer");
    }
  }
  if (detail != nullptr) {
    detail->clear();
  }
  return true;
}

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
    std::string* detail) {
  if (groups == nullptr || physical_pages == nullptr || spans == nullptr ||
      output == nullptr || physical_page_count == 0 || slot_bytes == 0 ||
      span_capacity < source_count) {
    return fail(detail, "invalid manager-page capture request");
  }
  if (!validate_sources(sources, source_count, group_count, detail)) {
    return false;
  }

  std::uint32_t expected_page_offset = 0;
  for (std::uint32_t group = 0; group < group_count; ++group) {
    const auto& descriptor = groups[group];
    if (descriptor.physical_page_offset != expected_page_offset ||
        descriptor.page_count == 0 || descriptor.reserved[0] != 0 ||
        descriptor.reserved[1] != 0 ||
        descriptor.page_count > physical_page_count - expected_page_offset) {
      return fail(
          detail,
          "manager-page groups must cover the physical-page table exactly");
    }
    expected_page_offset += descriptor.page_count;
  }
  if (expected_page_offset != physical_page_count) {
    return fail(
        detail,
        "manager-page groups leave trailing physical-page entries");
  }

  std::array<std::uint64_t, SPARK_CACHE_PAGE_CAPTURE_MAX_GROUPS>
      source_pages{};
  for (std::uint32_t index = 0; index < source_count; ++index) {
    source_pages[sources[index].group_index] = sources[index].source_pages;
  }
  for (std::uint32_t group = 0; group < group_count; ++group) {
    const auto& descriptor = groups[group];
    for (std::uint32_t page = 0; page < descriptor.page_count; ++page) {
      if (physical_pages[descriptor.physical_page_offset + page] >=
          source_pages[group]) {
        return fail(
            detail,
            "manager-page request references a page outside its source group");
      }
    }
  }

  std::array<SparkCachePageCaptureSpan,
             SPARK_CACHE_PAGE_CAPTURE_MAX_SOURCES>
      planned_spans{};
  std::uint64_t cursor = 0;
  for (std::uint32_t index = 0; index < source_count; ++index) {
    const auto& source = sources[index];
    const auto& group = groups[source.group_index];
    std::uint64_t length = 0;
    std::uint64_t end = 0;
    if (!checked_multiply(
            group.page_count,
            source.bytes_per_page,
            &length) ||
        !checked_add(cursor, length, &end) || end > slot_bytes) {
      return fail(detail, "manager-page payload exceeds the bounded ring slot");
    }
    planned_spans[index] = SparkCachePageCaptureSpan{
        cursor,
        length,
        index,
        group.physical_page_offset,
        group.page_count,
        0,
    };
    cursor = end;
  }

  SparkCachePageCapturePlan plan{
      cursor,
      source_count,
      group_count,
      source_count,
      0,
  };
  for (std::uint32_t index = 0; index < source_count; ++index) {
    spans[index] = planned_spans[index];
  }
  *output = plan;
  if (detail != nullptr) {
    detail->clear();
  }
  return true;
}

}  // namespace spark_cache::page_capture
