#include "spark_cache_page_capture_layout.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

namespace {

using spark_cache::page_capture::plan_capture;
using spark_cache::page_capture::validate_sources;

static_assert(sizeof(SparkCachePageCaptureSource) == 40);
static_assert(sizeof(SparkCachePageCaptureGroup) == 16);
static_assert(sizeof(SparkCachePageCaptureSpan) == 32);
static_assert(sizeof(SparkCachePageCapturePlan) == 24);
static_assert(sizeof(SparkCachePageCaptureSubmission) == 32);
static_assert(sizeof(SparkCachePageCaptureAbiInfo) == 48);

std::array<SparkCachePageCaptureSource, 3> sources() {
  return {{
      {0x1000, 1024, 4096, 4096, 0, 0, 0},
      {0x2000, 1024, 8192, 6144, 0, 1, 0},
      {0x3000, 1024, 256, 192, 1, 0, 0},
  }};
}

std::array<SparkCachePageCaptureGroup, 2> groups() {
  return {{{0, 3, {0, 0}}, {3, 2, {0, 0}}}};
}

void test_multigroup_layout_matches_block_page_body_order() {
  const auto inventory = sources();
  const auto request_groups = groups();
  const std::array<std::uint32_t, 5> pages{{17, 3, 91, 8, 5}};
  std::array<SparkCachePageCaptureSpan, 3> spans{};
  SparkCachePageCapturePlan plan{};
  std::string detail;
  assert(validate_sources(
      inventory.data(), inventory.size(), request_groups.size(), &detail));
  assert(plan_capture(
      inventory.data(),
      inventory.size(),
      request_groups.data(),
      request_groups.size(),
      pages.data(),
      pages.size(),
      64 * 1024,
      spans.data(),
      spans.size(),
      &plan,
      &detail));

  assert(plan.source_count == 3);
  assert(plan.group_count == 2);
  assert(plan.span_count == 3);
  assert(plan.used_bytes == 3 * 4096 + 3 * 6144 + 2 * 192);
  assert(spans[0].source_index == 0);
  assert(spans[0].physical_page_offset == 0);
  assert(spans[0].page_count == 3);
  assert(spans[0].destination_offset_bytes == 0);
  assert(spans[0].length_bytes == 3 * 4096);
  assert(spans[1].destination_offset_bytes == spans[0].length_bytes);
  assert(spans[1].physical_page_offset == 0);
  assert(spans[1].length_bytes == 3 * 6144);
  assert(spans[2].destination_offset_bytes ==
         spans[0].length_bytes + spans[1].length_bytes);
  assert(spans[2].physical_page_offset == 3);
  assert(spans[2].page_count == 2);
}

void test_invalid_inventory_is_rejected() {
  std::string detail;
  auto inventory = sources();
  inventory[1].layer_ordinal = 2;
  assert(!validate_sources(inventory.data(), inventory.size(), 2, &detail));

  inventory = sources();
  inventory[1].source_pages = 512;
  assert(!validate_sources(inventory.data(), inventory.size(), 2, &detail));

  inventory = sources();
  inventory[0].source_page_stride_bytes = 4095;
  assert(!validate_sources(inventory.data(), inventory.size(), 2, &detail));

  inventory = sources();
  inventory[0].source_page_stride_bytes =
      std::numeric_limits<std::uint64_t>::max();
  assert(!validate_sources(inventory.data(), inventory.size(), 2, &detail));

  inventory = sources();
  inventory[0].source_base = std::numeric_limits<std::uint64_t>::max() - 8;
  assert(!validate_sources(inventory.data(), inventory.size(), 2, &detail));
}

void test_request_geometry_is_exact_and_bounded() {
  const auto inventory = sources();
  auto request_groups = groups();
  std::array<std::uint32_t, 5> pages{{17, 3, 91, 8, 5}};
  std::array<SparkCachePageCaptureSpan, 3> spans{};
  SparkCachePageCapturePlan plan{};
  std::string detail;

  request_groups[1].physical_page_offset = 4;
  assert(!plan_capture(
      inventory.data(), inventory.size(), request_groups.data(),
      request_groups.size(), pages.data(), pages.size(), 64 * 1024,
      spans.data(), spans.size(), &plan, &detail));

  request_groups = groups();
  pages[4] = 1024;
  assert(!plan_capture(
      inventory.data(), inventory.size(), request_groups.data(),
      request_groups.size(), pages.data(), pages.size(), 64 * 1024,
      spans.data(), spans.size(), &plan, &detail));

  pages[4] = 5;
  assert(!plan_capture(
      inventory.data(), inventory.size(), request_groups.data(),
      request_groups.size(), pages.data(), pages.size(), 1024,
      spans.data(), spans.size(), &plan, &detail));

  assert(!plan_capture(
      inventory.data(), inventory.size(), request_groups.data(),
      request_groups.size(), pages.data(), pages.size(), 64 * 1024,
      spans.data(), 2, &plan, &detail));
}

void test_failure_leaves_caller_outputs_unchanged() {
  const auto inventory = sources();
  const auto request_groups = groups();
  const std::array<std::uint32_t, 5> pages{{17, 3, 91, 8, 1024}};
  std::array<SparkCachePageCaptureSpan, 3> spans{};
  spans[0].destination_offset_bytes = 77;
  SparkCachePageCapturePlan plan{};
  plan.used_bytes = 99;
  std::string detail;
  assert(!plan_capture(
      inventory.data(), inventory.size(), request_groups.data(),
      request_groups.size(), pages.data(), pages.size(), 64 * 1024,
      spans.data(), spans.size(), &plan, &detail));
  assert(spans[0].destination_offset_bytes == 77);
  assert(plan.used_bytes == 99);
}

}  // namespace

int main() {
  test_multigroup_layout_matches_block_page_body_order();
  test_invalid_inventory_is_rejected();
  test_request_geometry_is_exact_and_bounded();
  test_failure_leaves_caller_outputs_unchanged();
  return 0;
}
