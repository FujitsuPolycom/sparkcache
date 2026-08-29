#include "spark_cache_placement.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

bool cuda_ok(cudaError_t result, const char* operation) {
  if (result == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
  return false;
}

bool placement_ok(
    SparkCachePlacementStatus status,
    SparkCachePlacement* placement,
    const char* operation) {
  if (status == SPARK_CACHE_PLACEMENT_OK) {
    return true;
  }
  std::fprintf(
      stderr,
      "%s failed: status=%d detail=%s\n",
      operation,
      static_cast<int>(status),
      spark_cache_placement_last_error(placement));
  return false;
}

}  // namespace

int main() {
  SparkCachePlacementConfig config{};
  config.abi_version = SPARK_CACHE_PLACEMENT_ABI_VERSION;
  config.arena_mode = SPARK_CACHE_ARENA_MAPPED_HOST;
  config.arena_bytes = 64ULL * 1024ULL * 1024ULL;
  config.max_destinations = 4;
  config.max_slots = 8;
  config.max_chunks_per_slab = 8;
  config.device_ordinal = 0;

  SparkCachePlacement* placement = nullptr;
  bool ok = placement_ok(
      spark_cache_placement_create(&config, &placement), placement, "create");
  std::uint8_t* destination = nullptr;
  ok = ok && cuda_ok(
      cudaMalloc(reinterpret_cast<void**>(&destination), 16),
      "cudaMalloc(destination)");
  ok = ok && cuda_ok(cudaMemset(destination, 0xEE, 16), "cudaMemset(destination)");

  const SparkCachePageDestinationDescriptor page_destination{
      reinterpret_cast<std::uintptr_t>(destination), 4, 4, 4, 0, 0};
  const SparkCachePageGroupDescriptor group{0, 2, 0, 0};
  const std::array<std::uint32_t, 2> slots{2, 0};
  ok = ok && placement_ok(
      spark_cache_placement_configure_page_destinations(
          placement, &page_destination, 1),
      placement,
      "configure page destinations");
  ok = ok && placement_ok(
      spark_cache_placement_begin_page_restore(
          placement, &group, 1, slots.data(), slots.size(), 8),
      placement,
      "begin page restore");

  void* arena = nullptr;
  std::uint64_t capacity = 0;
  ok = ok && placement_ok(
      spark_cache_placement_acquire_arena(
          placement, 0, &arena, &capacity),
      placement,
      "acquire arena");
  const std::array<std::uint8_t, 8> source{1, 2, 3, 4, 5, 6, 7, 8};
  if (ok && capacity >= source.size()) {
    std::memcpy(arena, source.data(), source.size());
  } else {
    ok = false;
  }
  const SparkCachePageCopySpan span{0, 0, 0, source.size(), 0, 0};
  ok = ok && placement_ok(
      spark_cache_placement_submit_page_slab(
          placement, 0, source.size(), &span, 1),
      placement,
      "submit page slab");
  SparkCachePlacementStats stats{};
  ok = ok && placement_ok(
      spark_cache_placement_finish_restore(placement, &stats),
      placement,
      "finish restore");

  std::array<std::uint8_t, 16> got{};
  ok = ok && cuda_ok(
      cudaMemcpy(got.data(), destination, got.size(), cudaMemcpyDeviceToHost),
      "copy destination");
  const std::array<std::uint8_t, 16> expected{
      5, 6, 7, 8,
      0xEE, 0xEE, 0xEE, 0xEE,
      1, 2, 3, 4,
      0xEE, 0xEE, 0xEE, 0xEE};
  ok = ok && got == expected && stats.slot_uploads == 1 &&
       stats.destination_table_uploads == 1 && stats.slabs_submitted == 1 &&
       stats.scatter_kernel_launches == 1 && stats.device_error == 0 &&
       stats.staged_h2d_bytes == 0;

  std::printf("mapped hybrid-page probe %s\n", ok ? "PASS" : "FAIL");
  cudaFree(destination);
  spark_cache_placement_destroy(placement);
  return ok ? 0 : 1;
}
