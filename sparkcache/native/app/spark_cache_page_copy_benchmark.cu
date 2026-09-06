// Compare placement libraries through their public ABI, including finish fences.
#include "spark_cache_placement.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr std::uint64_t MiB = 1024 * 1024;
constexpr std::uint64_t arena_bytes = 64 * MiB;

void cuda_check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template <typename T> T load(void* handle, const char* name) {
  auto symbol = dlsym(handle, name);
  if (!symbol) throw std::runtime_error(std::string("missing ABI symbol: ") + name);
  return reinterpret_cast<T>(symbol);
}

struct Api {
  void* handle;
#define ENTRY(name) decltype(&spark_cache_placement_##name) name
  ENTRY(create); ENTRY(destroy); ENTRY(configure_page_destinations);
  ENTRY(begin_page_restore); ENTRY(acquire_arena_view); ENTRY(submit_page_slab);
  ENTRY(finish_restore); ENTRY(last_error); ENTRY(abort_restore);
#undef ENTRY
  decltype(&spark_cache_reference_scatter_pages) reference;
  explicit Api(const char* path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!handle) throw std::runtime_error(dlerror());
#define BIND(name) name = load<decltype(name)>(handle, "spark_cache_placement_" #name)
    BIND(create); BIND(destroy); BIND(configure_page_destinations);
    BIND(begin_page_restore); BIND(acquire_arena_view); BIND(submit_page_slab);
    BIND(finish_restore); BIND(last_error); BIND(abort_restore);
#undef BIND
    reference = load<decltype(reference)>(handle, "spark_cache_reference_scatter_pages");
  }
  ~Api() { dlclose(handle); }
  void check(SparkCachePlacementStatus result, SparkCachePlacement* placement) {
    if (result != SPARK_CACHE_PLACEMENT_OK) throw std::runtime_error(last_error(placement));
  }
};

struct Case {
  const char* name;
  std::uint32_t page_bytes, pages, layers, span_bytes, padding, gap;
  bool irregular, split;
};

void run(Api& api, const Case& shape, int repetitions, std::uint32_t arena_mode) {
  const std::uint32_t physical_pages = shape.pages + (shape.padding ? 3 : 0);
  const std::uint32_t stride = shape.page_bytes + shape.padding;
  const std::uint64_t layer_bytes = std::uint64_t(shape.page_bytes) * shape.pages;
  const std::uint64_t pool_layer_bytes = std::uint64_t(stride) * physical_pages + 128;
  const std::uint64_t pool_bytes = pool_layer_bytes * shape.layers;
  const std::uint64_t explicit_allocations = 2 * arena_bytes + 2 * pool_bytes + 8 * MiB +
      (arena_mode == SPARK_CACHE_ARENA_STAGED_DEVICE ? 2 * arena_bytes : 0);
  if (explicit_allocations >= 512 * MiB) throw std::runtime_error("allocation ceiling exceeded");
  std::vector<std::uint8_t> expected(pool_bytes, 0xa5);
  std::vector<std::uint8_t> readback(8 * MiB);
  std::uint8_t* device = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&device), pool_bytes));
  cuda_check(cudaMemset(device, 0xa5, pool_bytes));
  SparkCachePlacement* placement = nullptr;
  SparkCachePlacementConfig config{};
  config.abi_version = SPARK_CACHE_PLACEMENT_ABI_VERSION;
  config.arena_mode = arena_mode;
  config.arena_bytes = arena_bytes;
  config.max_destinations = shape.layers;
  config.max_slots = physical_pages;
  config.max_chunks_per_slab = 4096;
  api.check(api.create(&config, &placement), placement);
  try {
    std::vector<SparkCachePageDestinationDescriptor> destinations(shape.layers);
    std::vector<SparkCachePageDestinationDescriptor> reference_destinations(shape.layers);
    for (std::uint32_t layer = 0; layer < shape.layers; ++layer) {
      destinations[layer] = {
          reinterpret_cast<std::uint64_t>(device + layer * pool_layer_bytes + 64),
          physical_pages, stride, shape.page_bytes, 0, 0};
      reference_destinations[layer] = destinations[layer];
      reference_destinations[layer].destination_base = reinterpret_cast<std::uint64_t>(
          expected.data() + layer * pool_layer_bytes + 64);
    }
    api.check(api.configure_page_destinations(placement, destinations.data(), destinations.size()), placement);
    std::vector<std::uint32_t> slots(physical_pages);
    std::iota(slots.begin(), slots.end(), 0);
    std::mt19937 random(1729);
    std::shuffle(slots.begin(), slots.end(), random);
    slots.resize(shape.pages);
    SparkCachePageGroupDescriptor group{0, shape.pages, 0, 0};
    std::vector<SparkCachePageCopySpan> spans;
    std::uint64_t used = 0, snapshot = 0;
    constexpr std::uint64_t lengths[] = {17, 65537, 131071, 8191};
    for (std::uint32_t layer = 0; layer < shape.layers; ++layer) {
      for (std::uint64_t offset = 0; offset < layer_bytes;) {
        const std::uint64_t requested = shape.irregular ? lengths[spans.size() % 4] : shape.span_bytes;
        const std::uint64_t bytes = std::min(layer_bytes - offset, requested);
        used += shape.gap;
        spans.push_back({used, snapshot, offset, bytes, layer, 0});
        used += bytes;
        snapshot += bytes;
        offset += bytes;
      }
    }
    if (used > arena_bytes || spans.size() > 4096) throw std::runtime_error("fixture exceeds arena");
    SparkCacheArenaView arena{};
    api.check(api.begin_page_restore(placement, &group, 1, slots.data(), slots.size(), snapshot), placement);
    api.check(api.acquire_arena_view(placement, 0, &arena), placement);
    auto* source = reinterpret_cast<std::uint8_t*>(arena.host_address);
    for (std::uint64_t index = 0; index < used; ++index) {
      source[index] = static_cast<std::uint8_t>(
          index * 131 + (index >> 8) * 17 + (index >> 16) * 31 + (index >> 24) * 73);
    }
    char detail[512]{};
    if (api.reference(source, used, snapshot, spans.data(), spans.size(),
                      reference_destinations.data(), reference_destinations.size(),
                      &group, 1, slots.data(), slots.size(), detail, sizeof(detail)) != SPARK_CACHE_PLACEMENT_OK) {
      throw std::runtime_error(detail);
    }
    api.check(api.abort_restore(placement), placement);
    std::vector<double> milliseconds;
    SparkCachePlacementStats stats{};
    for (int iteration = -3; iteration < repetitions; ++iteration) {
      // Clear outside the timed region so a prior successful iteration cannot
      // conceal a missing write in the final byte comparison.
      cuda_check(cudaMemset(device, 0xa5, pool_bytes));
      cuda_check(cudaDeviceSynchronize());
      const auto started = std::chrono::steady_clock::now();
      api.check(api.begin_page_restore(placement, &group, 1, slots.data(), slots.size(), snapshot), placement);
      const std::size_t batch_size = shape.split ? (spans.size() + 1) / 2 : spans.size();
      for (std::size_t first = 0; first < spans.size(); first += batch_size) {
        api.check(api.acquire_arena_view(placement, 0, &arena), placement);
        api.check(api.submit_page_slab(placement, 0, used, spans.data() + first,
                                      std::min(batch_size, spans.size() - first)), placement);
      }
      api.check(api.finish_restore(placement, &stats), placement);
      const double elapsed = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
      if (iteration >= 0) milliseconds.push_back(elapsed);
    }
    for (std::uint64_t offset = 0; offset < pool_bytes; offset += readback.size()) {
      const auto bytes = std::min<std::uint64_t>(readback.size(), pool_bytes - offset);
      cuda_check(cudaMemcpy(readback.data(), device + offset, bytes, cudaMemcpyDeviceToHost));
      if (std::memcmp(readback.data(), expected.data() + offset, bytes) != 0) {
        throw std::runtime_error(std::string("byte mismatch including canaries: ") + shape.name);
      }
    }
    if (stats.device_error != 0) throw std::runtime_error("device error after finish");
    std::sort(milliseconds.begin(), milliseconds.end());
    const double median = milliseconds[milliseconds.size() / 2];
    std::cout << "{\"case\":\"" << shape.name << "\",\"mode\":" << arena_mode
              << ",\"payload_bytes\":" << snapshot << ",\"spans\":" << spans.size()
              << ",\"explicit_allocation_bytes\":" << explicit_allocations
              << ",\"median_ms\":" << median
              << ",\"p95_ms\":" << milliseconds[(milliseconds.size() * 95 + 99) / 100 - 1]
              << ",\"gib_per_second\":" << snapshot * 1000.0 / (median * 1024 * 1024 * 1024)
              << ",\"byte_equal\":true,\"device_error\":0}" << std::endl;
  } catch (...) {
    api.abort_restore(placement);
    api.destroy(placement);
    cudaFree(device);
    throw;
  }
  api.destroy(placement);
  cuda_check(cudaFree(device));
}

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4) throw std::runtime_error("usage: benchmark LIBRARY [ITERATIONS=15] [ARENA_MODE=1]");
    const int repetitions = argc > 2 ? std::stoi(argv[2]) : 15;
    const std::uint32_t mode = argc > 3 ? std::stoul(argv[3]) : 1;
    if (repetitions < 1 || repetitions > 200 || mode < 1 || mode > 3) throw std::runtime_error("invalid bounds");
    cuda_check(cudaSetDevice(0));
    Api api(argv[1]);
    const Case cases[] = {
        {"single_span_64mib", 4096, 16384, 1, 64 * 1024 * 1024, 0, 0, false, false},
        {"huge_pages_64mib", 32 * 1024 * 1024, 2, 1, 64 * 1024 * 1024, 0, 0, false, false},
        {"layers_many_spans_64mib", 2048, 4096, 8, 65536, 0, 0, false, false},
        {"framed_irregular_32mib", 2048, 2048, 8, 0, 16, 3, true, false},
        {"odd_pages_split_slabs", 257, 4093, 3, 0, 15, 3, true, true},
    };
    for (const auto& shape : cases) run(api, shape, repetitions, mode);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_failed: " << error.what() << '\n';
    return 1;
  }
}
