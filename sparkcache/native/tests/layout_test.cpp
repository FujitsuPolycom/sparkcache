#include "spark_cache_placement.h"
#include "spark_cache_placement_layout.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

void append_u32_le(std::vector<std::uint8_t>* output, std::uint32_t value) {
  output->push_back(static_cast<std::uint8_t>(value));
  output->push_back(static_cast<std::uint8_t>(value >> 8U));
  output->push_back(static_cast<std::uint8_t>(value >> 16U));
  output->push_back(static_cast<std::uint8_t>(value >> 24U));
}

std::vector<std::uint8_t> make_chunk() {
  const std::string sha(64, '0');
  const std::string header =
      "{\"format_abi\":1,\"logical_end\":8,\"logical_start\":0,"
      "\"records\":["
      "{\"kind\":\"logical_positions\",\"length\":16,\"offset\":0,"
      "\"sha256\":\"" +
      sha +
      "\"},"
      "{\"kind\":\"sparse_indexer\",\"length\":4,\"offset\":16,"
      "\"sha256\":\"" +
      sha +
      "\"},"
      "{\"kind\":\"target_ckv\",\"length\":16,\"offset\":20,"
      "\"sha256\":\"" +
      sha + "\"}]}";
  std::vector<std::uint8_t> encoded{
      'S', 'P', 'C', 'K', 'V', '0', '0', '1'};
  append_u32_le(&encoded, 1);
  append_u32_le(&encoded, static_cast<std::uint32_t>(header.size()));
  encoded.insert(encoded.end(), header.begin(), header.end());
  for (const std::uint32_t position : {0U, 2U, 4U, 6U}) {
    append_u32_le(&encoded, position);
  }
  encoded.insert(encoded.end(), {21, 22, 23, 24});
  encoded.insert(
      encoded.end(), {1, 2, 3, 4, 5, 6, 7, 8});
  encoded.insert(
      encoded.end(), {11, 12, 13, 14, 15, 16, 17, 18});
  return encoded;
}

void test_inventory_and_slab_math() {
  constexpr std::uint64_t global_tokens = 393216;
  constexpr std::uint64_t local_rows = global_tokens / 4;
  constexpr std::uint64_t target_layers = 79;
  constexpr std::uint64_t target_width = 368;
  constexpr std::uint64_t indexer_layers = 22;
  constexpr std::uint64_t indexer_width = 132;
  constexpr std::uint64_t bytes =
      local_rows *
      (target_layers * target_width + indexer_layers * indexer_width);
  static_assert(bytes == 3143368704ULL);

  const auto slab_count = [&](std::uint64_t cap) {
    const std::uint64_t target_per =
        cap / (local_rows * target_width);
    const std::uint64_t indexer_per =
        cap / (local_rows * indexer_width);
    return (target_layers + target_per - 1) / target_per +
           (indexer_layers + indexer_per - 1) / indexer_per;
  };
  assert(slab_count(128 * spark_cache::placement::kMiB) == 30);
  assert(slab_count(256 * spark_cache::placement::kMiB) == 14);

  // The direct path processes encoded chunks instead of transposing layers.
  // The exact live entry has 1,535 chunks / 3,142,449,596 encoded bytes.
  constexpr std::uint64_t encoded_bytes = 3142449596ULL;
  constexpr std::uint64_t chunks = 1535;
  const std::uint64_t average_chunk =
      (encoded_bytes + chunks - 1) / chunks;
  assert(average_chunk > 1900 * 1024);
  assert(average_chunk < 2100 * 1024);
}

void test_config_validation() {
  SparkCachePlacementConfig config{};
  config.abi_version = SPARK_CACHE_PLACEMENT_ABI_VERSION;
  config.arena_mode = SPARK_CACHE_ARENA_MAPPED_HOST;
  config.arena_bytes = 128 * spark_cache::placement::kMiB;
  config.max_destinations = 128;
  config.max_slots = 131072;
  config.max_chunks_per_slab = 128;
  std::string error;
  assert(spark_cache::placement::validate_config(config, &error));
  config.arena_bytes = 192 * spark_cache::placement::kMiB;
  assert(!spark_cache::placement::validate_config(config, &error));
}

void test_parse_and_byte_exact_scatter() {
  auto encoded = make_chunk();
  SparkCacheChunkDescriptor chunk{};
  std::array<char, 256> error{};
  const auto parsed = spark_cache_parse_verified_v1_chunk(
      encoded.data(),
      encoded.size(),
      0,
      static_cast<std::uint32_t>(encoded.size()),
      0,
      2,
      0,
      0,
      (1U << SPARK_CACHE_RECORD_TARGET_CKV) |
          (1U << SPARK_CACHE_RECORD_SPARSE_INDEXER),
      &chunk,
      error.data(),
      error.size());
  assert(parsed == SPARK_CACHE_PLACEMENT_OK);
  assert(chunk.row_count == 4);
  assert(chunk.first_slot_index == 0);
  assert(chunk.record_offset_bytes[SPARK_CACHE_RECORD_SPARSE_INDEXER] == 16);
  assert(chunk.record_offset_bytes[SPARK_CACHE_RECORD_TARGET_CKV] == 20);
  assert(chunk.record_length_bytes[SPARK_CACHE_RECORD_TARGET_CKV] == 16);

  std::array<std::uint8_t, 16> target0{};
  std::array<std::uint8_t, 16> target1{};
  std::array<std::uint8_t, 8> indexer{};
  const std::array<std::uint32_t, 4> slots{5, 1, 7, 3};
  const std::array<SparkCacheDestinationDescriptor, 3> destinations{{
      {
          reinterpret_cast<std::uintptr_t>(target0.data()),
          8,
          2,
          2,
          SPARK_CACHE_RECORD_TARGET_CKV,
          0,
      },
      {
          reinterpret_cast<std::uintptr_t>(target1.data()),
          8,
          2,
          2,
          SPARK_CACHE_RECORD_TARGET_CKV,
          1,
      },
      {
          reinterpret_cast<std::uintptr_t>(indexer.data()),
          8,
          1,
          1,
          SPARK_CACHE_RECORD_SPARSE_INDEXER,
          0,
      },
  }};
  const auto scattered = spark_cache_reference_scatter_direct(
      encoded.data(),
      encoded.size(),
      &chunk,
      1,
      destinations.data(),
      destinations.size(),
      slots.data(),
      slots.size(),
      error.data(),
      error.size());
  assert(scattered == SPARK_CACHE_PLACEMENT_OK);
  assert((std::array<std::uint8_t, 2>{
              target0[slots[0] * 2], target0[slots[0] * 2 + 1]} ==
          std::array<std::uint8_t, 2>{1, 2}));
  assert((std::array<std::uint8_t, 2>{
              target0[slots[3] * 2], target0[slots[3] * 2 + 1]} ==
          std::array<std::uint8_t, 2>{7, 8}));
  assert((std::array<std::uint8_t, 2>{
              target1[slots[1] * 2], target1[slots[1] * 2 + 1]} ==
          std::array<std::uint8_t, 2>{13, 14}));
  assert(indexer[slots[2]] == 23);
}

void test_parser_rejects_wrong_positions() {
  auto encoded = make_chunk();
  const std::uint32_t header_bytes =
      static_cast<std::uint32_t>(encoded[12]) |
      (static_cast<std::uint32_t>(encoded[13]) << 8U) |
      (static_cast<std::uint32_t>(encoded[14]) << 16U) |
      (static_cast<std::uint32_t>(encoded[15]) << 24U);
  encoded[16 + header_bytes + 4] = 3;
  SparkCacheChunkDescriptor chunk{};
  std::array<char, 256> error{};
  const auto result = spark_cache_parse_verified_v1_chunk(
      encoded.data(),
      encoded.size(),
      0,
      static_cast<std::uint32_t>(encoded.size()),
      0,
      2,
      0,
      0,
      (1U << SPARK_CACHE_RECORD_TARGET_CKV) |
          (1U << SPARK_CACHE_RECORD_SPARSE_INDEXER),
      &chunk,
      error.data(),
      error.size());
  assert(result == SPARK_CACHE_PLACEMENT_FORMAT_ERROR);
  assert(std::strstr(error.data(), "logical_positions") != nullptr);
}

void test_duplicate_slots_are_rejected() {
  std::array<std::uint8_t, 8> destination_storage{};
  const SparkCacheDestinationDescriptor destination{
      reinterpret_cast<std::uintptr_t>(destination_storage.data()),
      8,
      1,
      1,
      SPARK_CACHE_RECORD_TARGET_CKV,
      0,
  };
  const std::array<std::uint32_t, 2> duplicate{2, 2};
  std::string error;
  assert(!spark_cache::placement::validate_slots(
      duplicate.data(),
      duplicate.size(),
      8,
      &destination,
      1,
      &error));
}

void test_page_abi_and_byte_exact_scatter() {
  SparkCachePagePlacementAbiInfo abi{};
  assert(
      spark_cache_placement_query_page_abi(&abi) ==
      SPARK_CACHE_PLACEMENT_OK);
  assert(abi.abi_version == SPARK_CACHE_PAGE_PLACEMENT_ABI_VERSION);
  assert(abi.sizeof_destination == 32);
  assert(abi.sizeof_group == 16);
  assert(abi.sizeof_copy_span == 40);
  assert(
      (abi.capability_flags & SPARK_CACHE_PAGE_CAP_REFERENCE_SCATTER) != 0);

  std::array<std::uint8_t, 64> arena{};
  arena.fill(0xee);
  const auto fill = [&](std::size_t offset, std::uint8_t first,
                        std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
      arena[offset + index] = static_cast<std::uint8_t>(first + index);
    }
  };
  fill(2, 1, 5);
  fill(20, 6, 3);
  fill(30, 9, 4);
  fill(40, 13, 9);

  std::array<std::uint8_t, 20> destination0{};
  std::array<std::uint8_t, 12> destination1{};
  std::array<std::uint8_t, 22> destination2{};
  destination0.fill(0xcc);
  destination1.fill(0xcc);
  destination2.fill(0xcc);
  const std::array<SparkCachePageDestinationDescriptor, 3> destinations{{
      {
          reinterpret_cast<std::uintptr_t>(destination0.data() + 1),
          3,
          6,
          4,
          0,
          0,
      },
      {
          reinterpret_cast<std::uintptr_t>(destination1.data() + 1),
          3,
          3,
          2,
          0,
          0,
      },
      {
          reinterpret_cast<std::uintptr_t>(destination2.data() + 1),
          4,
          5,
          3,
          1,
          0,
      },
  }};
  const std::array<SparkCachePageGroupDescriptor, 2> groups{{
      {0, 2, 0, 0},
      {2, 3, 0, 0},
  }};
  const std::array<std::uint32_t, 5> slots{2, 0, 1, 3, 0};
  const std::array<SparkCachePageCopySpan, 4> spans{{
      {2, 0, 0, 5, 0, 0},
      {20, 5, 5, 3, 0, 0},
      {30, 8, 0, 4, 1, 0},
      {40, 12, 0, 9, 2, 0},
  }};
  std::array<char, 256> error{};
  assert(
      spark_cache_reference_scatter_pages(
          arena.data(),
          arena.size(),
          21,
          spans.data(),
          spans.size(),
          destinations.data(),
          destinations.size(),
          groups.data(),
          groups.size(),
          slots.data(),
          slots.size(),
          error.data(),
          error.size()) == SPARK_CACHE_PLACEMENT_OK);

  // Group 0 remaps logical pages [0, 1] to physical pages [2, 0].
  assert((std::array<std::uint8_t, 4>{
              destination0[13], destination0[14], destination0[15],
              destination0[16]} ==
          std::array<std::uint8_t, 4>{1, 2, 3, 4}));
  assert((std::array<std::uint8_t, 4>{
              destination0[1], destination0[2], destination0[3],
              destination0[4]} ==
          std::array<std::uint8_t, 4>{5, 6, 7, 8}));
  assert((std::array<std::uint8_t, 2>{
              destination1[7], destination1[8]} ==
          std::array<std::uint8_t, 2>{9, 10}));
  assert((std::array<std::uint8_t, 2>{
              destination1[1], destination1[2]} ==
          std::array<std::uint8_t, 2>{11, 12}));
  // Group 1 independently remaps [0, 1, 2] to [1, 3, 0].
  assert((std::array<std::uint8_t, 3>{
              destination2[6], destination2[7], destination2[8]} ==
          std::array<std::uint8_t, 3>{13, 14, 15}));
  assert((std::array<std::uint8_t, 3>{
              destination2[16], destination2[17], destination2[18]} ==
          std::array<std::uint8_t, 3>{16, 17, 18}));
  assert((std::array<std::uint8_t, 3>{
              destination2[1], destination2[2], destination2[3]} ==
          std::array<std::uint8_t, 3>{19, 20, 21}));

  // Outer canaries and stride padding prove no adjacent bytes were touched.
  assert(destination0.front() == 0xcc && destination0.back() == 0xcc);
  assert(destination1.front() == 0xcc && destination1.back() == 0xcc);
  assert(destination2.front() == 0xcc && destination2.back() == 0xcc);
  assert(destination0[5] == 0xcc && destination0[6] == 0xcc);
  assert(destination1[3] == 0xcc && destination1[6] == 0xcc);
  assert(destination2[4] == 0xcc && destination2[5] == 0xcc);
}

void test_page_validation_is_atomic_before_copy() {
  std::array<std::uint8_t, 8> arena{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<std::uint8_t, 10> destination{};
  destination.fill(0xa5);
  const SparkCachePageDestinationDescriptor descriptor{
      reinterpret_cast<std::uintptr_t>(destination.data() + 1),
      2,
      4,
      2,
      0,
      0,
  };
  const SparkCachePageGroupDescriptor group{0, 2, 0, 0};
  const std::array<std::uint32_t, 2> duplicate_slots{1, 1};
  const SparkCachePageCopySpan span{0, 0, 0, 4, 0, 0};
  std::array<char, 128> error{};
  assert(
      spark_cache_reference_scatter_pages(
          arena.data(),
          arena.size(),
          4,
          &span,
          1,
          &descriptor,
          1,
          &group,
          1,
          duplicate_slots.data(),
          duplicate_slots.size(),
          error.data(),
          error.size()) == SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT);
  assert(std::strstr(error.data(), "unique within each group") != nullptr);
  assert(std::all_of(
      destination.begin(),
      destination.end(),
      [](std::uint8_t value) { return value == 0xa5; }));
}

void test_page_completion_requires_every_snapshot_and_destination_byte() {
  const std::array<SparkCachePageDestinationDescriptor, 2> destinations{{
      {1, 4, 4, 4, 0, 0},
      {1, 4, 2, 2, 0, 0},
  }};
  const SparkCachePageGroupDescriptor group{0, 2, 0, 0};
  std::string error;

  assert(!spark_cache::placement::validate_page_completion(
      12,
      8,
      destinations.data(),
      destinations.size(),
      &group,
      1,
      {8, 4},
      &error));
  assert(error == "page slabs do not cover the complete snapshot");

  assert(!spark_cache::placement::validate_page_completion(
      12,
      12,
      destinations.data(),
      destinations.size(),
      &group,
      1,
      {8, 3},
      &error));
  assert(error == "page slabs do not cover every destination");

  const std::array<SparkCachePageGroupDescriptor, 2> extra_group{{
      {0, 2, 0, 0},
      {2, 1, 0, 0},
  }};
  assert(!spark_cache::placement::validate_page_completion(
      12,
      12,
      destinations.data(),
      destinations.size(),
      extra_group.data(),
      extra_group.size(),
      {8, 4},
      &error));
  assert(error == "every page group must have at least one destination");

  assert(spark_cache::placement::validate_page_completion(
      12,
      12,
      destinations.data(),
      destinations.size(),
      &group,
      1,
      {8, 4},
      &error));
}

}  // namespace

int main() {
  test_inventory_and_slab_math();
  test_config_validation();
  test_parse_and_byte_exact_scatter();
  test_parser_rejects_wrong_positions();
  test_duplicate_slots_are_rejected();
  test_page_abi_and_byte_exact_scatter();
  test_page_validation_is_atomic_before_copy();
  test_page_completion_requires_every_snapshot_and_destination_byte();
  return 0;
}
