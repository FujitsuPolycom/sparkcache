#include "spark_cache_snapshot.h"
#include "spark_cache_page_capture_layout.hpp"
#include "spark_cache_snapshot_ring.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

using spark_cache::snapshot::RingState;
using spark_cache::snapshot::calculate_payload_layout;
using spark_cache::snapshot::validate_config;
using spark_cache::snapshot::validate_sources;
using spark_cache::page_capture::plan_capture;

constexpr std::uint32_t kThreads = 256;
thread_local std::array<char, 512> g_runtime_error{};

struct NativeSlot {
  void* host = nullptr;
  std::uint8_t* device = nullptr;
  std::uint32_t* host_physical_slots = nullptr;
  std::uint32_t* device_physical_slots = nullptr;
  SparkCachePageCaptureGroup* host_page_groups = nullptr;
  SparkCachePageCaptureGroup* device_page_groups = nullptr;
  SparkCachePageCaptureSpan* host_page_spans = nullptr;
  SparkCachePageCaptureSpan* device_page_spans = nullptr;
  cudaEvent_t producer_ready = nullptr;
  cudaEvent_t complete = nullptr;
  cudaStream_t capture_stream = nullptr;
  cudaStream_t quarantined_stream = nullptr;
  bool requires_stream_drain = false;
  SparkCacheSnapshotReadyView view{};
};

}  // namespace

struct SparkCacheSnapshot {
  explicit SparkCacheSnapshot(std::uint32_t slot_count)
      : ring(slot_count) {}

  SparkCacheSnapshotConfig config{};
  RingState ring;
  std::array<NativeSlot, SPARK_CACHE_SNAPSHOT_MAX_SLOTS> slots{};
  SparkCacheSnapshotSource* device_sources = nullptr;
  std::vector<SparkCacheSnapshotSource> sources;
  SparkCachePageCaptureSource* device_page_sources = nullptr;
  std::vector<SparkCachePageCaptureSource> page_sources;
  std::uint32_t page_group_count = 0;
  SparkCacheSnapshotStats stats{};
  std::array<char, 512> last_error{};
  std::mutex mutex;
  bool shutdown_complete = false;
};

namespace {

void set_error(SparkCacheSnapshot* snapshot, const std::string& message) {
  std::snprintf(
      g_runtime_error.data(), g_runtime_error.size(), "%s", message.c_str());
  if (snapshot != nullptr) {
    std::snprintf(
        snapshot->last_error.data(),
        snapshot->last_error.size(),
        "%s",
        message.c_str());
  }
}

SparkCacheSnapshotStatus cuda_failure(
    SparkCacheSnapshot* snapshot,
    const char* operation,
    cudaError_t result) {
  std::string message(operation);
  message += ": ";
  message += cudaGetErrorString(result);
  set_error(snapshot, message);
  return SPARK_CACHE_SNAPSHOT_CUDA_ERROR;
}

cudaError_t record_snapshot_completion_event(
    cudaEvent_t event,
    cudaStream_t stream) {
#if defined(SPARK_CACHE_SNAPSHOT_TEST_FORCE_EVENT_RECORD_FAILURE)
  (void)event;
  (void)stream;
  return cudaErrorUnknown;
#else
  return cudaEventRecord(event, stream);
#endif
}

cudaError_t synchronize_quarantined_stream(cudaStream_t stream) {
#if defined(SPARK_CACHE_SNAPSHOT_TEST_FORCE_STREAM_DRAIN_FAILURE)
  (void)stream;
  return cudaErrorUnknown;
#else
  return cudaStreamSynchronize(stream);
#endif
}

void release_native(SparkCacheSnapshot* snapshot) {
  if (snapshot == nullptr) {
    return;
  }
  for (std::uint32_t index = 0;
       index < snapshot->config.slot_count;
       ++index) {
    auto& slot = snapshot->slots[index];
    if (slot.complete != nullptr) {
      (void)cudaEventSynchronize(slot.complete);
      (void)cudaEventDestroy(slot.complete);
      slot.complete = nullptr;
    }
    if (slot.producer_ready != nullptr) {
      (void)cudaEventDestroy(slot.producer_ready);
      slot.producer_ready = nullptr;
    }
    if (slot.capture_stream != nullptr) {
      (void)cudaStreamDestroy(slot.capture_stream);
      slot.capture_stream = nullptr;
    }
    if (slot.host_physical_slots != nullptr) {
      (void)cudaFreeHost(slot.host_physical_slots);
      slot.host_physical_slots = nullptr;
      slot.device_physical_slots = nullptr;
    }
    if (slot.host_page_groups != nullptr) {
      (void)cudaFreeHost(slot.host_page_groups);
      slot.host_page_groups = nullptr;
      slot.device_page_groups = nullptr;
    }
    if (slot.host_page_spans != nullptr) {
      (void)cudaFreeHost(slot.host_page_spans);
      slot.host_page_spans = nullptr;
      slot.device_page_spans = nullptr;
    }
    if (slot.host != nullptr) {
      if (snapshot->config.arena_mode ==
          SPARK_CACHE_SNAPSHOT_MANAGED) {
        (void)cudaFree(slot.device);
      } else {
        (void)cudaFreeHost(slot.host);
      }
      slot.host = nullptr;
      slot.device = nullptr;
    }
  }
  if (snapshot->device_sources != nullptr) {
    (void)cudaFree(snapshot->device_sources);
    snapshot->device_sources = nullptr;
  }
  if (snapshot->device_page_sources != nullptr) {
    (void)cudaFree(snapshot->device_page_sources);
    snapshot->device_page_sources = nullptr;
  }
}

SparkCacheSnapshotStatus drain_quarantined_slot_locked(
    SparkCacheSnapshot* snapshot,
    std::uint32_t index) {
  auto& slot = snapshot->slots[index];
  if (!slot.requires_stream_drain) {
    return SPARK_CACHE_SNAPSHOT_OK;
  }
  const auto* state = snapshot->ring.inspect(index);
  if (state == nullptr ||
      state->state != SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING ||
      !state->discard) {
    set_error(snapshot, "quarantined snapshot slot state mismatch");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }

  const cudaError_t synchronized =
      synchronize_quarantined_stream(slot.quarantined_stream);
  if (synchronized != cudaSuccess) {
    return cuda_failure(
        snapshot,
        "cudaStreamSynchronize(quarantined snapshot)",
        synchronized);
  }
  const auto reaped = snapshot->ring.reap_discarded(index);
  if (reaped != SPARK_CACHE_SNAPSHOT_OK) {
    set_error(snapshot, "quarantined snapshot reaper state mismatch");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  slot.requires_stream_drain = false;
  slot.quarantined_stream = nullptr;
  return SPARK_CACHE_SNAPSHOT_OK;
}

SparkCacheSnapshotStatus quarantine_post_launch_failure_locked(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket& ticket,
    std::uint64_t context_sequence,
    cudaStream_t stream,
    const char* operation,
    cudaError_t failure) {
  auto& slot = snapshot->slots[ticket.slot_index];
  slot.requires_stream_drain = true;
  slot.quarantined_stream = stream;
  snapshot->stats.abandoned +=
      snapshot->ring.abandon(context_sequence);

  std::string failure_message(operation);
  failure_message += ": ";
  failure_message += cudaGetErrorString(failure);
  const auto drained =
      drain_quarantined_slot_locked(snapshot, ticket.slot_index);
  if (drained != SPARK_CACHE_SNAPSHOT_OK) {
    std::string combined = failure_message;
    combined += "; quarantine retained: ";
    combined += snapshot->last_error.data();
    set_error(snapshot, combined);
    return drained;
  }
  set_error(snapshot, failure_message);
  return SPARK_CACHE_SNAPSHOT_CUDA_ERROR;
}

bool ticket_matches(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket& ticket) {
  const auto* state = snapshot->ring.inspect(ticket.slot_index);
  return state != nullptr &&
         state->state != SPARK_CACHE_SNAPSHOT_SLOT_FREE &&
         state->generation == ticket.generation;
}

SparkCacheSnapshotStatus reap_abandoned_locked(
    SparkCacheSnapshot* snapshot,
    std::uint32_t skip_index) {
  for (std::uint32_t index = 0;
       index < snapshot->config.slot_count;
       ++index) {
    if (index == skip_index) {
      continue;
    }
    const auto* state = snapshot->ring.inspect(index);
    if (state->state != SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING ||
        !state->discard) {
      continue;
    }
    if (snapshot->slots[index].requires_stream_drain) {
      const auto drained =
          drain_quarantined_slot_locked(snapshot, index);
      if (drained != SPARK_CACHE_SNAPSHOT_OK) {
        return drained;
      }
      continue;
    }
    const cudaError_t result =
        cudaEventQuery(snapshot->slots[index].complete);
    if (result == cudaErrorNotReady) {
      continue;
    }
    if (result != cudaSuccess) {
      return cuda_failure(
          snapshot, "cudaEventQuery(abandoned snapshot)", result);
    }
    const auto reaped = snapshot->ring.reap_discarded(index);
    if (reaped != SPARK_CACHE_SNAPSHOT_OK) {
      set_error(snapshot, "abandoned snapshot reaper state mismatch");
      return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
    }
  }
  return SPARK_CACHE_SNAPSHOT_OK;
}

SparkCacheSnapshotStatus shutdown_locked(
    SparkCacheSnapshot* snapshot) {
  if (snapshot->shutdown_complete) {
    return SPARK_CACHE_SNAPSHOT_OK;
  }
  if (snapshot->ring.has_writing()) {
    set_error(
        snapshot,
        "snapshot shutdown requires every WRITING view to be released");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  (void)snapshot->ring.abandon_all();
  for (std::uint32_t index = 0;
       index < snapshot->config.slot_count;
       ++index) {
    const auto* state = snapshot->ring.inspect(index);
    if (state->state != SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING) {
      continue;
    }
    if (snapshot->slots[index].requires_stream_drain) {
      const auto drained =
          drain_quarantined_slot_locked(snapshot, index);
      if (drained != SPARK_CACHE_SNAPSHOT_OK) {
        return drained;
      }
      continue;
    }
    const cudaError_t result =
        cudaEventSynchronize(snapshot->slots[index].complete);
    if (result != cudaSuccess) {
      return cuda_failure(
          snapshot, "cudaEventSynchronize(snapshot shutdown)", result);
    }
    const auto reaped = snapshot->ring.reap_discarded(index);
    if (reaped != SPARK_CACHE_SNAPSHOT_OK) {
      set_error(snapshot, "snapshot shutdown reaper state mismatch");
      return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
    }
  }
  snapshot->shutdown_complete = true;
  set_error(snapshot, "");
  return SPARK_CACHE_SNAPSHOT_OK;
}

void fill_view_state(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket& ticket,
    SparkCacheSnapshotReadyView* output) {
  *output = snapshot->slots[ticket.slot_index].view;
  const auto* state = snapshot->ring.inspect(ticket.slot_index);
  output->state =
      state == nullptr ? SPARK_CACHE_SNAPSHOT_SLOT_FREE : state->state;
}

SparkCacheSnapshotStatus poll_locked(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket& ticket,
    SparkCacheSnapshotReadyView* output) {
  const auto reaped = reap_abandoned_locked(
      snapshot, ticket.slot_index);
  if (reaped != SPARK_CACHE_SNAPSHOT_OK) {
    return reaped;
  }
  if (ticket.slot_index < snapshot->config.slot_count &&
      snapshot->slots[ticket.slot_index].requires_stream_drain) {
    const auto drained = drain_quarantined_slot_locked(
        snapshot, ticket.slot_index);
    if (drained != SPARK_CACHE_SNAPSHOT_OK) {
      return drained;
    }
  }
  if (!ticket_matches(snapshot, ticket)) {
    snapshot->stats.stale_tickets += 1;
    return SPARK_CACHE_SNAPSHOT_DROPPED;
  }
  const auto* state = snapshot->ring.inspect(ticket.slot_index);
  if (state->state == SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING) {
    const cudaError_t result =
        cudaEventQuery(snapshot->slots[ticket.slot_index].complete);
    if (result == cudaErrorNotReady) {
      return SPARK_CACHE_SNAPSHOT_NOT_READY;
    }
    if (result != cudaSuccess) {
      return cuda_failure(snapshot, "cudaEventQuery(snapshot)", result);
    }
    const auto completed = snapshot->ring.complete(ticket);
    if (completed == SPARK_CACHE_SNAPSHOT_DROPPED) {
      return completed;
    }
    if (completed != SPARK_CACHE_SNAPSHOT_OK) {
      return completed;
    }
    snapshot->stats.completed_bytes +=
        snapshot->slots[ticket.slot_index].view.used_bytes;
  }
  if (output != nullptr) {
    fill_view_state(snapshot, ticket, output);
  }
  return SPARK_CACHE_SNAPSHOT_OK;
}

__global__ void gather_snapshot_kernel(
    const SparkCacheSnapshotSource* sources,
    std::uint32_t source_count,
    const std::uint32_t* physical_slots,
    std::uint32_t row_count,
    SparkCacheSnapshotReadyView layout,
    std::uint8_t* output) {
  const std::uint32_t source_index = blockIdx.x;
  if (source_index >= source_count) {
    return;
  }
  const auto source = sources[source_index];
  const std::uint64_t source_bytes =
      static_cast<std::uint64_t>(row_count) * source.bytes_per_token;
  const std::uint64_t output_base =
      layout.record_offset_bytes[source.record_kind] +
      static_cast<std::uint64_t>(source.source_layer_ordinal) * source_bytes;

  for (std::uint64_t byte_index =
           static_cast<std::uint64_t>(threadIdx.x);
       byte_index < source_bytes;
       byte_index += blockDim.x) {
    const auto local_row =
        static_cast<std::uint32_t>(byte_index / source.bytes_per_token);
    const auto row_byte =
        static_cast<std::uint32_t>(byte_index % source.bytes_per_token);
    const auto physical_row = physical_slots[local_row];
    const auto* source_base = reinterpret_cast<const std::uint8_t*>(
        static_cast<std::uintptr_t>(source.source_base));
    output[output_base + byte_index] =
        source_base[
            static_cast<std::uint64_t>(physical_row) *
                source.source_row_stride_bytes +
            row_byte];
  }
}

__global__ void gather_manager_pages_kernel(
    const SparkCachePageCaptureSource* sources,
    const SparkCachePageCaptureSpan* spans,
    std::uint32_t span_count,
    const std::uint32_t* physical_pages,
    std::uint8_t* output) {
  const std::uint32_t span_index = blockIdx.x;
  if (span_index >= span_count) {
    return;
  }
  const auto span = spans[span_index];
  const auto source = sources[span.source_index];
  const auto request_page = static_cast<std::uint32_t>(blockIdx.y);
  if (request_page >= span.page_count) {
    return;
  }
  const auto* source_base = reinterpret_cast<const std::uint8_t*>(
      static_cast<std::uintptr_t>(source.source_base));
  const auto output_base =
      span.destination_offset_bytes +
      static_cast<std::uint64_t>(request_page) * source.bytes_per_page;
  for (std::uint32_t page_byte = threadIdx.x;
       page_byte < source.bytes_per_page;
       page_byte += blockDim.x) {
    const auto physical_page =
        physical_pages[span.physical_page_offset + request_page];
    output[output_base + page_byte] =
        source_base[
            static_cast<std::uint64_t>(physical_page) *
                source.source_page_stride_bytes +
            page_byte];
  }
}

}  // namespace

extern "C" SparkCacheSnapshotStatus spark_cache_snapshot_query_abi(
    SparkCacheSnapshotAbiInfo* output) {
  if (output == nullptr) {
    set_error(nullptr, "snapshot ABI output pointer is null");
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  SparkCacheSnapshotAbiInfo info{};
  info.abi_version = SPARK_CACHE_SNAPSHOT_ABI_VERSION;
  info.cudart_version = CUDART_VERSION;
  info.min_slots = SPARK_CACHE_SNAPSHOT_MIN_SLOTS;
  info.max_slots = SPARK_CACHE_SNAPSHOT_MAX_SLOTS;
  info.max_record_kinds = SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS;
  info.sizeof_config = sizeof(SparkCacheSnapshotConfig);
  info.sizeof_source = sizeof(SparkCacheSnapshotSource);
  info.sizeof_submission = sizeof(SparkCacheSnapshotSubmission);
  info.sizeof_ticket = sizeof(SparkCacheSnapshotTicket);
  info.sizeof_ready_view = sizeof(SparkCacheSnapshotReadyView);
  info.sizeof_stats = sizeof(SparkCacheSnapshotStats);
  info.capability_flags =
      SPARK_CACHE_SNAPSHOT_CAP_MAPPED_HOST |
      SPARK_CACHE_SNAPSHOT_CAP_MANAGED |
      SPARK_CACHE_SNAPSHOT_CAP_EXTERNAL_STREAM |
      SPARK_CACHE_SNAPSHOT_CAP_NONBLOCKING_ACQUIRE |
      SPARK_CACHE_SNAPSHOT_CAP_CONTEXT_ABANDON |
      SPARK_CACHE_SNAPSHOT_CAP_ORDERLY_SHUTDOWN |
      SPARK_CACHE_SNAPSHOT_CAP_MANAGER_PAGE_CAPTURE |
      SPARK_CACHE_SNAPSHOT_CAP_LOW_PRIORITY_CAPTURE_STREAM;
  *output = info;
  set_error(nullptr, "");
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" SparkCacheSnapshotStatus
spark_cache_snapshot_query_page_capture_abi(
    SparkCachePageCaptureAbiInfo* output) {
  if (output == nullptr) {
    set_error(nullptr, "manager-page ABI output pointer is null");
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  SparkCachePageCaptureAbiInfo info{};
  info.contract_version = SPARK_CACHE_PAGE_CAPTURE_CONTRACT_VERSION;
  info.max_groups = SPARK_CACHE_PAGE_CAPTURE_MAX_GROUPS;
  info.max_sources = SPARK_CACHE_PAGE_CAPTURE_MAX_SOURCES;
  info.sizeof_source = sizeof(SparkCachePageCaptureSource);
  info.sizeof_group = sizeof(SparkCachePageCaptureGroup);
  info.sizeof_span = sizeof(SparkCachePageCaptureSpan);
  info.sizeof_plan = sizeof(SparkCachePageCapturePlan);
  info.sizeof_submission = sizeof(SparkCachePageCaptureSubmission);
  info.capability_flags =
      SPARK_CACHE_SNAPSHOT_CAP_MANAGER_PAGE_CAPTURE |
      SPARK_CACHE_SNAPSHOT_CAP_LOW_PRIORITY_CAPTURE_STREAM;
  *output = info;
  set_error(nullptr, "");
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" SparkCacheSnapshotStatus spark_cache_snapshot_create(
    const SparkCacheSnapshotConfig* config,
    SparkCacheSnapshot** output) {
  if (config == nullptr || output == nullptr) {
    set_error(nullptr, "snapshot config or output pointer is null");
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  *output = nullptr;
  std::string detail;
  if (!validate_config(*config, &detail)) {
    set_error(nullptr, detail);
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  auto* snapshot =
      new (std::nothrow) SparkCacheSnapshot(config->slot_count);
  if (snapshot == nullptr) {
    set_error(nullptr, "cannot allocate snapshot handle");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  snapshot->config = *config;
  set_error(snapshot, "");

  cudaError_t result = cudaSetDevice(config->device_ordinal);
  if (result != cudaSuccess) {
    const auto status = cuda_failure(snapshot, "cudaSetDevice", result);
    delete snapshot;
    return status;
  }
  result = cudaMalloc(
      reinterpret_cast<void**>(&snapshot->device_sources),
      static_cast<std::size_t>(config->max_sources) *
          sizeof(SparkCacheSnapshotSource));
  if (result != cudaSuccess) {
    const auto status =
        cuda_failure(snapshot, "cudaMalloc(snapshot sources)", result);
    delete snapshot;
    return status;
  }
  result = cudaMalloc(
      reinterpret_cast<void**>(&snapshot->device_page_sources),
      static_cast<std::size_t>(config->max_sources) *
          sizeof(SparkCachePageCaptureSource));
  if (result != cudaSuccess) {
    const auto status =
        cuda_failure(snapshot, "cudaMalloc(manager-page sources)", result);
    release_native(snapshot);
    delete snapshot;
    return status;
  }

  int least_priority = 0;
  int greatest_priority = 0;
  result = cudaDeviceGetStreamPriorityRange(
      &least_priority, &greatest_priority);
  if (result != cudaSuccess) {
    const auto status = cuda_failure(
        snapshot, "cudaDeviceGetStreamPriorityRange", result);
    release_native(snapshot);
    delete snapshot;
    return status;
  }
  (void)greatest_priority;

  for (std::uint32_t index = 0; index < config->slot_count; ++index) {
    auto& slot = snapshot->slots[index];
    if (config->arena_mode == SPARK_CACHE_SNAPSHOT_MANAGED) {
      result = cudaMallocManaged(
          reinterpret_cast<void**>(&slot.device),
          static_cast<std::size_t>(config->slot_bytes),
          cudaMemAttachGlobal);
      slot.host = slot.device;
    } else {
      result = cudaHostAlloc(
          &slot.host,
          static_cast<std::size_t>(config->slot_bytes),
          cudaHostAllocMapped | cudaHostAllocPortable);
      if (result == cudaSuccess) {
        result = cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&slot.device), slot.host, 0);
      }
    }
    if (result == cudaSuccess) {
      result = cudaHostAlloc(
          reinterpret_cast<void**>(&slot.host_physical_slots),
          static_cast<std::size_t>(config->max_rows) *
              sizeof(std::uint32_t),
          cudaHostAllocMapped | cudaHostAllocPortable);
    }
    if (result == cudaSuccess) {
      result = cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&slot.device_physical_slots),
          slot.host_physical_slots,
          0);
    }
    if (result == cudaSuccess) {
      result = cudaHostAlloc(
          reinterpret_cast<void**>(&slot.host_page_groups),
          SPARK_CACHE_PAGE_CAPTURE_MAX_GROUPS *
              sizeof(SparkCachePageCaptureGroup),
          cudaHostAllocMapped | cudaHostAllocPortable);
    }
    if (result == cudaSuccess) {
      result = cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&slot.device_page_groups),
          slot.host_page_groups,
          0);
    }
    if (result == cudaSuccess) {
      result = cudaHostAlloc(
          reinterpret_cast<void**>(&slot.host_page_spans),
          static_cast<std::size_t>(config->max_sources) *
              sizeof(SparkCachePageCaptureSpan),
          cudaHostAllocMapped | cudaHostAllocPortable);
    }
    if (result == cudaSuccess) {
      result = cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&slot.device_page_spans),
          slot.host_page_spans,
          0);
    }
    if (result == cudaSuccess) {
      result = cudaEventCreateWithFlags(
          &slot.producer_ready, cudaEventDisableTiming);
    }
    if (result == cudaSuccess) {
      result = cudaEventCreateWithFlags(
          &slot.complete, cudaEventDisableTiming);
    }
    if (result == cudaSuccess) {
      result = cudaStreamCreateWithPriority(
          &slot.capture_stream,
          cudaStreamNonBlocking,
          least_priority);
    }
    if (result != cudaSuccess) {
      const auto status =
          cuda_failure(snapshot, "allocate snapshot slot", result);
      release_native(snapshot);
      delete snapshot;
      return status;
    }
  }
  *output = snapshot;
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" void spark_cache_snapshot_destroy(
    SparkCacheSnapshot* snapshot) {
  if (snapshot == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(snapshot->mutex);
    if (shutdown_locked(snapshot) != SPARK_CACHE_SNAPSHOT_OK) {
      return;
    }
    release_native(snapshot);
  }
  delete snapshot;
}

extern "C" SparkCacheSnapshotStatus spark_cache_snapshot_shutdown(
    SparkCacheSnapshot* snapshot) {
  if (snapshot == nullptr) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  return shutdown_locked(snapshot);
}

extern "C" SparkCacheSnapshotStatus
spark_cache_snapshot_configure_sources(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotSource* sources,
    std::uint32_t source_count) {
  if (snapshot == nullptr) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  if (snapshot->shutdown_complete) {
    set_error(snapshot, "snapshot handle is shut down");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  if (!snapshot->page_sources.empty()) {
    set_error(snapshot, "manager-page sources are already configured");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  std::string detail;
  if (!validate_sources(
          sources,
          source_count,
          snapshot->config.max_sources,
          &detail)) {
    set_error(snapshot, detail);
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  for (std::uint32_t index = 0;
       index < snapshot->config.slot_count;
       ++index) {
    const auto* state = snapshot->ring.inspect(index);
    if (state->state != SPARK_CACHE_SNAPSHOT_SLOT_FREE) {
      set_error(snapshot, "cannot reconfigure active snapshot sources");
      return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
    }
  }
  const cudaError_t result = cudaMemcpy(
      snapshot->device_sources,
      sources,
      static_cast<std::size_t>(source_count) *
          sizeof(SparkCacheSnapshotSource),
      cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    return cuda_failure(snapshot, "cudaMemcpy(snapshot sources)", result);
  }
  snapshot->sources.assign(sources, sources + source_count);
  set_error(snapshot, "");
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" SparkCacheSnapshotStatus
spark_cache_snapshot_configure_page_sources(
    SparkCacheSnapshot* snapshot,
    const SparkCachePageCaptureSource* sources,
    std::uint32_t source_count,
    std::uint32_t group_count) {
  if (snapshot == nullptr) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  if (snapshot->shutdown_complete) {
    set_error(snapshot, "snapshot handle is shut down");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  if (!snapshot->sources.empty() || source_count > snapshot->config.max_sources) {
    set_error(
        snapshot,
        "row sources are configured or page source capacity differs");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  std::string detail;
  if (!spark_cache::page_capture::validate_sources(
          sources, source_count, group_count, &detail)) {
    set_error(snapshot, detail);
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  for (std::uint32_t index = 0;
       index < snapshot->config.slot_count;
       ++index) {
    const auto* state = snapshot->ring.inspect(index);
    if (state->state != SPARK_CACHE_SNAPSHOT_SLOT_FREE) {
      set_error(snapshot, "cannot reconfigure active manager-page sources");
      return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
    }
  }
  const cudaError_t result = cudaMemcpy(
      snapshot->device_page_sources,
      sources,
      static_cast<std::size_t>(source_count) *
          sizeof(SparkCachePageCaptureSource),
      cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    return cuda_failure(snapshot, "cudaMemcpy(manager-page sources)", result);
  }
  snapshot->page_sources.assign(sources, sources + source_count);
  snapshot->page_group_count = group_count;
  set_error(snapshot, "");
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" SparkCacheSnapshotStatus spark_cache_snapshot_try_submit(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotSubmission* submission,
    const std::uint32_t* physical_slots,
    std::uint64_t producer_stream,
    SparkCacheSnapshotTicket* output) {
  if (snapshot == nullptr || submission == nullptr ||
      physical_slots == nullptr || output == nullptr) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  if (snapshot->shutdown_complete) {
    set_error(snapshot, "snapshot handle is shut down");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  const auto reaped = reap_abandoned_locked(
      snapshot, SPARK_CACHE_SNAPSHOT_MAX_SLOTS);
  if (reaped != SPARK_CACHE_SNAPSHOT_OK) {
    return reaped;
  }
  if (snapshot->sources.empty() || submission->context_sequence == 0 ||
      submission->row_count == 0 ||
      submission->row_count > snapshot->config.max_rows ||
      submission->flags != 0 || submission->reserved[0] != 0 ||
      submission->reserved[1] != 0) {
    set_error(snapshot, "invalid snapshot submission");
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  SparkCacheSnapshotReadyView layout{};
  std::string detail;
  if (!calculate_payload_layout(
          snapshot->sources.data(),
          static_cast<std::uint32_t>(snapshot->sources.size()),
          submission->row_count,
          snapshot->config.slot_bytes,
          &layout,
          &detail)) {
    set_error(snapshot, detail);
    snapshot->stats.abandoned += 1;
    return SPARK_CACHE_SNAPSHOT_DROPPED;
  }
  for (std::uint32_t row = 0; row < submission->row_count; ++row) {
    for (const auto& source : snapshot->sources) {
      if (physical_slots[row] >= source.source_rows) {
        set_error(snapshot, "snapshot physical slot is out of bounds");
        return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
      }
    }
  }

  SparkCacheSnapshotTicket ticket{};
  const auto reserved = snapshot->ring.reserve(
      submission->context_sequence, layout.used_bytes, &ticket);
  if (reserved == SPARK_CACHE_SNAPSHOT_WOULD_BLOCK) {
    snapshot->stats.would_block += 1;
    return reserved;
  }
  if (reserved != SPARK_CACHE_SNAPSHOT_OK) {
    return reserved;
  }
  auto& slot = snapshot->slots[ticket.slot_index];
  std::memcpy(
      slot.host_physical_slots,
      physical_slots,
      static_cast<std::size_t>(submission->row_count) *
          sizeof(std::uint32_t));
  std::atomic_thread_fence(std::memory_order_release);
  layout.host_address = reinterpret_cast<std::uintptr_t>(slot.host);
  layout.device_address = reinterpret_cast<std::uintptr_t>(slot.device);
  layout.context_sequence = submission->context_sequence;
  layout.logical_start = submission->logical_start;
  layout.generation = ticket.generation;
  layout.slot_index = ticket.slot_index;
  layout.state = SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING;
  slot.view = layout;

  auto stream = reinterpret_cast<cudaStream_t>(
      static_cast<std::uintptr_t>(producer_stream));
  gather_snapshot_kernel<<<
      static_cast<unsigned int>(snapshot->sources.size()),
      kThreads,
      0,
      stream>>>(
      snapshot->device_sources,
      static_cast<std::uint32_t>(snapshot->sources.size()),
      slot.device_physical_slots,
      submission->row_count,
      layout,
      slot.device);
  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) {
    return quarantine_post_launch_failure_locked(
        snapshot,
        ticket,
        submission->context_sequence,
        stream,
        "launch snapshot gather",
        result);
  }
  result = record_snapshot_completion_event(slot.complete, stream);
  if (result != cudaSuccess) {
    return quarantine_post_launch_failure_locked(
        snapshot,
        ticket,
        submission->context_sequence,
        stream,
        "cudaEventRecord(snapshot gather)",
        result);
  }
  snapshot->stats.submissions += 1;
  snapshot->stats.submitted_bytes += layout.used_bytes;
  *output = ticket;
  set_error(snapshot, "");
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" SparkCacheSnapshotStatus
spark_cache_snapshot_try_submit_pages(
    SparkCacheSnapshot* snapshot,
    const SparkCachePageCaptureSubmission* submission,
    const SparkCachePageCaptureGroup* groups,
    const std::uint32_t* physical_pages,
    std::uint64_t producer_stream,
    SparkCacheSnapshotTicket* output) {
  if (snapshot == nullptr || submission == nullptr || groups == nullptr ||
      physical_pages == nullptr || output == nullptr) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  if (snapshot->shutdown_complete) {
    set_error(snapshot, "snapshot handle is shut down");
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  const auto reaped = reap_abandoned_locked(
      snapshot, SPARK_CACHE_SNAPSHOT_MAX_SLOTS);
  if (reaped != SPARK_CACHE_SNAPSHOT_OK) {
    return reaped;
  }
  if (snapshot->page_sources.empty() || submission->context_sequence == 0 ||
      submission->physical_page_count == 0 ||
      submission->physical_page_count > snapshot->config.max_rows ||
      submission->group_count != snapshot->page_group_count ||
      submission->flags != 0 || submission->reserved != 0) {
    set_error(snapshot, "invalid manager-page submission");
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }

  std::array<SparkCachePageCaptureSpan,
             SPARK_CACHE_PAGE_CAPTURE_MAX_SOURCES>
      planned_spans{};
  SparkCachePageCapturePlan page_plan{};
  std::string detail;
  if (!plan_capture(
          snapshot->page_sources.data(),
          static_cast<std::uint32_t>(snapshot->page_sources.size()),
          groups,
          submission->group_count,
          physical_pages,
          submission->physical_page_count,
          snapshot->config.slot_bytes,
          planned_spans.data(),
          static_cast<std::uint32_t>(planned_spans.size()),
          &page_plan,
          &detail)) {
    set_error(snapshot, detail);
    snapshot->stats.abandoned += 1;
    return SPARK_CACHE_SNAPSHOT_DROPPED;
  }
  std::uint32_t max_group_pages = 0;
  for (std::uint32_t index = 0; index < submission->group_count; ++index) {
    max_group_pages =
        std::max(max_group_pages, groups[index].page_count);
  }
  if (max_group_pages > 65535) {
    set_error(snapshot, "manager-page group exceeds CUDA grid-y capacity");
    snapshot->stats.abandoned += 1;
    return SPARK_CACHE_SNAPSHOT_DROPPED;
  }

  SparkCacheSnapshotTicket ticket{};
  const auto reserved = snapshot->ring.reserve(
      submission->context_sequence, page_plan.used_bytes, &ticket);
  if (reserved == SPARK_CACHE_SNAPSHOT_WOULD_BLOCK) {
    snapshot->stats.would_block += 1;
    return reserved;
  }
  if (reserved != SPARK_CACHE_SNAPSHOT_OK) {
    return reserved;
  }

  auto& slot = snapshot->slots[ticket.slot_index];
  std::memcpy(
      slot.host_physical_slots,
      physical_pages,
      static_cast<std::size_t>(submission->physical_page_count) *
          sizeof(std::uint32_t));
  std::memcpy(
      slot.host_page_groups,
      groups,
      static_cast<std::size_t>(submission->group_count) *
          sizeof(SparkCachePageCaptureGroup));
  std::memcpy(
      slot.host_page_spans,
      planned_spans.data(),
      static_cast<std::size_t>(page_plan.span_count) *
          sizeof(SparkCachePageCaptureSpan));
  std::atomic_thread_fence(std::memory_order_release);

  SparkCacheSnapshotReadyView layout{};
  layout.host_address = reinterpret_cast<std::uintptr_t>(slot.host);
  layout.device_address = reinterpret_cast<std::uintptr_t>(slot.device);
  layout.capacity_bytes = snapshot->config.slot_bytes;
  layout.used_bytes = page_plan.used_bytes;
  layout.context_sequence = submission->context_sequence;
  layout.logical_start = submission->logical_start;
  layout.generation = ticket.generation;
  layout.row_count = submission->physical_page_count;
  layout.slot_index = ticket.slot_index;
  layout.state = SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING;
  slot.view = layout;

  auto producer = reinterpret_cast<cudaStream_t>(
      static_cast<std::uintptr_t>(producer_stream));
  cudaError_t result = cudaEventRecord(slot.producer_ready, producer);
  if (result != cudaSuccess) {
    (void)snapshot->ring.abandon(submission->context_sequence);
    (void)snapshot->ring.reap_discarded(ticket.slot_index);
    return cuda_failure(snapshot, "cudaEventRecord(producer readiness)", result);
  }
  result = cudaStreamWaitEvent(slot.capture_stream, slot.producer_ready, 0);
  if (result != cudaSuccess) {
    (void)snapshot->ring.abandon(submission->context_sequence);
    (void)snapshot->ring.reap_discarded(ticket.slot_index);
    return cuda_failure(snapshot, "cudaStreamWaitEvent(page capture)", result);
  }

  gather_manager_pages_kernel<<<
      dim3(page_plan.span_count, max_group_pages),
      kThreads,
      0,
      slot.capture_stream>>>(
      snapshot->device_page_sources,
      slot.device_page_spans,
      page_plan.span_count,
      slot.device_physical_slots,
      slot.device);
  result = cudaGetLastError();
  if (result != cudaSuccess) {
    return quarantine_post_launch_failure_locked(
        snapshot,
        ticket,
        submission->context_sequence,
        slot.capture_stream,
        "launch manager-page gather",
        result);
  }
  result = record_snapshot_completion_event(slot.complete, slot.capture_stream);
  if (result != cudaSuccess) {
    return quarantine_post_launch_failure_locked(
        snapshot,
        ticket,
        submission->context_sequence,
        slot.capture_stream,
        "cudaEventRecord(manager-page gather)",
        result);
  }
  snapshot->stats.submissions += 1;
  snapshot->stats.submitted_bytes += page_plan.used_bytes;
  *output = ticket;
  set_error(snapshot, "");
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" SparkCacheSnapshotStatus spark_cache_snapshot_poll(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket* ticket,
    SparkCacheSnapshotReadyView* output) {
  if (snapshot == nullptr || ticket == nullptr || output == nullptr) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  return poll_locked(snapshot, *ticket, output);
}

extern "C" SparkCacheSnapshotStatus spark_cache_snapshot_claim(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket* ticket,
    SparkCacheSnapshotReadyView* output) {
  if (snapshot == nullptr || ticket == nullptr || output == nullptr) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  const auto polled = poll_locked(snapshot, *ticket, nullptr);
  if (polled != SPARK_CACHE_SNAPSHOT_OK) {
    return polled;
  }
  const auto claimed = snapshot->ring.claim(*ticket);
  if (claimed != SPARK_CACHE_SNAPSHOT_OK) {
    return claimed;
  }
  snapshot->stats.claims += 1;
  fill_view_state(snapshot, *ticket, output);
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" SparkCacheSnapshotStatus spark_cache_snapshot_release(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket* ticket) {
  if (snapshot == nullptr || ticket == nullptr) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  if (!ticket_matches(snapshot, *ticket)) {
    snapshot->stats.stale_tickets += 1;
    return SPARK_CACHE_SNAPSHOT_DROPPED;
  }
  const auto bytes = snapshot->slots[ticket->slot_index].view.used_bytes;
  const auto released = snapshot->ring.release(*ticket);
  if (released == SPARK_CACHE_SNAPSHOT_OK) {
    snapshot->stats.releases += 1;
    snapshot->stats.released_bytes += bytes;
  }
  return released;
}

extern "C" SparkCacheSnapshotStatus
spark_cache_snapshot_abandon_context(
    SparkCacheSnapshot* snapshot,
    std::uint64_t context_sequence) {
  if (snapshot == nullptr || context_sequence == 0) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  if (snapshot->shutdown_complete) {
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  snapshot->stats.abandoned += snapshot->ring.abandon(context_sequence);
  return reap_abandoned_locked(
      snapshot, SPARK_CACHE_SNAPSHOT_MAX_SLOTS);
}

extern "C" SparkCacheSnapshotStatus spark_cache_snapshot_drain_context(
    SparkCacheSnapshot* snapshot,
    std::uint64_t context_sequence) {
  if (snapshot == nullptr || context_sequence == 0) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  if (snapshot->shutdown_complete) {
    return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  }
  snapshot->stats.abandoned += snapshot->ring.abandon(context_sequence);
  for (std::uint32_t index = 0;
       index < snapshot->config.slot_count;
       ++index) {
    const auto* state = snapshot->ring.inspect(index);
    if (state == nullptr ||
        state->state != SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING ||
        state->context_sequence != context_sequence) {
      continue;
    }
    if (snapshot->slots[index].requires_stream_drain) {
      const auto drained = drain_quarantined_slot_locked(snapshot, index);
      if (drained != SPARK_CACHE_SNAPSHOT_OK) {
        return drained;
      }
      continue;
    }
    const cudaError_t result =
        cudaEventSynchronize(snapshot->slots[index].complete);
    if (result != cudaSuccess) {
      return cuda_failure(
          snapshot, "cudaEventSynchronize(manager-page preemption)", result);
    }
    const auto reaped_slot = snapshot->ring.reap_discarded(index);
    if (reaped_slot != SPARK_CACHE_SNAPSHOT_OK) {
      set_error(snapshot, "manager-page preemption reaper state mismatch");
      return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
    }
  }
  set_error(snapshot, "");
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" SparkCacheSnapshotStatus spark_cache_snapshot_get_stats(
    SparkCacheSnapshot* snapshot,
    SparkCacheSnapshotStats* output) {
  if (snapshot == nullptr || output == nullptr) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(snapshot->mutex);
  *output = snapshot->stats;
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" SparkCacheSnapshotStatus
spark_cache_snapshot_copy_last_error(
    SparkCacheSnapshot* snapshot,
    char* output,
    std::size_t output_capacity) {
  if (output == nullptr || output_capacity == 0) {
    return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
  }
  const char* source =
      snapshot == nullptr ? g_runtime_error.data()
                          : snapshot->last_error.data();
  std::snprintf(output, output_capacity, "%s", source);
  return SPARK_CACHE_SNAPSHOT_OK;
}

extern "C" const char* spark_cache_snapshot_runtime_last_error(void) {
  return g_runtime_error.data();
}

extern "C" const char* spark_cache_snapshot_status_string(
    SparkCacheSnapshotStatus status) {
  switch (status) {
    case SPARK_CACHE_SNAPSHOT_OK:
      return "ok";
    case SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT:
      return "invalid_argument";
    case SPARK_CACHE_SNAPSHOT_INVALID_STATE:
      return "invalid_state";
    case SPARK_CACHE_SNAPSHOT_CUDA_ERROR:
      return "cuda_error";
    case SPARK_CACHE_SNAPSHOT_WOULD_BLOCK:
      return "would_block";
    case SPARK_CACHE_SNAPSHOT_NOT_READY:
      return "not_ready";
    case SPARK_CACHE_SNAPSHOT_DROPPED:
      return "dropped";
  }
  return "unknown";
}
