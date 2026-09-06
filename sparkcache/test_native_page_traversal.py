"""Execute the CUDA page traversal as serial C++ without a CUDA runtime.

The test compiles the actual kernel body and copy helpers with simulated launch
indices. It checks address arithmetic and byte coverage, not GPU concurrency,
CUDA compilation, or device memory ordering.
"""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


def test_native_page_traversal_bytes_and_grid_stride(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("GPU-free native traversal requires a C++17 compiler")
    native = Path(__file__).parent / "native"
    source = (native / "src/spark_cache_placement.cu").read_text(encoding="utf-8")
    start = source.index("__device__ __forceinline__ std::uint64_t page_min_bytes(")
    end = source.index("\nvoid release_arena(", start)
    constants = "\n".join(
        re.findall(
            r"constexpr std::uint(?:32|64)_t k(?:PageTileBytes|MaximumPageBlocks) = [^;]+;",
            source,
        )
    )
    assert "kPageTileBytes" in constants and "kMaximumPageBlocks" in constants
    harness = r"""
#include "spark_cache_placement.h"
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>
#define __device__
#define __global__
#define __forceinline__ inline
struct alignas(16) uint4 { std::uint32_t a,b,c,d; };
struct Dim { std::uint32_t x; };
Dim threadIdx{}, blockIdx{}, blockDim{256}, gridDim{};
enum { kDeviceChunkBounds=1, kDeviceDestinationBounds=2, kDeviceSlotBounds=3 };
void set_device_error(std::uint32_t* error, std::uint32_t code) { *error=code; }
"""
    cases = r"""
void check(bool ok) { if (!ok) throw std::runtime_error("page traversal mismatch"); }
void run(std::uint32_t page, std::uint32_t pages, std::uint32_t offset,
         std::uint32_t fragment, std::uint32_t blocks) {
  const std::uint64_t bytes=std::uint64_t(page)*pages;
  std::vector<std::uint8_t> arena(bytes+64, 0xAB), output(bytes+64, 0xCD);
  for (std::uint64_t i=0;i<bytes;++i) arena[offset+i]=std::uint8_t(i*17+3);
  std::vector<SparkCachePageCopySpan> spans;
  for (std::uint64_t i=0;i<bytes;i+=fragment)
    spans.push_back({offset+i, 13+i, i, std::min<std::uint64_t>(fragment,bytes-i),0,0});
  std::vector<std::uint32_t> slots(pages);
  for(std::uint32_t i=0;i<pages;++i) slots[i]=pages-1-i;
  SparkCachePageGroupDescriptor group{0,pages,0,0};
  SparkCachePageDestinationDescriptor destination{
    reinterpret_cast<std::uintptr_t>(output.data()+16),pages,page,page,0,0};
  std::uint32_t error=0;
  gridDim.x=blocks;
  for(blockIdx.x=0;blockIdx.x<blocks;++blockIdx.x)
    for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
      scatter_page_kernel(arena.data(),arena.size(),spans.data(),spans.size(),bytes,
                          &destination,1,&group,1,slots.data(),slots.size(),&error);
  check(error==0);
  for (std::uint64_t i=0;i<bytes;++i) {
    const auto physical=std::uint64_t(slots[i/page])*page+i%page;
    check(output[16+physical]==std::uint8_t(i*17+3));
  }
  for(std::uint32_t i=0;i<16;++i) check(output[i]==0xCD);
  for(std::uint64_t i=16+bytes;i<output.size();++i) check(output[i]==0xCD);
  // The same traversal must reject an out-of-range allocator slot.
  slots[0]=pages;
  blockIdx.x=0;threadIdx.x=0;
  scatter_page_kernel(arena.data(),arena.size(),spans.data(),spans.size(),bytes,
                      &destination,1,&group,1,slots.data(),slots.size(),&error);
  check(error==kDeviceDestinationBounds);
}
int main() {
  for(auto offset : {0U,1U,4U,16U}) run(257,1201,offset,997,2);
  run(4096,64,0,65536,2);
  run(4096,64,4,65536,2);
  // More than 4096 tiles exercises the production capped-grid continuation.
  run(65536,4097,0,65536*4097U,kMaximumPageBlocks);
}
"""
    program = tmp_path / "page_traversal.cpp"
    program.write_text(
        harness + constants + "\n" + source[start:end] + cases, encoding="utf-8"
    )
    executable = tmp_path / "page_traversal"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(native / "include"),
            str(program),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=60
    )
