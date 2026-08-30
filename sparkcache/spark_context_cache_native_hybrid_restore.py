"""Compatibility aliases for the SparkCache CUDA hybrid-restore interface."""

from sparkcache.spark_context_cache_cuda_hybrid_restore import (
    CudaHybridRestoreError,
    CudaHybridRestoreResult,
    CudaPageObject,
    CudaPageRestoreError,
    CudaPageRestoreResult,
    CudaPageSlab,
    build_page_copy_spans,
    build_page_object_spans,
    execute_cuda_direct_restore,
    execute_cuda_hybrid_placement,
    execute_cuda_hybrid_restore,
    execute_cuda_page_placement,
    plan_page_slabs,
)

NativeHybridRestoreError = CudaHybridRestoreError
NativeHybridRestoreResult = CudaHybridRestoreResult
NativePageObject = CudaPageObject
NativePageSlab = CudaPageSlab
execute_native_hybrid_placement = execute_cuda_hybrid_placement
execute_native_hybrid_restore = execute_cuda_hybrid_restore


__all__ = [
    "CudaHybridRestoreError",
    "CudaHybridRestoreResult",
    "CudaPageObject",
    "CudaPageRestoreError",
    "CudaPageRestoreResult",
    "CudaPageSlab",
    "NativeHybridRestoreError",
    "NativeHybridRestoreResult",
    "NativePageObject",
    "NativePageSlab",
    "build_page_copy_spans",
    "build_page_object_spans",
    "execute_cuda_direct_restore",
    "execute_cuda_hybrid_placement",
    "execute_cuda_hybrid_restore",
    "execute_cuda_page_placement",
    "execute_native_hybrid_placement",
    "execute_native_hybrid_restore",
    "plan_page_slabs",
]
