"""GPU-free compatibility tests for SparkCache CUDA restore terminology."""

from sparkcache import spark_cache_cuda, spark_cache_native
from sparkcache import spark_context_cache_cuda_hybrid_restore as cuda_hybrid
from sparkcache import spark_context_cache_cuda_page_restore as cuda_page
from sparkcache import spark_context_cache_cuda_placement as cuda_placement
from sparkcache import spark_context_cache_cuda_restore as cuda_restore
from sparkcache import spark_context_cache_native_hybrid_restore as legacy_page
from sparkcache import spark_context_cache_native_placement as legacy_placement
from sparkcache import spark_context_cache_native_restore as legacy_restore


def test_canonical_cuda_symbols_retain_compatibility_aliases() -> None:
    assert (
        spark_cache_cuda.CudaPlacementError is spark_cache_native.NativePlacementError
    )
    assert (
        cuda_placement.CudaPlacementAdapter is legacy_placement.NativePlacementAdapter
    )
    assert cuda_restore.CudaRestoreError is legacy_restore.NativeRestoreError
    assert cuda_restore.execute_cuda_restore is legacy_restore.execute_native_restore
    assert cuda_restore.plan_cuda_restore is legacy_restore.plan_native_restore
    assert cuda_hybrid.CudaHybridRestoreError is legacy_page.NativeHybridRestoreError
    assert cuda_hybrid.CudaHybridRestoreResult is legacy_page.NativeHybridRestoreResult
    assert cuda_hybrid.CudaPageObject is legacy_page.NativePageObject
    assert cuda_hybrid.CudaPageSlab is legacy_page.NativePageSlab
    assert (
        cuda_hybrid.execute_cuda_hybrid_placement
        is legacy_page.execute_native_hybrid_placement
    )
    assert (
        cuda_hybrid.execute_cuda_hybrid_restore
        is legacy_page.execute_native_hybrid_restore
    )
    assert cuda_page.CudaPageRestoreError is cuda_hybrid.CudaHybridRestoreError
    assert (
        cuda_page.execute_cuda_page_placement
        is cuda_hybrid.execute_cuda_hybrid_placement
    )
    assert (
        cuda_page.execute_cuda_direct_restore is cuda_hybrid.execute_cuda_hybrid_restore
    )
