from pathlib import Path
from types import SimpleNamespace

import pytest

from sparkcache.spark_context_cache_hybrid import PageGroup, PageLayer, PageLayout
from sparkcache.streaming import manager_page_factory as factory


@pytest.mark.parametrize("mode,expected", [("mapped", 1), ("managed", 2)])
def test_capture_arena_choice_reaches_native_allocation(monkeypatch, mode, expected):
    settings = factory.ManagerPageCaptureSettings(
        Path("/attested.so"), "a" * 64, 3758096384, slot_count=1, arena_mode=mode
    )
    tensor = SimpleNamespace(
        device=SimpleNamespace(type="cuda", index=0),
        shape=(8, 2),
        data_ptr=lambda: 4096,
        stride=lambda dimension: 2,
        element_size=lambda: 1,
    )
    connector = SimpleNamespace(
        _page_layout=PageLayout(
            (PageGroup(256, (PageLayer("layer", "u8", (2,), 2),)),)
        ),
        _layer_tensors={"layer": tensor},
        _worker_rank=lambda: 0,
    )
    observed = []
    ring = SimpleNamespace(configure_sources=lambda *a, **kw: None)
    monkeypatch.setattr(
        factory, "verify_manager_page_lease_contract", lambda settings: ()
    )

    def build(config, **kwargs):
        observed.append(config)
        return ring

    factory.build_manager_page_runtime(
        connector,
        settings,
        ring_builder=build,
        progress_thread_initializer=lambda: None,
    )
    assert observed[0].arena_mode == expected
    assert observed[0].slot_bytes == 3758096384
    assert observed[0].slot_count == 1


def test_capture_arena_default_and_invalid_values():
    assert (
        factory.ManagerPageCaptureSettings(Path("/a.so"), "a" * 64, 1024).arena_mode
        == "mapped"
    )
    for value in ("", "pinned", 2, None):
        with pytest.raises(RuntimeError, match="arena mode"):
            factory.ManagerPageCaptureSettings(
                Path("/a.so"), "a" * 64, 1024, arena_mode=value
            )


def test_connector_config_selects_managed_capture_without_changing_geometry():
    extra = {
        "spark_cache_async_page_capture_library": "/attested.so",
        "spark_cache_async_page_capture_library_sha256": "a" * 64,
        "spark_cache_async_page_capture_slot_bytes": 3758096384,
        "spark_cache_async_page_capture_slot_count": 1,
        "spark_cache_async_page_capture_arena_mode": "managed",
    }
    connector = SimpleNamespace(
        _kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda key, default: extra.get(key, default)
        )
    )
    settings = factory.ManagerPageCaptureSettings.from_connector(connector)
    assert settings.arena_mode == "managed"
    assert settings.slot_bytes == 3758096384
