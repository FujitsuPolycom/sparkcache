"""Flat history restore reads each surviving layer tail once, one at a time."""

import weakref

from sparkcache.persistent_context_cache import cache_manifest

from .test_page_history_reconstruction import _history


def test_history_reads_only_surviving_tails_and_releases_each_before_the_next(
    tmp_path, monkeypatch
):
    store, identity, layout, digest, snapshots = _history(tmp_path, stages=4)
    lookup = store.lookup(identity, digest, verify_chunks=False)
    final_objects = {
        f"{descriptor['sha256']}.spcc"
        for descriptor in lookup._manifest["delta_stages"][-1]["delta_objects"]
    }
    original = cache_manifest._read_object_extent
    previous = []
    read_objects = []

    class TrackedTail(bytearray):
        pass

    def read(pieces):
        assert all(reference() is None for reference in previous)
        read_objects.extend(path.name for path, *_rest in pieces)
        result = TrackedTail(original(pieces))
        previous.append(weakref.ref(result))
        return result

    monkeypatch.setattr(cache_manifest, "_read_object_extent", read)
    actual = store.restore_page_snapshot(
        lookup, layout=layout, result_block_counts=(6,), result_boundary_tokens=1536,
    )
    assert actual == snapshots[-1]
    # Every stage of this fixture rewrites every page, so only the newest
    # stage's two layer tails survive.
    assert len(previous) == 2
    assert set(read_objects) <= final_objects
    assert all(reference() is None for reference in previous)
