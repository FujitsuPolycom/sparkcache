"""Large incoming delta objects must not overlap across history stages."""

import weakref

from .test_page_history_reconstruction import _history


def test_history_releases_incoming_delta_before_reading_next(tmp_path, monkeypatch):
    store, identity, layout, digest, snapshots = _history(tmp_path, stages=4)
    original = store._read_page_delta_objects
    previous = []

    class TrackedDelta(bytearray):
        pass

    def read(*args, **kwargs):
        assert all(reference() is None for reference in previous)
        result = TrackedDelta(original(*args, **kwargs))
        previous.append(weakref.ref(result))
        return result

    monkeypatch.setattr(store, "_read_page_delta_objects", read)
    actual = store.restore_page_snapshot(
        store.lookup(identity, digest, verify_chunks=False),
        layout=layout, result_block_counts=(6,), result_boundary_tokens=1536,
    )
    assert actual == snapshots[-1]
    assert len(previous) == 4
    assert all(reference() is None for reference in previous)
