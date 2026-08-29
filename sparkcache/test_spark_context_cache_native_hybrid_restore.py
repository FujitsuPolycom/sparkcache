from __future__ import annotations

from sparkcache.spark_context_cache_hybrid import (
    PageGroup,
    PageLayer,
    PageLayout,
    encode_page_snapshot,
    plan_page_snapshot,
)
from sparkcache.spark_context_cache_native_hybrid_restore import (
    build_page_copy_spans,
)


def test_page_copy_spans_cover_payload_once_in_layout_order() -> None:
    layout = PageLayout(
        (
            PageGroup(
                256,
                (
                    PageLayer("a", "torch.uint8", (4,), 4),
                    PageLayer("b", "torch.uint8", (2,), 2),
                ),
            ),
            PageGroup(1, (PageLayer("state", "torch.uint8", (3,), 3),)),
        )
    )
    encoded = encode_page_snapshot(
        layout,
        (2, 1),
        {"a": bytes(8), "b": bytes(4), "state": bytes(3)},
    )
    plan = plan_page_snapshot(layout, encoded, (2, 1))

    spans = build_page_copy_spans(plan)

    assert [span.destination_index for span in spans] == [0, 1, 2]
    assert [span.snapshot_offset_bytes for span in spans] == [0, 8, 12]
    assert [span.destination_byte_offset for span in spans] == [0, 0, 0]
    assert sum(span.byte_count for span in spans) == 15
    assert spans[0].arena_offset_bytes == plan.header_bytes
