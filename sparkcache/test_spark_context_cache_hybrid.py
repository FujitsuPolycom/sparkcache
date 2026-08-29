from __future__ import annotations

import unittest

from sparkcache.spark_context_cache_hybrid import (
    HybridCodecError,
    PageGroup,
    PageLayer,
    PageLayout,
    decode_page_snapshot,
    encode_page_snapshot,
    plan_page_snapshot,
    split_snapshot,
)


def _layout() -> PageLayout:
    return PageLayout(
        (
            PageGroup(
                256,
                (
                    PageLayer("compressed", "torch.uint8", (2, 8), 16),
                    PageLayer("full", "torch.uint8", (64, 8), 512),
                ),
            ),
            PageGroup(
                4,
                (PageLayer("state", "torch.float32", (4, 16), 256),),
            ),
        )
    )


class HybridPageCodecTests(unittest.TestCase):
    def test_page_plan_describes_payloads_without_slicing_them(self) -> None:
        layout = _layout()
        payloads = {
            "full": bytes(1024),
            "compressed": bytes(32),
            "state": bytes(768),
        }
        encoded = encode_page_snapshot(layout, (2, 3), payloads)

        plan = plan_page_snapshot(layout, encoded, (2, 3))

        self.assertEqual(plan.total_bytes, len(encoded))
        self.assertEqual(plan.block_counts, (2, 3))
        self.assertEqual(
            [(span.layer_name, span.group_index, span.page_count) for span in plan.spans],
            [
                ("compressed", 0, 2),
                ("full", 0, 2),
                ("state", 1, 3),
            ],
        )
        self.assertEqual(
            {
                span.layer_name: bytes(encoded[span.source_start : span.source_end])
                for span in plan.spans
            },
            payloads,
        )

    def test_page_plan_accepts_header_prefix_plus_declared_total(self) -> None:
        layout = _layout()
        encoded = encode_page_snapshot(
            layout,
            (1, 1),
            {"full": bytes(512), "compressed": bytes(16), "state": bytes(256)},
        )
        header_end = len(encoded) - 512 - 16 - 256

        plan = plan_page_snapshot(
            layout,
            encoded[:header_end],
            (1, 1),
            total_bytes=len(encoded),
        )

        self.assertEqual(plan.header_bytes, header_end)
        self.assertEqual(plan.spans[-1].source_end, len(encoded))

    def test_page_plan_rejects_truncated_header_or_wrong_total(self) -> None:
        layout = _layout()
        encoded = encode_page_snapshot(
            layout,
            (1, 1),
            {"full": bytes(512), "compressed": bytes(16), "state": bytes(256)},
        )
        with self.assertRaises(HybridCodecError):
            plan_page_snapshot(layout, encoded[:10], (1, 1), total_bytes=len(encoded))
        with self.assertRaises(HybridCodecError):
            plan_page_snapshot(layout, encoded, (1, 1), total_bytes=len(encoded) + 1)

    def test_multi_group_pages_round_trip_byte_exactly(self) -> None:
        layout = _layout()
        payloads = {
            "full": bytes(range(256)) * 4,
            "compressed": bytes(range(32)),
            "state": bytes(range(256)) * 3,
        }
        encoded = encode_page_snapshot(layout, (2, 3), payloads)
        parts = split_snapshot(encoded, 4)

        self.assertEqual(
            decode_page_snapshot(layout, b"".join(parts), (2, 3)),
            payloads,
        )

    def test_layout_or_block_count_change_fails_closed(self) -> None:
        layout = _layout()
        payloads = {
            "full": bytes(512),
            "compressed": bytes(16),
            "state": bytes(256),
        }
        encoded = encode_page_snapshot(layout, (1, 1), payloads)

        with self.assertRaises(HybridCodecError):
            decode_page_snapshot(layout, encoded, (2, 1))
        changed = PageLayout(
            (
                layout.groups[0],
                PageGroup(
                    8,
                    (PageLayer("state", "torch.float32", (4, 16), 256),),
                ),
            )
        )
        with self.assertRaises(HybridCodecError):
            decode_page_snapshot(changed, encoded, (1, 1))

    def test_payload_length_and_trailing_bytes_are_rejected(self) -> None:
        layout = _layout()
        with self.assertRaises(HybridCodecError):
            encode_page_snapshot(
                layout,
                (1, 1),
                {"full": bytes(511), "compressed": bytes(16), "state": bytes(256)},
            )
        encoded = encode_page_snapshot(
            layout,
            (1, 1),
            {"full": bytes(512), "compressed": bytes(16), "state": bytes(256)},
        )
        with self.assertRaises(HybridCodecError):
            decode_page_snapshot(layout, encoded + b"x", (1, 1))

    def test_recurrent_align_policy_is_part_of_layout_identity(self) -> None:
        recurrent = PageLayout(
            (
                PageGroup(
                    256,
                    (PageLayer("state", "torch.float32", (4, 16), 256),),
                    reuse_policy="recurrent_align",
                ),
            )
        )
        full = PageLayout(
            (
                PageGroup(
                    256,
                    (PageLayer("state", "torch.float32", (4, 16), 256),),
                ),
            )
        )
        self.assertNotEqual(recurrent.digest, full.digest)
        payloads = {"state": bytes(256)}
        encoded = encode_page_snapshot(recurrent, (1,), payloads)
        self.assertEqual(decode_page_snapshot(recurrent, encoded, (1,)), payloads)
        with self.assertRaises(HybridCodecError):
            decode_page_snapshot(full, encoded, (1,))


if __name__ == "__main__":
    unittest.main()
