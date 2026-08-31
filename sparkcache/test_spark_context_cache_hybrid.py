from __future__ import annotations

import unittest

from sparkcache.spark_context_cache_hybrid import (
    HybridCodecError,
    PageGroup,
    PageLayer,
    PageLayout,
    decode_page_snapshot,
    apply_page_delta,
    encode_page_delta,
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
    def test_page_delta_reuses_byte_identical_pages_and_round_trips(self) -> None:
        layout = PageLayout(
            (
                PageGroup(256, (PageLayer("full", "u8", (1024,), 1024),)),
                PageGroup(
                    256,
                    (PageLayer("recurrent", "u8", (512,), 512),),
                    reuse_policy="recurrent_align",
                ),
                PageGroup(
                    64,
                    (PageLayer("sliding", "u8", (256,), 256),),
                    reuse_policy="sliding",
                    reuse_window_tokens=512,
                ),
            )
        )
        base_payloads = {
            "full": b"A" * 2048,
            "recurrent": b"B" * 1024,
            "sliding": b"C" * 512,
        }
        result_payloads = {
            "full": base_payloads["full"] + b"D" * 2048,
            "recurrent": base_payloads["recurrent"] + b"E" * 512,
            "sliding": base_payloads["sliding"] + b"F" * 256,
        }
        base = encode_page_snapshot(layout, (2, 2, 2), base_payloads)
        result = encode_page_snapshot(layout, (4, 3, 3), result_payloads)

        delta = encode_page_delta(
            layout,
            base,
            result,
            base_block_counts=(2, 2, 2),
            result_block_counts=(4, 3, 3),
            base_boundary_tokens=512,
            result_boundary_tokens=768,
        )

        self.assertLess(len(delta), len(result))
        self.assertEqual(
            apply_page_delta(
                layout,
                base,
                delta,
                base_block_counts=(2, 2, 2),
                result_block_counts=(4, 3, 3),
                base_boundary_tokens=512,
                result_boundary_tokens=768,
            ),
            result,
        )

    def test_page_delta_rejects_wrong_base_corruption_and_unproven_boundary(
        self,
    ) -> None:
        layout = _layout()
        base_payloads = {
            "compressed": b"a" * 16,
            "full": b"b" * 512,
            "state": b"c" * 256,
        }
        result_payloads = {
            "compressed": base_payloads["compressed"] + b"d" * 16,
            "full": base_payloads["full"] + b"e" * 512,
            "state": base_payloads["state"] + b"f" * 256,
        }
        base = encode_page_snapshot(layout, (1, 1), base_payloads)
        result = encode_page_snapshot(layout, (2, 2), result_payloads)
        delta = encode_page_delta(
            layout,
            base,
            result,
            base_block_counts=(1, 1),
            result_block_counts=(2, 2),
            base_boundary_tokens=256,
            result_boundary_tokens=512,
        )

        wrong_base_payloads = dict(base_payloads)
        wrong_base_payloads["state"] = b"z" * 256
        wrong_base = encode_page_snapshot(layout, (1, 1), wrong_base_payloads)
        with self.assertRaisesRegex(HybridCodecError, "base differs"):
            apply_page_delta(
                layout,
                wrong_base,
                delta,
                base_block_counts=(1, 1),
                result_block_counts=(2, 2),
                base_boundary_tokens=256,
                result_boundary_tokens=512,
            )
        damaged = bytearray(delta)
        damaged[-1] ^= 1
        with self.assertRaisesRegex(HybridCodecError, "checksum mismatch"):
            apply_page_delta(
                layout,
                base,
                bytes(damaged),
                base_block_counts=(1, 1),
                result_block_counts=(2, 2),
                base_boundary_tokens=256,
                result_boundary_tokens=512,
            )
        with self.assertRaisesRegex(HybridCodecError, "boundaries are invalid"):
            encode_page_delta(
                layout,
                base,
                result,
                base_block_counts=(1, 1),
                result_block_counts=(2, 2),
                base_boundary_tokens=512,
                result_boundary_tokens=512,
            )

    def test_page_delta_does_not_reuse_changed_recurrent_prefix_pages(self) -> None:
        layout = PageLayout(
            (
                PageGroup(
                    256,
                    (PageLayer("state", "u8", (128,), 128),),
                    reuse_policy="recurrent_align",
                ),
            )
        )
        base = encode_page_snapshot(layout, (2,), {"state": b"A" * 256})
        result = encode_page_snapshot(
            layout,
            (3,),
            {"state": b"Z" * 128 + b"A" * 128 + b"B" * 128},
        )

        delta = encode_page_delta(
            layout,
            base,
            result,
            base_block_counts=(2,),
            result_block_counts=(3,),
            base_boundary_tokens=512,
            result_boundary_tokens=768,
        )

        self.assertEqual(
            apply_page_delta(
                layout,
                base,
                delta,
                base_block_counts=(2,),
                result_block_counts=(3,),
                base_boundary_tokens=512,
                result_boundary_tokens=768,
            ),
            result,
        )

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
            [
                (span.layer_name, span.group_index, span.page_count)
                for span in plan.spans
            ],
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
