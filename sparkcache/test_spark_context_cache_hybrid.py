from __future__ import annotations

import unittest

from sparkcache.spark_context_cache_hybrid import (
    HybridCodecError,
    PageGroup,
    PageLayer,
    PageLayout,
    decode_page_snapshot,
    encode_page_snapshot,
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


if __name__ == "__main__":
    unittest.main()
