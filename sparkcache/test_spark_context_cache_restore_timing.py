from __future__ import annotations

import json
import unittest

from sparkcache.spark_context_cache_restore_timing import (
    RESTORE_TIMING_PREFIX,
    RestoreTiming,
)


class RestoreTimingTests(unittest.TestCase):
    def test_record_is_stable_complete_and_machine_readable(self) -> None:
        timing = RestoreTiming(
            request_id="request-1",
            digest="a" * 64,
            span_tokens=16384,
            storage_mode="block_pages_v1",
            enqueued_ns=1_000_000,
        )
        timing.start_service(2_000_000)
        timing.add("manifest_lookup", 3_000_000)
        timing.add("prior_cuda_work", 2_000_000)
        timing.add("restore_read", 4_000_000)
        timing.add("reassembly_decode", 5_000_000)
        timing.add("h2d_submit", 6_000_000)
        timing.add("cuda_sync", 7_000_000)
        timing.page_bytes = 123456
        timing.chunk_count = 64
        timing.finish("verified", 30_000_000)

        rendered = timing.render()
        self.assertTrue(rendered.startswith(RESTORE_TIMING_PREFIX))
        record = json.loads(rendered.removeprefix(RESTORE_TIMING_PREFIX))
        self.assertEqual(record["schema"], "sparkcache-restore-timing/v1")
        self.assertEqual(record["request_id"], "request-1")
        self.assertEqual(record["digest"], "a" * 12)
        self.assertEqual(record["span_tokens"], 16384)
        self.assertEqual(record["queue_wait_ms"], 1.0)
        self.assertEqual(record["service_ms"], 28.0)
        self.assertEqual(record["end_to_end_ms"], 29.0)
        self.assertEqual(record["page_bytes"], 123456)
        self.assertEqual(record["chunk_count"], 64)
        self.assertEqual(
            set(record["phase_ms"]),
            {
                "manifest_lookup",
                "prior_cuda_work",
                "restore_read",
                "reassembly_decode",
                "h2d_submit",
                "cuda_sync",
            },
        )

    def test_unknown_or_duplicate_phase_is_rejected(self) -> None:
        timing = RestoreTiming(
            request_id="request-1",
            digest="a" * 64,
            span_tokens=256,
            storage_mode="rows",
            enqueued_ns=1,
        )
        with self.assertRaises(ValueError):
            timing.add("other", 1)
        timing.add("restore_read", 1)
        with self.assertRaises(ValueError):
            timing.add("restore_read", 2)


if __name__ == "__main__":
    unittest.main()
