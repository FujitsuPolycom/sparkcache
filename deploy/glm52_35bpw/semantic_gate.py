"""Store/restart/hit gate for the GLM-5.2 fixed-MTP4 serving recipe (``R7``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deploy.deployment_contract import (
    assistant_content as _content,
    request_chat as _request,
    run_semantic_hit as run_hit,
    run_semantic_miss as run_miss,
)


LONG_MAX_TOKENS = 256
SHORT_MAX_TOKENS = 128


def run_hit_after_quorum(
    endpoint: str,
    model: str,
    reference: Path,
    *,
    records: int | None = None,
) -> dict:
    """Prime worker stats with a non-cacheable request, then require the hit."""

    prime = _request(
        endpoint,
        model,
        "What is 1 + 1? Respond with only the integer.",
        SHORT_MAX_TOKENS,
    )
    if _content(prime) != "2":
        raise RuntimeError(
            "four-rank manifest-inventory publication request failed"
        )
    hit_options = {
        "long_max_tokens": LONG_MAX_TOKENS,
        "short_max_tokens": SHORT_MAX_TOKENS,
    }
    if records is not None:
        hit_options["records"] = records
    result = run_hit(endpoint, model, reference, **hit_options)
    result["quorum_prime"] = "2"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("miss", "hit"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm-5.2-exl3-r7-3.5bpw")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--records",
        type=int,
        help="archive records; miss defaults to 384 and hit reads the reference",
    )
    args = parser.parse_args()
    result = (
        run_miss(
            args.endpoint,
            args.model,
            args.reference,
            long_max_tokens=LONG_MAX_TOKENS,
            records=args.records or 384,
        )
        if args.phase == "miss"
        else run_hit_after_quorum(
            args.endpoint,
            args.model,
            args.reference,
            records=args.records,
        )
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
