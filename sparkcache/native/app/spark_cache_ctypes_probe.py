#!/usr/bin/env python3
"""Smoke-test the exported shared-library ABI from the canonical binding."""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path


NATIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE_ROOT.parent))

from sparkcache.spark_cache_native import (  # noqa: E402
    AbiInfo,
    ArenaView,
    NativePlacementError,
    PlacementConfig,
    check,
    load_library,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    arguments = parser.parse_args()
    library, info = load_library(arguments.library)
    print(
        "ctypes ABI PASS:"
        f" abi={info.abi_version}"
        f" cudart={info.cudart_version}"
        f" caps=0x{info.capability_flags:x}"
        f" abi_info={ctypes.sizeof(AbiInfo)}"
        f" arena_view={ctypes.sizeof(ArenaView)}"
    )

    invalid = PlacementConfig(abi_version=0xFFFFFFFF)
    handle = ctypes.c_void_p()
    try:
        check(
            library,
            library.spark_cache_placement_create(
                ctypes.byref(invalid), ctypes.byref(handle)
            ),
            "create",
            handle,
        )
    except NativePlacementError as error:
        if "ABI" not in str(error):
            raise
        print(f"create-error accessor PASS: {error}")
    else:
        raise AssertionError("invalid ABI unexpectedly created a handle")
    finally:
        if handle.value:
            library.spark_cache_placement_destroy(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
