"""Validate four DeepSeek TP4 source inspections as one portable cluster."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from deploy.deepseek_v4.launch_from_inspect import _vllm_args
from deploy.deepseek_v4.tp4_profile import (
    PROFILE,
    ProfileTransformError,
    transform_inspection,
)


_PORT_NAME = re.compile(r"(?:^|_)PORT\d*\Z")


def _option(arguments: list[str], name: str) -> str:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            values.append(arguments[index + 1])
        elif argument.startswith(name + "="):
            values.append(argument.split("=", 1)[1])
    if len(values) != 1:
        raise ProfileTransformError(f"transformed command requires one {name}")
    return values[0]


def _environment(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not name or name in result:
            raise ProfileTransformError(
                "transformed environment must contain unique NAME=VALUE entries"
            )
        result[name] = setting
    return result


def _collective_ports(environment: dict[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, raw in environment.items():
        if name in {"PORT", "MASTER_PORT"} or _PORT_NAME.search(name) is None:
            continue
        try:
            result[name] = int(raw)
        except ValueError as error:
            raise ProfileTransformError(
                f"transformed environment port {name} is invalid"
            ) from error
    return dict(sorted(result.items()))


def _model_source(inspection: dict[str, Any]) -> str:
    destination = str(PROFILE["model"]["container_path"])
    matches = [
        mount
        for mount in inspection.get("Mounts", ())
        if isinstance(mount, dict) and mount.get("Destination") == destination
    ]
    if len(matches) != 1:
        raise ProfileTransformError("cluster rank has no unique model bind")
    return str(matches[0].get("Source", ""))


def validate_cluster(
    inspections: Iterable[dict[str, Any]],
    *,
    checkpoint_sha256: str,
    api_port: int,
    master_port: int,
) -> dict[str, Any]:
    """Return a deterministic cross-rank plan or fail before Docker mutation."""

    sources = list(inspections)
    expected_nodes = int(PROFILE["serving"]["nodes"])
    if len(sources) != expected_nodes:
        raise ProfileTransformError(
            f"cluster preflight requires exactly {expected_nodes} inspections"
        )
    records: list[dict[str, Any]] = []
    for source in sources:
        transformed = transform_inspection(
            source,
            checkpoint_sha256=checkpoint_sha256,
            api_port=api_port,
            master_port=master_port,
        )
        arguments = _vllm_args(list(transformed["Config"]["Cmd"]))
        environment = _environment(transformed["Config"].get("Env", ()))
        rank = int(_option(arguments, "--node-rank"))
        records.append(
            {
                "rank": rank,
                "image": str(source.get("Image", "")),
                "master_addr": _option(arguments, "--master-addr"),
                "master_port": int(_option(arguments, "--master-port")),
                "api_port": (
                    int(_option(arguments, "--port")) if rank == 0 else None
                ),
                "headless": "--headless" in arguments,
                "model_host_path": _model_source(source),
                "collective_ports": _collective_ports(environment),
            }
        )
    records.sort(key=lambda record: record["rank"])
    ranks = [record["rank"] for record in records]
    if ranks != list(range(expected_nodes)):
        raise ProfileTransformError(
            f"cluster physical ranks must be 0..{expected_nodes - 1} exactly once"
        )
    images = {record["image"] for record in records}
    if len(images) != 1:
        raise ProfileTransformError("cluster source image IDs are not homogeneous")
    master_addresses = {record["master_addr"] for record in records}
    if len(master_addresses) != 1:
        raise ProfileTransformError("cluster master addresses differ")
    collective_maps = {
        json.dumps(record["collective_ports"], sort_keys=True)
        for record in records
    }
    if len(collective_maps) != 1:
        raise ProfileTransformError("cluster collective port assignments differ")
    return {
        "schema": "sparkcache-deepseek0731-tp4-cluster-plan/v1",
        "profile_id": str(PROFILE["profile_id"]),
        "checkpoint_sha256": checkpoint_sha256,
        "image": records[0]["image"],
        "master_addr": records[0]["master_addr"],
        "master_port": master_port,
        "api_port": api_port,
        "collective_ports": records[0]["collective_ports"],
        "ranks": records,
    }


def _read_inspection(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileTransformError(f"cannot read inspection {path}") from error
    if isinstance(document, list):
        if len(document) != 1:
            raise ProfileTransformError(f"inspection {path} must contain one object")
        document = document[0]
    if not isinstance(document, dict):
        raise ProfileTransformError(f"inspection {path} must contain one object")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="append", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--api-port", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        plan = validate_cluster(
            (_read_inspection(path) for path in args.inspect),
            checkpoint_sha256=args.checkpoint_sha256,
            api_port=args.api_port,
            master_port=args.master_port,
        )
    except ProfileTransformError as error:
        parser.error(str(error))
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
