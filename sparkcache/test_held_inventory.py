"""Inventory mutation tracking and bounded quorum-report work."""

import operator
from unittest import mock

import pytest

from sparkcache.held_inventory import HeldInventory
from sparkcache.test_spark_context_cache_connector import (
    KVConnectorRole,
    _make_connector,
    connector_module,
)


def test_unchanged_large_inventory_report_does_not_copy_or_iterate(tmp_path):
    worker = _make_connector(tmp_path, 0)
    worker._held = {f"{value:064x}" for value in range(100_000)}
    worker._build_quorum_report_locked()
    with (
        mock.patch.object(HeldInventory, "copy", side_effect=AssertionError("inventory copied")),
        mock.patch.object(HeldInventory, "__iter__", side_effect=AssertionError("inventory iterated")),
        mock.patch.object(HeldInventory, "__eq__", side_effect=AssertionError("inventory compared")),
    ):
        report = worker._build_quorum_report_locked()
    assert report["held_count"] == 100_000
    assert len(report["checkpoint"]["held"]) <= connector_module._QUORUM_REPORT_BATCH_SIZE


def _mutate(inventory, name):
    a, b, c = (f"{n:064x}" for n in range(3))
    operations = {
        "add": lambda: inventory.add(c),
        "discard": lambda: inventory.discard(a),
        "remove": lambda: inventory.remove(a),
        "pop": inventory.pop,
        "clear": inventory.clear,
        "update": lambda: inventory.update([c]),
        "difference_update": lambda: inventory.difference_update([a]),
        "intersection_update": lambda: inventory.intersection_update([b]),
        "symmetric_difference_update": lambda: inventory.symmetric_difference_update([a, c]),
        "ior": lambda: operator.ior(inventory, {c}),
        "isub": lambda: operator.isub(inventory, {a}),
        "iand": lambda: operator.iand(inventory, {b}),
        "ixor": lambda: operator.ixor(inventory, {a, c}),
    }
    operations[name]()


@pytest.mark.parametrize(
    "mutation",
    [
        "add", "discard", "remove", "pop", "clear", "update", "difference_update",
        "intersection_update", "symmetric_difference_update", "ior", "isub", "iand", "ixor",
    ],
)
def test_every_inventory_mutation_is_advertised_and_withdraws_membership(tmp_path, mutation):
    import types

    worker = _make_connector(tmp_path / "worker", 0)
    scheduler = _make_connector(tmp_path / "scheduler", 0, role=KVConnectorRole.SCHEDULER)
    initial = {f"{n:064x}" for n in range(2)}
    worker._held = initial

    def deliver():
        report = worker._build_quorum_report_locked()
        scheduler._absorb_quorum(
            types.SimpleNamespace(
                kv_connector_stats=connector_module.SparkCacheStats(data={"reports": [report]})
            )
        )
        return report

    first = deliver()
    _mutate(worker._held, mutation)
    expected = set(worker._held)
    report = deliver()
    assert report["checkpoint"]["state_sequence"] == first["checkpoint"]["state_sequence"] + 1
    assert set(report["checkpoint"]["held"]) == expected
    assert scheduler._worker_held[0] == expected
    assert all(0 not in scheduler._quorum.get(digest, ()) for digest in initial - expected)


def test_inventory_assignment_cannot_reuse_a_previous_revision(tmp_path):
    worker = _make_connector(tmp_path, 0)
    worker._held = {"a" * 64}
    first = worker._build_quorum_report_locked()
    worker._held = {"b" * 64}
    second = worker._build_quorum_report_locked()
    assert second["checkpoint"]["state_sequence"] == first["checkpoint"]["state_sequence"] + 1
    assert second["checkpoint"]["held"] == ["b" * 64]


def test_inventory_replacement_with_equal_content_preserves_sequence(tmp_path):
    worker = _make_connector(tmp_path, 0)
    external = {"a" * 64}
    worker._held = external
    first = worker._build_quorum_report_locked()
    external.add("b" * 64)
    assert worker._held == {"a" * 64}
    worker._held = {"a" * 64}
    second = worker._build_quorum_report_locked()
    assert second["checkpoint"]["state_sequence"] == first["checkpoint"]["state_sequence"]


def test_in_place_assignment_preserves_inventory_identity_and_tracks_changes(tmp_path):
    worker = _make_connector(tmp_path, 0)
    worker._held = {"a" * 64}
    worker._build_quorum_report_locked()
    inventory = worker._held
    worker._held |= {"b" * 64}
    assert worker._held is inventory
    report = worker._build_quorum_report_locked()
    assert report["checkpoint"]["held"] == ["a" * 64, "b" * 64]


def test_inventory_no_op_mutations_keep_revision_stable():
    inventory = HeldInventory({"a"})
    revision = inventory.revision
    inventory.add("a")
    inventory.discard("absent")
    inventory.update({"a"})
    inventory.difference_update({"absent"})
    inventory.intersection_update({"a", "b"})
    inventory.symmetric_difference_update([])
    assert inventory.revision == revision


@pytest.mark.parametrize("operation", ["update", "difference_update"])
def test_partial_iterator_failure_still_advances_inventory_revision(operation):
    inventory = HeldInventory({"a"})
    revision = inventory.revision

    def faulty_values():
        yield "b" if operation == "update" else "a"
        raise RuntimeError("iterator failed")

    with pytest.raises(RuntimeError, match="iterator failed"):
        getattr(inventory, operation)(faulty_values())
    assert inventory.revision > revision
    assert inventory == ({"a", "b"} if operation == "update" else set())


@pytest.mark.parametrize("operation", ["difference_update", "symmetric_difference_update"])
def test_inventory_can_remove_itself(operation):
    inventory = HeldInventory({"a", "b"})
    revision = inventory.revision
    getattr(inventory, operation)(inventory)
    assert not inventory
    assert inventory.revision > revision
