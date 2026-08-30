"""TinyLFU-style shadow cache for admission experiments.

Replays a key sequence through an in-memory two-band cache whose admission
comparison reads the ``HitRing`` frequency sketch. The shadow decides what
frequency-aware admission would have retained; it never serves, never
persists, and shares no state with any serving component.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from research.heat_ssd_control.heat_model import (
    HeatKey,
    HitRing,
    HitRingConfig,
    ResearchFormatError,
)


@dataclass(frozen=True)
class ShadowConfig:
    window_capacity: int = 1024
    main_capacity: int = 65536
    ring: HitRingConfig | None = None

    def __post_init__(self) -> None:
        if (
            type(self.window_capacity) is not int
            or type(self.main_capacity) is not int
            or self.window_capacity <= 0
            or self.main_capacity <= 0
        ):
            raise ValueError("shadow capacities must be positive integers")
        if self.ring is not None and not isinstance(self.ring, HitRingConfig):
            raise ValueError("ring must be a HitRingConfig or None")


@dataclass(frozen=True)
class ShadowDecision:
    """What one access would have done under frequency-aware admission."""

    admitted: bool
    hit: bool
    reason: str
    estimate: int
    victim_estimate: int | None


@dataclass(frozen=True)
class TraceReport:
    """Rollup of one ``evaluate_trace`` run.

    Attributes:
        requests: replayed access count.
        window_hits: accesses served by the window band.
        main_hits: accesses served by the main band; the shadow's hits.
        misses: accesses not resident in either band.
        admitted: new keys that entered or replaced in the main cache.
        rejected: keys that lost the admission comparison.
        final_resident: entries remaining across both bands.

    hit_rate is (window_hits + main_hits) / requests, zero when no
    accesses were replayed.
    """

    requests: int
    window_hits: int
    main_hits: int
    misses: int
    admitted: int
    rejected: int
    final_resident: int

    @property
    def hit_rate(self) -> float:
        if not self.requests:
            return 0.0
        return (self.window_hits + self.main_hits) / self.requests


class TinyLFUShadow:
    """Window + main-cache shadows controlled by the 8-bit heat ring.

    Deliberate simplifications, stated so results are read correctly:
    single-band main cache (no protected/probationary segmentation), no
    ghost admission on rejection, direct-mapped ring instead of a Cuckoo
    filter. Each deviation biases the shadow toward retaining slightly fewer
    reusable keys than full W-TinyLFU.
    """

    def __init__(self, config: ShadowConfig | None = None) -> None:
        self._config = config or ShadowConfig()
        self._ring = HitRing(self._config.ring)
        self._window: OrderedDict[HeatKey, None] = OrderedDict()
        self._main: OrderedDict[HeatKey, None] = OrderedDict()

    @property
    def config(self) -> ShadowConfig:
        return self._config

    def access(self, key: HeatKey) -> ShadowDecision:
        """Run one access and return the decision it would have produced."""
        if not isinstance(key, HeatKey):
            raise ResearchFormatError("key must be a HeatKey")
        self._ring.record_hit(key)
        if key in self._window:
            self._window.move_to_end(key)
            return ShadowDecision(True, True, "resident_window", self._ring.estimate(key), None)
        if key in self._main:
            self._main.move_to_end(key)
            return ShadowDecision(True, True, "resident_main", self._ring.estimate(key), None)

        self._window[key] = None
        if len(self._window) <= self._config.window_capacity:
            return ShadowDecision(False, False, "window", self._ring.estimate(key), None)

        candidate, _ = self._window.popitem(last=False)
        if len(self._main) < self._config.main_capacity:
            self._main[candidate] = None
            return ShadowDecision(True, False, "spare_capacity", self._ring.estimate(candidate), None)
        victim = next(iter(self._main))  # least-recently-used main entry
        victim_estimate = self._ring.estimate(victim)
        candidate_estimate = self._ring.estimate(candidate)
        if candidate_estimate > victim_estimate:
            self._main.popitem(last=False)
            self._main[candidate] = None
            return ShadowDecision(
                True, False, "admission_win", candidate_estimate, victim_estimate
            )
        return ShadowDecision(False, False, "admission_loss", candidate_estimate, victim_estimate)

    def evaluate_trace(self, keys: Iterable[HeatKey]) -> TraceReport:
        requests = 0
        window_hits = 0
        main_hits = 0
        misses = 0
        admitted = 0
        rejected = 0
        for key in keys:
            decision = self.access(key)
            requests += 1
            window_hits += decision.reason == "resident_window"
            main_hits += decision.reason == "resident_main"
            misses += not decision.hit
            admitted += decision.admitted and not decision.hit
            rejected += decision.reason == "admission_loss"
        return TraceReport(
            requests=requests,
            window_hits=window_hits,
            main_hits=main_hits,
            misses=misses,
            admitted=admitted,
            rejected=rejected,
            final_resident=len(self._window) + len(self._main),
        )


def evaluate_trace(keys: Iterable[HeatKey], config: ShadowConfig | None = None) -> TraceReport:
    """One-shot convenience: fresh shadow, replay, report."""
    return TinyLFUShadow(config).evaluate_trace(keys)
