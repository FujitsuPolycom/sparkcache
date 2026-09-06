"""SparkCache heat-aware admission and SSD-control prototype.

**Research-only. Offline and excluded from SparkCache packages.**

No production SparkCache module imports this package, and this module imports
nothing from SparkCache production code. It models heat counters,
shadow admission, and write-budget accounting entirely in process memory with
no I/O and no serving dependency. Constructing any object here has no effect
on storage or serving behavior. GPU-free behavior and import-isolation
regressions live in ``research/heat_ssd_control/test_prototype.py``.
"""
