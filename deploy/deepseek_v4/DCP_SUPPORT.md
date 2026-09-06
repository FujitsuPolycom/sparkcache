# DeepSeek-V4 DCP support boundary

The qualified profile is TP4/DCP1. DCP2 and DCP4 are **unsupported** and
rejected before Docker mutation. The live evidence in
`DEEPSEEK_V4_TP4_LIVE_VALIDATION.md` covers DCP1 only.

Two independent contracts make higher DCP unsafe:

1. The DSpark implementation in SparkRing serving-image lineage `R7` requires
   decode-context-parallel size one.
   Enabling DCP changes the draft model's rolling-context ownership without a
   validated sharding/restore contract.
2. SparkCache's `block_pages_v1` codec stores opaque hybrid-memory-allocator
   (HMA) pages from five cache groups. `ModelProfile.validate_for_deployment`
   rejects any DCP degree other
   than one because the codec does not define which logical page fragments
   belong to each DCP rank or how recycled sliding-window pages map back into
   group block tables.

This is unsupported because ownership cannot be proven, rather than because of
capacity. The model fits at TP4;
the missing work is a correctness contract. DCP2/DCP4 require:

- a per-group logical-page ownership map for every HMA block size and reuse
  window;
- an identity/wire-format revision that distinguishes DCP page sharding;
- store and restore assembly that never duplicates or drops page fragments;
- DSpark rolling-context state ownership compatible with the target groups;
- TP4/DCP2 and TP4/DCP4 GPU-free layout oracles, corruption tests, and
  four-rank miss/restart/hit qualification; and
- proof that all four physical ranks, not only DCP-local ranks, satisfy the
  current-generation quorum.

Until those conditions exist, the deployment transformer and profile-level
validation must continue to reject DCP2/DCP4. Silently reusing the DCP1 opaque
page format would risk coherent-looking but incorrect generations.
