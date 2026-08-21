# SparkCache domain context

## Deployment vocabulary

### Connector configuration

Connector configuration is the immutable, validated interpretation of vLLM
parallel settings, SparkCache extra configuration, environment fallbacks,
model-profile policy, cache capacity, and KV-group topology. The
`sparkcache/spark_context_cache_config.py` module owns this interpretation and
constructs cache identities only after both DCP-local and physical TP shard
ranks are explicit.

Scheduler and worker connectors consume the same connector configuration so
their wire identities and deployment constraints cannot drift.

### Deployment contract

A deployment contract is the complete set of model-neutral mechanics that
turn a Docker inspection record into a reproducible serving launch. It covers
inspection structure, vLLM command normalization, environment parsing, port
validity, immutable source identities, overlay receipts, and create-only
container launch behavior.

The `deploy/deployment_contract/` module owns these mechanics. Its interface
must preserve deterministic output, fail-closed validation, and the public
imports and command-line behavior used by existing deployments.

### Profile adapter

A profile adapter applies model-specific policy to a deployment contract.
DeepSeek-V4 and GLM-5.2 adapters own their accepted model arguments,
checkpoint and cache identities, tensor and decode-context parallel geometry,
quantization rules, environment requirements, and model-specific overlay
hashes. Shared deployment mechanics must not weaken or replace those rules.

### Transformed inspection

A transformed inspection is a deep copy of one Docker inspection record with
the exact command, environment, labels, and mounts required by a profile
adapter. Transformations do not mutate the source record.

### Overlay receipt

An overlay receipt is a deterministic JSON record that binds generated vLLM
source files to their exact preimages, transformations, resulting hashes, and
SparkCache source-tree digest. A launcher rejects missing, incompatible, or
modified receipt inputs.
