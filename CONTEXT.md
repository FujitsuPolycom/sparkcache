# SparkCache domain context

## Connector configuration

Connector configuration is the validated interpretation of vLLM parallel
settings, SparkCache options, environment fallbacks, model-profile policy,
cache capacity, and KV-group topology.

`sparkcache/spark_context_cache_config.py` owns this interpretation. It creates
a cache identity only after DCP-local and physical TP shard ranks are explicit.

Scheduler and worker connectors consume the same configuration. Their wire
identities and deployment constraints cannot drift.

## Deployment contract

A deployment contract contains the model-neutral mechanics that turn one
Docker inspection record into a repeatable launch.

It covers inspection structure, vLLM command normalization, environment
parsing, ports, immutable source identities, overlay receipts, and create-only
container launch behavior.

`deploy/deployment_contract/` owns these mechanics. Its output must be
deterministic, preserve verified-or-recompute behavior, and retain public
imports used by existing deployment tools.

## Profile adapter

A profile adapter adds model-specific policy to a deployment contract. It owns
accepted model arguments, checkpoints, cache identity, parallel geometry,
quantization rules, environment requirements, and overlay hashes.

Shared deployment code must not infer or weaken profile policy. Adapters,
launch procedures, and live test records live under `deploy/`.

## Transformed inspection

A transformed inspection is a deep copy of one Docker inspection record with
the command, environment, labels, and mounts required by a profile adapter.
The transformation never mutates its input.

## Overlay receipt

An overlay receipt is deterministic JSON that binds generated vLLM files to
their exact inputs, transformations, output hashes, and SparkCache source-tree
digest.

A launcher rejects a missing, incompatible, or modified receipt input.
