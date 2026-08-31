# GLM-5.2 deployment failure points

This table covers the SparkCache composition of the public GLM-5.2 EXL3
3.5-bpw profile whose CUDA graph exact-state receipt covers capture sizes 1
through 40 (`exact-Q40`). It is a setup and recovery reference;
passing an individual check does not qualify the four-rank deployment.
The Spark inter-rank collective layer is abbreviated `SIRCL` below.

| Failure signature | Unsafe assumption | Required guard and recovery |
|---|---|---|
| A rank cannot reach the rendezvous address, or communicator initialization stalls after TCP bootstrap. | An available SSH path is also suitable for GLOO and NCCL bootstrap. | Give every rank a LAN management interface outside the direct-ring subnets. Run SparkRing site validation and the read-only preflight before container creation. Do not use Tailscale or a ring port as the management interface. |
| The model mount exists after reboot but `config.json` or the model index is absent. | A one-shot bind mount survives power loss, or directory existence proves the checkpoint is mounted. | Use a model-specific persistent bind/automount. Require an exact mountpoint check and rerun the pinned checkpoint verifier after reboot. Do not share a host mountpoint with another model profile. |
| Runtime attestation reports multiple failed file hashes. | The base image contains every compatibility file used by the accepted operator profile. | Treat the runtime as an explicit closure. Mount or bake the entrypoint, weight loader, CUDA-graph helper, QuACK helpers, DCP audit, shared parallel state, ARM64 TVM-FFI package, SIRCL files, and target-only Q40 files. Keep every bind read-only except documented caches. |
| `ModuleNotFoundError: spark_tp4_query_row_provider` appears during `verify_runtime.py`. | The SIRCL backend, port namespace, and capacity pool are a complete Python module set. | Include `spark_tp4_query_row_provider.py` beside the other `/opt/spark-vllm` SIRCL modules. Exercise `sitecustomize` under the serving environment during create-only attestation. |
| Exact-Q40 startup rejects an existing receipt, or a receipt names another image ID. | Q40 receipts transfer between image identities. | Preserve the receipt under an archive name, then let the exact target-only Q40 profile regenerate it. The mounted Q40 source, environment image ID, Docker image ID, and receipt image ID must agree. |
| Overlay preparation rejects `scheduler.py` even though vLLM reports commit `e2666d9a6`. | One vLLM commit has one scheduler preimage across all serving images. | Use the GLM stock preimage `1ea341…` and patch 011, which produces `d4ebec…`. The DeepSeek/adaptive scheduler uses patch 010 and is not interchangeable. |
| `git apply` succeeds from a checkout but fails from a staged archive. | Patch files retain one line-ending representation on every controller. | Overlay preparation canonicalizes patch input to LF and verifies the exact postimage. Never bypass the preimage or postimage hash. |
| The staged SparkCache source digest differs between Windows and Linux. | Filesystem path ordering and checkout line endings are platform-independent. | Source receipts canonicalize CRLF to LF and sort by POSIX relative path. Stage an immutable commit archive and require the same digest on every rank. |
| Only some ranks remain running after a power event or failed start. | Starting one rank repairs an incomplete collective. | Stop and start all physical ranks as one coordinated operation. Preserve failed containers under rollback names and generate all replacements with `docker create` before the cutover. |
| The miss semantic check succeeds, but no durable hit is available after restart. | The HTTP response waits for background cache publication. | Do not restart until every rank logs a matching snapshot and commit digest and its manifest/chunks pass filesystem checks. The miss command returning is not the store barrier. |
| A hit returns the right text without a four-rank external restore. | Semantic equality alone proves SparkCache persistence. | After a coordinated restart, require four-rank manifest discovery, the scheduler quorum-hit log, a restore log on every rank, the exact response, and a post-restore canary. |

For the qualification baseline, keep streaming snapshots and SparkCache CUDA
restore disabled. The bounded rank-local NVMe policy remains 200 GiB high,
180 GiB low, and TTL zero.
