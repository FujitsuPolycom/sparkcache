# Deployment contract

Status: **implemented**.

The deployment-contract module owns model-neutral mechanics used to transform
Docker inspection records into reproducible SparkCache launches. DeepSeek-V4
and GLM-5.2 profile adapters call this interface while retaining their own
model, topology, quantization, cache-identity, and accepted-source policy.

## Interface

| Module | Behavior |
| --- | --- |
| `command.py` | deterministic JSON encoding, vLLM command normalization, and value-bearing option parsing |
| `inspection.py` | Docker environment parsing and single-record inspection loading |
| `ports.py` | numeric port-range validation |
| `source.py` | file SHA-256 and line-ending-independent deployable-source identity |
| `patches.py` | exact-preimage, exact-postimage Git patch application |
| `receipts.py` | overlay schema, inventory, source identity, and generated-file verification |
| `container.py` | deterministic Docker command construction and execution |
| `semantic.py` | deterministic miss, restart/hit, and post-restore semantic gates |

`deploy/deepseek_v4/` and `deploy/glm52_35bpw/` are profile adapters at this
seam. They must reject any source inspection that violates their explicit
model contract. The shared module does not infer or relax model policy.

Compatibility imports remain available from the model-specific modules used
by existing scripts. Receipt schema identifiers and serialized receipt bytes
are unchanged.

## Validation

```bash
python -m pytest deploy/deployment_contract -q
python -m pytest sparkcache deploy -q
python -m ruff check sparkcache deploy
```
