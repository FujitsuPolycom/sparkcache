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

The semantic response reader records the final `content` answer separately
from an evidence body that concatenates the assistant message fields
`reasoning`, `reasoning_content`, and `content` in that order. It also records
the choice's `finish_reason`. A token-limited response
(`finish_reason == "length"`) or a response without a non-whitespace assistant
body raises `SemanticGateInconclusive`; `SemanticGateInconclusive.as_result()`
provides a JSON-compatible `INCONCLUSIVE` report. Successful miss and hit
result dictionaries retain their established fields. The miss reference
stores the combined body so the hit gate can require deterministic evidence.
Exact expected-answer checks use final `content` and still determine semantic
success. References written before the optional `assistant_body` field remain
readable.

## Validation

```bash
python -m pytest deploy/deployment_contract -q
python -m pytest sparkcache deploy -q
python -m ruff check sparkcache deploy
```
