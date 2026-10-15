# Checkpoint identity from pinned inputs

Status: **implemented**, CPU tested. The function
`sparkcache.checkpoint_identity.derive_checkpoint_identity(inputs)` and CLI
derive a 64-character lowercase SHA-256. Neither changes a deployment's
recorded identity or verifies model weights on disk.

```bash
python -m sparkcache.checkpoint_identity --input deploy/checkpoint_identity/glm53-csf-inputs.json
sparkcache-checkpoint-identity --input inputs.json --output identity-receipt.json
```

The console command is installed by the SparkCache package. Output contains
the schema, complete inputs and digest. An output file must not exist.
The input is a JSON object with exactly six fields:

| Field | Contract |
|---|---|
| `repository` | Exact repository identifier; no URL or case normalization |
| `revision` | Full lowercase 40- or 64-hex immutable commit; never a branch, tag or abbreviation |
| `artifacts` | Either `{"manifest_sha256": "..."}` hashing the exact manifest bytes, or `{"files": {"relative/path": "sha256", ...}}` |
| `quantization` | Object with explicit `method` and `encoding`, plus any required loader settings; use `none` when absent |
| `model_config` | Either `{"sha256": "..."}` for exact config bytes or `{"values": {...}}` for the complete relevant configuration |
| `draft` | `{"mode":"none"}`, or `{"mode":"embedded","config":{...}}`, or `{"mode":"separate","config":{...},"checkpoint_sha256":"..."}` |

The hash preimage is `{"schema":"sparkcache-checkpoint-identity/v1","inputs":...}`
serialized with Python JSON's `sort_keys=True`, `separators=(",", ":")`,
`ensure_ascii=True`, `allow_nan=False`, encoded as ASCII with no newline.
Object key order does not matter; list order, number representation, strings
and all supplied values do. No Unicode, numeric or repository normalization
is performed. Duplicate JSON keys, unknown top-level fields, mutable revisions,
nonfinite numbers, malformed digests and unsafe artifact paths are refused.
Changing this serialization requires a different schema.

Use the complete artifact inventory or a manifest that binds every required
weight, config and tokenizer file. The function cannot prove completeness of
caller-selected files or config values. Config-byte hashing is preferred to
a selected field list. Include effective loader/quantization overrides and
speculative method, token count, adaptive settings, draft backend and dtype.
For a separate draft, derive its checkpoint identity with this same function
(normally `draft.mode=none`), then bind it in the target's draft object.
Embedded MTP binds the target's own checkpoint and its explicit MTP settings.
Filesystem paths and timestamps are not checkpoint identities.

This digest is an input to `CacheIdentity`, not its complete runtime namespace:
KV geometry, topology and runtime numerics still require their own pins.
Adopting a v1 digest in a deployment intentionally causes cache misses against
entries using another digest. No automatic migration or legacy alias exists.

## Qwen installer inputs

The release 2026.10.4 serving inputs and complete LF-normalized SparkRing pin
manifests are copied from SparkRing `bfff0adc978acb41dd70ba569cf11c0d071f4e09`.
Each receipt is reproducible with `python -m sparkcache.checkpoint_identity
--input deploy/checkpoint_identity/qwen38-<name>-inputs.json`:

| Name | Revision | Target and embedded draft identity |
|---|---|---|
| `csf` | `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` | `7906c93c59bf153168170b601cf36ae7cdd31927797dc4608320fe1ba8ec2c78` |
| `step5500` | `60215d26cf5e42c2db6128774032d57fc62678da` | `759a6ce3138e95cddaa4dc8e53480274d2a3e75154cf1893ff275237b1ded2b6` |

Inputs use the effective CSF TP2 and QAD step-5500 TP4 loader settings and the
complete speculative configuration, including BF16 draft KV (`auto`). Adjacent
`-pins.json` and `-receipt.json` files preserve the preimages and outputs.
These identities intentionally miss entries from either the other checkpoint
or step 4000. They do not replace any recorded identity.

## Recorded identity audit

The fixtures copy public pin manifests and serving inputs from SparkRing commit
`68a995125bef`. Their provenance and the full outputs of five derivation
attempts are in [recorded-identities.json](recorded-identities.json).

| Checkpoint | Repository revision | Recorded SparkCache digest |
|---|---|---|
| GLM-5.3-Flash CSF | `dec48abd33efa73c3bb7c95b74eee10cad34f9be` | `6af0343976eefbfa0fb59d87befcf6d10f02fef90ba0e7881dd1201fe0ecd76e` |
| Qwen3.8-Flash-Next step 4000 | `629bc3218833a38b475b719f34aa571666f4a03e` | `036c2f7994466d32514130f0417ccd117705ca01e2038c1ce5745f84813829ba` |

`dec48abd` abbreviates a repository revision, not a SparkCache SHA-256.
Neither recorded digest reproduces from the available derivations:

1. `deploy/glm53_flash/profile.py`'s `hf-revision-v1` hash.
2. SHA-256 of the exact SparkRing pin-manifest bytes.
3. SHA-256 of that manifest serialized as canonical JSON.
4. The DeepSeek `sparkcache-checkpoint-manifest/v1` preimage, converting all
   pinned files into sorted `{path, bytes, sha256}` rows. This tests the
   available manifest; the original on-disk checkpoint inventory is absent.
5. The explicit inputs and v1 function specified here.

The GLM profile helper validates a supplied digest; it does not derive it
from a pin manifest. Neither repository records the original algorithm,
serialization, salt or exact preimage of the two recorded values. That missing
provenance prevents reproduction; a digest alone cannot identify its preimage.
These attempts do not prove that no other algorithm could produce the values.
Both recorded digests remain pinned. The computed fixture digests are audit
results, not replacements for the recorded deployment values. The tests assert
both the recorded values and the mismatches instead of rewriting either.
