# Publish the GLM-5.3 SparkCache image

Status: **implemented** for image construction and private GHCR publication.
Four-rank TP4/DCP1 serving remains unqualified until the exact registry digest
passes the cache-enabled and cache-disabled procedures.

The published artifact is a FujitsuPolycom community derivative. It is not an
official NVIDIA, vLLM, local-inference-lab, B12X, Inco AI, Z.AI, or SparkCache
release artifact. The image does not contain either model checkpoint.

## Required parent

Build and publish the SparkRing GLM-5.3 runtime first. Supply its immutable
registry reference, not a tag:

```bash
parent='ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:<runtime-digest>'
```

The parent must carry the exact source, platform, transport, and license labels
verified by `build_public_image.py`.

## Build

Use an immutable SparkCache checkout. Choose a local tag only as a build handle;
the Docker image ID and registry digest are the artifact identities:

```bash
git clone https://github.com/FujitsuPolycom/sparkcache.git sparkcache
git -C sparkcache checkout --detach <immutable-sparkcache-revision>

python sparkcache/deploy/glm53_flash/build_public_image.py \
  --repository sparkcache \
  --base-image "${parent}" \
  --output-image sparkring-glm53-sparkcache:da4d7be-dflash2-bf16-arm64 \
  --output glm53-sparkcache-build-receipt.json
```

## SBOM

Generate and inspect an SPDX JSON software bill of materials with Syft before
publication:

```bash
syft scan sparkring-glm53-sparkcache:da4d7be-dflash2-bf16-arm64 \
  --output spdx-json=glm53-sparkcache.spdx.json
```

The SBOM supplements rather than replaces the immutable source pins, Git trees,
license files, and image labels.

## Private publication

Authenticate to GHCR with a token carrying `write:packages`, then publish a
semantic tag and record its digest:

```bash
python sparkcache/deploy/glm53_flash/publish_image.py \
  --image sparkring-glm53-sparkcache:da4d7be-dflash2-bf16-arm64 \
  --destination ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:da4d7be-dflash2-bf16-arm64 \
  --build-receipt glm53-sparkcache-build-receipt.json \
  --sbom glm53-sparkcache.spdx.json \
  --output glm53-sparkcache-publication-receipt.json
```

Keep the package private while its status is implemented. Pull the receipt's
`registry_digest` on all four ranks, verify one local image ID, and run the
complete TP4/DCP1 qualification. Publish the qualification receipt before
making the GHCR package public.

Public GHCR visibility is irreversible. Confirm the image contains no model
weights, credentials, site configuration, cache entries, or unlicensed source
before changing visibility.

## Required public record

A public announcement names:

- community-derivative status;
- registry digest and Linux ARM64 manifest identity;
- parent digest and local parent image ID;
- SparkRing, SparkCache, vLLM, B12X, NCCL, and InstantTensor revisions;
- every applied patch and Git tree;
- combined licenses and third-party notices;
- build command, build receipt, SPDX SBOM, and provenance attestation;
- four-Spark hardware, TP/DCP, checkpoints, DFlash depth, KV mode and size,
  scheduler settings, graph mode, and limits;
- store, restart, restore, semantic, DFlash, health, RDMA, and fatal-log results;
- unsupported configurations; and
- SparkRing and SparkCache issue trackers.
