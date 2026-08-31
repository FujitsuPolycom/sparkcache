# Publish the GLM-5.3 SparkCache image

This procedure documents the superseded da4d7be overlay publication path. The
canonical public Jovian Judgement r7 ARM64 image route is
[`JJ_R7_ARM64_IMAGE.md`](JJ_R7_ARM64_IMAGE.md). Do not substitute the historical
digests below into that quickstart.

Status: **implemented** for image construction and private GHCR publication.
An exact registry digest remains unqualified until all four TP4/DCP1 ranks pull
that digest and pass persistent store, coordinated restart, verified restore,
exact semantic output, cache-disabled serving, health, and fatal-log checks.

The qualified public artifact is
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943`.
Its recorded parent is
`ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd`.
Another build remains implemented until its own digest passes the named checks.

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

Install Syft before publication. `publish_image.py` invokes Syft against the
immutable image ID recorded by the build receipt. The `--sbom` argument names a
new output path and must not identify an existing file.

The generated SPDX JSON document supplements rather than replaces immutable
source pins, Git trees, license files, and image labels. The publication receipt
records its SHA-256 and the source image ID.

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

Keep each registry digest private until that digest has a TP4/DCP1 qualification
receipt. Pull the receipt's `registry_digest` on all four ranks, verify each
local image ID, and run the named serving checks above. Publish the qualification
receipt before making the GHCR package public.

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
