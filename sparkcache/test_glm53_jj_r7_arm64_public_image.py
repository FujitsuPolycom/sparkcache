from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "evidence"
    / "glm53-jj-r7-arm64"
    / "jj-r7-arm64-public-image-c4-smoke.json"
)
IMAGE_RECORD = ROOT / "deploy" / "glm53_flash" / "JJ_R7_ARM64_IMAGE.md"
README = ROOT / "README.md"
PACKAGE_README = ROOT / "sparkcache" / "README.md"
DEPLOYMENT_GUIDE = ROOT / "deploy" / "glm53_flash" / "README.md"
HISTORICAL_IMAGE = ROOT / "deploy" / "glm53_flash" / "IMAGE_ANNOUNCEMENT.md"
HISTORICAL_PUBLISHING = ROOT / "deploy" / "glm53_flash" / "PUBLISHING.md"
EXPLAINER = ROOT / "docs" / "sparkcache-prefix-explainer.html"

MANIFEST = (
    "sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5"
)
IMAGE_CONFIG = (
    "sha256:6af83baabb239db6b05e379401daf93c8f51694f81483c2781f6014c30e31db4"
)
PARENT = (
    "sha256:11922064b342de1fc98f0ef85e6648843c8fa7eb3e4f4353c6ad82d6e457dde0"
)
PARENT_CONFIG = (
    "sha256:8cff7a250f16bfb89df23d29f9233dbb1c700a780dcec86a64c535a71aee88be"
)
QUICKSTART = (
    "https://github.com/FujitsuPolycom/sparkring/blob/main/"
    "docs/GLM53_JJ_R7_GB10_TP4_QUICKSTART.md"
)


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_public_arm64_receipt_binds_artifact_and_status() -> None:
    receipt = _receipt()

    assert receipt["schema"] == "sparkcache-public-arm64-smoke/v1"
    assert receipt["status"] == "implemented"
    assert receipt["scope"] == {
        "tp4_smoke_verified": True,
        "general_qualification": False,
        "topology": "TP4/DCP1 on four NVIDIA GB10 systems",
        "platform": "linux/arm64",
    }
    assert receipt["artifact"] == {
        "repository": "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache",
        "manifest_digest": MANIFEST,
        "image_config_digest": IMAGE_CONFIG,
        "parent_manifest_digest": PARENT,
        "parent_image_config_digest": PARENT_CONFIG,
        "contains_model_weights": False,
    }
    assert receipt["quickstart"] == QUICKSTART


def test_public_arm64_receipt_binds_runtime_and_model_sources() -> None:
    receipt = _receipt()
    sources = receipt["sources"]
    models = receipt["models"]

    assert sources["vllm"] == {
        "upstream_role": "Local Inference Lab Jovian Judgement runtime",
        "public_repository": "FujitsuPolycom/vllm",
        "commit": "331573d20bd47e78327ed8d8b4d2e6d350bbb1ab",
        "tree": "927f52a0085bcecfd2ba679e5abebe1a62623daf",
    }
    assert sources["b12x"] == {
        "commit": "6255090a03b12c3f7d552102a02fac0b542fb8c9",
        "tree": "0bb58d0dcc10e29e00ff9850c0d719fca1aba5ad",
    }
    assert sources["nccl_library_sha256"] == (
        "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3"
    )
    assert sources["sparkcache"] == {
        "commit": "dcbe040d339f243621163b0c6ed4ce96462403d8",
        "tree": "861562a7f5cb867be4313a2979027bc4f499cb31",
        "deployable_source_sha256": (
            "9cf50afd04e385975a487a0129645bd294e0395012424995569a9b50a7c447f1"
        ),
        "cuda_placement_sha256": (
            "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c"
        ),
    }
    assert models["target"]["revision"] == (
        "520de24eabf507659eaef7c70f14fd584527facc"
    )
    assert models["draft"]["revision"] == (
        "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"
    )
    assert models["draft"]["speculative_depth"] == 7


def test_public_arm64_receipt_distinguishes_composition_from_lineage_labels() -> None:
    labels = _receipt()["observed_labels"]

    assert labels["active_composition"] == {
        "org.sparkring.vllm.sparkcache-composition": (
            "331573d20bd47e78327ed8d8b4d2e6d350bbb1ab"
        ),
        "org.sparkring.vllm.tree": "927f52a0085bcecfd2ba679e5abebe1a62623daf",
        "org.sparkring.b12x.composition": (
            "6255090a03b12c3f7d552102a02fac0b542fb8c9"
        ),
        "org.sparkring.b12x.tree": "0bb58d0dcc10e29e00ff9850c0d719fca1aba5ad",
        "org.sparkring.nccl.sha256": (
            "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3"
        ),
        "org.sparkring.runtime.status": "tp4-smoke-verified",
        "org.sparkcache.commit": "dcbe040d339f243621163b0c6ed4ce96462403d8",
        "org.sparkcache.tree": "861562a7f5cb867be4313a2979027bc4f499cb31",
        "org.sparkcache.source-sha256": (
            "9cf50afd04e385975a487a0129645bd294e0395012424995569a9b50a7c447f1"
        ),
        "org.sparkcache.cuda-placement-sha256": (
            "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c"
        ),
        "org.sparkcache.publication-schema": "page-tail-cow-v1",
        "org.sparkcache.default": "disabled-until-explicitly-configured",
        "org.sparkring.parent.image": PARENT_CONFIG,
    }
    assert labels["inherited_lineage_not_active_composition"] == [
        "ai.vllm.build.commit",
        "org.jovian.vllm.commit",
        "org.jovian.b12x.commit",
        "org.sparkring.native-parent.vllm",
        (
            "org.glm53.dflash2.checkpoint-revision="
            "b6d33aa93fc1ac5b23a88251a1c0ce0bfe2ad17c"
        ),
        "org.glm53.dflash2.mxfp8-quant-plumbing=v2",
    ]

    image_record = IMAGE_RECORD.read_text(encoding="utf-8")
    assert "not the mounted BF16" in image_record
    assert "incoai/GLM-5.3-Flash-DFlash2@dc77ff1c" in image_record


def test_public_arm64_receipt_binds_bounded_c4_smoke() -> None:
    smoke = _receipt()["smoke"]
    publication = smoke["publication"]
    restore = smoke["restart_restore"]

    assert publication == {
        "fresh_cache_root": True,
        "concurrency": 4,
        "expected_codewords": ["red", "blue", "green", "black"],
        "exact_results": True,
        "result_set_sha256": (
            "edb9c082fc6fe1b99004fa4c04d9e4b53d0525fe5410313ba13f18f2dc09ffbc"
        ),
        "root_bytes_per_rank": 605690671,
        "same_root_bytes_on_all_ranks": True,
        "manifests_per_rank": 4,
        "clear_completion_marker_per_rank": True,
    }
    assert restore == {
        "full_engine_restart": True,
        "readiness_inference_completed": True,
        "concurrency": 4,
        "exact_results": True,
        "result_set_sha256": (
            "02a0c0fafa95294008cd1b9a8a6269dabc0d161c10c307bc1922f1b9aa20c100"
        ),
        "external_restore_requests": 4,
        "external_restore_fraction": 1.0,
        "client_seconds": {"minimum": 0.561595, "maximum": 1.582937},
        "worker_cache_service_ms": {"minimum": 277, "maximum": 394},
    }
    assert "shared-base read coalescing" in smoke["not_established"]
    assert "general deployment qualification" in smoke["not_established"]


def test_public_arm64_docs_route_by_digest_without_overclaiming() -> None:
    image_record = " ".join(IMAGE_RECORD.read_text(encoding="utf-8").split())
    readme = " ".join(README.read_text(encoding="utf-8").split())

    for text in (image_record, readme):
        assert MANIFEST in text
        assert QUICKSTART in text
        assert "implemented and tp4 smoke-verified" in text.casefold()
        assert "not generally qualified" in text.casefold()

    assert IMAGE_CONFIG in image_record
    assert PARENT in image_record
    assert PARENT_CONFIG in image_record
    assert "does not prove shared-base read coalescing" in image_record
    assert "not generally qualified" in readme.casefold()


def test_canonical_docs_route_to_public_arm64_manifest() -> None:
    documents = (README, PACKAGE_README, DEPLOYMENT_GUIDE, IMAGE_RECORD)

    for path in documents:
        text = path.read_text(encoding="utf-8")
        assert MANIFEST in text, path
        assert "not generally qualified" in text.casefold(), path

    readme = README.read_text(encoding="utf-8")
    deployment = DEPLOYMENT_GUIDE.read_text(encoding="utf-8")
    assert "The canonical public image is" in readme
    assert "canonical public image route" in deployment
    assert "export GLM53_IMAGE='" + (
        "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@" + MANIFEST
    ) in readme
    assert 'docker pull "$GLM53_IMAGE"' in readme


def test_historical_image_docs_cannot_be_mistaken_for_current_route() -> None:
    announcement = HISTORICAL_IMAGE.read_text(encoding="utf-8")
    publishing = HISTORICAL_PUBLISHING.read_text(encoding="utf-8")

    assert announcement.startswith("# Historical GLM-5.3")
    assert "canonical public image and run procedure" in announcement
    assert "superseded da4d7be overlay publication path" in publishing
    assert "JJ_R7_ARM64_IMAGE.md" in announcement
    assert "JJ_R7_ARM64_IMAGE.md" in publishing


def test_canonical_docs_exclude_superseded_local_image_identities() -> None:
    canonical = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (README, PACKAGE_README, DEPLOYMENT_GUIDE, EXPLAINER)
    )

    for stale in (
        "sha256:f2723a71b49509294072f5886b4fe081ac1f87dd1f931cc3cb8f538bc3eb037d",
        "sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e",
        "sha256:df4e09a32cdbf1c0e69cc7c4c9e95d890d6c7a1e3eaac84f969912a16fd27dd3",
        "sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818",
    ):
        assert stale not in canonical

    assert "Physical-page delta publication and direct restore | **research-only**" not in canonical
    assert "participants=7" not in canonical
    assert "avoided_base_reads=6" not in canonical


def test_front_door_names_an_existing_pypi_artifact() -> None:
    readme = README.read_text(encoding="utf-8")
    prose = " ".join(readme.split())

    assert "sparkcache[connector]==0.1.0a2" in readme
    assert "sparkcache==0.1.0a3" not in readme
    assert "sparkcache[connector]==0.1.0a3" not in readme
    assert "Live serving status is artifact-bound" in prose
