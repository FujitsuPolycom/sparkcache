"""Pinned identities are deterministic and cannot alias changed checkpoint inputs."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sparkcache.checkpoint_identity import SCHEMA, derive_checkpoint_identity, main
from deploy.glm53_flash.profile import immutable_revision_identity

FIXTURES = Path(__file__).resolve().parents[1] / "deploy/checkpoint_identity"


@pytest.mark.parametrize("name,expected", [
    ("csf", "7906c93c59bf153168170b601cf36ae7cdd31927797dc4608320fe1ba8ec2c78"),
    ("step5500", "759a6ce3138e95cddaa4dc8e53480274d2a3e75154cf1893ff275237b1ded2b6"),
])
def test_qwen_installer_identities_bind_pins_and_embedded_bf16_mtp(name, expected):
    raw = (FIXTURES / f"qwen38-{name}-pins.json").read_bytes()
    pins = json.loads(raw)
    value = json.loads((FIXTURES / f"qwen38-{name}-inputs.json").read_text())
    receipt = json.loads((FIXTURES / f"qwen38-{name}-receipt.json").read_text())
    assert value["artifacts"] == {"manifest_sha256": hashlib.sha256(raw).hexdigest()}
    assert value["repository"] == pins["repository"]
    assert value["revision"] == pins["revision"]
    assert value["model_config"] == {"sha256": pins["files"]["config.json"]["sha256"]}
    assert value["draft"]["mode"] == "embedded"
    assert value["draft"]["config"]["kv_cache_dtype"] == "auto"
    assert receipt["inputs"] == value
    assert receipt["checkpoint_sha256"] == derive_checkpoint_identity(value) == expected
    assert expected != "036c2f7994466d32514130f0417ccd117705ca01e2038c1ce5745f84813829ba"


def inputs():
    return json.loads((FIXTURES / "glm53-csf-inputs.json").read_text())


def test_canonical_envelope_matches_independent_serialization():
    value = inputs()
    expected = hashlib.sha256(json.dumps({"schema": SCHEMA, "inputs": value}, sort_keys=True,
                                        separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    assert derive_checkpoint_identity(value) == expected
    assert derive_checkpoint_identity(dict(reversed(list(value.items())))) == expected


@pytest.mark.parametrize("field,replacement", [
    ("repository", "another/model"), ("revision", "1" * 40),
    ("artifacts", {"files": {"model.safetensors": "2" * 64}}),
    ("quantization", {"method": "nvfp4_csf", "encoding": "safetensors"}),
    ("model_config", {"sha256": "3" * 64}),
    ("draft", {"mode": "embedded", "config": {"num_speculative_tokens": 4}}),
    ("draft", {"mode": "none"}),
    ("draft", {"mode": "separate", "config": {"tokens": 3}, "checkpoint_sha256": "4" * 64}),
])
def test_each_identity_dimension_changes_namespace(field, replacement):
    value = inputs()
    changed = copy.deepcopy(value)
    changed[field] = replacement
    assert derive_checkpoint_identity(changed) != derive_checkpoint_identity(value)


def test_file_order_is_irrelevant_but_name_and_content_are_bound():
    value = inputs()
    value["artifacts"] = {"files": {"a": "1" * 64, "b": "2" * 64}}
    first = derive_checkpoint_identity(value)
    value["artifacts"]["files"] = {"b": "2" * 64, "a": "1" * 64}
    assert derive_checkpoint_identity(value) == first
    value["artifacts"]["files"]["a"] = "3" * 64
    assert derive_checkpoint_identity(value) != first
    value["artifacts"]["files"] = {"c": "1" * 64, "b": "2" * 64}
    assert derive_checkpoint_identity(value) != first


@pytest.mark.parametrize("field,replacement", [
    ("revision", "main"), ("revision", "dec48abd"), ("revision", "A" * 40),
    ("artifacts", {}), ("artifacts", {"files": {}}),
    ("artifacts", {"files": {"../weights": "1" * 64}}),
    ("artifacts", {"files": {"C:/weights": "1" * 64}}),
    ("artifacts", {"files": {"/weights": "1" * 64}}),
    ("artifacts", {"files": {"a//b": "1" * 64}}),
    ("artifacts", {"files": {"a": "abc"}}),
    ("quantization", {}), ("model_config", {"values": {}}),
    ("model_config", {"values": {"x": float("nan")}}),
    ("model_config", {"values": {1: "value"}}),
    ("draft", {}), ("draft", {"mode": "none", "config": {}}),
    ("draft", {"mode": "separate", "config": {"tokens": 3}}),
])
def test_incomplete_or_ambiguous_inputs_are_rejected(field, replacement):
    value = inputs()
    value[field] = replacement
    with pytest.raises(ValueError):
        derive_checkpoint_identity(value)


def test_missing_and_unknown_top_level_fields_are_rejected():
    for key in inputs():
        value = inputs()
        del value[key]
        with pytest.raises(ValueError):
            derive_checkpoint_identity(value)
    with pytest.raises(ValueError):
        derive_checkpoint_identity({**inputs(), "checkpoint_sha256": "1" * 64})


def test_cli_receipt_and_exclusive_output(tmp_path, capsys):
    source = FIXTURES / "glm53-csf-inputs.json"
    assert main(["--input", str(source)]) == 0
    stdout = capsys.readouterr().out
    assert json.loads(stdout)["checkpoint_sha256"] == derive_checkpoint_identity(inputs())
    output = tmp_path / "receipt.json"
    assert main(["--input", str(source), "--output", str(output)]) == 0
    assert output.read_text() == stdout
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--output", str(output)])
    assert output.read_text() == stdout


def test_cli_rejects_duplicate_json_keys(tmp_path):
    source = tmp_path / "duplicate.json"
    source.write_text('{"repository":"one","repository":"two"}')
    with pytest.raises(SystemExit):
        main(["--input", str(source)])


@pytest.mark.parametrize("name", ["glm53-csf", "qwen38-step4000"])
def test_recorded_values_remain_pinned_and_attempts_are_reproducible(name):
    records = json.loads((FIXTURES / "recorded-identities.json").read_text())
    expected_recorded = {
        "glm53-csf": "6af0343976eefbfa0fb59d87befcf6d10f02fef90ba0e7881dd1201fe0ecd76e",
        "qwen38-step4000": "036c2f7994466d32514130f0417ccd117705ca01e2038c1ce5745f84813829ba",
    }
    record = records[name]
    raw = (FIXTURES / (name + "-pins.json")).read_bytes()
    pins = json.loads(raw)
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")

    def sha(value):
        return hashlib.sha256(value).hexdigest()
    manifest = {"schema": "sparkcache-checkpoint-manifest/v1", "repository": pins["repository"],
                "revision": pins["revision"], "files": [
                    {"path": path, "bytes": row["size"], "sha256": row["sha256"]}
                    for path, row in sorted(pins["files"].items())]}
    value = json.loads((FIXTURES / (name + "-inputs.json")).read_text())
    assert value["artifacts"]["manifest_sha256"] == sha(raw)
    assert record["recorded_checkpoint_sha256"] == expected_recorded[name]
    attempts = {
        "hf-revision-v1": immutable_revision_identity(pins["repository"], pins["revision"]),
        "manifest-file-sha256": sha(raw),
        "canonical-pin-manifest-sha256": sha(canonical(pins)),
        "deepseek-manifest-v1-all-pinned-files": sha(canonical(manifest)),
        SCHEMA: derive_checkpoint_identity(value),
    }
    assert attempts == record["attempts"]
    assert expected_recorded[name] not in attempts.values()
