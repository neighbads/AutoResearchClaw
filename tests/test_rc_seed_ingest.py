# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.executor import execute_stage
from researchclaw.pipeline.stages import Stage, StageStatus


def _config_with_seeds(
    *,
    tmp_path: Path,
    spec_path: Path | str,
    repo_path: Path | str,
    project_root: Path | None = None,
) -> RCConfig:
    data: dict[str, Any] = {
        "project": {"name": "rc-seed-test", "mode": "docs-first"},
        "research": {
            "topic": "seed ingest test",
            "domains": [],
            "daily_paper_count": 0,
            "quality_threshold": 0.0,
            "seed_spec_path": str(spec_path),
            "seed_repo_path": str(repo_path),
        },
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local", "on_gate_required": True},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {"use_memory": False, "use_message": False},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
        },
        "security": {"hitl_required_stages": []},
        "experiment": {"mode": "sandbox"},
    }
    return RCConfig.from_dict(
        data, project_root=(project_root or tmp_path), check_paths=False
    )


def _write_seed_repo(repo_path: Path) -> None:
    (repo_path / "README.md").write_text("# Seed Repo\n", encoding="utf-8")
    (repo_path / "include").mkdir(parents=True, exist_ok=True)
    (repo_path / "src").mkdir(parents=True, exist_ok=True)
    (repo_path / "include" / "crypto.h").write_text(
        "\n".join(
            [
                "#pragma once",
                "int mlkem_init(void);",
                "void tls_handshake(void);",
                "int x509_parse_cert(const unsigned char* buf, int len);",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_path / "src" / "crypto.c").write_text(
        "\n".join(
            [
                '#include "crypto.h"',
                "int mlkem_init(void) { return 0; }",
                "void tls_handshake(void) {}",
                "int x509_parse_cert(const unsigned char* buf, int len) { (void)buf; (void)len; return 0; }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_seed_ingest_stage_success_writes_all_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "seed_spec.md"
    spec_path.write_text(
        "\n".join(
            [
                "# Seed Spec",
                "",
                "## Crypto Profile",
                "This implementation MUST support ML-KEM and TLS.",
                "It SHOULD support X.509 for certificates.",
                "Refer to RFC 8446 and GB/T 38636.",
                "",
                "### OIDs",
                "The OID 1.2.840.113549.1.1.1 SHALL be recognized.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    repo_path = tmp_path / "seed_repo"
    repo_path.mkdir()
    _write_seed_repo(repo_path)

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=repo_path)
    adapters = AdapterBundle()

    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-0",
        config=config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.DONE
    stage_dir = run_dir / "stage-00"
    expected = (
        "seed_manifest.json",
        "seed_spec_outline.md",
        "seed_claims.json",
        "seed_open_questions.md",
        "seed_repo_inventory.json",
        "seed_repo_keyfiles.md",
        "seed_api_map.json",
        "seed_spec_code_alignment.md",
    )
    for name in expected:
        p = stage_dir / name
        assert p.exists(), name
        assert p.stat().st_size > 0, name


def test_seed_ingest_outputs_have_basic_structure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "seed_spec.md"
    spec_path.write_text(
        "\n".join(
            [
                "# Title",
                "## Section",
                "Clients MUST validate certificates.",
                "Servers SHOULD implement ML-DSA.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    repo_path = tmp_path / "seed_repo"
    repo_path.mkdir()
    _write_seed_repo(repo_path)

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=repo_path)
    adapters = AdapterBundle()

    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-1",
        config=config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.DONE

    stage_dir = run_dir / "stage-00"

    claims = json.loads((stage_dir / "seed_claims.json").read_text(encoding="utf-8"))
    assert isinstance(claims, list)
    assert len(claims) >= 1
    assert all(isinstance(c, dict) for c in claims)
    assert any("MUST" in str(c.get("strength", "")) or "MUST" in str(c.get("text", "")) for c in claims)

    inv = json.loads((stage_dir / "seed_repo_inventory.json").read_text(encoding="utf-8"))
    assert isinstance(inv, dict)
    assert inv.get("total_files", 0) >= 3
    by_ext = inv.get("by_extension")
    assert isinstance(by_ext, dict)
    assert by_ext.get(".h", 0) >= 1

    api = json.loads((stage_dir / "seed_api_map.json").read_text(encoding="utf-8"))
    assert isinstance(api, dict)
    functions = api.get("functions")
    assert isinstance(functions, list)
    assert any("mlkem_init" in str(fn.get("name", "")) for fn in functions)


def test_seed_ingest_fails_when_seed_spec_path_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "missing_seed_spec.md"
    repo_path = tmp_path / "seed_repo"
    repo_path.mkdir()
    _write_seed_repo(repo_path)

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=repo_path)
    adapters = AdapterBundle()

    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-missing-spec",
        config=config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.FAILED
    assert result.error
    msg = result.error.lower()
    assert "seed spec" in msg
    assert "not found" in msg or "missing" in msg


def test_seed_ingest_fails_when_seed_repo_path_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "seed_spec.md"
    spec_path.write_text("# X\n", encoding="utf-8")
    repo_path = tmp_path / "missing_repo"

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=repo_path)
    adapters = AdapterBundle()

    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-missing-repo",
        config=config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.FAILED
    assert result.error
    msg = result.error.lower()
    assert "seed repo" in msg
    assert "not found" in msg or "missing" in msg


def test_seed_ingest_fails_when_seed_spec_path_blank(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    repo_path = tmp_path / "seed_repo"
    repo_path.mkdir()
    _write_seed_repo(repo_path)

    config = _config_with_seeds(tmp_path=tmp_path, spec_path="   ", repo_path=repo_path)
    adapters = AdapterBundle()

    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-blank-spec",
        config=config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.FAILED
    assert result.error
    assert "seed spec path" in result.error.lower()
    assert "blank" in result.error.lower()


def test_seed_ingest_fails_when_seed_repo_path_blank(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "seed_spec.md"
    spec_path.write_text("# X\n", encoding="utf-8")

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=" \n")
    adapters = AdapterBundle()

    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-blank-repo",
        config=config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.FAILED
    assert result.error
    assert "seed repo path" in result.error.lower()
    assert "blank" in result.error.lower()


def test_seed_ingest_fails_when_repo_effectively_empty_under_skip_rules(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "seed_spec.md"
    spec_path.write_text("# Title\n\nClients MUST do X.\n", encoding="utf-8")

    repo_path = tmp_path / "seed_repo"
    repo_path.mkdir()
    # Repo only contains skipped metadata; should be considered empty.
    (repo_path / ".git").mkdir()
    (repo_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=repo_path)
    adapters = AdapterBundle()

    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-empty-repo",
        config=config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.FAILED
    assert result.error
    assert "seed repo" in result.error.lower()
    assert "no usable files" in result.error.lower() or "contains no files" in result.error.lower()


def test_seed_ingest_relative_seed_paths_resolve_against_project_root_even_if_cwd_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    seeds_dir = project_root / "seeds"
    seeds_dir.mkdir()

    spec_path = seeds_dir / "seed_spec.md"
    spec_path.write_text(
        "\n".join(
            [
                "# Seed Spec",
                "",
                "Clients MUST validate certificates.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    repo_path = seeds_dir / "repo"
    repo_path.mkdir()
    _write_seed_repo(repo_path)

    # Provide relative paths that are intended to resolve under project_root.
    config = _config_with_seeds(
        tmp_path=tmp_path,
        project_root=project_root,
        spec_path="seeds/seed_spec.md",
        repo_path="seeds/repo",
    )

    # Change cwd to prove execution doesn't depend on process working directory.
    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-relative",
        config=config,
        adapters=AdapterBundle(),
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.DONE
    assert (run_dir / "stage-00" / "seed_manifest.json").exists()


def test_seed_ingest_skips_llm_and_prompt_manager_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "seed_spec.md"
    spec_path.write_text("# Seed\n\nClients MUST do X.\n", encoding="utf-8")

    repo_path = tmp_path / "seed_repo"
    repo_path.mkdir()
    _write_seed_repo(repo_path)

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=repo_path)

    def boom_llm(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("LLM should not be constructed for SEED_SPEC_INGEST")

    class BoomPrompts:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PromptManager should not be constructed for SEED_SPEC_INGEST")

    monkeypatch.setattr("researchclaw.pipeline.executor.LLMClient.from_rc_config", boom_llm)
    monkeypatch.setattr("researchclaw.pipeline.executor.create_llm_client", boom_llm)
    monkeypatch.setattr("researchclaw.pipeline.executor.PromptManager", BoomPrompts)

    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-no-llm",
        config=config,
        adapters=AdapterBundle(),
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.DONE
    assert (run_dir / "stage-00" / "seed_manifest.json").exists()


def test_seed_ingest_hashes_and_ingests_full_seed_spec_not_truncated(
    tmp_path: Path,
) -> None:
    """Regression test: seed spec ingest must not truncate at 2MB."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "seed_spec.md"
    # Build a spec slightly larger than the old 2_000_000 byte limit.
    # The normative claim is at the tail so truncation would drop it.
    head = "# Seed Spec\n\n## Body\n\n"
    pad = "A" * (2_000_100)  # > 2MB
    tail = "\nTAIL REQUIREMENT: Clients MUST include tail content.\n"
    text = head + pad + tail
    spec_path.write_text(text, encoding="utf-8")

    repo_path = tmp_path / "seed_repo"
    repo_path.mkdir()
    _write_seed_repo(repo_path)

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=repo_path)
    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-full-spec",
        config=config,
        adapters=AdapterBundle(),
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.DONE

    stage_dir = run_dir / "stage-00"
    manifest = json.loads((stage_dir / "seed_manifest.json").read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert manifest["spec_sha256"] == expected_sha

    claims = json.loads((stage_dir / "seed_claims.json").read_text(encoding="utf-8"))
    assert any(c.get("strength") == "MUST" for c in claims), claims


def test_seed_ingest_repo_only_broken_symlink_is_effectively_empty_and_fails(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "seed_spec.md"
    spec_path.write_text("# Seed\n\nClients MUST do X.\n", encoding="utf-8")

    repo_path = tmp_path / "seed_repo"
    repo_path.mkdir()
    try:
        os.symlink(repo_path / "no_such_target", repo_path / "dangling_link")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not supported in this environment: {exc}")

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=repo_path)
    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-broken-symlink",
        config=config,
        adapters=AdapterBundle(),
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert result.error
    assert "no usable files" in result.error.lower()


def test_seed_ingest_repo_only_external_symlink_is_effectively_empty_and_fails(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    spec_path = tmp_path / "seed_spec.md"
    spec_path.write_text("# Seed\n\nClients MUST do X.\n", encoding="utf-8")

    outside = tmp_path / "outside.txt"
    outside.write_text("outside content\n", encoding="utf-8")

    repo_path = tmp_path / "seed_repo"
    repo_path.mkdir()
    try:
        os.symlink(outside, repo_path / "external_link")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not supported in this environment: {exc}")

    config = _config_with_seeds(tmp_path=tmp_path, spec_path=spec_path, repo_path=repo_path)
    result = execute_stage(
        Stage.SEED_SPEC_INGEST,
        run_dir=run_dir,
        run_id="run-seed-external-symlink",
        config=config,
        adapters=AdapterBundle(),
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert result.error
    assert "no usable files" in result.error.lower()
