"""Stage 0: Deterministic seed spec + repository ingestion.

This stage is intentionally static and deterministic:
- No subprocesses
- No network
- No LLM dependency for correctness

Outputs are simple normalized artifacts for downstream planning stages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import StageResult
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_NORMATIVE_RE = re.compile(
    r"\b(MUST NOT|SHALL NOT|SHOULD NOT|MUST|SHALL|SHOULD|MAY|REQUIRED|RECOMMENDED)\b"
)
_OID_RE = re.compile(r"\b\d+(?:\.\d+){2,}\b")
_REF_RE = re.compile(
    r"\b(RFC\s*\d+|GB/T\s*\d+(?:\.\d+)?|GM/T\s*\d+(?:\.\d+)?|ISO\s*\d+(?:-\d+)?)\b",
    re.IGNORECASE,
)

_PQC_KEYWORDS = (
    "ML-KEM",
    "ML-DSA",
    "TLS",
    "TLCP",
    "IPSec",
    "IKE",
    "X.509",
    "PKCS8",
    "ASN.1",
    "OID",
)

_REPO_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


def _iter_repo_files(repo_path: Path) -> list[Path]:
    """List repo files deterministically, pruning skipped directories early.

    Uses os.walk with in-place dir pruning so we do not recurse into skipped dirs.
    """
    files: list[Path] = []
    for root, dirs, filenames in os.walk(repo_path, topdown=True, followlinks=False):
        # Prune skipped directories in-place (effective for performance and correctness).
        dirs[:] = [d for d in dirs if d not in _REPO_SKIP_DIRS]
        for fn in filenames:
            p = Path(root) / fn
            # Usable-file semantics: regular, readable in-repo files only.
            # Exclude symlinks entirely (including symlinks to files, and broken symlinks).
            try:
                if p.is_symlink():
                    continue
                if not p.is_file():
                    continue
                with p.open("rb") as _h:
                    _h.read(1)
            except (OSError, PermissionError):
                continue
            files.append(p)
    files.sort(key=lambda x: x.relative_to(repo_path).as_posix())
    return files


def _validate_seed_inputs(
    spec_path_raw: str,
    repo_path_raw: str,
) -> tuple[Path, Path, list[Path]]:
    """Input validation boundary.

    Raises on invalid inputs; executor catches and converts to StageResult.FAILED.
    """
    # IMPORTANT: reject blank before Path(...) conversion.
    # Path("") becomes ".", which can lead to accidental ingestion of CWD.
    if not isinstance(spec_path_raw, str) or not spec_path_raw.strip():
        raise ValueError("Seed spec path is blank.")
    if not isinstance(repo_path_raw, str) or not repo_path_raw.strip():
        raise ValueError("Seed repo path is blank.")

    spec_path = Path(spec_path_raw)
    repo_path = Path(repo_path_raw)

    if not spec_path.exists() or not spec_path.is_file():
        raise FileNotFoundError(f"Seed spec not found: {spec_path}")
    if spec_path.stat().st_size == 0:
        raise ValueError(f"Seed spec is empty: {spec_path}")

    if not repo_path.exists() or not repo_path.is_dir():
        raise FileNotFoundError(f"Seed repo not found: {repo_path}")

    # Consider an effectively empty repo as invalid for ingestion.
    # Must use the same skip rules as _iter_repo_files().
    repo_files = _iter_repo_files(repo_path)
    if not repo_files:
        raise ValueError(f"Seed repo contains no usable files: {repo_path}")

    return spec_path, repo_path, repo_files


def _strip_fenced_blocks(md: str) -> str:
    return _FENCE_RE.sub("", md)


def _extract_heading_tree(spec_text: str) -> list[dict[str, object]]:
    """Heading extraction boundary.

    Returns a tree-like list of nodes: {"level": int, "title": str, "children": [...]}
    """
    headings: list[tuple[int, str]] = [
        (len(m.group(1)), m.group(2).strip()) for m in _HEADING_RE.finditer(spec_text)
    ]
    root: list[dict[str, object]] = []
    stack: list[dict[str, object]] = []
    for level, title in headings:
        node: dict[str, object] = {"level": level, "title": title, "children": []}
        while stack and int(stack[-1]["level"]) >= level:
            stack.pop()
        if stack:
            cast_children = stack[-1]["children"]
            assert isinstance(cast_children, list)
            cast_children.append(node)
        else:
            root.append(node)
        stack.append(node)
    return root


def _build_spec_outline(spec_text: str) -> str:
    """Spec outline boundary.

    Produces a markdown outline from headings.
    """
    tree = _extract_heading_tree(spec_text)
    lines: list[str] = ["# Seed Spec Outline", ""]

    def _walk(nodes: list[dict[str, object]], indent: int) -> None:
        for n in nodes:
            title = str(n.get("title", "")).strip()
            if not title:
                continue
            lines.append(f"{'  ' * indent}- {title}")
            children = n.get("children")
            if isinstance(children, list) and children:
                _walk(children, indent + 1)

    _walk(tree, 0)
    if len(lines) <= 2:
        lines.extend(["- (no headings detected)", ""])
    else:
        lines.append("")
    return "\n".join(lines)


def _extract_claims(spec_text: str) -> list[dict[str, object]]:
    """Claims extraction boundary.

    Heuristic: a "claim" is a line containing RFC 2119-ish normative keywords.
    """
    text = _strip_fenced_blocks(spec_text)
    lines = text.splitlines()

    current_heading = ""
    claims: list[dict[str, object]] = []
    claim_idx = 0

    for raw in lines:
        m_h = _HEADING_RE.match(raw)
        if m_h:
            current_heading = m_h.group(2).strip()
            continue

        line = raw.strip()
        if not line:
            continue

        m = _NORMATIVE_RE.search(line)
        if not m:
            continue

        strength = m.group(1).upper()
        refs = sorted(set(_REF_RE.findall(line)))
        oids = sorted(set(_OID_RE.findall(line)))
        keywords = sorted(
            {kw for kw in _PQC_KEYWORDS if kw.lower() in line.lower()}
        )

        claim_idx += 1
        claims.append(
            {
                "id": f"C{claim_idx:04d}",
                "strength": strength,
                "text": line[:500],
                "heading": current_heading,
                "refs": refs,
                "oids": oids,
                "keywords": keywords,
            }
        )

    if not claims:
        # Ensure downstream always has something to reason about.
        claims.append(
            {
                "id": "C0000",
                "strength": "INFO",
                "text": "No normative requirements detected in seed spec.",
                "heading": "",
                "refs": [],
                "oids": [],
                "keywords": [],
            }
        )
    return claims


def _extract_open_questions(spec_text: str, claims: list[dict[str, object]]) -> str:
    """Open questions generation boundary."""
    _ = spec_text
    lines: list[str] = ["# Seed Open Questions", ""]
    questions: list[str] = []

    for c in claims[:8]:
        text = str(c.get("text", "")).strip()
        strength = str(c.get("strength", "")).strip()
        heading = str(c.get("heading", "")).strip() or "General"
        kws = c.get("keywords") if isinstance(c.get("keywords"), list) else []
        kw_hint = f" ({', '.join(str(x) for x in kws)})" if kws else ""
        questions.append(
            f"- [{strength}] In '{heading}', where is this implemented in the repo{kw_hint}?  \n"
            f"  Spec: {text}"
        )

    if not questions:
        questions.append("- What are the key MUST/SHOULD requirements in the seed spec?")
        questions.append("- Which repo modules implement those requirements?")

    lines.extend(questions)
    lines.append("")
    return "\n".join(lines)


def _inventory_repo(repo_path: Path, *, repo_files: list[Path] | None = None) -> dict[str, object]:
    """Repo inventory boundary."""
    files = repo_files if repo_files is not None else _iter_repo_files(repo_path)
    by_ext: dict[str, int] = {}
    total_bytes = 0

    for p in files:
        ext = p.suffix.lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass

    # Keyfiles heuristic: well-known metadata + include/ + crypto-ish names
    keyfiles: list[str] = []
    want_names = {"readme.md", "license", "copying", "cmakelists.txt", "makefile"}
    for p in files:
        rel = p.relative_to(repo_path).as_posix()
        name = p.name.lower()
        if name in want_names:
            keyfiles.append(rel)
            continue
        if rel.startswith("include/") and p.suffix.lower() in {".h", ".hpp"}:
            keyfiles.append(rel)
            continue
        if any(tok in name for tok in ("tls", "x509", "kem", "dsa", "oid", "asn1", "asn.1")):
            keyfiles.append(rel)

    keyfiles = sorted(set(keyfiles))

    return {
        "root": str(repo_path),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "by_extension": {k: by_ext[k] for k in sorted(by_ext.keys())},
        "key_files": keyfiles,
    }


def _read_text_lossy(path: Path, *, limit_bytes: int = 256_000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > limit_bytes:
        data = data[:limit_bytes]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _build_repo_keyfiles_md(repo_path: Path, repo_inventory: dict[str, object]) -> str:
    key_files = repo_inventory.get("key_files")
    if not isinstance(key_files, list):
        key_files = []
    lines: list[str] = ["# Seed Repo Key Files", ""]
    if not key_files:
        lines.append("_No key files detected by heuristic._")
        lines.append("")
        return "\n".join(lines)

    for rel in key_files[:50]:
        rel_s = str(rel)
        p = repo_path / rel_s
        snippet = _read_text_lossy(p, limit_bytes=16_384)
        snippet_lines = [ln.rstrip() for ln in snippet.splitlines()[:30]]
        lines.append(f"## `{rel_s}`")
        if snippet_lines:
            lines.append("")
            lines.append("```")
            lines.extend(snippet_lines)
            lines.append("```")
        else:
            lines.append("")
            lines.append("_Unable to read file._")
        lines.append("")
    return "\n".join(lines)


def _extract_api_map(repo_path: Path, *, repo_files: list[Path] | None = None) -> dict[str, object]:
    """API map extraction boundary.

    Heuristic: scan C/C++ headers and sources for function declarations/definitions.
    """
    files = repo_files if repo_files is not None else _iter_repo_files(repo_path)
    candidates = [p for p in files if p.suffix.lower() in {".h", ".hpp", ".c", ".cc", ".cpp"}]

    proto_re = re.compile(
        r"^\s*(?:extern\s+)?(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w\s\*]*\s+"
        r"([A-Za-z_]\w*)\s*\([^;{]*\)\s*;",
        re.MULTILINE,
    )
    def_re = re.compile(
        r"^\s*(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w\s\*]*\s+"
        r"([A-Za-z_]\w*)\s*\([^;]*\)\s*\{",
        re.MULTILINE,
    )
    blacklist = {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "sizeof",
    }

    fn_to_files: dict[str, set[str]] = {}
    for p in candidates:
        rel = p.relative_to(repo_path).as_posix()
        text = _read_text_lossy(p)
        for m in proto_re.finditer(text):
            name = m.group(1)
            if name in blacklist:
                continue
            fn_to_files.setdefault(name, set()).add(rel)
        for m in def_re.finditer(text):
            name = m.group(1)
            if name in blacklist:
                continue
            fn_to_files.setdefault(name, set()).add(rel)

    functions = [
        {"name": name, "files": sorted(fn_to_files[name])}
        for name in sorted(fn_to_files.keys())
    ]
    return {
        "root": str(repo_path),
        "functions": functions,
        "function_count": len(functions),
    }


def _build_alignment(
    claims: list[dict[str, object]],
    repo_inventory: dict[str, object],
    api_map: dict[str, object],
) -> str:
    """Spec-code alignment generation boundary."""
    _ = repo_inventory
    functions = api_map.get("functions")
    if not isinstance(functions, list):
        functions = []

    fn_names = [str(f.get("name", "")) for f in functions if isinstance(f, dict)]

    lines: list[str] = ["# Seed Spec-Code Alignment", ""]
    lines.append("| Claim | Strength | Heading | Candidate APIs |")
    lines.append("| --- | --- | --- | --- |")

    for c in claims[:50]:
        cid = str(c.get("id", ""))
        strength = str(c.get("strength", ""))
        heading = str(c.get("heading", ""))
        text = str(c.get("text", ""))

        # Candidate match: look for short crypto-ish substrings in function names.
        needles: set[str] = set()
        for kw in _PQC_KEYWORDS:
            if kw.lower() in text.lower():
                needles.add(re.sub(r"[^a-z0-9]", "", kw.lower()))
        for tok in ("tls", "x509", "kem", "dsa", "oid", "asn1", "cert"):
            if tok in text.lower():
                needles.add(tok)

        cands: list[str] = []
        if needles:
            for fn in fn_names:
                fn_l = fn.lower()
                if any(n in fn_l for n in needles):
                    cands.append(fn)
        cands = sorted(set(cands))[:6]
        cands_s = ", ".join(cands) if cands else "(none found)"
        lines.append(f"| {cid} | {strength} | {heading} | {cands_s} |")

    lines.append("")
    return "\n".join(lines)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8", errors="replace"))


def _fingerprint_repo(repo_path: Path, *, repo_files: list[Path] | None = None) -> str:
    """Deterministic repo fingerprint (paths + bytes)."""
    h = hashlib.sha256()
    files = repo_files if repo_files is not None else _iter_repo_files(repo_path)
    for p in files:
        rel = p.relative_to(repo_path).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        try:
            data = p.read_bytes()
        except OSError:
            data = b""
        h.update(hashlib.sha256(data).digest())
        h.update(b"\n")
    return h.hexdigest()


def _execute_seed_spec_ingest(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """Execute Stage 0: ingest a seed spec + seed repo into normalized artifacts."""
    _ = run_dir
    _ = adapters
    _ = llm
    _ = prompts

    spec_path, repo_path, repo_files = _validate_seed_inputs(
        config.research.seed_spec_path, config.research.seed_repo_path
    )

    # Canonical seed spec ingest must use the full file contents (no truncation),
    # since Stage 0 artifacts are the downstream source of truth.
    try:
        spec_bytes = spec_path.read_bytes()
    except OSError as exc:
        raise OSError(f"Failed to read seed spec: {spec_path}") from exc
    try:
        spec_text = spec_bytes.decode("utf-8")
    except UnicodeDecodeError:
        spec_text = spec_bytes.decode("utf-8", errors="replace")
    if not spec_text.strip():
        raise ValueError(f"Seed spec is empty after decoding: {spec_path}")

    outline_md = _build_spec_outline(spec_text)
    claims = _extract_claims(spec_text)
    open_questions_md = _extract_open_questions(spec_text, claims)

    repo_inventory = _inventory_repo(repo_path, repo_files=repo_files)
    repo_keyfiles_md = _build_repo_keyfiles_md(repo_path, repo_inventory)
    api_map = _extract_api_map(repo_path, repo_files=repo_files)
    alignment_md = _build_alignment(claims, repo_inventory, api_map)

    manifest = {
        "stage": int(Stage.SEED_SPEC_INGEST),
        "spec_path": str(spec_path),
        "repo_path": str(repo_path),
        "spec_sha256": _sha256_bytes(spec_bytes),
        "repo_fingerprint_sha256": _fingerprint_repo(repo_path, repo_files=repo_files),
        "claims_count": len(claims),
        "repo_total_files": int(repo_inventory.get("total_files", 0) or 0),
        "api_function_count": int(api_map.get("function_count", 0) or 0),
    }

    # Write artifacts (all non-empty by construction).
    (stage_dir / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (stage_dir / "seed_spec_outline.md").write_text(outline_md, encoding="utf-8")
    (stage_dir / "seed_claims.json").write_text(
        json.dumps(claims, indent=2, sort_keys=True), encoding="utf-8"
    )
    (stage_dir / "seed_open_questions.md").write_text(
        open_questions_md, encoding="utf-8"
    )
    (stage_dir / "seed_repo_inventory.json").write_text(
        json.dumps(repo_inventory, indent=2, sort_keys=True), encoding="utf-8"
    )
    (stage_dir / "seed_repo_keyfiles.md").write_text(
        repo_keyfiles_md, encoding="utf-8"
    )
    (stage_dir / "seed_api_map.json").write_text(
        json.dumps(api_map, indent=2, sort_keys=True), encoding="utf-8"
    )
    (stage_dir / "seed_spec_code_alignment.md").write_text(
        alignment_md, encoding="utf-8"
    )

    artifacts = (
        "seed_manifest.json",
        "seed_spec_outline.md",
        "seed_claims.json",
        "seed_open_questions.md",
        "seed_repo_inventory.json",
        "seed_repo_keyfiles.md",
        "seed_api_map.json",
        "seed_spec_code_alignment.md",
    )
    evidence = tuple(f"stage-00/{a}" for a in artifacts)

    return StageResult(
        stage=Stage.SEED_SPEC_INGEST,
        status=StageStatus.DONE,
        artifacts=artifacts,
        evidence_refs=evidence,
    )
