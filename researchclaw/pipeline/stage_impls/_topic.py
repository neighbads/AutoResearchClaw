"""Stages 1-2: Topic initialization and problem decomposition."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.hardware import detect_hardware, ensure_torch_available
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._domain import _detect_domain
from researchclaw.pipeline._helpers import (
    StageResult,
    _get_evolution_overlay,
    _read_prior_artifact,
    _safe_json_loads,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _read_seed_artifact(run_dir: Path, filename: str) -> str:
    """Read a Stage 0 seed artifact if present, otherwise return empty string.

    This is intentionally lenient to preserve legacy entrypoints that call
    stage functions directly without running Stage 0 first.
    """
    found = _read_prior_artifact(run_dir, filename)
    if not isinstance(found, str):
        return ""
    return found


def _parse_seed_claims(seed_claims_raw: str) -> list[dict[str, object]]:
    data = _safe_json_loads(seed_claims_raw, [])
    if not isinstance(data, list):
        return []
    out: list[dict[str, object]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


def _format_seed_claims_bullets(
    claims: list[dict[str, object]], *, limit: int = 12
) -> str:
    lines: list[str] = []
    for c in claims[:limit]:
        cid = str(c.get("id", "")).strip() or "C????"
        strength = str(c.get("strength", "")).strip() or "INFO"
        heading = str(c.get("heading", "")).strip()
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        heading_s = f" ({heading})" if heading else ""
        lines.append(f"- {cid} [{strength}]{heading_s}: {text}")
    if not lines:
        lines.append("- (no claims parsed)")
    return "\n".join(lines)


def _topic_seed_focus_block() -> str:
    return (
        "Implementation gaps to investigate:\n"
        "- Which seed-defined requirements do not yet have an obvious implementation location?\n"
        "- Which APIs, modules, or tests need to be inspected to prove coverage?\n\n"
        "Compatibility questions to resolve:\n"
        "- Which wire formats, integrations, or versions could break if the seed requirements are enforced?\n"
        "- Which backward-compatibility constraints must shape the research plan?\n"
    )


def _problem_seed_question_block() -> str:
    return (
        "Validation Questions:\n"
        "- What evidence is needed to prove each seed claim?\n"
        "- Which claims remain ambiguous or underspecified?\n\n"
        "Implementation Questions (implementation gaps):\n"
        "- Where in the codebase should each claim be implemented or verified?\n"
        "- Which implementation gaps, APIs, call paths, or tests demonstrate missing coverage?\n\n"
        "Compatibility Questions:\n"
        "- Which existing integrations, formats, or versions could break?\n"
        "- What backward-compatibility policy should later stages evaluate?\n"
    )


def _execute_topic_init(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    topic = config.research.topic
    domains = (
        ", ".join(config.research.domains) if config.research.domains else "general"
    )

    seed_claims_raw = _read_seed_artifact(run_dir, "seed_claims.json")
    seed_open_questions_md = _read_seed_artifact(run_dir, "seed_open_questions.md")
    seed_spec_outline_md = _read_seed_artifact(run_dir, "seed_spec_outline.md")
    seed_present = bool(
        seed_claims_raw.strip()
        or seed_open_questions_md.strip()
        or seed_spec_outline_md.strip()
    )
    seed_claims = _parse_seed_claims(seed_claims_raw) if seed_present else []

    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "topic_init")
        topic_for_prompt = topic
        if seed_present:
            claims_bullets = _format_seed_claims_bullets(seed_claims, limit=8)
            topic_for_prompt = (
                f"{topic}\n\n"
                "Seed context (Stage 0 artifacts):\n\n"
                f"{seed_spec_outline_md.strip()}\n\n"
                "Seed claims (subset):\n"
                f"{claims_bullets}\n\n"
                "Seed open questions:\n\n"
                f"{seed_open_questions_md.strip()}\n\n"
                f"{_topic_seed_focus_block()}"
            )
        sp = _pm.for_stage(
            "topic_init",
            evolution_overlay=_overlay,
            topic=topic_for_prompt,
            domains=domains,
            project_name=config.project.name,
            quality_threshold=config.research.quality_threshold,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
        )
        goal_md = resp.content
    else:
        if seed_present:
            claims_bullets = _format_seed_claims_bullets(seed_claims, limit=12)
            open_questions_block = seed_open_questions_md.strip() or "_(missing)_"
            outline_block = seed_spec_outline_md.strip() or "_(missing)_"
            goal_md = f"""# Research Goal

## Topic
{topic}

## Seed Context (Stage 0)

### Seed Spec Outline
{outline_block}

### Seed Claims To Validate (subset)
{claims_bullets}

### Seed Open Questions
{open_questions_block}

### Implementation Gaps
- Identify where the seed-defined requirements should be implemented or verified in the codebase.
- Highlight modules, APIs, or tests that appear to be missing or incomplete.

### Compatibility Questions
- Track which integrations, wire formats, or versions may break if the seed requirements are enforced.
- Preserve the questions that later stages must validate around interoperability and migration.

## Scope
Investigate the topic with emphasis on validating the seed specification against code reality,
implementation gaps, and compatibility constraints.

## SMART Goal
- Specific: Validate the extracted seed claims for {topic} and map the highest-risk implementation gaps
- Measurable: Produce evidence-backed notes on claims, open questions, compatibility risks, and repo touchpoints
- Achievable: Complete through staged pipeline with static seed artifacts plus later experiment validation
- Relevant: Aligned with project {config.project.name} and the supplied seed specification
- Time-bound: Constrained by pipeline execution budget

## Constraints
- Quality threshold: {config.research.quality_threshold}
- Daily paper target: {config.research.daily_paper_count}

## Success Criteria
- Every high-priority seed claim is traced to evidence, missing implementation, or an explicit open question
- Compatibility-sensitive integrations are identified for later validation
- Downstream stages receive a goal that stays anchored to the seed materials

## Generated
{_utcnow_iso()}
"""
        else:
            goal_md = f"""# Research Goal

## Topic
{topic}

## Scope
Investigate the topic with emphasis on reproducible methods and measurable outcomes.

## SMART Goal
- Specific: Build a focused research plan for {topic}
- Measurable: Produce literature shortlist, hypotheses, experiment plan, and final paper
- Achievable: Complete through staged pipeline with gate checks
- Relevant: Aligned with project {config.project.name}
- Time-bound: Constrained by pipeline execution budget

## Constraints
- Quality threshold: {config.research.quality_threshold}
- Daily paper target: {config.research.daily_paper_count}

## Success Criteria
- At least 2 falsifiable hypotheses
- Executable experiment code and results analysis
- Revised paper passing quality gate

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "goal.md").write_text(goal_md, encoding="utf-8")

    # --- Hardware detection (GPU / MPS / CPU) ---
    hw = detect_hardware()
    (stage_dir / "hardware_profile.json").write_text(
        json.dumps(hw.to_dict(), indent=2), encoding="utf-8"
    )
    if hw.warning:
        logger.warning("Hardware advisory: %s", hw.warning)
    else:
        logger.info("Hardware detected: %s (%s, %s MB VRAM)", hw.gpu_name, hw.gpu_type, hw.vram_mb)

    # --- Optionally ensure PyTorch is available ---
    if hw.has_gpu and config.experiment.mode == "sandbox":
        torch_ok = ensure_torch_available(config.experiment.sandbox.python_path, hw.gpu_type)
        if torch_ok:
            logger.info("PyTorch is available for sandbox experiments")
        else:
            logger.warning("PyTorch could not be installed; sandbox will use CPU-only packages")
    elif hw.has_gpu and config.experiment.mode == "docker":
        logger.info("Docker sandbox: PyTorch pre-installed in container image")

    return StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=("goal.md", "hardware_profile.json"),
        evidence_refs=("stage-01/goal.md", "stage-01/hardware_profile.json"),
    )


def _execute_problem_decompose(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    goal_text = _read_prior_artifact(run_dir, "goal.md") or ""

    seed_claims_raw = _read_seed_artifact(run_dir, "seed_claims.json")
    seed_open_questions_md = _read_seed_artifact(run_dir, "seed_open_questions.md")
    seed_spec_outline_md = _read_seed_artifact(run_dir, "seed_spec_outline.md")
    seed_present = bool(
        seed_claims_raw.strip()
        or seed_open_questions_md.strip()
        or seed_spec_outline_md.strip()
    )
    seed_claims = _parse_seed_claims(seed_claims_raw) if seed_present else []

    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "problem_decompose")
        goal_for_prompt = goal_text
        if seed_present:
            claims_bullets = _format_seed_claims_bullets(seed_claims, limit=8)
            goal_for_prompt = (
                f"{goal_text.strip()}\n\n"
                "Seed spec outline:\n\n"
                f"{seed_spec_outline_md.strip()}\n\n"
                "Seed claims (subset):\n"
                f"{claims_bullets}\n\n"
                "Seed open questions:\n\n"
                f"{seed_open_questions_md.strip()}\n\n"
                f"{_problem_seed_question_block()}\n"
            )
        sp = _pm.for_stage(
            "problem_decompose",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            goal_text=goal_for_prompt,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
        )
        body = resp.content
    else:
        if seed_present:
            claims_bullets = _format_seed_claims_bullets(seed_claims, limit=10)
            open_q_lines = [
                ln.strip()
                for ln in seed_open_questions_md.splitlines()
                if ln.strip().startswith("-")
            ][:6]
            open_q_bullets = "\n".join(open_q_lines) if open_q_lines else "- (none)"

            body = f"""# Problem Decomposition

## Source
Derived from `goal.md` + Stage 0 seed artifacts for topic: {config.research.topic}

## Seed Spec Outline
{seed_spec_outline_md.strip() or "- (missing)"}

## Seed Claims (subset)
{claims_bullets}

## Seed Open Questions (subset)
{open_q_bullets}

## Sub-questions
1. Validation Questions: For each MUST/SHOULD claim, what evidence demonstrates correctness (tests, invariants, or standards citations), and which claims remain ambiguous?
2. Implementation Questions (implementation gaps): Where in the repo should each claim be implemented or verified, and which implementation gaps remain across modules, APIs, call paths, or fixtures?
3. Compatibility Questions: Which existing integrations, formats, or versions might break when enforcing the seed requirements?
4. Deployment Questions: What rollout, CI, packaging, or runtime constraints must be satisfied to land the changes safely?
5. Comparison Questions: If multiple approaches satisfy the same claim, what acceptance criteria differentiate them (security, performance, complexity, maintainability)?

## Priority Ranking
1. Validation evidence for MUST claims and unresolved ambiguities
2. Implementation mapping and missing test coverage
3. Compatibility-sensitive integrations and migration constraints
4. Deployment readiness and reproducibility requirements
5. Comparative evaluation criteria for alternative implementations

## Risks
- Seed claims may map to multiple partial implementations, making ownership ambiguous
- Compatibility requirements may conflict with strict enforcement of the seed specification

## Generated
{_utcnow_iso()}
"""
        else:
            body = f"""# Problem Decomposition

## Source
Derived from `goal.md` for topic: {config.research.topic}

## Sub-questions
1. Which problem settings and benchmarks define current SOTA?
2. Which methodological gaps remain unresolved?
3. Which hypotheses are testable under realistic constraints?
4. Which datasets and metrics best discriminate method quality?
5. Which failure modes can invalidate expected gains?

## Priority Ranking
1. Problem framing and benchmark setup
2. Gap identification and hypothesis formulation
3. Experiment and metric design
4. Failure analysis and robustness checks

## Risks
- Ambiguous task definition
- Dataset leakage or metric mismatch

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "problem_tree.md").write_text(body, encoding="utf-8")

    # IMP-35: Topic/title quality pre-evaluation
    # Quick LLM check: is the topic well-scoped for a conference paper?
    if llm is not None:
        try:
            _eval_resp = llm.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Evaluate this research topic for a top ML conference paper. "
                            "Score 1-10 on: (a) novelty, (b) specificity, (c) feasibility. "
                            "If overall score < 5, suggest a refined topic.\n\n"
                            f"Topic: {config.research.topic}\n\n"
                            "Reply as JSON: {\"novelty\": N, \"specificity\": N, "
                            "\"feasibility\": N, \"overall\": N, \"suggestion\": \"...\"}"
                        ),
                    }
                ],
                system=(
                    f"You are a senior {_detect_domain(config.research.topic, config.research.domains)[1]} "
                    f"researcher evaluating research topic quality."
                ),
            )
            _eval_data = _safe_json_loads(_eval_resp.content, {})
            if isinstance(_eval_data, dict):
                overall = _eval_data.get("overall", 10)
                if isinstance(overall, (int, float)) and overall < 5:
                    logger.warning(
                        "IMP-35: Topic quality score %s/10 — consider refining: %s",
                        overall,
                        _eval_data.get("suggestion", ""),
                    )
                else:
                    logger.info("IMP-35: Topic quality score %s/10", overall)
                (stage_dir / "topic_evaluation.json").write_text(
                    json.dumps(_eval_data, indent=2), encoding="utf-8"
                )
        except Exception:  # noqa: BLE001
            logger.debug("IMP-35: Topic evaluation skipped (non-blocking)")

    return StageResult(
        stage=Stage.PROBLEM_DECOMPOSE,
        status=StageStatus.DONE,
        artifacts=("problem_tree.md",),
        evidence_refs=("stage-02/problem_tree.md",),
    )
