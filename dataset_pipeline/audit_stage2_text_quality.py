#!/usr/bin/env python3
"""Fail closed on synthetic Stage 2 text that is not publication quality."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


META_TEXT = re.compile(
    r"^(?:A note mentions|The source describes|Information about|"
    r"This update refers to|A source item names|One post mentions|"
    r"The text mentions|A brief note records|This source names|"
    r"The item references|A short post notes|The record includes|"
    r"One source entry states|The archive mentions|A written update names|"
    r"This entry covers|One note records|The post references|"
    r"A source text mentions|This record names|An entry refers to|"
    r"The source item notes|A brief source note mentions|"
    r"This historical text references|The available text includes)\b",
    re.IGNORECASE,
)

TRAILING_FUNCTION_WORD = re.compile(
    # Copulas can legitimately close an embedded clause (for example,
    # "who the breadwinner is").  Short-fragment checks catch genuinely
    # clipped outputs such as "final s/o is" without rejecting those clauses.
    r"\b(?:a|an|the|and|or|as|at|by|for|from|in|into|of|on|to|with)\.?$",
    re.IGNORECASE,
)
COMPLETE_IN_PHRASAL_VERB = re.compile(
    r"\b(?:jump(?:s|ed|ing)?|step(?:s|ped|ping)?|join(?:s|ed|ing)?|"
    r"chime(?:s|d)?|chip(?:s|ped|ping)?|check(?:s|ed|ing)?|"
    r"weigh(?:s|ed|ing)?)\s+in\.?$",
    re.IGNORECASE,
)


def has_incomplete_trailing_phrase(text: str) -> bool:
    if not TRAILING_FUNCTION_WORD.search(text):
        return False
    # "Jumping in" and similar phrasal verbs are complete, even without a period.
    return not COMPLETE_IN_PHRASAL_VERB.search(text)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(records: list[dict], generator_report: dict) -> dict:
    by_label = Counter(record.get("label_name") for record in records)
    llm_audits = {
        item["base_profile_id"]: item
        for item in generator_report.get("llm_bot_audits", [])
    }
    agent_audits = {
        item["base_profile_id"]: item
        for item in generator_report.get("agent_audits", [])
    }
    errors: list[str] = []
    stats = Counter()
    for record in records:
        base_id = record.get("base_profile_id")
        label = record.get("label_name")
        posts = record.get("posts") or []
        if len(posts) != 25:
            errors.append(f"{base_id}:{label}: expected 25 posts, got {len(posts)}")
        for index, post in enumerate(posts):
            text = str(post.get("text") or "").strip()
            if META_TEXT.search(text):
                errors.append(f"{base_id}:{label}: meta-text at post {index + 1}: {text[:100]}")
                stats[f"{label}:meta_text"] += 1
            words = re.findall(r"[A-Za-z]+", text.lower())
            if len(words) >= 4 and len(set(words)) <= 2:
                errors.append(f"{base_id}:{label}: repeated short words at post {index + 1}")
                stats[f"{label}:repeated_words"] += 1
            action_type = str(post.get("action_type") or "")
            if label == "full_stack_agent" and action_type != "repost" and len(words) < 5:
                errors.append(f"{base_id}:{label}: short fragment at post {index + 1}: {text[:100]}")
                stats[f"{label}:short_fragments"] += 1
            # Reposts are deterministic copies of the observed OASIS source.
            # Preserve source fidelity even when the source itself ends in a
            # function word (for example, a clipped historical post).
            if (
                label == "full_stack_agent"
                and action_type != "repost"
                and has_incomplete_trailing_phrase(text)
            ):
                errors.append(f"{base_id}:{label}: incomplete trailing phrase at post {index + 1}: {text[:100]}")
                stats[f"{label}:trailing_fragments"] += 1
        if label == "llm_bot":
            audit_item = llm_audits.get(base_id)
            if audit_item is None:
                errors.append(f"{base_id}:llm_bot: missing generator audit")
                continue
            rejections = audit_item.get("quality_rejections") or {}
            fallback_count = int(rejections.get("controlled_fallback_used", 0)) + int(
                rejections.get("closed_world_surface_used", 0)
            ) + int(audit_item.get("deterministic_fallback_count", 0))
            stats["llm:fallback_posts"] += fallback_count
            if fallback_count > 5:
                errors.append(f"{base_id}:llm_bot: {fallback_count}/25 fallback posts (limit 5)")
        elif label == "full_stack_agent":
            audit_item = agent_audits.get(base_id)
            if audit_item is None:
                errors.append(f"{base_id}:full_stack_agent: missing generator audit")
            else:
                # Controller-selected reposts are valid environment actions;
                # only deterministic text repairs are a text-quality fallback.
                controller_count = int(audit_item.get("controller_fallback_action_count", 0))
                text_count = int(audit_item.get("deterministic_text_fallback_action_count", 0))
                neutral_count = int(audit_item.get("no_claim_surface_repair_action_count", 0))
                stats["agent:controller_fallback_actions"] += controller_count
                stats["agent:deterministic_text_fallback_actions"] += text_count
                stats["agent:no_claim_surface_repairs"] += neutral_count
                if text_count > 5:
                    errors.append(f"{base_id}:full_stack_agent: {text_count}/25 deterministic text fallbacks (limit 5)")
                if neutral_count > 0:
                    errors.append(f"{base_id}:full_stack_agent: neutral surface repairs are not allowed in the final release")
    if by_label != {"llm_bot": len(llm_audits), "full_stack_agent": len(agent_audits)}:
        errors.append(f"record/report label counts disagree: {dict(by_label)}")
    return {
        "schema_version": "quadbot-v3-stage2-text-quality-1",
        "record_count": len(records),
        "label_counts": dict(sorted(by_label.items())),
        "counts": dict(sorted(stats.items())),
        "error_count": len(errors),
        "errors_preview": errors[:30],
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--generator-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_jsonl(args.records), json.loads(args.generator_report.read_text(encoding="utf-8")))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
