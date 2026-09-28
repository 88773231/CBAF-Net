#!/usr/bin/env python3
"""Independently audit paired QuadBot-v3 LLM Bot and Agent records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


URL_RE = re.compile(
    r"https?:/{1,2}\S+|www\.\S+|\b(?:[A-Z0-9-]+\.)+[A-Z]{2,}(?:/\S*)?",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
FIRST_PERSON_SINGULAR_RE = re.compile(
    r"(?<![A-Za-z])(?:i|i'm|i've|i'd|i'll|me|my|mine)(?![A-Za-z])",
    re.IGNORECASE,
)
ORGANIZATION_MARKERS = (
    "association",
    "center",
    "centre",
    "company",
    "foundation",
    "institute",
    "institution",
    "laboratory",
    "non-profit",
    "nonprofit",
    "organization",
    "official account",
    "research group",
    "university",
)
GENERATOR_KEYS = {
    "generator_family",
    "temperature",
    "top_p",
    "repetition_penalty",
    "max_new_tokens",
    "do_sample",
    "generation_rounds",
    "items_per_round",
    "grounding_validator_version",
}
GENERIC_TEMPLATE_PATTERNS = (
    ("cant_wait", re.compile(r"\bcan(?:t|\s+t) wait\b")),
    ("excited_to", re.compile(r"\bexcited to\b")),
    ("lets_continue", re.compile(r"\blet(?:s|\s+s) continue\b")),
    ("stay_tuned", re.compile(r"\bstay tuned\b")),
    ("initial_absolutely", re.compile(r"^absolutely\b")),
    ("initial_great_post", re.compile(r"^great post\b")),
)
HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
NUMBER_RE = re.compile(r"\b\d+(?:[.:/-]\d+)*\b")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9'-]*\b")
UNSUPPORTED_CLAIM_PATTERNS = (
    ("breakthrough_claim", re.compile(
        r"\b(?:breakthrough|groundbreaking|game changer|leading the charge)\b"
    )),
    ("speculative_impact", re.compile(
        r"\b(?:could|may|might) (?:lead to|impact|revolutionize|reshape|unlock)\b"
    )),
    ("unsupported_ownership", re.compile(
        r"\b(?:our team|our latest|our (?:event|report|research|study|session|work)|"
        r"we (?:found|discovered|developed|uncovered|hosted|organized|presented|"
        r"published|supported|partnered|collaborated)|we teamed up)\b"
    )),
)
EVIDENCE_ONLY_CLAIM_PATTERNS = (
    ("unsupported_temporal_status", re.compile(
        r"\b(?:latest|recent|recently|upcoming|coming soon|due soon|"
        r"next (?:year|month|week)|this (?:year|month|week)|"
        r"new (?:report|publication|event|initiative))\b"
    )),
    ("unsupported_research_status", re.compile(
        r"\b(?:new (?:study|research|findings|results)|"
        r"study (?:shows|finds|discusses|reports)|"
        r"research (?:shows|finds|demonstrates)|"
        r"findings (?:show|suggest)|results (?:show|suggest))\b"
    )),
    ("unsupported_event_outcome", re.compile(
        r"\b(?:winners? (?:were|are|have been) announced|"
        r"announced (?:the )?winners?|successful (?:event|conference|"
        r"convention|session)|(?:event|conference|convention|session) success)\b"
    )),
    ("unsupported_effect_claim", re.compile(
        r"\b(?:significantly (?:enhance|improve|increase|reduce|boost)|"
        r"proven to|shown to|opens? (?:more )?opportunities|"
        r"(?:to|can|could|may|will) (?:improve|enhance|increase|reduce|boost)\b|"
        r"(?:mentorship|training|education|research) (?:is |are )?key for\b)"
    )),
    ("unsupported_entity_or_relation", re.compile(
        r"\bfellow (?:[a-z0-9'-]+ ){0,3}members?\b"
    )),
    ("unsupported_quantity_or_comparison", re.compile(
        r"\bpioneering (?:figures?|leaders?|people)\b"
    )),
    ("unsupported_evaluation", re.compile(
        r"\b(?:important|importance|essential|vital|critical|crucial|pioneering|"
        r"inspiring|eye-opening|transformative|dynamic)\b"
    )),
    ("unsupported_trust_claim", re.compile(
        r"\btrusted (?:[a-z0-9'-]+ ){0,3}"
        r"(?:care|treatment|services?|providers?)\b"
    )),
    ("unsupported_participation_or_role", re.compile(
        r"\b(?:i|we) (?:attended|participated|joined|listened|hosted|organized|"
        r"presented|published|supported)\b"
    )),
    ("unsupported_audience", re.compile(
        r"\b(?:students?|educators?|policymakers?|communities|members?|"
        r"attendees?|participants?|audiences?)\b"
    )),
    ("unsupported_experience_or_intent", re.compile(
        r"\b(?:listening to|reflecting on|inspired by|our knowledge|our understanding|"
        r"our (?:[a-z0-9'-]+ ){0,2}journey)\b"
    )),
    ("unsupported_action_or_benefit", re.compile(
        r"\b(?:dive in|for insights?|we need more|need more initiatives?|"
        r"for all|advocacy efforts?)\b"
    )),
    ("unsupported_presupposition", re.compile(
        r"\bchallenges persist\b"
    )),
)
MONTH_ALIASES = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "june": "jun", "july": "jul", "august": "aug", "september": "sep",
    "october": "oct", "november": "nov", "december": "dec",
}
DATE_WORDS = set(MONTH_ALIASES) | set(MONTH_ALIASES.values()) | {
    "monday", "mon", "tuesday", "tue", "wednesday", "wed",
    "thursday", "thu", "friday", "fri", "saturday", "sat",
    "sunday", "sun", "today", "tomorrow", "yesterday",
}
ENTITY_CONNECTORS = {"and", "de", "of", "the", "van", "von"}
ENTITY_LEADING_STOPWORDS = {
    "a", "an", "here", "i", "it", "our", "that", "the", "these",
    "this", "those", "today", "tomorrow", "we", "yesterday",
}
ORGANIZATION_ENTITY_WORDS = {
    "academy", "agency", "association", "center", "centre", "college",
    "committee", "company", "corporation", "council", "department",
    "foundation", "group", "inc", "institute", "institution", "lab",
    "laboratory", "labs", "llc", "ltd", "network", "school", "society",
    "university",
}
GROUNDING_TOKEN_STOP_WORDS = {
    "about", "action", "after", "again", "all", "also", "among", "apply",
    "before", "being", "below", "between", "check", "come", "could",
    "discuss", "event", "from", "great", "have", "here", "how", "important",
    "into", "join", "just", "learn", "make", "more", "most", "need", "news",
    "now", "only", "opportunity", "other", "over", "please", "post", "read",
    "right", "share", "should", "some", "such", "take", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "today", "under", "update", "using", "very", "visit", "want",
    "were", "what", "when", "where", "which", "while", "will", "with",
    "would", "your", "thanks", "thank", "http", "https", "user", "url", "amp",
}
MIN_EVIDENCE_TOKEN_PRECISION = 0.50
V8_PIPELINE_VERSION = "quadbot-v3-llm-agent-generation-8"
CURRENT_PIPELINE_VERSION = "quadbot-v3-llm-agent-generation-33"
SEMANTIC_VERIFIER_VERSION = "qwen-closed-world-grounding-v5"
SEMANTIC_RESPONSE_PREFILL_VERSION = "json-verdict-reasons-v1"
SEMANTIC_PROMPT_SHA256 = (
    "ba948091b49f65a4cb2c0ee59958bd99abdd4b9b1d985e821baa4f521b5d96f0"
)
SEMANTIC_POLICY = "closed_world_single_bound_evidence"
QWEN_VERIFICATION_METHOD = "qwen_closed_world"
DETERMINISTIC_GROUNDING_METHOD = "deterministic_closed_world_fallback"
REPOST_VERIFICATION_METHOD = "deterministic_oasis_repost_binding"
SEMANTIC_DECODE = {
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.0,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEMANTIC_CASE_ID_RE = re.compile(r"^case_[0-9a-f]{20}$")
SEMANTIC_PROVENANCE_FIELDS = {
    "semantic_grounding_policy",
    "semantic_grounding_verifier_version",
    "semantic_grounding_prompt_sha256",
    "semantic_grounding_response_prefill_version",
    "semantic_grounding_model_name",
    "semantic_grounding_model_path",
    "semantic_grounding_model_fingerprint",
    "semantic_grounding_decode",
    "semantic_grounding_fail_closed",
    "semantic_grounding_stats",
}
SEMANTIC_EVIDENCE_FIELDS = {
    "semantic_case_id",
    "semantic_verdict",
    "semantic_reason",
    "semantic_reason_codes",
    "semantic_unsupported_spans",
    "semantic_case_sha256",
    "semantic_batch_sha256",
    "semantic_judge_seed",
    "semantic_judge_output_sha256",
    "semantic_verification_method",
    "semantic_parse_error",
    "semantic_cache_hit",
}
SEMANTIC_STATS_FIELDS = {
    "cases_checked",
    "pass_count",
    "fail_count",
    "uncertain_count",
    "parse_error_count",
    "cache_hit_count",
    "qwen_cases_checked",
    "deterministic_reposts_checked",
    "reason_counts",
}
SEMANTIC_CONFIG_KEYS = {
    "pipeline_version",
    "semantic_grounding_policy",
    "semantic_grounding_verifier_version",
    "semantic_grounding_prompt_sha256",
    "semantic_grounding_response_prefill_version",
    "semantic_grounding_model_name",
    "semantic_grounding_model_path",
    "semantic_grounding_model_fingerprint",
    "semantic_grounding_decode",
    "semantic_grounding_fail_closed",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-posts", type=int, default=25)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--token-jaccard-threshold", type=float, default=0.72)
    parser.add_argument("--source-similarity-threshold", type=float, default=0.90)
    parser.add_argument("--expected-base-profiles", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def clean_text(value: Any, max_chars: int = 280) -> str:
    text = str(value or "").strip().strip('"').strip()
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = URL_RE.sub("<URL>", text)
    text = HANDLE_RE.sub("<USER>", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def expected_repost_text(evidence_text: str) -> str:
    target_text = clean_text(evidence_text)
    if target_text.casefold().startswith("rt <user>:"):
        return target_text
    return clean_text(f"RT <USER>: {target_text}")


def semantic_case_payload(
    action_type: str,
    candidate_text: str,
    profile_support: str,
    evidence_id: object,
    evidence_text: str,
    evidence_role: str,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "candidate_text": clean_text(candidate_text),
        "profile_support": clean_text(profile_support),
        "evidence": {
            "id": str(evidence_id),
            "text": clean_text(evidence_text),
            "role": evidence_role,
        },
    }


def semantic_case_sha256(
    provenance: dict[str, Any], case_payload: dict[str, Any]
) -> str:
    return canonical_sha256({
        "verifier_version": provenance.get(
            "semantic_grounding_verifier_version"
        ),
        "prompt_sha256": provenance.get(
            "semantic_grounding_prompt_sha256"
        ),
        "response_prefill_version": provenance.get(
            "semantic_grounding_response_prefill_version"
        ),
        "model_fingerprint": provenance.get(
            "semantic_grounding_model_fingerprint"
        ),
        "case": case_payload,
    })


def repost_execution_binding(
    case_payload: dict[str, Any],
    expected_target_post_id: Any,
    evidence_posts: dict[Any, dict[str, Any]],
    trace_row: dict[str, Any] | None,
) -> dict[str, Any]:
    trace_rowid = trace_row.get("rowid") if trace_row is not None else None
    trace_info = trace_row.get("info", {}) if trace_row is not None else {}
    if not isinstance(trace_info, dict):
        trace_info = {}
    trace_reposted_id = trace_info.get("reposted_id")
    trace_new_post_id = trace_info.get("new_post_id")
    result_post_id = trace_new_post_id
    target_row = evidence_posts.get(expected_target_post_id)
    new_post_row = evidence_posts.get(result_post_id)

    target_author_id = (
        target_row.get("author_id") if target_row is not None else None
    )
    target_text = (
        clean_text(target_row.get("text")) if target_row is not None else ""
    )
    new_post_user_id = (
        new_post_row.get("author_id") if new_post_row is not None else None
    )
    new_post_original_post_id = (
        new_post_row.get("original_post_id")
        if new_post_row is not None
        else None
    )
    new_post_text = (
        clean_text(new_post_row.get("text")) if new_post_row is not None else ""
    )
    expected_display_text = expected_repost_text(target_text)
    checks = {
        "case_action_is_repost": case_payload.get("action_type") == "repost",
        "case_evidence_id_matches_target": (
            case_payload.get("evidence", {}).get("id")
            == str(expected_target_post_id)
        ),
        "case_evidence_text_matches_database": (
            case_payload.get("evidence", {}).get("text") == target_text
        ),
        "candidate_matches_database_target": (
            case_payload.get("candidate_text") == expected_display_text
        ),
        "trace_target_matches_expected": (
            trace_reposted_id == expected_target_post_id
        ),
        "result_post_id_is_valid": (
            isinstance(result_post_id, int)
            and not isinstance(result_post_id, bool)
        ),
        "trace_new_post_matches_result": (
            trace_new_post_id == result_post_id
        ),
        "target_exists_and_is_external": (
            target_row is not None and target_author_id != 0
        ),
        "new_post_exists_and_is_actor_owned": (
            new_post_row is not None and new_post_user_id == 0
        ),
        "new_post_original_matches_target": (
            new_post_original_post_id == expected_target_post_id
        ),
        "new_post_content_matches_oasis_repost_storage": (
            new_post_row is not None
            and new_post_text in {"", target_text}
        ),
    }
    return {
        "expected_target_post_id": expected_target_post_id,
        "trace_rowid": trace_rowid,
        "trace_reposted_id": trace_reposted_id,
        "trace_new_post_id": trace_new_post_id,
        "result_post_id": result_post_id,
        "new_post_user_id": new_post_user_id,
        "new_post_original_post_id": new_post_original_post_id,
        "target_author_id": target_author_id,
        "target_text_sha256": hashlib.sha256(
            target_text.encode("utf-8")
        ).hexdigest(),
        "expected_display_text_sha256": hashlib.sha256(
            expected_display_text.encode("utf-8")
        ).hexdigest(),
        "new_post_content_sha256": hashlib.sha256(
            new_post_text.encode("utf-8")
        ).hexdigest(),
        "checks": checks,
    }


def normalized_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalized_text(left).split())
    right_tokens = set(normalized_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def opening_signature(text: str, size: int = 3) -> tuple[str, ...]:
    return tuple(normalized_text(text).split()[:size])


def generic_template_signatures(text: str) -> set[str]:
    normalized = normalized_text(text)
    return {
        name for name, pattern in GENERIC_TEMPLATE_PATTERNS
        if pattern.search(normalized)
    }


def sentence_initial_word(text: str, start: int) -> bool:
    prefix = text[:start].rstrip()
    return not prefix or prefix[-1:] in ".!?"


def grounding_marker_findings(
    candidate: str, support_texts: list[str]
) -> tuple[list[str], list[str]]:
    support = " ".join(support_texts)
    support_normalized = normalized_text(support)
    support_tokens = set(normalized_text(support).split())
    for full_name, abbreviation in MONTH_ALIASES.items():
        if full_name in support_tokens or abbreviation in support_tokens:
            support_tokens.update((full_name, abbreviation))
    hard = set()
    soft = set()
    for hashtag in HASHTAG_RE.findall(candidate):
        marker = normalized_text(hashtag)
        if marker and marker not in support_tokens:
            soft.add(f"hashtag:{hashtag.casefold()}")
    support_numbers = set(NUMBER_RE.findall(support))
    for number in NUMBER_RE.findall(candidate):
        if number not in support_numbers:
            hard.add(f"number:{number}")

    matches = list(WORD_RE.finditer(candidate))
    entity_parts: list[str] = []
    entity_words: list[str] = []
    entity_started_sentence_initial = False

    def flush_entity() -> None:
        nonlocal entity_started_sentence_initial
        if not entity_words:
            entity_parts.clear()
            entity_started_sentence_initial = False
            return
        while entity_parts and entity_parts[-1].casefold() in ENTITY_CONNECTORS:
            entity_parts.pop()
        span = " ".join(entity_parts)
        normalized_span = normalized_text(span)
        normalized_words = [word.casefold() for word in entity_words]
        supported = (
            normalized_span in support_normalized
            or all(word in support_tokens for word in normalized_words)
        )
        organization_like = any(
            word in ORGANIZATION_ENTITY_WORDS for word in normalized_words
        )
        if not supported and (len(entity_words) >= 2 or organization_like):
            hard.add(f"entity:{normalized_span}")
        elif not supported and (
            not entity_started_sentence_initial
            or any(word.isupper() and len(word) > 1 for word in entity_words)
        ):
            soft.add(f"capitalized:{normalized_span}")
        entity_parts.clear()
        entity_words.clear()
        entity_started_sentence_initial = False

    previous_end = 0
    for match in matches:
        word = match.group(0)
        separator = candidate[previous_end : match.start()]
        if re.search(r"[.!?;:\n]", separator):
            flush_entity()
        if match.start() > 0 and candidate[match.start() - 1] == "#":
            flush_entity()
            previous_end = match.end()
            continue
        normalized_word = word.casefold()
        capitalized = (word.isupper() and len(word) > 1) or word[0].isupper()
        if capitalized:
            if (
                not entity_words
                and sentence_initial_word(candidate, match.start())
                and normalized_word in ENTITY_LEADING_STOPWORDS
            ):
                previous_end = match.end()
                continue
            if not entity_words:
                entity_started_sentence_initial = sentence_initial_word(
                    candidate, match.start()
                )
            entity_parts.append(word)
            entity_words.append(word)
        elif normalized_word in ENTITY_CONNECTORS and entity_words:
            entity_parts.append(word)
        else:
            flush_entity()
        if normalized_word in DATE_WORDS and normalized_word not in support_tokens:
            hard.add(f"date:{normalized_word}")
        previous_end = match.end()
    flush_entity()

    support_symbols = {
        char for char in support if unicodedata.category(char) in {"So", "Sk"}
    }
    for char in candidate:
        if (
            unicodedata.category(char) in {"So", "Sk"}
            and char not in support_symbols
        ):
            hard.add(f"symbol:u+{ord(char):04x}")
    return sorted(hard), sorted(soft)


def unsupported_grounding_markers(
    candidate: str, support_texts: list[str]
) -> list[str]:
    hard, _ = grounding_marker_findings(candidate, support_texts)
    return hard


def grounding_soft_warnings(
    candidate: str, support_texts: list[str]
) -> list[str]:
    _, soft = grounding_marker_findings(candidate, support_texts)
    return soft


def grounding_content_tokens(text: str) -> set[str]:
    cleaned = clean_text(text)
    cleaned = HASHTAG_RE.sub(" ", cleaned)
    cleaned = (
        cleaned.replace("<URL>", " ")
        .replace("<USER>", " ")
        .replace("<EMAIL>", " ")
    )
    return {
        word.casefold().strip("'-")
        for word in WORD_RE.findall(cleaned)
        if len(word.strip("'-")) >= 3
        and word.casefold().strip("'-") not in GROUNDING_TOKEN_STOP_WORDS
    }


def evidence_token_precision(candidate: str, evidence_texts: list[str]) -> float:
    candidate_tokens = grounding_content_tokens(candidate)
    if not candidate_tokens:
        return 1.0
    evidence_tokens = grounding_content_tokens(" ".join(evidence_texts))
    return len(candidate_tokens & evidence_tokens) / len(candidate_tokens)


def grounding_rejection_reason(
    candidate: str,
    support_texts: list[str],
    ownership_support_texts: list[str] | None = None,
    evidence_support_texts: list[str] | None = None,
    minimum_token_precision: float | None = None,
) -> str | None:
    if unsupported_grounding_markers(candidate, support_texts):
        return "unsupported_grounding_marker"
    candidate_normalized = normalized_text(candidate)
    support_normalized = normalized_text(" ".join(support_texts))
    for name, pattern in UNSUPPORTED_CLAIM_PATTERNS:
        match = pattern.search(candidate_normalized)
        allowed_support = (
            normalized_text(" ".join(ownership_support_texts))
            if name == "unsupported_ownership" and ownership_support_texts is not None
            else support_normalized
        )
        if match and match.group(0) not in allowed_support:
            return name
    evidence_support_texts = evidence_support_texts or support_texts
    evidence_normalized = normalized_text(" ".join(evidence_support_texts))
    for name, pattern in EVIDENCE_ONLY_CLAIM_PATTERNS:
        for match in pattern.finditer(candidate_normalized):
            if match.group(0) not in evidence_normalized:
                return name
    if (
        minimum_token_precision is not None
        and evidence_token_precision(candidate, evidence_support_texts)
        < minimum_token_precision
    ):
        return "low_evidence_token_precision"
    return None


def organization_voice(profile: dict[str, Any]) -> bool:
    description = str(profile.get("description", "")).casefold()
    return any(marker in description for marker in ORGANIZATION_MARKERS)


def raw_identifier_leaks(text: str) -> list[str]:
    scrubbed = text.replace("<URL>", "").replace("<EMAIL>", "").replace("<USER>", "")
    leaks = []
    if URL_RE.search(scrubbed):
        leaks.append("url_or_domain")
    if EMAIL_RE.search(scrubbed):
        leaks.append("email")
    if HANDLE_RE.search(scrubbed):
        leaks.append("handle")
    return leaks


def trace_rows(db_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT rowid, user_id, created_at, action, info FROM trace "
            "WHERE user_id = ? ORDER BY rowid",
            (0,),
        ).fetchall()
    return [
        {
            "rowid": row[0],
            "user_id": row[1],
            "created_at": row[2],
            "action": row[3],
            "info": json.loads(row[4]) if row[4] else {},
        }
        for row in rows
    ]


def duplicate_audit(
    texts: list[str],
    action_types: list[str],
    sequence_threshold: float,
    jaccard_threshold: float,
) -> dict[str, Any]:
    normalized = [normalized_text(text) for text in texts]
    # Reposts are bound to their target post and can legitimately render the
    # same text for different targets. Duplicate-text constraints apply only
    # to authored post/comment content.
    authored_indices = [
        index for index, action_type in enumerate(action_types)
        if action_type != "repost"
    ]
    exact_duplicates = [
        {"normalized_text": text, "count": count}
        for text, count in Counter(
            normalized[index] for index in authored_indices
        ).items()
        if count > 1
    ]
    near_duplicates = []
    lexical_duplicates = []
    repeated_openings = []
    generic_template_occurrences: dict[str, list[int]] = {
        name: [] for name, _ in GENERIC_TEMPLATE_PATTERNS
    }
    for index, (text, action_type) in enumerate(zip(texts, action_types)):
        if action_type == "repost":
            continue
        for signature in generic_template_signatures(text):
            generic_template_occurrences[signature].append(index)
    maximum = {"similarity": 0.0, "left_index": None, "right_index": None}
    for right_index, right in enumerate(normalized):
        for left_index in range(right_index):
            if (
                action_types[left_index] == "repost"
                or action_types[right_index] == "repost"
            ):
                continue
            similarity = SequenceMatcher(None, normalized[left_index], right).ratio()
            if similarity > maximum["similarity"]:
                maximum = {
                    "similarity": round(similarity, 6),
                    "left_index": left_index,
                    "right_index": right_index,
                }
            if similarity >= sequence_threshold:
                near_duplicates.append(
                    {
                        "similarity": round(similarity, 6),
                        "left_index": left_index,
                        "right_index": right_index,
                    }
                )
            jaccard = token_jaccard(normalized[left_index], right)
            if jaccard >= jaccard_threshold:
                lexical_duplicates.append(
                    {
                        "jaccard": round(jaccard, 6),
                        "left_index": left_index,
                        "right_index": right_index,
                    }
                )
            if (
                action_types[left_index] != "repost"
                and action_types[right_index] != "repost"
                and len(opening_signature(texts[left_index])) == 3
                and opening_signature(texts[left_index])
                == opening_signature(texts[right_index])
            ):
                repeated_openings.append(
                    {
                        "opening": list(opening_signature(texts[left_index])),
                        "left_index": left_index,
                        "right_index": right_index,
                    }
                )
    return {
        "normalized_unique_count": (
            len(set(normalized[index] for index in authored_indices))
            + len(action_types) - len(authored_indices)
        ),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "lexical_duplicates": lexical_duplicates,
        "repeated_openings": repeated_openings,
        "repeated_generic_templates": [
            {"template": name, "count": len(indices), "indices": indices}
            for name, indices in generic_template_occurrences.items()
            if len(indices) > 1
        ],
        "generic_template_occurrences": {
            name: indices
            for name, indices in generic_template_occurrences.items()
            if indices
        },
        "maximum_pairwise_similarity": maximum,
    }


def source_similarity_audit(
    texts: list[str], source_posts: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    generated = [normalized_text(text) for text in texts]
    sources = [normalized_text(post["text"]) for post in source_posts]
    violations = []
    maximum = {"similarity": 0.0, "generated_index": None, "source_index": None}
    for generated_index, candidate in enumerate(generated):
        for source_index, source in enumerate(sources):
            similarity = SequenceMatcher(None, candidate, source).ratio()
            if similarity > maximum["similarity"]:
                maximum = {
                    "similarity": round(similarity, 6),
                    "generated_index": generated_index,
                    "source_index": source_index,
                }
            if similarity >= threshold:
                violations.append(
                    {
                        "similarity": round(similarity, 6),
                        "generated_index": generated_index,
                        "source_index": source_index,
                    }
                )
    return {"maximum": maximum, "violations": violations}


def semantic_grounding_audit(
    record: dict[str, Any], seed: dict[str, Any]
) -> dict[str, Any]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        return {
            "applicable": True,
            "passed": False,
            "error": "provenance_missing_or_invalid",
            "case_ids": [],
        }

    pipeline_version = provenance.get("pipeline_version")
    if pipeline_version == V8_PIPELINE_VERSION:
        return {
            "applicable": False,
            "passed": True,
            "pipeline_version": pipeline_version,
            "compatibility_mode": "explicit_legacy_v8",
            "semantic_guarantee": False,
            "warning": "V8 predates persisted semantic-verifier provenance",
            "case_ids": [],
        }
    if pipeline_version != CURRENT_PIPELINE_VERSION:
        return {
            "applicable": True,
            "passed": False,
            "pipeline_version": pipeline_version,
            "error": "unsupported_or_missing_pipeline_version",
            "expected_pipeline_version": CURRENT_PIPELINE_VERSION,
            "legacy_pipeline_version": V8_PIPELINE_VERSION,
            "case_ids": [],
        }

    actions = record.get("actions", [])
    posts = record.get("posts", [])
    if not isinstance(actions, list) or not isinstance(posts, list):
        return {
            "applicable": True,
            "passed": False,
            "pipeline_version": pipeline_version,
            "error": "posts_or_actions_invalid",
            "case_ids": [],
        }

    missing_provenance_fields = sorted(
        SEMANTIC_PROVENANCE_FIELDS - provenance.keys()
    )
    config_violations = []
    if provenance.get("semantic_grounding_policy") != SEMANTIC_POLICY:
        config_violations.append("semantic_grounding_policy")
    if (
        provenance.get("semantic_grounding_verifier_version")
        != SEMANTIC_VERIFIER_VERSION
    ):
        config_violations.append("semantic_grounding_verifier_version")
    if (
        provenance.get("semantic_grounding_prompt_sha256")
        != SEMANTIC_PROMPT_SHA256
    ):
        config_violations.append("semantic_grounding_prompt_sha256")
    if (
        provenance.get("semantic_grounding_response_prefill_version")
        != SEMANTIC_RESPONSE_PREFILL_VERSION
    ):
        config_violations.append("semantic_grounding_response_prefill_version")
    if provenance.get("semantic_grounding_fail_closed") is not True:
        config_violations.append("semantic_grounding_fail_closed")
    if provenance.get("semantic_grounding_decode") != SEMANTIC_DECODE:
        config_violations.append("semantic_grounding_decode")

    model_name = provenance.get("semantic_grounding_model_name")
    model_path = provenance.get("semantic_grounding_model_path")
    model_fingerprint = provenance.get("semantic_grounding_model_fingerprint")
    if not isinstance(model_name, str) or not model_name.strip():
        config_violations.append("semantic_grounding_model_name")
    if not isinstance(model_path, str) or not model_path.strip():
        config_violations.append("semantic_grounding_model_path")
    elif isinstance(model_name, str) and Path(model_path).name != model_name:
        config_violations.append("semantic_model_name_path_mismatch")
    if (
        not isinstance(model_fingerprint, str)
        or not SHA256_RE.fullmatch(model_fingerprint)
    ):
        config_violations.append("semantic_grounding_model_fingerprint")

    stats = provenance.get("semantic_grounding_stats")
    stats_violations = []
    if not isinstance(stats, dict):
        missing_stats_fields = sorted(SEMANTIC_STATS_FIELDS)
        stats_violations.append("semantic_grounding_stats_not_object")
        stats = {}
    else:
        missing_stats_fields = sorted(SEMANTIC_STATS_FIELDS - stats.keys())
    numeric_stat_fields = SEMANTIC_STATS_FIELDS - {"reason_counts"}
    for field in sorted(numeric_stat_fields):
        value = stats.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            stats_violations.append(f"invalid_stat:{field}")
    reason_counts = stats.get("reason_counts")
    if (
        not isinstance(reason_counts, dict)
        or not all(isinstance(key, str) for key in reason_counts)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in reason_counts.values()
        )
    ):
        stats_violations.append("invalid_stat:reason_counts")
    if not missing_stats_fields and not stats_violations:
        if (
            stats["pass_count"]
            + stats["fail_count"]
            + stats["uncertain_count"]
            != stats["cases_checked"]
        ):
            stats_violations.append("verdict_counts_do_not_sum_to_cases_checked")
        # The generator records three semantic verification methods: Qwen,
        # deterministic OASIS repost binding, and the fail-closed deterministic
        # evidence-fragment fallback.  The latter is intentionally an extra
        # stat for backward-compatible reports, so include it in the method
        # conservation check when present.
        deterministic_fallback_count = stats.get(
            "deterministic_fallback_count", 0
        )
        if (
            not isinstance(deterministic_fallback_count, int)
            or isinstance(deterministic_fallback_count, bool)
            or deterministic_fallback_count < 0
        ):
            stats_violations.append("invalid_stat:deterministic_fallback_count")
            deterministic_fallback_count = 0
        if (
            stats["qwen_cases_checked"]
            + stats["deterministic_reposts_checked"]
            + deterministic_fallback_count
            != stats["cases_checked"]
        ):
            stats_violations.append("method_counts_do_not_sum_to_cases_checked")
        if stats["pass_count"] < len(actions):
            stats_violations.append("pass_count_below_final_action_count")
        if stats["cases_checked"] < len(actions):
            stats_violations.append("cases_checked_below_final_action_count")

    raw_entries = provenance.get("grounding_evidence")
    if not isinstance(raw_entries, list):
        return {
            "applicable": True,
            "passed": False,
            "pipeline_version": pipeline_version,
            "error": "grounding_evidence_missing_or_invalid",
            "missing_provenance_fields": missing_provenance_fields,
            "config_violations": sorted(set(config_violations)),
            "missing_stats_fields": missing_stats_fields,
            "stats_violations": sorted(set(stats_violations)),
            "case_ids": [],
        }

    invalid_entry_positions = [
        index for index, entry in enumerate(raw_entries)
        if not isinstance(entry, dict)
    ]
    entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    valid_indices = [
        entry.get("action_index")
        for entry in entries
        if isinstance(entry.get("action_index"), int)
        and not isinstance(entry.get("action_index"), bool)
    ]
    invalid_action_indices = [
        entry.get("action_index")
        for entry in entries
        if not isinstance(entry.get("action_index"), int)
        or isinstance(entry.get("action_index"), bool)
    ]
    index_counts = Counter(valid_indices)
    duplicate_action_indices = sorted(
        index for index, count in index_counts.items() if count > 1
    )
    expected_indices = set(range(len(actions)))
    missing_action_indices = sorted(expected_indices - set(valid_indices))
    unexpected_action_indices = sorted(set(valid_indices) - expected_indices)
    by_index: dict[int, dict[str, Any]] = {}
    for entry in entries:
        index = entry.get("action_index")
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and index not in by_index
        ):
            by_index[index] = entry

    label_name = record.get("label_name")
    evidence_context_error = None
    source_posts: dict[str, dict[str, Any]] = {}
    evidence_posts: dict[Any, dict[str, Any]] = {}
    repost_trace_rows: list[dict[str, Any]] = []
    if label_name == "llm_bot":
        human_replay_posts = seed.get("human_replay_posts")
        if not isinstance(human_replay_posts, list):
            evidence_context_error = "seed_human_replay_posts_missing_or_invalid"
        else:
            for item in human_replay_posts:
                if isinstance(item, dict) and "source_tweet_id" in item:
                    source_posts[str(item["source_tweet_id"])] = item
    elif label_name == "full_stack_agent":
        db_value = provenance.get("oasis_database")
        db_path = Path(db_value) if isinstance(db_value, str) else Path("")
        if not isinstance(db_value, str) or not db_path.is_file():
            evidence_context_error = "agent_evidence_database_missing"
        else:
            try:
                with sqlite3.connect(db_path) as connection:
                    evidence_posts = {
                        post_id: {
                            "text": content,
                            "author_id": user_id,
                            "original_post_id": original_post_id,
                        }
                        for post_id, content, user_id, original_post_id
                        in connection.execute(
                            "SELECT post_id, content, user_id, original_post_id "
                            "FROM post"
                        ).fetchall()
                    }
                    repost_trace_rows = [
                        {
                            "rowid": rowid,
                            "info": json.loads(info) if info else {},
                        }
                        for rowid, info in connection.execute(
                            "SELECT rowid, info FROM trace "
                            "WHERE user_id = ? AND action = 'repost' "
                            "ORDER BY rowid",
                            (0,),
                        ).fetchall()
                    ]
            except (sqlite3.Error, json.JSONDecodeError) as exc:
                evidence_context_error = f"agent_evidence_database_error:{exc}"
    else:
        evidence_context_error = f"unsupported_label:{label_name!r}"

    missing_entry_fields = {}
    invalid_entry_fields = {}
    unknown_actions = []
    evidence_binding_violations = []
    evidence_hash_violations = []
    case_hash_violations = []
    deterministic_batch_hash_violations = []
    deterministic_output_hash_violations = []
    execution_binding_violations = []
    case_ids = []
    case_hashes = []
    profile = record.get("profile")
    profile_support = (
        str(profile.get("description", ""))
        if isinstance(profile, dict)
        else ""
    )
    if not isinstance(profile, dict):
        evidence_context_error = evidence_context_error or "profile_invalid"
    repost_action_indices = [
        index for index, action in enumerate(actions)
        if isinstance(action, dict) and action.get("type") == "repost"
    ]
    repost_trace_by_action = {
        action_index: (
            repost_trace_rows[ordinal]
            if ordinal < len(repost_trace_rows)
            else None
        )
        for ordinal, action_index in enumerate(repost_action_indices)
    }
    repost_trace_count_violation = (
        label_name == "full_stack_agent"
        and len(repost_trace_rows) != len(repost_action_indices)
    )
    if not missing_stats_fields and not stats_violations:
        final_repost_count = len(repost_action_indices)
        final_qwen_count = len(actions) - final_repost_count
        if stats["qwen_cases_checked"] < final_qwen_count:
            # Deterministic closed-world fallback records are valid semantic
            # checks for final non-repost actions too.  Count them alongside
            # Qwen cases when checking final-action coverage.
            verified_nonrepost = (
                stats["qwen_cases_checked"]
                + int(stats.get("deterministic_fallback_count", 0))
            )
            if verified_nonrepost < final_qwen_count:
                stats_violations.append("qwen_count_below_final_qwen_action_count")
        if stats["deterministic_reposts_checked"] < final_repost_count:
            stats_violations.append(
                "repost_count_below_final_repost_action_count"
            )

    for index in sorted(expected_indices):
        entry = by_index.get(index)
        if entry is None:
            continue
        required_fields = set(SEMANTIC_EVIDENCE_FIELDS)
        if label_name == "llm_bot":
            required_fields.update(("source_tweet_id", "source_text_sha256"))
        elif label_name == "full_stack_agent":
            required_fields.update(("oasis_post_id", "evidence_text_sha256"))
        missing = sorted(required_fields - entry.keys())
        if missing:
            missing_entry_fields[str(index)] = missing

        invalid = []
        case_id = entry.get("semantic_case_id")
        if not isinstance(case_id, str) or not SEMANTIC_CASE_ID_RE.fullmatch(case_id):
            invalid.append("semantic_case_id")
        else:
            case_ids.append(case_id)
        if entry.get("semantic_verdict") != "pass":
            invalid.append("semantic_verdict")
        if entry.get("semantic_reason") != "supported":
            invalid.append("semantic_reason")
        if entry.get("semantic_reason_codes") != []:
            invalid.append("semantic_reason_codes")
        if entry.get("semantic_unsupported_spans") != []:
            invalid.append("semantic_unsupported_spans")
        if entry.get("semantic_parse_error") is not None:
            invalid.append("semantic_parse_error")
        if not isinstance(entry.get("semantic_cache_hit"), bool):
            invalid.append("semantic_cache_hit")
        for field in (
            "semantic_case_sha256",
            "semantic_batch_sha256",
            "semantic_judge_output_sha256",
        ):
            value = entry.get(field)
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                invalid.append(field)
        judge_seed = entry.get("semantic_judge_seed")
        if (
            not isinstance(judge_seed, int)
            or isinstance(judge_seed, bool)
            or judge_seed < 0
        ):
            invalid.append("semantic_judge_seed")

        action = actions[index] if index < len(actions) else None
        post = posts[index] if index < len(posts) else None
        if not isinstance(action, dict) or not isinstance(post, dict):
            evidence_binding_violations.append({
                "action_index": index,
                "reason": "final_post_or_action_missing",
            })
            if invalid:
                invalid_entry_fields[str(index)] = sorted(set(invalid))
            continue
        action_type = action.get("type")
        allowed_actions = (
            {"post"} if label_name == "llm_bot"
            else {"post", "comment", "repost"}
        )
        if action_type not in allowed_actions:
            unknown_actions.append({
                "action_index": index,
                "action_type": action_type,
            })
            if invalid:
                invalid_entry_fields[str(index)] = sorted(set(invalid))
            continue
        if action_type == "repost":
            if "semantic_execution_binding" not in entry:
                missing_entry_fields.setdefault(str(index), []).append(
                    "semantic_execution_binding"
                )
        elif "semantic_execution_binding" in entry:
            invalid.append("unexpected_semantic_execution_binding")
        if post.get("action_type") != action_type:
            evidence_binding_violations.append({
                "action_index": index,
                "reason": "post_action_type_mismatch",
            })

        evidence_id = None
        evidence_text = None
        if label_name == "llm_bot":
            evidence_id = str(entry.get("source_tweet_id", ""))
            source_post = source_posts.get(evidence_id)
            if (
                action.get("grounding_source_id") != evidence_id
                or source_post is None
            ):
                evidence_binding_violations.append({
                    "action_index": index,
                    "reason": "llm_source_binding_mismatch",
                    "evidence_id": evidence_id,
                })
            else:
                evidence_text = clean_text(source_post.get("text"))
                expected_evidence_hash = hashlib.sha256(
                    evidence_text.encode("utf-8")
                ).hexdigest()
                if entry.get("source_text_sha256") != expected_evidence_hash:
                    evidence_hash_violations.append(index)
        else:
            evidence_id = entry.get("oasis_post_id")
            evidence_post = evidence_posts.get(evidence_id)
            if evidence_post is None or evidence_post.get("author_id") == 0:
                evidence_binding_violations.append({
                    "action_index": index,
                    "reason": "agent_evidence_binding_mismatch",
                    "evidence_id": evidence_id,
                })
            else:
                evidence_text = clean_text(evidence_post.get("text"))
                expected_evidence_hash = hashlib.sha256(
                    evidence_text.encode("utf-8")
                ).hexdigest()
                if entry.get("evidence_text_sha256") != expected_evidence_hash:
                    evidence_hash_violations.append(index)

        expected_methods = (
            {REPOST_VERIFICATION_METHOD}
            if action_type == "repost"
            else {QWEN_VERIFICATION_METHOD, DETERMINISTIC_GROUNDING_METHOD}
        )
        if entry.get("semantic_verification_method") not in expected_methods:
            invalid.append("semantic_verification_method")

        if evidence_text is not None:
            case_payload = semantic_case_payload(
                action_type=action_type,
                candidate_text=post.get("text", ""),
                profile_support=profile_support,
                evidence_id=evidence_id,
                evidence_text=evidence_text,
                evidence_role=(
                    "same_profile_historical_post"
                    if label_name == "llm_bot"
                    else "third_party_topic_evidence"
                ),
            )
            expected_case_hash = semantic_case_sha256(provenance, case_payload)
            stored_case_hash = entry.get("semantic_case_sha256")
            if stored_case_hash != expected_case_hash:
                case_hash_violations.append({
                    "action_index": index,
                    "stored": stored_case_hash,
                    "expected": expected_case_hash,
                })
            elif isinstance(stored_case_hash, str):
                case_hashes.append(stored_case_hash)

            if action_type == "repost":
                expected_binding = repost_execution_binding(
                    case_payload,
                    evidence_id,
                    evidence_posts,
                    repost_trace_by_action.get(index),
                )
                stored_binding = entry.get("semantic_execution_binding")
                if stored_binding != expected_binding:
                    execution_binding_violations.append({
                        "action_index": index,
                        "stored": stored_binding,
                        "expected": expected_binding,
                    })
                if not all(
                    value is True
                    for value in expected_binding["checks"].values()
                ):
                    evidence_binding_violations.append({
                        "action_index": index,
                        "reason": "repost_execution_checks_failed",
                        "checks": expected_binding["checks"],
                    })
                expected_batch_hash = canonical_sha256({
                    "verification_method": REPOST_VERIFICATION_METHOD,
                    "case_sha256": expected_case_hash,
                    "execution_binding": expected_binding,
                })
                if entry.get("semantic_batch_sha256") != expected_batch_hash:
                    deterministic_batch_hash_violations.append(index)
                expected_output_hash = canonical_sha256({
                    "verdict": "pass",
                    "reason_codes": [],
                    "unsupported_spans": [],
                    "execution_binding": expected_binding,
                })
                if (
                    entry.get("semantic_judge_output_sha256")
                    != expected_output_hash
                ):
                    deterministic_output_hash_violations.append(index)

        if invalid:
            invalid_entry_fields[str(index)] = sorted(set(invalid))

    duplicate_case_ids = sorted(
        case_id for case_id, count in Counter(case_ids).items() if count > 1
    )
    duplicate_case_hashes = sorted(
        case_hash for case_hash, count in Counter(case_hashes).items()
        if count > 1
    )
    passed = not any((
        missing_provenance_fields,
        config_violations,
        missing_stats_fields,
        stats_violations,
        invalid_entry_positions,
        len(raw_entries) != len(actions),
        invalid_action_indices,
        duplicate_action_indices,
        missing_action_indices,
        unexpected_action_indices,
        evidence_context_error,
        missing_entry_fields,
        invalid_entry_fields,
        unknown_actions,
        evidence_binding_violations,
        evidence_hash_violations,
        case_hash_violations,
        deterministic_batch_hash_violations,
        deterministic_output_hash_violations,
        execution_binding_violations,
        repost_trace_count_violation,
        duplicate_case_ids,
        duplicate_case_hashes,
    ))
    return {
        "applicable": True,
        "passed": passed,
        "pipeline_version": pipeline_version,
        "semantic_guarantee": passed,
        "evidence_count": len(raw_entries),
        "expected_evidence_count": len(actions),
        "missing_provenance_fields": missing_provenance_fields,
        "config_violations": sorted(set(config_violations)),
        "missing_stats_fields": missing_stats_fields,
        "stats_violations": sorted(set(stats_violations)),
        "invalid_entry_positions": invalid_entry_positions,
        "invalid_action_indices": invalid_action_indices,
        "duplicate_action_indices": duplicate_action_indices,
        "missing_action_indices": missing_action_indices,
        "unexpected_action_indices": unexpected_action_indices,
        "evidence_context_error": evidence_context_error,
        "missing_entry_fields": missing_entry_fields,
        "invalid_entry_fields": invalid_entry_fields,
        "unknown_actions": unknown_actions,
        "evidence_binding_violations": evidence_binding_violations,
        "evidence_hash_violations": evidence_hash_violations,
        "case_hash_violations": case_hash_violations,
        "deterministic_batch_hash_violations": (
            deterministic_batch_hash_violations
        ),
        "deterministic_output_hash_violations": (
            deterministic_output_hash_violations
        ),
        "execution_binding_violations": execution_binding_violations,
        "repost_trace_count_violation": repost_trace_count_violation,
        "expected_repost_trace_count": len(repost_action_indices),
        "actual_repost_trace_count": len(repost_trace_rows),
        "duplicate_case_ids": duplicate_case_ids,
        "duplicate_case_hashes": duplicate_case_hashes,
        "case_ids": case_ids,
    }


def grounding_audit(
    record: dict[str, Any], seed: dict[str, Any]
) -> dict[str, Any]:
    provenance = record.get("provenance", {})
    evidence_entries = provenance.get("grounding_evidence")
    if evidence_entries is None:
        return {"applicable": False, "passed": True}
    evidence_entries = [entry for entry in evidence_entries if isinstance(entry, dict)]
    evidence_indices = [entry.get("action_index") for entry in evidence_entries]
    expected_indices = set(range(len(record.get("actions", []))))
    index_counts = Counter(evidence_indices)
    duplicate_indices = sorted(
        (index for index, count in index_counts.items() if count > 1),
        key=lambda value: str(value),
    )
    missing_indices = sorted(expected_indices - set(evidence_indices))
    unexpected_indices = sorted(
        index for index in set(evidence_indices) - expected_indices
        if isinstance(index, int)
    )
    invalid_indices = [index for index in evidence_indices if not isinstance(index, int)]
    by_index = {
        entry.get("action_index"): entry
        for entry in evidence_entries
    }
    profile_support = str(record.get("profile", {}).get("description", ""))
    violations = []
    soft_warnings = []
    evidence_hash_violations = []
    evidence_id_violations = []
    evidence_author_violations = []
    evidence_reuse_violations = []

    if record["label_name"] == "llm_bot":
        source_posts = {
            str(item["source_tweet_id"]): item
            for item in seed["human_replay_posts"]
        }
        used_source_ids = []
        for index, (post, action) in enumerate(
            zip(record.get("posts", []), record.get("actions", []))
        ):
            entry = by_index.get(index, {})
            source_id = str(entry.get("source_tweet_id", ""))
            if (
                action.get("grounding_source_id") != source_id
                or source_id not in source_posts
            ):
                evidence_id_violations.append(index)
                continue
            used_source_ids.append(source_id)
            source_post = source_posts[source_id]
            source_text = clean_text(source_post["text"])
            expected_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            if entry.get("source_text_sha256") != expected_hash:
                evidence_hash_violations.append(index)
            reason = grounding_rejection_reason(
                post["text"],
                [profile_support, source_text],
                evidence_support_texts=[source_text],
                minimum_token_precision=MIN_EVIDENCE_TOKEN_PRECISION,
            )
            if reason is not None:
                violations.append({
                    "post_index": index,
                    "reason": reason,
                    "markers": unsupported_grounding_markers(
                        post["text"], [profile_support, source_text]
                    ),
                })
            warnings = grounding_soft_warnings(
                post["text"], [profile_support, source_text]
            )
            if warnings:
                soft_warnings.append({"post_index": index, "warnings": warnings})
        evidence_reuse_violations = sorted(
            source_id
            for source_id, count in Counter(used_source_ids).items()
            if count > 1
        )
    else:
        db_path = Path(provenance.get("oasis_database", ""))
        if not db_path.is_file():
            return {
                "applicable": True,
                "passed": False,
                "error": "database_missing",
                "database": str(db_path),
            }
        with sqlite3.connect(db_path) as connection:
            evidence_posts = {
                post_id: {"text": content, "author_id": user_id}
                for post_id, content, user_id in connection.execute(
                    "SELECT post_id, content, user_id FROM post"
                ).fetchall()
            }
        used_evidence_by_type: dict[str, list[int]] = {
            "post": [], "comment": [], "repost": []
        }
        for index, (post, action) in enumerate(
            zip(record.get("posts", []), record.get("actions", []))
        ):
            entry = by_index.get(index, {})
            evidence_post_id = entry.get("oasis_post_id")
            evidence_post = evidence_posts.get(evidence_post_id)
            if evidence_post is None:
                evidence_id_violations.append(index)
                continue
            if evidence_post["author_id"] == 0:
                evidence_author_violations.append(index)
            used_evidence_by_type.setdefault(action.get("type", ""), []).append(
                evidence_post_id
            )
            cleaned_evidence = clean_text(evidence_post["text"])
            expected_hash = hashlib.sha256(
                cleaned_evidence.encode("utf-8")
            ).hexdigest()
            if entry.get("evidence_text_sha256") != expected_hash:
                evidence_hash_violations.append(index)
            if action.get("type") == "repost":
                continue
            reason = grounding_rejection_reason(
                post["text"],
                [profile_support, cleaned_evidence],
                ownership_support_texts=[profile_support],
                evidence_support_texts=[cleaned_evidence],
                minimum_token_precision=MIN_EVIDENCE_TOKEN_PRECISION,
            )
            if reason is not None:
                violations.append({
                    "post_index": index,
                    "reason": reason,
                    "markers": unsupported_grounding_markers(
                        post["text"], [profile_support, cleaned_evidence]
                    ),
                })
            warnings = grounding_soft_warnings(
                post["text"], [profile_support, cleaned_evidence]
            )
            if warnings:
                soft_warnings.append({"post_index": index, "warnings": warnings})
        evidence_reuse_violations = {
            action_type: sorted(
                evidence_id
                for evidence_id, count in Counter(evidence_ids).items()
                if count > 1
            )
            for action_type, evidence_ids in used_evidence_by_type.items()
            if any(count > 1 for count in Counter(evidence_ids).values())
        }

    passed = (
        len(evidence_entries) == len(record.get("actions", []))
        and not duplicate_indices
        and not missing_indices
        and not unexpected_indices
        and not invalid_indices
        and not violations
        and not evidence_hash_violations
        and not evidence_id_violations
        and not evidence_author_violations
        and not evidence_reuse_violations
    )
    return {
        "applicable": True,
        "passed": passed,
        "evidence_count": len(evidence_entries),
        "duplicate_action_indices": duplicate_indices,
        "missing_action_indices": missing_indices,
        "unexpected_action_indices": unexpected_indices,
        "invalid_action_indices": invalid_indices,
        "grounding_violations": violations,
        "soft_grounding_warnings": soft_warnings,
        "evidence_hash_violations": evidence_hash_violations,
        "evidence_id_violations": evidence_id_violations,
        "evidence_author_violations": evidence_author_violations,
        "evidence_reuse_violations": evidence_reuse_violations,
    }


def structural_audit(record: dict[str, Any], expected_posts: int) -> dict[str, Any]:
    posts = record.get("posts", [])
    actions = record.get("actions", [])
    nodes = {node.get("id") for node in record.get("graph", {}).get("nodes", [])}
    alignment_violations = []
    for index, (post, action) in enumerate(zip(posts, actions)):
        if (
            post.get("action_type") != action.get("type")
            or post.get("interaction_targets") != action.get("target_ids")
            or post.get("created_at") != action.get("created_at")
            or post.get("time_step") != action.get("time_step")
        ):
            alignment_violations.append(index)
    missing_targets = sorted(
        {
            target
            for action in actions
            for target in action.get("target_ids", [])
            if target not in nodes
        }
    )
    actual_edges = Counter(
        (edge.get("target"), edge.get("type"), edge.get("time_step"))
        for edge in record.get("graph", {}).get("edges", [])
    )
    expected_edges = Counter(
        (action["target_ids"][0], action.get("type"), action.get("time_step"))
        for action in actions
        if action.get("target_ids")
    )
    passed = (
        len(posts) == expected_posts
        and len(actions) == expected_posts
        and len(posts) == len(actions)
        and not alignment_violations
        and not missing_targets
        and actual_edges == expected_edges
    )
    return {
        "passed": passed,
        "post_count": len(posts),
        "action_count": len(actions),
        "alignment_violations": alignment_violations,
        "missing_graph_targets": missing_targets,
        "graph_edges_match_actions": actual_edges == expected_edges,
    }


def agent_trace_audit(
    record: dict[str, Any], observation_copy_threshold: float
) -> dict[str, Any]:
    db_path = Path(record["provenance"]["oasis_database"])
    if not db_path.is_file():
        return {"passed": False, "database": str(db_path), "error": "database_missing"}
    rows = trace_rows(db_path)
    with sqlite3.connect(db_path) as connection:
        external_posts = connection.execute(
            "SELECT post_id, content FROM post WHERE user_id != ? ORDER BY post_id",
            (0,),
        ).fetchall()
        actor_comments = connection.execute(
            "SELECT content, post_id FROM comment WHERE user_id = ? ORDER BY created_at, comment_id",
            (0,),
        ).fetchall()
    evidence_by_index = {
        entry.get("action_index"): entry
        for entry in record.get("provenance", {}).get("grounding_evidence", [])
        if isinstance(entry, dict)
    }
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    trace_counts = Counter(row["action"] for row in rows)
    action_counts = Counter(action["type"] for action in record["actions"])
    generation_rounds = int(record["provenance"]["generation_rounds"])
    max_refreshes_per_round = int(
        record["provenance"].get("max_observation_refreshes_per_round", 1)
    )
    refresh_count = trace_counts["refresh"]
    unexpected_trace_actions = sorted(
        set(trace_counts) - {"sign_up", "refresh", "create_post", "create_comment", "repost"}
    )
    refresh_count_valid = (
        generation_rounds <= refresh_count
        <= generation_rounds * max_refreshes_per_round
    )
    expected_counts = Counter(
        {
            "sign_up": 1,
            "create_post": action_counts["post"],
            "create_comment": action_counts["comment"],
            "repost": action_counts["repost"],
        }
    )
    manifest_original = Counter(
        post["text"]
        for post in record["posts"]
        if post["action_type"] in {"post", "comment"}
    )
    trace_original = Counter(
        row["info"].get("content", "")
        for row in rows
        if row["action"] in {"create_post", "create_comment"}
    )
    copy_violations = []
    external_texts = [(post_id, normalized_text(content)) for post_id, content in external_posts]
    for index, post in enumerate(record["posts"]):
        if post["action_type"] != "post":
            continue
        candidate = normalized_text(post["text"])
        for target_post_id, target_text in external_texts:
            similarity = SequenceMatcher(None, candidate, target_text).ratio()
            if similarity >= observation_copy_threshold:
                copy_violations.append(
                    {
                        "action_index": index,
                        "action_type": "post",
                        "target_post_id": target_post_id,
                        "similarity": round(similarity, 6),
                    }
                )
    manifest_comments = [
        (index, post)
        for index, post in enumerate(record["posts"])
        if post["action_type"] == "comment"
    ]
    comments_align = len(manifest_comments) == len(actor_comments)
    for (index, post), (content, target_post_id) in zip(manifest_comments, actor_comments):
        comments_align = comments_align and post["text"] == content
        comments_align = comments_align and (
            evidence_by_index.get(index, {}).get("oasis_post_id") == target_post_id
        )
        target_text = dict(external_texts).get(target_post_id)
        if target_text is None:
            comments_align = False
            continue
        similarity = SequenceMatcher(
            None, normalized_text(post["text"]), target_text
        ).ratio()
        if similarity >= observation_copy_threshold:
            copy_violations.append(
                {
                    "action_index": index,
                    "action_type": "comment",
                    "target_post_id": target_post_id,
                    "similarity": round(similarity, 6),
                }
            )
    manifest_reposts = [
        index
        for index, post in enumerate(record["posts"])
        if post["action_type"] == "repost"
    ]
    trace_repost_targets = [
        row["info"].get("reposted_id")
        for row in rows
        if row["action"] == "repost"
    ]
    reposts_align = len(manifest_reposts) == len(trace_repost_targets) and all(
        evidence_by_index.get(index, {}).get("oasis_post_id") == target_post_id
        for index, target_post_id in zip(manifest_reposts, trace_repost_targets)
    )
    passed = (
        digest == record["provenance"].get("oasis_trace_sha256")
        and all(
            trace_counts[action] == expected_counts[action]
            for action in ("sign_up", "create_post", "create_comment", "repost")
        )
        and refresh_count_valid
        and not unexpected_trace_actions
        and manifest_original == trace_original
        and comments_align
        and reposts_align
        and not copy_violations
    )
    return {
        "passed": passed,
        "database": str(db_path),
        "trace_sha256_matches": digest
        == record["provenance"].get("oasis_trace_sha256"),
        "trace_action_counts": dict(sorted(trace_counts.items())),
        "expected_action_counts": {
            **dict(sorted(expected_counts.items())),
            "refresh": generation_rounds,
        },
        "refresh_count": refresh_count,
        "refresh_count_range": [
            generation_rounds,
            generation_rounds * max_refreshes_per_round,
        ],
        "refresh_count_valid": refresh_count_valid,
        "unexpected_trace_actions": unexpected_trace_actions,
        "original_texts_match_trace": manifest_original == trace_original,
        "comments_match_database_targets": comments_align,
        "reposts_match_trace_targets": reposts_align,
        "observation_copy_violations": copy_violations,
    }


def pair_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(record["label_name"] for record in records)
    duplicate_labels = sorted(
        label for label, count in label_counts.items() if count > 1
    )
    by_label = {record["label_name"]: record for record in records}
    missing = sorted({"llm_bot", "full_stack_agent"} - set(by_label))
    if missing or duplicate_labels:
        return {
            "passed": False,
            "missing_labels": missing,
            "duplicate_labels": duplicate_labels,
        }
    llm_record = by_label["llm_bot"]
    agent_record = by_label["full_stack_agent"]
    same_generator = all(
        llm_record["provenance"].get(key) == agent_record["provenance"].get(key)
        for key in GENERATOR_KEYS
    )
    same_semantic_verifier = all(
        llm_record["provenance"].get(key)
        == agent_record["provenance"].get(key)
        for key in SEMANTIC_CONFIG_KEYS
    )
    result = {
        "same_profile": llm_record["profile"] == agent_record["profile"],
        "same_timestamps": [post["created_at"] for post in llm_record["posts"]]
        == [post["created_at"] for post in agent_record["posts"]],
        "same_generator_distribution": same_generator,
        "same_semantic_verifier": same_semantic_verifier,
        "missing_labels": missing,
        "duplicate_labels": duplicate_labels,
    }
    result["passed"] = all(result[key] for key in (
        "same_profile",
        "same_timestamps",
        "same_generator_distribution",
        "same_semantic_verifier",
    )) and not missing and not duplicate_labels
    return result


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.records)
    seeds = {record["base_profile_id"]: record for record in load_jsonl(args.seeds)}
    sample_id_counts = Counter(record.get("sample_id") for record in records)
    duplicate_sample_ids = sorted(
        str(sample_id)
        for sample_id, count in sample_id_counts.items()
        if count > 1
    )
    base_label_counts = {
        base_id: Counter(
            record.get("label_name")
            for record in records
            if record.get("base_profile_id") == base_id
        )
        for base_id in {record.get("base_profile_id") for record in records}
    }
    invalid_base_label_sets = {
        str(base_id): dict(sorted(counts.items(), key=lambda item: str(item[0])))
        for base_id, counts in base_label_counts.items()
        if counts != Counter({"llm_bot": 1, "full_stack_agent": 1})
    }
    missing_seed_base_ids = sorted(
        str(base_id) for base_id in base_label_counts if base_id not in seeds
    )
    expected_base_count_matches = (
        args.expected_base_profiles <= 0
        or len(base_label_counts) == args.expected_base_profiles
    )
    dataset_integrity = {
        "nonempty": bool(records),
        "duplicate_sample_ids": duplicate_sample_ids,
        "invalid_base_label_sets": invalid_base_label_sets,
        "missing_seed_base_ids": missing_seed_base_ids,
        "expected_base_profiles": args.expected_base_profiles,
        "actual_base_profiles": len(base_label_counts),
        "expected_base_count_matches": expected_base_count_matches,
    }
    dataset_integrity["passed"] = (
        dataset_integrity["nonempty"]
        and not duplicate_sample_ids
        and not invalid_base_label_sets
        and not missing_seed_base_ids
        and expected_base_count_matches
    )
    record_audits = []
    semantic_case_ids = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        base_id = record["base_profile_id"]
        grouped.setdefault(base_id, []).append(record)
        texts = [post["text"] for post in record.get("posts", [])]
        action_types = [post["action_type"] for post in record.get("posts", [])]
        duplicates = duplicate_audit(
            texts,
            action_types,
            args.near_duplicate_threshold,
            args.token_jaccard_threshold,
        )
        identity_violations = (
            [
                index
                for index, (text, action_type) in enumerate(zip(texts, action_types))
                if action_type != "repost" and FIRST_PERSON_SINGULAR_RE.search(text)
            ]
            if organization_voice(record.get("profile", {}))
            else []
        )
        leaks = [
            {"post_index": index, "types": raw_identifier_leaks(text)}
            for index, text in enumerate(texts)
            if raw_identifier_leaks(text)
        ]
        source_similarity = None
        if record["label_name"] == "llm_bot":
            source_similarity = source_similarity_audit(
                texts,
                seeds[base_id]["human_replay_posts"],
                args.source_similarity_threshold,
            )
        trace = (
            agent_trace_audit(record, args.source_similarity_threshold)
            if record["label_name"] == "full_stack_agent"
            else None
        )
        grounding = (
            grounding_audit(record, seeds[base_id])
            if base_id in seeds
            else {"applicable": True, "passed": False, "error": "seed_missing"}
        )
        semantic_grounding = (
            semantic_grounding_audit(record, seeds[base_id])
            if base_id in seeds
            else {
                "applicable": True,
                "passed": False,
                "error": "seed_missing",
                "case_ids": [],
            }
        )
        semantic_case_ids.extend(semantic_grounding.get("case_ids", []))
        structure = structural_audit(record, args.expected_posts)
        passed = (
            structure["passed"]
            and duplicates["normalized_unique_count"] == len(texts)
            and not duplicates["near_duplicates"]
            and not duplicates["lexical_duplicates"]
            and not duplicates["repeated_openings"]
            and not duplicates["repeated_generic_templates"]
            and not leaks
            and not identity_violations
            and (source_similarity is None or not source_similarity["violations"])
            and (trace is None or trace["passed"])
            and grounding["passed"]
            and semantic_grounding["passed"]
        )
        record_audits.append(
            {
                "sample_id": record["sample_id"],
                "base_profile_id": base_id,
                "label_name": record["label_name"],
                "passed": passed,
                "structure": structure,
                "duplicates": duplicates,
                "raw_identifier_leaks": leaks,
                "organization_voice_identity_violations": identity_violations,
                "source_similarity": source_similarity,
                "agent_trace": trace,
                "grounding": grounding,
                "semantic_grounding": semantic_grounding,
            }
        )

    duplicate_semantic_case_ids = sorted(
        case_id
        for case_id, count in Counter(semantic_case_ids).items()
        if count > 1
    )
    dataset_integrity["duplicate_semantic_case_ids"] = (
        duplicate_semantic_case_ids
    )
    dataset_integrity["passed"] = (
        dataset_integrity["passed"] and not duplicate_semantic_case_ids
    )
    pair_audits = [
        {"base_profile_id": base_id, **pair_audit(base_records)}
        for base_id, base_records in sorted(grouped.items())
    ]
    report = {
        "schema_version": "quadbot-v3-independent-stage2-audit-6",
        "records": str(args.records),
        "seeds": str(args.seeds),
        "record_count": len(records),
        "base_profile_count": len(grouped),
        "thresholds": {
            "near_duplicate": args.near_duplicate_threshold,
            "token_jaccard": args.token_jaccard_threshold,
            "source_similarity": args.source_similarity_threshold,
        },
        "record_audits": record_audits,
        "pair_audits": pair_audits,
        "dataset_integrity": dataset_integrity,
        "all_records_passed": all(item["passed"] for item in record_audits),
        "all_pairs_passed": all(item["passed"] for item in pair_audits),
    }
    report["passed"] = (
        report["dataset_integrity"]["passed"]
        and report["all_records_passed"]
        and report["all_pairs_passed"]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "record_count": report["record_count"],
        "base_profile_count": report["base_profile_count"],
        "all_records_passed": report["all_records_passed"],
        "all_pairs_passed": report["all_pairs_passed"],
        "passed": report["passed"],
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
