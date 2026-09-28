#!/usr/bin/env python3
"""CPU-only regression for generation/audit parity and release budgets."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import sqlite3
import types
import unittest
from collections import Counter
from pathlib import Path

import audit_quadbot_stage2 as independent
import audit_stage2_text_quality as quality


ROOT = Path(__file__).resolve().parent
STANDARD_IMPORTS = {
    "argparse", "asyncio", "hashlib", "json", "os", "random", "re",
    "sqlite3", "unicodedata", "collections", "datetime", "difflib",
    "pathlib", "typing", "__future__",
}


def load_generator_helpers() -> dict:
    # Exercise the real functions without loading model/GPU dependencies.
    path = ROOT / "generate_llm_agent_pilot.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.AsyncFunctionDef)):
            body.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module in STANDARD_IMPORTS:
            body.append(node)
        elif isinstance(node, ast.Import) and all(alias.name in STANDARD_IMPORTS for alias in node.names):
            body.append(node)
    namespace = {"__file__": str(path)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ReleaseGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = load_generator_helpers()

    def test_source_sentence_fallback_remains_bound_to_evidence(self) -> None:
        source = (
            "<USER> <USER> Right. They are not the solution and convincing "
            "60 senators would be necessary. The earlier bill passed by a whisker."
        )
        candidates = self.generator["source_sentence_candidates"](source)
        self.assertIn(
            "They are not the solution and convincing 60 senators would be necessary.",
            candidates,
        )
        self.assertTrue(all("<USER>" not in candidate for candidate in candidates))
        self.assertTrue(all("new outcome" not in candidate for candidate in candidates))

    def test_trailing_phrase_audit_keeps_complete_phrasal_verbs(self) -> None:
        self.assertFalse(quality.has_incomplete_trailing_phrase(
            "RT <USER>: Couldn't risk another team jumping in"
        ))
        self.assertTrue(quality.has_incomplete_trailing_phrase(
            "RT <USER>: The latest discussion is about a plan in"
        ))
        self.assertTrue(quality.has_incomplete_trailing_phrase(
            "A proposal for"
        ))

    def test_source_bound_repost_is_not_rejected_for_source_tail(self) -> None:
        # Reposts must reproduce the observed source exactly, even when that
        # source ends in a function word because of upstream historical data.
        self.assertTrue(quality.has_incomplete_trailing_phrase(
            "RT <USER>: Check who is paying the price for."
        ))

    def test_grounding_policy_matches_independent_audit(self) -> None:
        self.assertEqual(
            self.generator["GROUNDING_TOKEN_STOP_WORDS"],
            independent.GROUNDING_TOKEN_STOP_WORDS,
        )
        for text in (
            "Worth noting: Climate research expands in Paris.",
            "A useful point around interesting discussion.",
            "RT @alice: Visit https://example.com now! #Science",
            "We've discussed education; it is a closer look at science.",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    self.generator["grounding_content_tokens"](text),
                    independent.grounding_content_tokens(text),
                )

    def test_generic_filler_cannot_hide_weak_evidence(self) -> None:
        candidate = "Worth noting: research is an interesting useful point."
        evidence = ["Research continues."]
        self.assertLess(self.generator["evidence_token_precision"](candidate, evidence), 0.5)
        self.assertEqual(
            self.generator["grounding_rejection_reason"](
                candidate, evidence, minimum_token_precision=0.5,
            ),
            "low_evidence_token_precision",
        )

    def test_valid_llm_variants_survive_one_bad_variant(self) -> None:
        value = {"items": [
            {"text": "bad"}, {"text": "Research continues in Paris."},
            {"text": "Paris research continues."},
        ]}
        self.assertEqual(len(self.generator["validate_llm_items"](value, 1)), 2)
        with self.assertRaises(ValueError):
            self.generator["validate_llm_items"]({"items": [{"text": "bad"}]}, 1)

    def test_grounding_retry_exposes_audit_vocabulary_and_blocks_hashtag_inference(self) -> None:
        anchor = (
            "A lesson for all comedians to stick to comedy. Politics is not their "
            "cup of tea! #Zelensky #UkraineConflict"
        )
        messages = self.generator["llm_single_messages"](
            {"profile": {}, "human_replay_posts": []},
            [],
            anchor,
            2,
            4,
            1,
            {
                "low_evidence_token_precision": 2,
                "semantic:unsupported_entity_or_relation": 1,
            },
        )
        system = messages[0]["content"]
        user = messages[1]["content"]
        self.assertIn("Treat hashtags as topical metadata", system)
        self.assertIn("use only the content words listed", system)
        self.assertIn('"allowed_grounding_content_words"', user)
        self.assertNotIn('"zelensky"', user.casefold().split('"allowed_grounding_content_words"', 1)[1].split(']', 1)[0])
        self.assertIn('"comedians"', user.casefold())
        self.assertIn('"politics"', user.casefold())

    def test_normal_and_rescue_selection_enforce_remaining_budget(self) -> None:
        g = self.generator
        g["agent_candidate_rejection_reason"] = lambda *args: None
        g["agent_deterministic_rejection_reason"] = lambda *args: None
        g["matches_existing_text"] = lambda *args: False
        g["repeats_opening"] = lambda *args: False
        g["repeats_generic_template"] = lambda *args: False
        candidates = [
            {
                "type": "post", "content": f"Evidence variant number {index} here.",
                "evidence_post_id": index, "_semantic_grounding": {"verdict": "pass"},
                **({"_deterministic_text_fallback": "evidence_phrase"} if index < 2 else {}),
            }
            for index in range(4)
        ]
        for selector in ("select_agent_batch", "force_select_agent_batch"):
            for budget in (0, 1, 2):
                with self.subTest(selector=selector, budget=budget):
                    result = g[selector](
                        candidates, 2, {}, set(), False, "", set(), set(), {},
                        maximum_text_fallbacks=budget,
                    )
                    self.assertIsNotNone(result)
                    self.assertLessEqual(
                        sum(item.get("_deterministic_text_fallback") is not None for item in result),
                        budget,
                    )
            result = g[selector](
                candidates[:2], 2, {}, set(), False, "", set(), set(), {"post": 2},
                maximum_text_fallbacks=1,
            )
            self.assertIsNone(result)

    def test_rescue_cannot_bypass_within_batch_opening_checks(self) -> None:
        g = self.generator
        g["agent_deterministic_rejection_reason"] = lambda *args: None
        g["matches_existing_text"] = lambda *args: False
        g["repeats_generic_template"] = lambda *args: False
        g["repeats_opening"] = lambda text, previous: bool(previous)
        candidates = [
            {
                "type": "post", "content": f"Worth noting research variant {index}.",
                "evidence_post_id": index, "_semantic_grounding": {"verdict": "pass"},
            }
            for index in range(2)
        ]
        self.assertIsNone(g["force_select_agent_batch"](
            candidates, 2, {}, set(), False, "", set(), set(), {},
        ))

    def test_rescue_cannot_exhaust_early_repost_targets(self) -> None:
        g = self.generator
        pool = [
            {"type": "repost", "target_post_id": target}
            for target in range(10, 20)
        ] + [{"type": "comment", "target_post_id": 1, "content": "A grounded comment."}]
        limited = g["limit_early_repost_candidates"](
            pool, round_index=2, rounds=5, reposts_already_executed=6,
        )
        self.assertEqual(
            {item["target_post_id"] for item in limited if item["type"] == "repost"},
            set(range(10, 20)),
        )
        self.assertEqual(
            g["limit_early_repost_candidates"](
                pool, round_index=3, rounds=5, reposts_already_executed=20,
            ),
            [pool[-1]],
        )
        self.assertEqual(
            g["limit_early_repost_candidates"](
                pool, round_index=4, rounds=5, reposts_already_executed=20,
            ),
            pool,
        )

    def test_required_agent_types_are_satisfied_by_second_round(self) -> None:
        minimums = self.generator["agent_round_minimums"]
        empty = Counter()
        self.assertEqual(minimums(empty, round_index=0, rounds=5), {})
        self.assertEqual(
            minimums(empty, round_index=1, rounds=5),
            {"post": 1, "repost": 1},
        )
        post_only = Counter({"post": 1})
        self.assertEqual(
            minimums(post_only, round_index=1, rounds=5),
            {"repost": 1},
        )
        complete = Counter({"post": 1, "repost": 1})
        self.assertEqual(minimums(complete, round_index=4, rounds=5), {})

    def test_llm_generator_stops_before_sixth_text_fallback(self) -> None:
        g = self.generator
        sources = [
            {"text": f"Research evidence variant {index} remains available.",
             "source_tweet_id": str(index), "created_at": "2022-01-01T00:00:00Z"}
            for index in range(25)
        ]
        g["llm_rejection_reason"] = lambda *args: None
        g["deterministic_llm_fallback_candidates"] = lambda text: [text]
        g["safe_deterministic_fallback_candidate"] = lambda text: True
        g["update_local_semantic_stats"] = lambda *args: None
        g["semantic_evidence_fields"] = lambda *args: {}

        def failed_generation(*args):
            raise ValueError("test generation failure")

        verifier = types.SimpleNamespace(verify_many=lambda cases: {
            case["case_id"]: {"verdict": "pass", "reason": "supported"} for case in cases
        })
        seed = {"base_profile_id": "regression", "profile": {}, "human_replay_posts": sources}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "text_fallbacks=5/5"):
            g["generate_llm_bot"](
                types.SimpleNamespace(generate_json=failed_generation), verifier, seed, 1, 5, 5,
            )


def check_corpus(records_path: Path, seeds_path: Path) -> dict:
    g = load_generator_helpers()
    records = independent.load_jsonl(records_path)
    seeds = {seed["base_profile_id"]: seed for seed in independent.load_jsonl(seeds_path)}
    checked = 0
    rejected = 0
    for record in records:
        seed = seeds[record["base_profile_id"]]
        if record["label_name"] == "llm_bot":
            sources = {
                str(item["source_tweet_id"]): item["text"] for item in seed["human_replay_posts"]
            }
            id_key = "source_tweet_id"
        else:
            db_path = Path(record["provenance"]["oasis_database"])
            if not db_path.is_file():
                raise FileNotFoundError(db_path)
            with sqlite3.connect(db_path) as connection:
                sources = dict(connection.execute("SELECT post_id, content FROM post"))
            id_key = "oasis_post_id"
        for entry in record["provenance"]["grounding_evidence"]:
            index = entry["action_index"]
            if record["actions"][index]["type"] == "repost":
                continue
            text = record["posts"][index]["text"]
            evidence = [g["clean_text"](sources[entry[id_key]])]
            actual = g["evidence_token_precision"](text, evidence)
            expected = independent.evidence_token_precision(text, evidence)
            if actual != expected:
                raise AssertionError(f"precision mismatch: {record['sample_id']}/{index}: {actual} != {expected}")
            checked += 1
            rejected += actual < g["MIN_EVIDENCE_TOKEN_PRECISION"]
    return {"checked_text_evidence_pairs": checked, "weak_evidence_cases_now_rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path)
    parser.add_argument("--seeds", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReleaseGateTests))
    summary = {"unit_tests": result.testsRun, "passed": result.wasSuccessful()}
    if args.records or args.seeds:
        if not args.records or not args.seeds:
            parser.error("--records and --seeds must be supplied together")
        summary.update(check_corpus(args.records, args.seeds))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
