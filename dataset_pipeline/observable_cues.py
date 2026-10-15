"""Label-free post observations; no actions/provenance/graph serialization."""

import math
from collections import Counter

import numpy as np

from dataset_pipeline.export_quadbot_v3_features import (
    HASHTAG_RE, TOKEN_RE, URL_RE, clean_text, hash_embed, profile_text,
)
from dataset_pipeline.export_relative_timing import gap_statistics, parse_time


TEXT_DIM = 256
STEPS = 5
BEHAVIOR_NAMES = (
    "post_fraction", "reply_fraction", "repost_fraction", "other_fraction",
    "target_presence_fraction", "log_mean_target_count", "action_entropy",
    "log_post_count",
)
STYLE_NAMES = (
    "log_mean_characters", "log_sd_characters", "unique_token_fraction",
    "punctuation_fraction", "urls_per_post", "hashtags_per_post",
    "uppercase_fraction", "non_ascii_fraction",
)
FEATURE_DIMENSIONS = {
    "description": TEXT_DIM, "numeric": 5, "text": TEXT_DIM,
    "behavior": len(BEHAVIOR_NAMES), "style": len(STYLE_NAMES), "timing": 7,
}


def public_action_type(post):
    kind = clean_text(post.get("action_type") or "post").casefold()
    if kind in ("post", "tweet", "original"):
        return "post"
    if kind in ("reply", "comment"):
        return "reply"
    if kind in ("repost", "retweet", "quote", "quote_tweet"):
        return "repost"
    return "other"


def behavior_cues(posts):
    counts = Counter(public_action_type(post) for post in posts)
    targets = [post.get("interaction_targets") or [] for post in posts]
    if any(not isinstance(value, list) for value in targets):
        raise ValueError("public interaction targets must be lists")
    count = len(posts)
    entropy = -sum((value / count) * math.log(value / count) for value in counts.values())
    return np.asarray([
        *(counts[kind] / count for kind in ("post", "reply", "repost", "other")),
        sum(bool(value) for value in targets) / count,
        math.log1p(sum(map(len, targets)) / count),
        entropy, math.log1p(count),
    ], dtype=np.float32)


def style_cues(texts):
    joined = " ".join(texts)
    tokens, characters = TOKEN_RE.findall(joined), max(len(joined), 1)
    return np.asarray([
        math.log1p(float(np.mean([len(text) for text in texts]))),
        math.log1p(float(np.std([len(text) for text in texts]))),
        len({token.casefold() for token in tokens}) / max(len(tokens), 1),
        sum(character in "!?.,;:" for character in joined) / characters,
        len(URL_RE.findall(joined)) / len(texts),
        len(HASHTAG_RE.findall(joined)) / len(texts),
        sum(character.isupper() for character in joined) / max(sum(character.isalpha() for character in joined), 1),
        sum(ord(character) > 127 for character in joined) / characters,
    ], dtype=np.float32)


def observe(row):
    posts = row.get("posts")
    if not isinstance(posts, list) or len(posts) < STEPS:
        raise ValueError("at least five timestamped public posts are required")
    times = [parse_time(post.get("created_at")) for post in posts]
    # Rank rounded relative timestamps, not simulator step numbers or epochs.
    rounded = np.round(np.asarray(times) - min(times))
    order = sorted(range(len(posts)), key=lambda index: (rounded[index], index))
    windows = np.array_split(order, STEPS)
    cues = {name: [] for name in ("text", "behavior", "style", "timing")}
    for indices in windows:
        window = [posts[int(index)] for index in indices]
        texts = [clean_text(post.get("text")) for post in window]
        vectors = [hash_embed(text, "observable-post-text", TEXT_DIM) for text in texts]
        vector = np.mean(vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            vector /= norm
        cues["text"].append(vector)
        cues["behavior"].append(behavior_cues(window))
        cues["style"].append(style_cues(texts))
        cues["timing"].append(gap_statistics([rounded[int(index)] for index in indices]))
    profile = row.get("profile") or {}
    metrics = profile.get("public_metrics") or {}
    result = {
        "description": hash_embed(profile_text(row), "observable-profile", TEXT_DIM),
        "numeric": np.asarray([
            *(math.log1p(max(0, float(metrics.get(key, 0) or 0))) for key in (
                "followers_count", "following_count", "tweet_count", "listed_count")),
            math.log1p(len(posts)),
        ], dtype=np.float32),
        **{name: np.stack(values) for name, values in cues.items()},
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("public observation features must be finite")
    return result
