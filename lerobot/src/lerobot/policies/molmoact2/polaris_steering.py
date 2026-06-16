"""Steerable / Quiz SFT for MolmoAct2 (polaris).

Loads the steering annotations produced by
``scripts/imitation_learning/generate_steering_annotations.py`` (polaris_real2sim) and
exposes the per-frame command swap (steerable) and quiz-target tokenization (quiz). Mirrors
``third_party/openpi/src/openpi/training/misc/polaris_steering.py``; the annotation JSON is
tokenizer-agnostic (it stores ``quiz_anchor``/``quiz_geometry`` strings), so the same file
drives both backends and MolmoAct2 re-tokenizes the strings with its Qwen2 tokenizer.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

# (frame_start, frame_end_exclusive, commands)
Segment = tuple[int, int, tuple[str, ...]]
# (frame_start, frame_end_exclusive, commands, anchor, geometry)
QuizSegment = tuple[int, int, tuple[str, ...], str, str]

# Per-process RNG: dataloader workers are spawned, so each process re-seeds from OS entropy
# and commands are re-sampled on every dataset pass (matches the reference implementation).
_rng = np.random.default_rng()


def load_steering_segments(path: str | pathlib.Path) -> dict[int, tuple[Segment, ...]]:
    payload = json.loads(pathlib.Path(path).read_text())
    table: dict[int, tuple[Segment, ...]] = {}
    for episode in payload["episodes"]:
        segments = tuple(
            (
                int(segment["frame_start"]),
                int(segment["frame_end"]),
                tuple(str(command["text"]) for command in segment["commands"]),
            )
            for segment in episode["segments"]
        )
        table[int(episode["episode_index"])] = segments
    return table


def load_quiz_segments(path: str | pathlib.Path) -> dict[int, tuple[QuizSegment, ...]]:
    payload = json.loads(pathlib.Path(path).read_text())
    table: dict[int, tuple[QuizSegment, ...]] = {}
    for episode in payload["episodes"]:
        segments = tuple(
            (
                int(segment["frame_start"]),
                int(segment["frame_end"]),
                tuple(str(command["text"]) for command in segment["commands"]),
                str(segment.get("quiz_anchor", "")),
                str(segment.get("quiz_geometry", "")),
            )
            for segment in episode["segments"]
        )
        table[int(episode["episode_index"])] = segments
    return table


def covering_segment(segments: tuple[Any, ...] | None, frame: int) -> Any | None:
    if not segments:
        return None
    return next((seg for seg in segments if seg[0] <= frame < seg[1]), None)


def sample_command(commands: tuple[str, ...], steer_prob: float) -> str | None:
    """Return a uniformly sampled command with probability ``steer_prob``, else None."""
    if not commands or float(_rng.random()) >= steer_prob:
        return None
    return str(commands[int(_rng.integers(len(commands)))])


def build_quiz_target_tokens(
    tokenizer: Any,
    anchor: str,
    geometry: str,
    *,
    steered: bool,
    length: int,
    pad_token_id: int,
) -> tuple[list[int], list[bool], list[bool]]:
    """Tokenize the hybrid quiz target ("anchor. geometry") to a fixed length.

    Mirrors ``QuizSampler._build`` (openpi). When the prompt was steered the loss mask covers
    only the geometry span (the anchor is in the input prompt and would otherwise leak). Unlike
    the PaliGemma tokenizer, Qwen2 with ``add_special_tokens=False`` appends no trailing tokens,
    so the geometry-span count needs no ``-2`` correction.
    """
    target = f"{anchor}. {geometry}".strip() if geometry else anchor
    ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])[:length]
    real = len(ids)
    token_mask = [True] * real + [False] * (length - real)
    ids = ids + [pad_token_id] * (length - real)
    loss_mask = list(token_mask)
    if steered and geometry:
        n_geom = len(tokenizer(geometry, add_special_tokens=False)["input_ids"])
        keep = [False] * length
        for i in range(max(0, real - n_geom), real):
            keep[i] = True
        loss_mask = [m and k for m, k in zip(loss_mask, keep, strict=True)]
    return ids, token_mask, loss_mask
