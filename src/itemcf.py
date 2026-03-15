from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm

MS_PER_DAY = 24 * 60 * 60 * 1000
MS_PER_HOUR = 60 * 60 * 1000
TYPE_TO_ID = {"clicks": 0, "carts": 1, "orders": 2}
ID_TO_TYPE = {value: key for key, value in TYPE_TO_ID.items()}
TARGETS = ("clicks", "carts", "orders")
TARGET_WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
EVENT_TYPE_WEIGHTS = {0: 1.0, 1: 3.0, 2: 6.0}
MATRIX_NAMES = ("cart_order", "click_buy", "click_cart_order")
MATRIX_WEIGHT_PRESETS = {
    "target_aware": {
        "cart_order": {
            "pair_weights": {
                (1, 1): 2.4,
                (1, 2): 2.4,
                (2, 1): 2.0,
                (2, 2): 2.8,
            },
            "backward_scale": 0.45,
            "time_scale_ms": 12 * 60 * 60 * 1000,
            "pop_alpha": 0.18,
        },
        "click_buy": {
            "pair_weights": {
                (0, 1): 1.0,
                (0, 2): 1.4,
                (1, 2): 0.7,
                (2, 1): 0.5,
                (1, 0): 0.3,
                (2, 0): 0.2,
            },
            "backward_scale": 0.15,
            "time_scale_ms": 8 * 60 * 60 * 1000,
            "pop_alpha": 0.14,
        },
        "click_cart_order": {
            "pair_weights": {
                (0, 0): 1.0,
                (0, 1): 0.8,
                (0, 2): 0.9,
                (1, 0): 0.5,
                (1, 1): 1.2,
                (1, 2): 1.4,
                (2, 0): 0.4,
                (2, 1): 1.1,
                (2, 2): 1.6,
            },
            "backward_scale": 0.65,
            "time_scale_ms": 24 * 60 * 60 * 1000,
            "pop_alpha": 0.10,
        },
    },
    "one_hot": {
        "cart_order": {
            "pair_weights": {
                (1, 1): 1.0,
                (1, 2): 1.0,
                (2, 1): 1.0,
                (2, 2): 1.0,
            },
            "backward_scale": 0.5,
            "time_scale_ms": 12 * 60 * 60 * 1000,
            "pop_alpha": 0.15,
        },
        "click_buy": {
            "pair_weights": {
                (0, 1): 1.0,
                (0, 2): 1.0,
                (1, 0): 1.0,
                (2, 0): 1.0,
            },
            "backward_scale": 0.2,
            "time_scale_ms": 8 * 60 * 60 * 1000,
            "pop_alpha": 0.12,
        },
        "click_cart_order": {
            "pair_weights": {
                (0, 0): 1.0,
                (0, 1): 1.0,
                (0, 2): 1.0,
                (1, 0): 1.0,
                (1, 1): 1.0,
                (1, 2): 1.0,
                (2, 0): 1.0,
                (2, 1): 1.0,
                (2, 2): 1.0,
            },
            "backward_scale": 0.7,
            "time_scale_ms": 24 * 60 * 60 * 1000,
            "pop_alpha": 0.08,
        },
    },
}
TARGET_MATRIX_WEIGHTS = {
    "target_aware": {
        "clicks": {"cart_order": 0.0, "click_buy": 0.25, "click_cart_order": 0.75},
        "carts": {"cart_order": 0.60, "click_buy": 0.30, "click_cart_order": 0.10},
        "orders": {"cart_order": 0.60, "click_buy": 0.25, "click_cart_order": 0.15},
    },
    "one_hot": {
        "clicks": {"cart_order": 0.0, "click_buy": 0.20, "click_cart_order": 0.80},
        "carts": {"cart_order": 0.50, "click_buy": 0.30, "click_cart_order": 0.20},
        "orders": {"cart_order": 0.70, "click_buy": 0.20, "click_cart_order": 0.10},
    },
}

Event = Tuple[int, int, int]
CovisitationMatrices = Dict[str, Dict[int, List[Tuple[int, float]]]]


@dataclass(slots=True)
class SessionRecord:
    session: int
    events: List[Event]
    labels: Optional[Dict[str, object]] = None


@dataclass(slots=True)
class PopularityArtifacts:
    all_rank: Dict[int, int]
    target_rank: Dict[str, Dict[int, int]]
    fallback: Dict[str, List[int]]


@dataclass(slots=True)
class CovisitationArtifacts:
    matrices: CovisitationMatrices
    target_matrix_weights: Dict[str, Dict[str, float]]
    pair_counts: Dict[int, Dict[int, float]]


def resolve_raw_files(raw_dir: Path) -> Dict[str, Path]:
    required = {
        "train": raw_dir / "train.jsonl",
        "test": raw_dir / "test.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing raw OTTO files: {joined}")
    return required


def iter_sessions(jsonl_path: Path) -> Iterator[Tuple[int, List[Event]]]:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            events = [
                (int(event["aid"]), int(event["ts"]), TYPE_TO_ID[event["type"]])
                for event in row["events"]
            ]
            yield int(row["session"]), events


def get_max_ts(train_path: Path) -> int:
    max_ts = 0
    for _, events in tqdm(iter_sessions(train_path), desc="scan max ts"):
        if events and events[-1][1] > max_ts:
            max_ts = events[-1][1]
    return max_ts


def update_reservoir(
    sample: List[SessionRecord],
    seen: int,
    session: SessionRecord,
    max_size: int,
    rng: random.Random,
) -> int:
    seen += 1
    if max_size <= 0:
        return seen
    if len(sample) < max_size:
        sample.append(session)
        return seen
    replace_at = rng.randrange(seen)
    if replace_at < max_size:
        sample[replace_at] = session
    return seen


def unique_recent_events(events: Sequence[Event], limit: int) -> List[Event]:
    seen = set()
    output: List[Event] = []
    for aid, ts, event_type in reversed(events):
        if aid in seen:
            continue
        seen.add(aid)
        output.append((aid, ts, event_type))
        if len(output) >= limit:
            break
    return output


def build_labels_from_suffix(suffix: Sequence[Event]) -> Dict[str, object]:
    labels: Dict[str, object] = {}
    next_click = None
    for aid, _, event_type in suffix:
        if event_type == TYPE_TO_ID["clicks"]:
            next_click = aid
            break
    if next_click is not None:
        labels["clicks"] = next_click

    carts = {aid for aid, _, event_type in suffix if event_type == TYPE_TO_ID["carts"]}
    if carts:
        labels["carts"] = carts

    orders = {aid for aid, _, event_type in suffix if event_type == TYPE_TO_ID["orders"]}
    if orders:
        labels["orders"] = orders
    return labels


def random_split_session(record: SessionRecord, rng: random.Random) -> Optional[SessionRecord]:
    if len(record.events) < 2:
        return None
    split_idx = rng.randint(1, len(record.events) - 1)
    prefix = list(record.events[:split_idx])
    suffix = record.events[split_idx:]
    labels = build_labels_from_suffix(suffix)
    if not labels:
        return None
    return SessionRecord(session=record.session, events=prefix, labels=labels)


def keep_known_aids(events: Sequence[Event], known_aids: set[int]) -> List[Event]:
    return [event for event in events if event[0] in known_aids]


class MultiCovisitationBuilder:
    def __init__(
        self,
        matrix_specs: Dict[str, Dict[str, object]],
        topk: int = 40,
        prune_topk: int = 80,
        forward_window: int = 5,
        pair_topk: int = 120,
        pair_prune_topk: int = 240,
    ) -> None:
        self.matrix_specs = matrix_specs
        self.topk = topk
        self.prune_topk = prune_topk
        self.forward_window = forward_window
        self.pair_topk = pair_topk
        self.pair_prune_topk = pair_prune_topk
        self.data: Dict[str, Dict[int, Dict[int, float]]] = {name: {} for name in MATRIX_NAMES}
        self.pair_counts: Dict[int, Dict[int, float]] = {}

    def add_session(self, events: Sequence[Event]) -> None:
        recent = list(reversed(unique_recent_events(events, limit=30)))
        if len(recent) < 2:
            return
        n_events = len(recent)
        for left_idx in range(n_events - 1):
            src_aid, src_ts, src_type = recent[left_idx]
            max_right = min(n_events, left_idx + 1 + self.forward_window)
            for right_idx in range(left_idx + 1, max_right):
                dst_aid, dst_ts, dst_type = recent[right_idx]
                gap = right_idx - left_idx
                delta_ms = max(0, dst_ts - src_ts)
                gap_weight = 1.0 / float(1 + gap)
                self._update_pair_count(src_aid, dst_aid)
                self._update_pair_count(dst_aid, src_aid)
                for matrix_name, spec in self.matrix_specs.items():
                    pair_weights: Dict[Tuple[int, int], float] = spec["pair_weights"]  # type: ignore[assignment]
                    time_scale_ms = float(spec["time_scale_ms"])
                    backward_scale = float(spec["backward_scale"])

                    forward_base = pair_weights.get((src_type, dst_type), 0.0)
                    if forward_base > 0.0:
                        time_weight = 1.0 / (1.0 + (delta_ms / time_scale_ms))
                        forward_weight = forward_base * time_weight * gap_weight
                        self._update(matrix_name, src_aid, dst_aid, forward_weight)

                    backward_base = pair_weights.get((dst_type, src_type), 0.0)
                    if backward_base > 0.0 and backward_scale > 0.0:
                        time_weight = 1.0 / (1.0 + (delta_ms / time_scale_ms))
                        backward_weight = backward_base * time_weight * gap_weight * backward_scale
                        self._update(matrix_name, dst_aid, src_aid, backward_weight)

    def _update(self, matrix_name: str, src_aid: int, dst_aid: int, weight: float) -> None:
        matrix = self.data[matrix_name]
        bucket = matrix.get(src_aid)
        if bucket is None:
            bucket = {}
            matrix[src_aid] = bucket
        bucket[dst_aid] = bucket.get(dst_aid, 0.0) + weight
        if len(bucket) > self.prune_topk * 2:
            matrix[src_aid] = dict(
                sorted(bucket.items(), key=lambda item: item[1], reverse=True)[: self.prune_topk]
            )

    def _update_pair_count(self, src_aid: int, dst_aid: int) -> None:
        bucket = self.pair_counts.get(src_aid)
        if bucket is None:
            bucket = {}
            self.pair_counts[src_aid] = bucket
        bucket[dst_aid] = bucket.get(dst_aid, 0.0) + 1.0
        if len(bucket) > self.pair_prune_topk * 2:
            self.pair_counts[src_aid] = dict(
                sorted(bucket.items(), key=lambda item: item[1], reverse=True)[: self.pair_prune_topk]
            )

    def finalize(self, counts_all: Counter[int]) -> Tuple[CovisitationMatrices, Dict[int, Dict[int, float]]]:
        finalized: CovisitationMatrices = {name: {} for name in MATRIX_NAMES}
        for matrix_name, matrix in self.data.items():
            pop_alpha = float(self.matrix_specs[matrix_name]["pop_alpha"])
            for aid, neighbors in matrix.items():
                rescored: List[Tuple[int, float]] = []
                for dst_aid, score in neighbors.items():
                    popularity_penalty = 1.0 + pop_alpha * math.log1p(counts_all.get(dst_aid, 0))
                    rescored.append((dst_aid, score / popularity_penalty))
                finalized[matrix_name][aid] = sorted(
                    rescored, key=lambda item: item[1], reverse=True
                )[: self.topk]
        finalized_pair_counts: Dict[int, Dict[int, float]] = {}
        for aid, neighbors in self.pair_counts.items():
            finalized_pair_counts[aid] = dict(
                sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[: self.pair_topk]
            )
        return finalized, finalized_pair_counts


def build_popularity_artifacts(
    counts_all: Counter[int],
    counts_by_target: Dict[str, Counter[int]],
    rank_size: int = 200_000,
    fallback_size: int = 200,
) -> PopularityArtifacts:
    all_rank = {aid: rank for rank, (aid, _) in enumerate(counts_all.most_common(rank_size), start=1)}
    global_fallback = [aid for aid, _ in counts_all.most_common(fallback_size)]
    target_rank = {
        target: {aid: rank for rank, (aid, _) in enumerate(counter.most_common(rank_size), start=1)}
        for target, counter in counts_by_target.items()
    }
    fallback: Dict[str, List[int]] = {}
    for target, counter in counts_by_target.items():
        ordered = [aid for aid, _ in counter.most_common(fallback_size)] + global_fallback
        deduped: List[int] = []
        seen = set()
        for aid in ordered:
            if aid in seen:
                continue
            seen.add(aid)
            deduped.append(aid)
            if len(deduped) >= fallback_size:
                break
        fallback[target] = deduped
    return PopularityArtifacts(all_rank=all_rank, target_rank=target_rank, fallback=fallback)


def resolve_matrix_weights(matrix_mode: str) -> Dict[str, Dict[str, object]]:
    try:
        return MATRIX_WEIGHT_PRESETS[matrix_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported matrix_mode={matrix_mode}") from exc


def prepare_cv_corpus(
    train_path: Path,
    sample_size: int,
    seed: int,
    matrix_mode: str = "target_aware",
) -> Tuple[CovisitationArtifacts, PopularityArtifacts, List[SessionRecord], List[SessionRecord], int, set[int]]:
    max_ts = get_max_ts(train_path)
    split_ts = max_ts - 2 * MS_PER_DAY
    known_aids: set[int] = set()
    covis_builder = MultiCovisitationBuilder(resolve_matrix_weights(matrix_mode))
    counts_all: Counter[int] = Counter()
    counts_by_target: Dict[str, Counter[int]] = {target: Counter() for target in TARGETS}
    sample_sessions: List[SessionRecord] = []
    valid_pool: List[SessionRecord] = []
    rng = random.Random(seed)
    sampled_seen = 0

    print(f"[cv] split_ts={split_ts}")
    for session_id, events in tqdm(iter_sessions(train_path), desc="prepare cv corpus"):
        if not events:
            continue
        if events[0][1] <= split_ts:
            trimmed = [event for event in events if event[1] < split_ts]
            if not trimmed:
                continue
            covis_builder.add_session(trimmed)
            for aid, _, event_type in trimmed:
                known_aids.add(aid)
                counts_all[aid] += 1
                counts_by_target[ID_TO_TYPE[event_type]][aid] += 1
            if len(trimmed) >= 2:
                sampled_seen = update_reservoir(
                    sample_sessions,
                    sampled_seen,
                    SessionRecord(session=session_id, events=trimmed),
                    sample_size,
                    rng,
                )
        else:
            valid_pool.append(SessionRecord(session=session_id, events=events))

    popularity = build_popularity_artifacts(counts_all, counts_by_target)
    valid_examples: List[SessionRecord] = []
    split_rng = random.Random(seed)
    for record in tqdm(valid_pool, desc="build cv valid"):
        filtered = keep_known_aids(record.events, known_aids)
        example = random_split_session(SessionRecord(record.session, filtered), split_rng)
        if example is not None:
            valid_examples.append(example)

    matrices, pair_counts = covis_builder.finalize(counts_all)
    return (
        CovisitationArtifacts(
            matrices=matrices,
            target_matrix_weights=TARGET_MATRIX_WEIGHTS[matrix_mode],
            pair_counts=pair_counts,
        ),
        popularity,
        sample_sessions,
        valid_examples,
        split_ts,
        known_aids,
    )


def prepare_full_corpus(
    train_path: Path,
    sample_size: int,
    seed: int,
    matrix_mode: str = "target_aware",
) -> Tuple[CovisitationArtifacts, PopularityArtifacts, List[SessionRecord], set[int]]:
    known_aids: set[int] = set()
    covis_builder = MultiCovisitationBuilder(resolve_matrix_weights(matrix_mode))
    counts_all: Counter[int] = Counter()
    counts_by_target: Dict[str, Counter[int]] = {target: Counter() for target in TARGETS}
    sample_sessions: List[SessionRecord] = []
    rng = random.Random(seed)
    sampled_seen = 0

    for session_id, events in tqdm(iter_sessions(train_path), desc="prepare full corpus"):
        if not events:
            continue
        covis_builder.add_session(events)
        for aid, _, event_type in events:
            known_aids.add(aid)
            counts_all[aid] += 1
            counts_by_target[ID_TO_TYPE[event_type]][aid] += 1
        if len(events) >= 2:
            sampled_seen = update_reservoir(
                sample_sessions,
                sampled_seen,
                SessionRecord(session=session_id, events=events),
                sample_size,
                rng,
            )
    popularity = build_popularity_artifacts(counts_all, counts_by_target)
    matrices, pair_counts = covis_builder.finalize(counts_all)
    return (
        CovisitationArtifacts(
            matrices=matrices,
            target_matrix_weights=TARGET_MATRIX_WEIGHTS[matrix_mode],
            pair_counts=pair_counts,
        ),
        popularity,
        sample_sessions,
        known_aids,
    )


def build_candidates(
    events: Sequence[Event],
    covis: CovisitationArtifacts,
    fallback: Sequence[int],
    target: str,
    candidate_limit: int = 60,
) -> Tuple[List[int], Dict[int, float], Dict[int, float], Dict[int, int]]:
    target_candidate_limit = 100 if target == "carts" else candidate_limit
    if not events:
        candidates = list(fallback[:target_candidate_limit])
        return candidates, {}, {}, {}

    recent = unique_recent_events(events, limit=20)
    history_recent = [aid for aid, _, _ in recent]
    matrices = covis.matrices
    matrix_weights = covis.target_matrix_weights[target]
    source_recent = recent[:10] if target == "carts" else recent[:5]

    score_sum: Dict[int, float] = Counter()
    score_max: Dict[int, float] = {}
    best_rank: Dict[int, int] = {}

    for idx, (aid, _, event_type) in enumerate(recent):
        history_score = EVENT_TYPE_WEIGHTS[event_type] / float(idx + 1)
        score_sum[aid] += history_score
        score_max[aid] = max(score_max.get(aid, 0.0), history_score)
        best_rank[aid] = min(best_rank.get(aid, 10**9), idx + 1)

    for src_idx, (src_aid, _, src_type) in enumerate(source_recent):
        src_decay = EVENT_TYPE_WEIGHTS[src_type] / float(src_idx + 1)
        for matrix_name, matrix_weight in matrix_weights.items():
            if matrix_weight <= 0.0:
                continue
            target_matrix = matrices[matrix_name]
            for neighbor_rank, (dst_aid, neighbor_score) in enumerate(target_matrix.get(src_aid, ()), start=1):
                total_score = matrix_weight * neighbor_score * src_decay
                score_sum[dst_aid] += total_score
                score_max[dst_aid] = max(score_max.get(dst_aid, 0.0), total_score)
                best_rank[dst_aid] = min(best_rank.get(dst_aid, 10**9), neighbor_rank)

    ordered = [aid for aid, _ in sorted(score_sum.items(), key=lambda item: item[1], reverse=True)]
    combined = history_recent + ordered + list(fallback)
    deduped: List[int] = []
    seen = set()
    for aid in combined:
        if aid in seen:
            continue
        seen.add(aid)
        deduped.append(aid)
        if len(deduped) >= target_candidate_limit:
            break
    return deduped, dict(score_sum), score_max, best_rank


def load_test_records(test_path: Path, known_aids: set[int]) -> Iterator[SessionRecord]:
    for session_id, events in iter_sessions(test_path):
        filtered = keep_known_aids(events, known_aids)
        yield SessionRecord(session=session_id, events=filtered)
