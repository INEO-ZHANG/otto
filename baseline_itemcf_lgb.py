#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

for thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    current = os.environ.get(thread_env)
    if current is None:
        os.environ[thread_env] = "1"
        continue
    try:
        if int(current) < 1:
            os.environ[thread_env] = "1"
    except ValueError:
        os.environ[thread_env] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm

MS_PER_DAY = 24 * 60 * 60 * 1000
TYPE_TO_ID = {"clicks": 0, "carts": 1, "orders": 2}
ID_TO_TYPE = {value: key for key, value in TYPE_TO_ID.items()}
TARGETS = ("clicks", "carts", "orders")
TARGET_WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
EVENT_TYPE_WEIGHTS = {0: 1.0, 1: 3.0, 2: 6.0}
FEATURE_COLUMNS = [
    "itemcf_score_sum",
    "itemcf_score_max",
    "itemcf_best_rank",
    "seen_count_all",
    "seen_count_clicks",
    "seen_count_carts",
    "seen_count_orders",
    "is_in_history",
    "is_last_aid",
    "last_pos_from_end",
    "first_pos_from_end",
    "time_since_last_seen",
    "session_len",
    "session_unique_aids",
    "last_event_type",
    "time_since_last_event",
    "global_pop_rank_all",
    "global_pop_rank_target",
]

Event = Tuple[int, int, int]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple OTTO itemCF + LightGBM baseline")
    parser.add_argument(
        "--mode",
        choices=("cv", "submit", "both"),
        default="both",
        help="Run local CV, submission, or both",
    )
    parser.add_argument(
        "--train-sample-sessions",
        type=int,
        default=500_000,
        help="Reservoir sample size for LightGBM training sessions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-threads", type=int, default=32, help="LightGBM threads")
    return parser.parse_args()


def ensure_dirs(base_dir: Path) -> Tuple[Path, Path, Path]:
    raw_dir = base_dir / "data" / "raw"
    cache_dir = base_dir / "cache"
    output_dir = base_dir / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, cache_dir, output_dir


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


class CovisitationBuilder:
    def __init__(self, topk: int = 40, prune_topk: int = 80, forward_window: int = 1) -> None:
        self.topk = topk
        self.prune_topk = prune_topk
        self.forward_window = forward_window
        self.data: Dict[int, Dict[int, float]] = {}

    def add_session(self, events: Sequence[Event]) -> None:
        recent = unique_recent_events(events, limit=30)
        if len(recent) < 2:
            return
        n_events = len(recent)
        for left_idx in range(n_events - 1):
            src_aid, _, src_type = recent[left_idx]
            max_right = min(n_events, left_idx + 1 + self.forward_window)
            for right_idx in range(left_idx + 1, max_right):
                dst_aid, _, dst_type = recent[right_idx]
                gap = right_idx - left_idx
                self._update(src_aid, dst_aid, EVENT_TYPE_WEIGHTS[dst_type] / (1.0 + gap))
                self._update(dst_aid, src_aid, EVENT_TYPE_WEIGHTS[src_type] / (1.0 + gap))

    def _update(self, src_aid: int, dst_aid: int, weight: float) -> None:
        bucket = self.data.get(src_aid)
        if bucket is None:
            bucket = {}
            self.data[src_aid] = bucket
        bucket[dst_aid] = bucket.get(dst_aid, 0.0) + weight
        if len(bucket) > self.prune_topk * 2:
            self.data[src_aid] = dict(
                sorted(bucket.items(), key=lambda item: item[1], reverse=True)[: self.prune_topk]
            )

    def finalize(self) -> Dict[int, List[Tuple[int, float]]]:
        finalized: Dict[int, List[Tuple[int, float]]] = {}
        for aid, neighbors in self.data.items():
            finalized[aid] = sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[: self.topk]
        return finalized


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


def prepare_cv_corpus(
    train_path: Path,
    sample_size: int,
    seed: int,
) -> Tuple[Dict[int, List[Tuple[int, float]]], PopularityArtifacts, List[SessionRecord], List[SessionRecord], int, set[int]]:
    max_ts = get_max_ts(train_path)
    split_ts = max_ts - 2 * MS_PER_DAY
    known_aids: set[int] = set()
    covis_builder = CovisitationBuilder()
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

    return covis_builder.finalize(), popularity, sample_sessions, valid_examples, split_ts, known_aids


def prepare_full_corpus(
    train_path: Path,
    sample_size: int,
    seed: int,
) -> Tuple[Dict[int, List[Tuple[int, float]]], PopularityArtifacts, List[SessionRecord], set[int]]:
    known_aids: set[int] = set()
    covis_builder = CovisitationBuilder()
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
    return covis_builder.finalize(), popularity, sample_sessions, known_aids


def build_session_context(events: Sequence[Event]) -> Dict[str, object]:
    session_len = len(events)
    last_ts = events[-1][1] if events else 0
    time_since_last_event = 0
    if session_len > 1:
        time_since_last_event = last_ts - events[-2][1]

    counts_all: Counter[int] = Counter()
    counts_by_type = {target: Counter() for target in TARGETS}
    last_pos: Dict[int, int] = {}
    first_pos: Dict[int, int] = {}
    last_seen_ts: Dict[int, int] = {}

    for idx, (aid, ts, event_type) in enumerate(events):
        pos_from_end = session_len - 1 - idx
        counts_all[aid] += 1
        counts_by_type[ID_TO_TYPE[event_type]][aid] += 1
        if aid not in first_pos:
            first_pos[aid] = pos_from_end
        last_pos[aid] = pos_from_end
        last_seen_ts[aid] = ts

    return {
        "session_len": session_len,
        "session_unique_aids": len(counts_all),
        "last_event_type": events[-1][2] if events else -1,
        "last_ts": last_ts,
        "time_since_last_event": time_since_last_event,
        "counts_all": counts_all,
        "counts_by_type": counts_by_type,
        "last_pos": last_pos,
        "first_pos": first_pos,
        "last_seen_ts": last_seen_ts,
    }


def build_candidates(
    events: Sequence[Event],
    covis: Dict[int, List[Tuple[int, float]]],
    fallback: Sequence[int],
    candidate_limit: int = 60,
) -> Tuple[List[int], Dict[int, float], Dict[int, float], Dict[int, int]]:
    if not events:
        candidates = list(fallback[:candidate_limit])
        return candidates, {}, {}, {}

    recent = unique_recent_events(events, limit=20)
    history_recent = [aid for aid, _, _ in recent]

    score_sum: Dict[int, float] = defaultdict(float)
    score_max: Dict[int, float] = defaultdict(float)
    best_rank: Dict[int, int] = {}

    for idx, (aid, _, event_type) in enumerate(recent):
        history_score = EVENT_TYPE_WEIGHTS[event_type] / float(idx + 1)
        score_sum[aid] += history_score
        score_max[aid] = max(score_max.get(aid, 0.0), history_score)
        best_rank[aid] = min(best_rank.get(aid, 10**9), idx + 1)

    for src_idx, (src_aid, _, src_type) in enumerate(recent[:5]):
        src_decay = EVENT_TYPE_WEIGHTS[src_type] / float(src_idx + 1)
        for neighbor_rank, (dst_aid, neighbor_score) in enumerate(covis.get(src_aid, ()), start=1):
            total_score = neighbor_score * src_decay
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
        if len(deduped) >= candidate_limit:
            break
    return deduped, score_sum, score_max, best_rank


def build_feature_rows(
    record: SessionRecord,
    covis: Dict[int, List[Tuple[int, float]]],
    popularity: PopularityArtifacts,
    target: str,
) -> Tuple[List[int], List[List[float]]]:
    events = record.events
    context = build_session_context(events)
    candidates, score_sum, score_max, best_rank = build_candidates(
        events, covis, popularity.fallback[target]
    )

    rows: List[List[float]] = []
    session_len = int(context["session_len"])
    last_ts = int(context["last_ts"])
    session_unique_aids = int(context["session_unique_aids"])
    last_event_type = int(context["last_event_type"])
    time_since_last_event = int(context["time_since_last_event"])
    counts_all: Counter[int] = context["counts_all"]  # type: ignore[assignment]
    counts_by_type: Dict[str, Counter[int]] = context["counts_by_type"]  # type: ignore[assignment]
    last_pos: Dict[int, int] = context["last_pos"]  # type: ignore[assignment]
    first_pos: Dict[int, int] = context["first_pos"]  # type: ignore[assignment]
    last_seen_ts: Dict[int, int] = context["last_seen_ts"]  # type: ignore[assignment]

    span_fallback = max(1, last_ts - events[0][1] + 1) if events else 1
    for aid in candidates:
        candidate_last_seen = last_seen_ts.get(aid)
        row = [
            float(score_sum.get(aid, 0.0)),
            float(score_max.get(aid, 0.0)),
            float(best_rank.get(aid, 999.0)),
            float(counts_all.get(aid, 0)),
            float(counts_by_type["clicks"].get(aid, 0)),
            float(counts_by_type["carts"].get(aid, 0)),
            float(counts_by_type["orders"].get(aid, 0)),
            float(aid in counts_all),
            float(session_len > 0 and events[-1][0] == aid),
            float(last_pos.get(aid, session_len + 1)),
            float(first_pos.get(aid, session_len + 1)),
            float(last_ts - candidate_last_seen if candidate_last_seen is not None else span_fallback),
            float(session_len),
            float(session_unique_aids),
            float(last_event_type),
            float(time_since_last_event),
            float(popularity.all_rank.get(aid, len(popularity.all_rank) + 1)),
            float(popularity.target_rank[target].get(aid, len(popularity.target_rank[target]) + 1)),
        ]
        rows.append(row)
    return candidates, rows


def label_candidates(candidates: Sequence[int], labels: Dict[str, object], target: str) -> List[int]:
    if target not in labels:
        return []
    if target == "clicks":
        positive = int(labels[target])  # type: ignore[arg-type]
        return [1 if aid == positive else 0 for aid in candidates]
    positive_set = labels[target]  # type: ignore[assignment]
    return [1 if aid in positive_set else 0 for aid in candidates]


def build_training_frames(
    sampled_sessions: Sequence[SessionRecord],
    covis: Dict[int, List[Tuple[int, float]]],
    popularity: PopularityArtifacts,
    seed: int,
    max_negatives: int = 20,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray], Dict[str, float]]:
    rows_by_target: Dict[str, List[List[float]]] = {target: [] for target in TARGETS}
    labels_by_target: Dict[str, List[int]] = {target: [] for target in TARGETS}
    coverage: Dict[str, Dict[str, int]] = {
        target: {"labeled_sessions": 0, "captured_sessions": 0} for target in TARGETS
    }
    rng = random.Random(seed)

    for record in tqdm(sampled_sessions, desc="build train rows"):
        example = random_split_session(record, rng)
        if example is None or example.labels is None:
            continue
        for target in TARGETS:
            if target not in example.labels:
                continue
            coverage[target]["labeled_sessions"] += 1
            candidates, features = build_feature_rows(example, covis, popularity, target)
            candidate_labels = label_candidates(candidates, example.labels, target)
            if not any(candidate_labels):
                continue
            coverage[target]["captured_sessions"] += 1
            positive_indices = [idx for idx, value in enumerate(candidate_labels) if value == 1]
            negative_indices = [idx for idx, value in enumerate(candidate_labels) if value == 0]
            if len(negative_indices) > max_negatives:
                negative_indices = rng.sample(negative_indices, max_negatives)
            keep_indices = positive_indices + negative_indices
            for idx in keep_indices:
                rows_by_target[target].append(features[idx])
                labels_by_target[target].append(candidate_labels[idx])

    frames = {
        target: pd.DataFrame(rows_by_target[target], columns=FEATURE_COLUMNS)
        for target in TARGETS
    }
    labels = {
        target: np.asarray(labels_by_target[target], dtype=np.int8)
        for target in TARGETS
    }
    coverage_ratio = {
        target: (
            coverage[target]["captured_sessions"] / coverage[target]["labeled_sessions"]
            if coverage[target]["labeled_sessions"]
            else 0.0
        )
        for target in TARGETS
    }
    return frames, labels, coverage_ratio


def train_models(
    frames: Dict[str, pd.DataFrame],
    labels: Dict[str, np.ndarray],
    num_threads: int,
    seed: int,
) -> Dict[str, object]:
    os.environ["OMP_NUM_THREADS"] = str(max(1, num_threads))
    import lightgbm as lgb

    models: Dict[str, object] = {}
    for offset, target in enumerate(TARGETS):
        frame = frames[target]
        target_labels = labels[target]
        if frame.empty or target_labels.sum() == 0:
            print(f"[train] skip {target}: empty training frame or no positives")
            continue
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=seed + offset,
            n_jobs=max(1, num_threads),
            verbosity=-1,
        )
        print(f"[train] {target}: rows={len(frame):,}, positives={int(target_labels.sum()):,}")
        model.fit(frame, target_labels)
        models[target] = model
    return models


def score_target(
    candidates: Sequence[int],
    features: Sequence[Sequence[float]],
    target: str,
    models: Dict[str, object],
    fallback: Sequence[int],
) -> List[int]:
    scores: List[Tuple[int, float]] = []
    if target in models and features:
        frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        probabilities = models[target].predict_proba(frame)[:, 1]
        scores = list(zip(candidates, probabilities.tolist()))
        scores.sort(key=lambda item: item[1], reverse=True)
    else:
        scores = list(zip(candidates, [row[0] for row in features]))
        scores.sort(key=lambda item: item[1], reverse=True)

    ranked = [aid for aid, _ in scores]
    seen = set(ranked)
    for aid in fallback:
        if len(ranked) >= 20:
            break
        if aid in seen:
            continue
        seen.add(aid)
        ranked.append(aid)
    return ranked[:20]


def evaluate_recall_at_20(
    valid_examples: Sequence[SessionRecord],
    covis: Dict[int, List[Tuple[int, float]]],
    popularity: PopularityArtifacts,
    models: Dict[str, object],
) -> Dict[str, float]:
    numerator = {target: 0.0 for target in TARGETS}
    denominator = {target: 0.0 for target in TARGETS}

    for record in tqdm(valid_examples, desc="evaluate cv"):
        labels = record.labels or {}
        for target in TARGETS:
            if target not in labels:
                continue
            candidates, features = build_feature_rows(record, covis, popularity, target)
            predictions = score_target(candidates, features, target, models, popularity.fallback[target])
            if target == "clicks":
                denominator[target] += 1.0
                numerator[target] += 1.0 if int(labels[target]) in predictions else 0.0  # type: ignore[arg-type]
            else:
                target_labels = labels[target]  # type: ignore[assignment]
                hits = sum(1 for aid in target_labels if aid in predictions)
                numerator[target] += float(hits)
                denominator[target] += float(min(20, len(target_labels)))

    metrics = {}
    total = 0.0
    for target in TARGETS:
        recall = numerator[target] / denominator[target] if denominator[target] else 0.0
        metrics[f"{target}_recall@20"] = recall
        total += TARGET_WEIGHTS[target] * recall
    metrics["weighted_recall@20"] = total
    metrics["valid_sessions"] = float(len(valid_examples))
    return metrics


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_test_records(test_path: Path, known_aids: set[int]) -> Iterator[SessionRecord]:
    for session_id, events in iter_sessions(test_path):
        filtered = keep_known_aids(events, known_aids)
        yield SessionRecord(session=session_id, events=filtered)


def write_submission(
    test_path: Path,
    known_aids: set[int],
    covis: Dict[int, List[Tuple[int, float]]],
    popularity: PopularityArtifacts,
    models: Dict[str, object],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session_type", "labels"])
        for record in tqdm(load_test_records(test_path, known_aids), desc="write submission"):
            for target in TARGETS:
                candidates, features = build_feature_rows(record, covis, popularity, target)
                predictions = score_target(candidates, features, target, models, popularity.fallback[target])
                writer.writerow([f"{record.session}_{target}", " ".join(str(aid) for aid in predictions)])


def run_cv(
    train_path: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    num_threads: int,
) -> Dict[str, float]:
    covis, popularity, sampled_sessions, valid_examples, split_ts, known_aids = prepare_cv_corpus(
        train_path=train_path,
        sample_size=sample_size,
        seed=seed,
    )
    print(
        f"[cv] train_samples={len(sampled_sessions):,} valid_examples={len(valid_examples):,} "
        f"known_aids={len(known_aids):,} split_ts={split_ts}"
    )
    frames, labels, coverage = build_training_frames(
        sampled_sessions=sampled_sessions,
        covis=covis,
        popularity=popularity,
        seed=seed,
    )
    for target, ratio in coverage.items():
        print(f"[cv] {target} candidate coverage={ratio:.4f}")
    models = train_models(frames, labels, num_threads=num_threads, seed=seed)
    metrics = evaluate_recall_at_20(valid_examples, covis, popularity, models)
    write_json(output_dir / "cv_metrics.json", {**metrics, "split_ts": split_ts, "coverage": coverage})
    print(f"[cv] metrics written to {output_dir / 'cv_metrics.json'}")
    return metrics


def run_submit(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    num_threads: int,
) -> Path:
    covis, popularity, sampled_sessions, known_aids = prepare_full_corpus(
        train_path=train_path,
        sample_size=sample_size,
        seed=seed,
    )
    print(
        f"[submit] train_samples={len(sampled_sessions):,} known_aids={len(known_aids):,}"
    )
    frames, labels, coverage = build_training_frames(
        sampled_sessions=sampled_sessions,
        covis=covis,
        popularity=popularity,
        seed=seed,
    )
    for target, ratio in coverage.items():
        print(f"[submit] {target} candidate coverage={ratio:.4f}")
    models = train_models(frames, labels, num_threads=num_threads, seed=seed)
    submission_path = output_dir / "submission.csv"
    write_submission(
        test_path=test_path,
        known_aids=known_aids,
        covis=covis,
        popularity=popularity,
        models=models,
        output_path=submission_path,
    )
    print(f"[submit] submission written to {submission_path}")
    return submission_path


def main() -> None:
    args = parse_args()
    base_dir = Path.cwd()
    raw_dir, _, output_dir = ensure_dirs(base_dir)
    extracted = resolve_raw_files(raw_dir)
    train_path = extracted["train"]
    test_path = extracted["test"]

    if args.mode in ("cv", "both"):
        metrics = run_cv(
            train_path=train_path,
            output_dir=output_dir,
            sample_size=args.train_sample_sessions,
            seed=args.seed,
            num_threads=args.num_threads,
        )
        print(f"[cv] weighted_recall@20={metrics['weighted_recall@20']:.6f}")

    if args.mode in ("submit", "both"):
        run_submit(
            train_path=train_path,
            test_path=test_path,
            output_dir=output_dir,
            sample_size=args.train_sample_sessions,
            seed=args.seed,
            num_threads=args.num_threads,
        )


if __name__ == "__main__":
    main()
