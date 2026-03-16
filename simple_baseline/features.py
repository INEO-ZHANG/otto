from __future__ import annotations

import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from recall import (
    BASE_COVIS_TOPK,
    CLICK2CART_TOPK,
    CLICK2CLICK_TOPK,
    ID_TO_TYPE,
    MS_PER_DAY,
    MS_PER_HOUR,
    PopularityArtifacts,
    RecallArtifacts,
    SessionRecord,
    TARGETS,
    TARGET_WEIGHTS,
    TYPE_TO_ID,
    build_candidates,
    random_split_session,
    unique_recent_events,
)

SIM_MATRIX_NAMES = ("base_covis", "click2click", "click2cart")
SIM_BUCKETS = ("all", "clicks", "carts", "orders")
SIM_AGGS = ("mean", "max", "last")
W2V_VECTOR_SIZE = 64
W2V_WINDOW = 10
W2V_EPOCHS = 10

FEATURE_COLUMNS = [
    "session_len",
    "session_unique_aids",
    "session_click_count",
    "session_cart_count",
    "session_order_count",
    "session_click_freq",
    "session_cart_freq",
    "session_order_freq",
    "last_behavior_type",
    "last_click_hour",
    "last_cart_hour",
    "last_order_hour",
    "aid_total_count",
    "aid_click_count",
    "aid_cart_count",
    "aid_order_count",
    "aid_click_ratio",
    "aid_cart_ratio",
    "aid_order_ratio",
    "aid_last_click_gap_days",
    "aid_last_cart_gap_days",
    "aid_last_order_gap_days",
    "aid_mean_behavior_type",
    "global_pop_rank_all",
    "global_pop_rank_target",
    "session_aid_total_count",
    "session_aid_click_count",
    "session_aid_cart_count",
    "session_aid_order_count",
    "session_aid_last_gap_days",
    "session_aid_last_click_gap_days",
    "session_aid_last_cart_gap_days",
    "session_aid_last_order_gap_days",
    "session_aid_mean_type",
    "session_aid_last_type",
    "delta_click_ratio",
    "delta_cart_ratio",
    "delta_order_ratio",
    "delta_click_gap",
    "delta_cart_gap",
    "delta_order_gap",
    "in_base_covis",
    "in_click2click",
    "in_click2cart",
    "base_covis_score_sum",
    "base_covis_score_max",
    "base_covis_best_rank",
    "click2click_score_sum",
    "click2click_score_max",
    "click2click_best_rank",
    "click2cart_score_sum",
    "click2cart_score_max",
    "click2cart_best_rank",
]
for matrix_name in SIM_MATRIX_NAMES:
    for bucket_name in SIM_BUCKETS:
        for agg_name in SIM_AGGS:
            FEATURE_COLUMNS.append(f"{matrix_name}_{bucket_name}_{agg_name}_sim")
FEATURE_COLUMNS.extend(
    [
        "w2v_last_aid_sim",
        "w2v_last_click_aid_sim",
        "w2v_last_cart_aid_sim",
        "w2v_last_order_aid_sim",
    ]
)


@dataclass(slots=True)
class W2VArtifacts:
    vectors: Dict[int, np.ndarray]
    cache_hit: bool

    def similarity(self, left_aid: Optional[int], right_aid: int) -> float:
        if left_aid is None:
            return 0.0
        left = self.vectors.get(left_aid)
        right = self.vectors.get(right_aid)
        if left is None or right is None:
            return 0.0
        return float(np.dot(left, right))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _save_w2v_cache(path: Path, vectors: Dict[int, np.ndarray]) -> None:
    if vectors:
        aids = np.asarray(sorted(vectors.keys()), dtype=np.int64)
        matrix = np.vstack([vectors[int(aid)] for aid in aids]).astype(np.float32, copy=False)
    else:
        aids = np.asarray([], dtype=np.int64)
        matrix = np.empty((0, W2V_VECTOR_SIZE), dtype=np.float32)
    np.savez_compressed(path, aids=aids, vectors=matrix)


def _load_w2v_cache(path: Path) -> Dict[int, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    aids = payload["aids"].astype(np.int64, copy=False)
    matrix = payload["vectors"].astype(np.float32, copy=False)
    return {int(aid): matrix[idx] for idx, aid in enumerate(aids.tolist())}


def _build_w2v_corpus(
    train_path: Path,
    split_ts: Optional[int],
    known_aids: set[int],
    corpus_path: Path,
) -> None:
    from recall import iter_sessions

    with corpus_path.open("w", encoding="utf-8") as handle:
        for _, events in iter_sessions(train_path):
            last_aid = None
            sequence: List[str] = []
            for aid, ts, _ in events:
                if split_ts is not None and ts >= split_ts:
                    break
                if aid not in known_aids:
                    continue
                if aid == last_aid:
                    continue
                sequence.append(str(aid))
                last_aid = aid
            if len(sequence) >= 2:
                handle.write(" ".join(sequence))
                handle.write("\n")


def prepare_w2v_artifacts(
    train_path: Path,
    cache_dir: Path,
    known_aids: set[int],
    split_ts: Optional[int],
    cache_prefix: str,
    seed: int,
) -> W2VArtifacts:
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence

    w2v_dir = cache_dir / "simple_baseline"
    w2v_dir.mkdir(parents=True, exist_ok=True)
    cache_path = w2v_dir / f"{cache_prefix}_w2v_dim{W2V_VECTOR_SIZE}.npz"
    corpus_path = w2v_dir / f"{cache_prefix}_w2v_corpus.txt"

    if cache_path.exists():
        return W2VArtifacts(vectors=_load_w2v_cache(cache_path), cache_hit=True)

    if not corpus_path.exists():
        _build_w2v_corpus(train_path, split_ts, known_aids, corpus_path)

    model = Word2Vec(
        sentences=LineSentence(str(corpus_path)),
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        sg=1,
        negative=10,
        min_count=1,
        epochs=W2V_EPOCHS,
        workers=1,
        seed=seed,
    )
    vectors: Dict[int, np.ndarray] = {}
    for aid in known_aids:
        token = str(aid)
        if token in model.wv:
            vectors[aid] = _normalize_vector(model.wv[token].astype(np.float32, copy=False))
    _save_w2v_cache(cache_path, vectors)
    return W2VArtifacts(vectors=vectors, cache_hit=False)


def build_session_context(events: Sequence[Tuple[int, int, int]]) -> Dict[str, object]:
    session_len = len(events)
    counts_all: Counter[int] = Counter()
    counts_by_type = {target: Counter() for target in TARGETS}
    last_seen_ts: Dict[int, int] = {}
    last_seen_ts_by_type: Dict[str, Dict[int, int]] = {target: {} for target in TARGETS}
    type_sum_by_aid: Counter[int] = Counter()
    last_type_by_aid: Dict[int, int] = {}
    last_aid_by_type: Dict[str, Optional[int]] = {target: None for target in TARGETS}
    last_ts_by_type: Dict[str, Optional[int]] = {target: None for target in TARGETS}

    for aid, ts, event_type in events:
        target = ID_TO_TYPE[event_type]
        counts_all[aid] += 1
        counts_by_type[target][aid] += 1
        last_seen_ts[aid] = ts
        last_seen_ts_by_type[target][aid] = ts
        type_sum_by_aid[aid] += float(event_type)
        last_type_by_aid[aid] = event_type
        last_aid_by_type[target] = aid
        last_ts_by_type[target] = ts

    recent = unique_recent_events(events, limit=20)
    all_recent_aids = [aid for aid, _, _ in recent]
    click_recent_aids = [aid for aid, _, event_type in recent if event_type == TYPE_TO_ID["clicks"]]
    cart_recent_aids = [aid for aid, _, event_type in recent if event_type == TYPE_TO_ID["carts"]]
    order_recent_aids = [aid for aid, _, event_type in recent if event_type == TYPE_TO_ID["orders"]]
    last_ts = events[-1][1] if events else 0

    def _hour(ts_value: Optional[int]) -> float:
        if ts_value is None:
            return -1.0
        return float((ts_value // MS_PER_HOUR) % 24)

    session_click_count = int(sum(1 for _, _, event_type in events if event_type == TYPE_TO_ID["clicks"]))
    session_cart_count = int(sum(1 for _, _, event_type in events if event_type == TYPE_TO_ID["carts"]))
    session_order_count = int(sum(1 for _, _, event_type in events if event_type == TYPE_TO_ID["orders"]))

    return {
        "session_len": session_len,
        "session_unique_aids": len(counts_all),
        "session_click_count": session_click_count,
        "session_cart_count": session_cart_count,
        "session_order_count": session_order_count,
        "last_behavior_type": events[-1][2] if events else -1,
        "last_ts": last_ts,
        "counts_all": counts_all,
        "counts_by_type": counts_by_type,
        "last_seen_ts": last_seen_ts,
        "last_seen_ts_by_type": last_seen_ts_by_type,
        "type_sum_by_aid": type_sum_by_aid,
        "last_type_by_aid": last_type_by_aid,
        "all_recent_aids": all_recent_aids,
        "click_recent_aids": click_recent_aids,
        "cart_recent_aids": cart_recent_aids,
        "order_recent_aids": order_recent_aids,
        "last_aid": events[-1][0] if events else None,
        "last_click_aid": last_aid_by_type["clicks"],
        "last_cart_aid": last_aid_by_type["carts"],
        "last_order_aid": last_aid_by_type["orders"],
        "last_click_hour": _hour(last_ts_by_type["clicks"]),
        "last_cart_hour": _hour(last_ts_by_type["carts"]),
        "last_order_hour": _hour(last_ts_by_type["orders"]),
    }


def _gap_days(last_ts: int, seen_ts: Optional[int]) -> float:
    if seen_ts is None:
        return 30.0
    return float((last_ts - seen_ts) / MS_PER_DAY)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _bucket_sim_features(
    candidate_aid: int,
    bucket_aids: Sequence[int],
    neighbor_lookup: Dict[int, Dict[int, float]],
) -> Tuple[float, float, float]:
    if not bucket_aids:
        return 0.0, 0.0, 0.0
    weighted_scores: List[float] = []
    for idx, src_aid in enumerate(bucket_aids):
        score = neighbor_lookup.get(src_aid, {}).get(candidate_aid, 0.0)
        weighted_scores.append(score / float(idx + 1))
    return (
        float(sum(weighted_scores) / len(weighted_scores)),
        float(max(weighted_scores)),
        float(weighted_scores[0]),
    )


def build_feature_rows(
    record: SessionRecord,
    recall_artifacts: RecallArtifacts,
    w2v_artifacts: W2VArtifacts,
    target: str,
) -> Tuple[List[int], List[List[float]], Dict[str, int]]:
    events = record.events
    context = build_session_context(events)
    bundle = build_candidates(events, recall_artifacts, target)
    candidates = bundle.candidates
    popularity = recall_artifacts.popularity
    aid_stats = recall_artifacts.aid_stats
    last_ts = int(context["last_ts"])

    all_recent_aids: List[int] = context["all_recent_aids"]  # type: ignore[assignment]
    click_recent_aids: List[int] = context["click_recent_aids"]  # type: ignore[assignment]
    cart_recent_aids: List[int] = context["cart_recent_aids"]  # type: ignore[assignment]
    order_recent_aids: List[int] = context["order_recent_aids"]  # type: ignore[assignment]
    matrix_source_aids = set(all_recent_aids)
    matrix_source_aids.update(click_recent_aids)
    matrix_source_aids.update(cart_recent_aids)
    matrix_source_aids.update(order_recent_aids)

    raw_matrices = {
        "base_covis": recall_artifacts.base_covis,
        "click2click": recall_artifacts.click2click,
        "click2cart": recall_artifacts.click2cart,
    }
    lookup = {
        matrix_name: {src_aid: dict(matrix.get(src_aid, ())) for src_aid in matrix_source_aids}
        for matrix_name, matrix in raw_matrices.items()
    }

    session_len = float(context["session_len"])
    session_click_count = float(context["session_click_count"])
    session_cart_count = float(context["session_cart_count"])
    session_order_count = float(context["session_order_count"])
    counts_all: Counter[int] = context["counts_all"]  # type: ignore[assignment]
    counts_by_type: Dict[str, Counter[int]] = context["counts_by_type"]  # type: ignore[assignment]
    last_seen_ts: Dict[int, int] = context["last_seen_ts"]  # type: ignore[assignment]
    last_seen_ts_by_type: Dict[str, Dict[int, int]] = context["last_seen_ts_by_type"]  # type: ignore[assignment]
    type_sum_by_aid: Counter[int] = context["type_sum_by_aid"]  # type: ignore[assignment]
    last_type_by_aid: Dict[int, int] = context["last_type_by_aid"]  # type: ignore[assignment]

    rows: List[List[float]] = []
    for aid in candidates:
        aid_total_count = float(aid_stats.counts_all.get(aid, 0))
        aid_click_count = float(aid_stats.counts_by_target["clicks"].get(aid, 0))
        aid_cart_count = float(aid_stats.counts_by_target["carts"].get(aid, 0))
        aid_order_count = float(aid_stats.counts_by_target["orders"].get(aid, 0))
        aid_click_ratio = _safe_ratio(aid_click_count, aid_total_count)
        aid_cart_ratio = _safe_ratio(aid_cart_count, aid_total_count)
        aid_order_ratio = _safe_ratio(aid_order_count, aid_total_count)
        aid_last_click_gap = _gap_days(last_ts, aid_stats.last_ts_by_target["clicks"].get(aid))
        aid_last_cart_gap = _gap_days(last_ts, aid_stats.last_ts_by_target["carts"].get(aid))
        aid_last_order_gap = _gap_days(last_ts, aid_stats.last_ts_by_target["orders"].get(aid))

        session_aid_total_count = float(counts_all.get(aid, 0))
        session_aid_click_count = float(counts_by_type["clicks"].get(aid, 0))
        session_aid_cart_count = float(counts_by_type["carts"].get(aid, 0))
        session_aid_order_count = float(counts_by_type["orders"].get(aid, 0))
        session_aid_last_gap = _gap_days(last_ts, last_seen_ts.get(aid))
        session_aid_last_click_gap = _gap_days(last_ts, last_seen_ts_by_type["clicks"].get(aid))
        session_aid_last_cart_gap = _gap_days(last_ts, last_seen_ts_by_type["carts"].get(aid))
        session_aid_last_order_gap = _gap_days(last_ts, last_seen_ts_by_type["orders"].get(aid))
        session_aid_mean_type = (
            float(type_sum_by_aid[aid] / session_aid_total_count) if session_aid_total_count > 0 else -1.0
        )
        session_aid_last_type = float(last_type_by_aid.get(aid, -1))

        session_click_ratio = _safe_ratio(session_aid_click_count, session_click_count)
        session_cart_ratio = _safe_ratio(session_aid_cart_count, session_cart_count)
        session_order_ratio = _safe_ratio(session_aid_order_count, session_order_count)

        row = [
            session_len,
            float(context["session_unique_aids"]),
            session_click_count,
            session_cart_count,
            session_order_count,
            _safe_ratio(session_click_count, session_len),
            _safe_ratio(session_cart_count, session_len),
            _safe_ratio(session_order_count, session_len),
            float(context["last_behavior_type"]),
            float(context["last_click_hour"]),
            float(context["last_cart_hour"]),
            float(context["last_order_hour"]),
            aid_total_count,
            aid_click_count,
            aid_cart_count,
            aid_order_count,
            aid_click_ratio,
            aid_cart_ratio,
            aid_order_ratio,
            aid_last_click_gap,
            aid_last_cart_gap,
            aid_last_order_gap,
            float(aid_stats.mean_behavior_type.get(aid, -1.0)),
            float(popularity.all_rank.get(aid, len(popularity.all_rank) + 1)),
            float(popularity.target_rank[target].get(aid, len(popularity.target_rank[target]) + 1)),
            session_aid_total_count,
            session_aid_click_count,
            session_aid_cart_count,
            session_aid_order_count,
            session_aid_last_gap,
            session_aid_last_click_gap,
            session_aid_last_cart_gap,
            session_aid_last_order_gap,
            session_aid_mean_type,
            session_aid_last_type,
            abs(session_click_ratio - aid_click_ratio),
            abs(session_cart_ratio - aid_cart_ratio),
            abs(session_order_ratio - aid_order_ratio),
            abs(math.log1p(session_aid_last_click_gap) - math.log1p(aid_last_click_gap)),
            abs(math.log1p(session_aid_last_cart_gap) - math.log1p(aid_last_cart_gap)),
            abs(math.log1p(session_aid_last_order_gap) - math.log1p(aid_last_order_gap)),
            float(aid in bundle.base_score_sum),
            float(aid in bundle.click2click_score_sum),
            float(aid in bundle.click2cart_score_sum),
            float(bundle.base_score_sum.get(aid, 0.0)),
            float(bundle.base_score_max.get(aid, 0.0)),
            float(bundle.base_best_rank.get(aid, BASE_COVIS_TOPK + 1)),
            float(bundle.click2click_score_sum.get(aid, 0.0)),
            float(bundle.click2click_score_max.get(aid, 0.0)),
            float(bundle.click2click_best_rank.get(aid, CLICK2CLICK_TOPK + 1)),
            float(bundle.click2cart_score_sum.get(aid, 0.0)),
            float(bundle.click2cart_score_max.get(aid, 0.0)),
            float(bundle.click2cart_best_rank.get(aid, CLICK2CART_TOPK + 1)),
        ]

        for matrix_name in SIM_MATRIX_NAMES:
            matrix_lookup = lookup[matrix_name]
            for bucket_aids in (all_recent_aids, click_recent_aids, cart_recent_aids, order_recent_aids):
                mean_value, max_value, last_value = _bucket_sim_features(aid, bucket_aids, matrix_lookup)
                row.extend([mean_value, max_value, last_value])

        row.extend(
            [
                w2v_artifacts.similarity(context["last_aid"], aid),  # type: ignore[arg-type]
                w2v_artifacts.similarity(context["last_click_aid"], aid),  # type: ignore[arg-type]
                w2v_artifacts.similarity(context["last_cart_aid"], aid),  # type: ignore[arg-type]
                w2v_artifacts.similarity(context["last_order_aid"], aid),  # type: ignore[arg-type]
            ]
        )
        rows.append(row)

    return candidates, rows, bundle.source_counts


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
    recall_artifacts: RecallArtifacts,
    w2v_artifacts: W2VArtifacts,
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
            candidates, features, _ = build_feature_rows(example, recall_artifacts, w2v_artifacts, target)
            candidate_labels = label_candidates(candidates, example.labels, target)
            if not any(candidate_labels):
                continue
            coverage[target]["captured_sessions"] += 1
            positive_indices = [idx for idx, value in enumerate(candidate_labels) if value == 1]
            negative_indices = [idx for idx, value in enumerate(candidate_labels) if value == 0]
            if len(negative_indices) > max_negatives:
                negative_indices = rng.sample(negative_indices, max_negatives)
            for idx in positive_indices + negative_indices:
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
        train_matrix = frame.to_numpy(dtype=np.float32, copy=False)
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
        model.fit(train_matrix, target_labels)
        models[target] = model
    return models


def score_target(
    candidates: Sequence[int],
    features: Sequence[Sequence[float]],
    target: str,
    models: Dict[str, object],
    fallback: Sequence[int],
) -> List[int]:
    if target in models and features:
        feature_matrix = np.asarray(features, dtype=np.float32)
        scores = models[target].booster_.predict(feature_matrix)
    else:
        scores = np.asarray([0.0] * len(candidates), dtype=np.float32)
    ranked = [aid for aid, _ in sorted(zip(candidates, scores.tolist()), key=lambda item: item[1], reverse=True)]
    if len(ranked) >= 20:
        return ranked[:20]
    seen = set(ranked)
    for aid in fallback:
        if aid in seen:
            continue
        seen.add(aid)
        ranked.append(aid)
        if len(ranked) >= 20:
            break
    return ranked[:20]


def evaluate_recall_at_20(
    valid_examples: Sequence[SessionRecord],
    recall_artifacts: RecallArtifacts,
    w2v_artifacts: W2VArtifacts,
    models: Dict[str, object],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, float]]:
    numerator = {target: 0.0 for target in TARGETS}
    denominator = {target: 0.0 for target in TARGETS}
    source_totals = {
        target: {"history": 0.0, "base_covis": 0.0, "click2click": 0.0, "click2cart": 0.0, "union": 0.0}
        for target in TARGETS
    }
    source_counts = {target: 0 for target in TARGETS}

    for record in tqdm(valid_examples, desc="evaluate cv"):
        labels = record.labels or {}
        for target in TARGETS:
            if target not in labels:
                continue
            candidates, features, candidate_sources = build_feature_rows(record, recall_artifacts, w2v_artifacts, target)
            source_counts[target] += 1
            for source_name, source_value in candidate_sources.items():
                source_totals[target][source_name] += float(source_value)
            predictions = score_target(candidates, features, target, models, recall_artifacts.popularity.fallback[target])
            if target == "clicks":
                denominator[target] += 1.0
                numerator[target] += 1.0 if int(labels[target]) in predictions else 0.0  # type: ignore[arg-type]
            else:
                target_labels = labels[target]  # type: ignore[assignment]
                hits = sum(1 for aid in target_labels if aid in predictions)
                numerator[target] += float(hits)
                denominator[target] += float(min(20, len(target_labels)))

    metrics: Dict[str, float] = {}
    total = 0.0
    for target in TARGETS:
        recall = numerator[target] / denominator[target] if denominator[target] else 0.0
        metrics[f"{target}_recall@20"] = recall
        total += TARGET_WEIGHTS[target] * recall
    metrics["weighted_recall@20"] = total
    metrics["valid_sessions"] = float(len(valid_examples))

    avg_candidates_by_target = {
        target: (source_totals[target]["union"] / source_counts[target] if source_counts[target] else 0.0)
        for target in TARGETS
    }
    candidate_source_stats = {
        target: {
            source_name: (source_value / source_counts[target] if source_counts[target] else 0.0)
            for source_name, source_value in source_totals[target].items()
        }
        for target in TARGETS
    }
    return metrics, candidate_source_stats, avg_candidates_by_target
