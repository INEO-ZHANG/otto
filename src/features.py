from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from embeddings import EmbeddingArtifacts
from itemcf import (
    MS_PER_HOUR,
    CovisitationArtifacts,
    ID_TO_TYPE,
    TARGETS,
    PopularityArtifacts,
    SessionRecord,
    TARGET_WEIGHTS,
    build_candidates,
    random_split_session,
)

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
    "session_click_count",
    "session_cart_count",
    "session_order_count",
    "last_event_type",
    "time_since_last_event",
    "global_pop_rank_all",
    "global_pop_rank_target",
    "last_item_itemcf_score",
    "last_hour_avg_itemcf_score",
    "last_item_cocount",
    "w2v_last_aid_cosine",
    "prone_session_aid_cosine",
]

RRF_K = 60.0
LGBM_RRF_WEIGHT = 0.6
CATBOOST_RRF_WEIGHT = 0.4


@dataclass(slots=True)
class PreparedRecord:
    record: SessionRecord
    features_by_target: Dict[str, Tuple[List[int], np.ndarray]]


@dataclass(slots=True)
class PredictionModel:
    backend: str
    predictor: object

    def predict_scores(self, feature_matrix: np.ndarray) -> np.ndarray:
        if self.backend == "lightgbm":
            return self.predictor.predict_proba(feature_matrix)[:, 1]
        if self.backend == "tl2cgen":
            import tl2cgen

            dmatrix = tl2cgen.DMatrix(feature_matrix)
            scores = self.predictor.predict(dmatrix)
            return np.asarray(scores, dtype=np.float32).reshape(-1)
        if self.backend == "catboost":
            scores = self.predictor.predict(feature_matrix)
            return np.asarray(scores, dtype=np.float32).reshape(-1)
        raise ValueError(f"unsupported prediction backend={self.backend}")


@dataclass(slots=True)
class ModelEnsemble:
    lgbm: Optional[PredictionModel] = None
    catboost: Optional[PredictionModel] = None


@dataclass(slots=True)
class TargetTrainingData:
    frame: pd.DataFrame
    labels: np.ndarray
    group_ids: np.ndarray
    group_sizes: np.ndarray


@dataclass(slots=True)
class TrainingArtifacts:
    by_target: Dict[str, TargetTrainingData]
    coverage: Dict[str, float]


def build_session_context(
    events: Sequence[Tuple[int, int, int]],
    embedding_artifacts: Optional[EmbeddingArtifacts] = None,
) -> Dict[str, object]:
    session_len = len(events)
    last_ts = events[-1][1] if events else 0
    time_since_last_event = 0
    if session_len > 1:
        time_since_last_event = last_ts - events[-2][1]

    counts_all: Counter[int] = Counter()
    counts_by_type = {target: Counter() for target in TARGETS}
    action_counts = {target: 0 for target in TARGETS}
    last_pos: Dict[int, int] = {}
    first_pos: Dict[int, int] = {}
    last_seen_ts: Dict[int, int] = {}

    for idx, (aid, ts, event_type) in enumerate(events):
        pos_from_end = session_len - 1 - idx
        counts_all[aid] += 1
        target_name = ID_TO_TYPE[event_type]
        counts_by_type[target_name][aid] += 1
        action_counts[target_name] += 1
        if aid not in first_pos:
            first_pos[aid] = pos_from_end
        last_pos[aid] = pos_from_end
        last_seen_ts[aid] = ts

    prone_session_vector = None
    if embedding_artifacts is not None:
        prone_session_vector = embedding_artifacts.build_prone_session_vector(events)

    return {
        "session_len": session_len,
        "session_unique_aids": len(counts_all),
        "session_click_count": action_counts["clicks"],
        "session_cart_count": action_counts["carts"],
        "session_order_count": action_counts["orders"],
        "last_event_type": events[-1][2] if events else -1,
        "last_ts": last_ts,
        "time_since_last_event": time_since_last_event,
        "counts_all": counts_all,
        "counts_by_type": counts_by_type,
        "last_pos": last_pos,
        "first_pos": first_pos,
        "last_seen_ts": last_seen_ts,
        "prone_session_vector": prone_session_vector,
    }


def build_feature_rows_from_context(
    events: Sequence[Tuple[int, int, int]],
    context: Dict[str, object],
    covis: CovisitationArtifacts,
    popularity: PopularityArtifacts,
    embedding_artifacts: Optional[EmbeddingArtifacts],
    target: str,
) -> Tuple[List[int], List[List[float]]]:
    candidates, score_sum, score_max, best_rank = build_candidates(
        events, covis, popularity.fallback[target], target
    )

    rows: List[List[float]] = []
    session_len = int(context["session_len"])
    last_ts = int(context["last_ts"])
    session_unique_aids = int(context["session_unique_aids"])
    session_click_count = int(context["session_click_count"])
    session_cart_count = int(context["session_cart_count"])
    session_order_count = int(context["session_order_count"])
    last_event_type = int(context["last_event_type"])
    time_since_last_event = int(context["time_since_last_event"])
    counts_all: Counter[int] = context["counts_all"]  # type: ignore[assignment]
    counts_by_type: Dict[str, Counter[int]] = context["counts_by_type"]  # type: ignore[assignment]
    last_pos: Dict[int, int] = context["last_pos"]  # type: ignore[assignment]
    first_pos: Dict[int, int] = context["first_pos"]  # type: ignore[assignment]
    last_seen_ts: Dict[int, int] = context["last_seen_ts"]  # type: ignore[assignment]
    prone_session_vector = context["prone_session_vector"]  # type: ignore[assignment]

    last_aid = events[-1][0] if events else None
    pair_counts = covis.pair_counts
    matrix_weights = covis.target_matrix_weights[target]
    source_aids = set()
    if last_aid is not None:
        source_aids.add(last_aid)
    recent_hour_aids: List[int] = []
    recent_hour_cutoff = last_ts - MS_PER_HOUR
    seen_recent_hour = set()
    for aid, ts, _ in reversed(events):
        if ts < recent_hour_cutoff:
            break
        if aid in seen_recent_hour:
            continue
        seen_recent_hour.add(aid)
        recent_hour_aids.append(aid)
        source_aids.add(aid)

    neighbor_lookup: Dict[str, Dict[int, Dict[int, float]]] = {}
    for matrix_name in matrix_weights:
        matrix = covis.matrices[matrix_name]
        matrix_cache: Dict[int, Dict[int, float]] = {}
        for src_aid in source_aids:
            matrix_cache[src_aid] = dict(matrix.get(src_aid, ()))
        neighbor_lookup[matrix_name] = matrix_cache

    def fused_direct_score(src_aid: Optional[int], dst_aid: int) -> float:
        if src_aid is None:
            return 0.0
        total = 0.0
        for matrix_name, matrix_weight in matrix_weights.items():
            if matrix_weight <= 0.0:
                continue
            total += matrix_weight * neighbor_lookup[matrix_name].get(src_aid, {}).get(dst_aid, 0.0)
        return total

    span_fallback = max(1, last_ts - events[0][1] + 1) if events else 1
    for aid in candidates:
        candidate_last_seen = last_seen_ts.get(aid)
        last_item_itemcf_score = fused_direct_score(last_aid, aid)
        if recent_hour_aids:
            last_hour_avg_itemcf_score = float(
                sum(fused_direct_score(src_aid, aid) for src_aid in recent_hour_aids) / len(recent_hour_aids)
            )
        else:
            last_hour_avg_itemcf_score = 0.0
        last_item_cocount = float(pair_counts.get(last_aid, {}).get(aid, 0.0)) if last_aid is not None else 0.0
        w2v_last_aid_cosine = 0.0
        prone_session_aid_cosine = 0.0
        if embedding_artifacts is not None:
            w2v_last_aid_cosine = embedding_artifacts.w2v_last_aid_cosine(last_aid, aid)
            prone_session_aid_cosine = embedding_artifacts.prone_session_aid_cosine(prone_session_vector, aid)

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
            float(session_click_count),
            float(session_cart_count),
            float(session_order_count),
            float(last_event_type),
            float(time_since_last_event),
            float(popularity.all_rank.get(aid, len(popularity.all_rank) + 1)),
            float(popularity.target_rank[target].get(aid, len(popularity.target_rank[target]) + 1)),
            float(last_item_itemcf_score),
            float(last_hour_avg_itemcf_score),
            float(last_item_cocount),
            float(w2v_last_aid_cosine),
            float(prone_session_aid_cosine),
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
    covis: CovisitationArtifacts,
    popularity: PopularityArtifacts,
    embedding_artifacts: Optional[EmbeddingArtifacts],
    seed: int,
    max_negatives: int = 20,
) -> TrainingArtifacts:
    rows_by_target: Dict[str, List[List[float]]] = {target: [] for target in TARGETS}
    labels_by_target: Dict[str, List[int]] = {target: [] for target in TARGETS}
    group_ids_by_target: Dict[str, List[int]] = {target: [] for target in TARGETS}
    group_sizes_by_target: Dict[str, List[int]] = {target: [] for target in TARGETS}
    coverage: Dict[str, Dict[str, int]] = {
        target: {"labeled_sessions": 0, "captured_sessions": 0} for target in TARGETS
    }
    query_ids = {target: 0 for target in TARGETS}
    rng = random.Random(seed)

    for record in tqdm(sampled_sessions, desc="build train rows"):
        example = random_split_session(record, rng)
        if example is None or example.labels is None:
            continue

        context = build_session_context(example.events, embedding_artifacts)
        for target in TARGETS:
            if target not in example.labels:
                continue
            coverage[target]["labeled_sessions"] += 1
            candidates, features = build_feature_rows_from_context(
                example.events,
                context,
                covis,
                popularity,
                embedding_artifacts,
                target,
            )
            candidate_labels = label_candidates(candidates, example.labels, target)
            positive_indices = [idx for idx, value in enumerate(candidate_labels) if value == 1]
            if not positive_indices:
                continue
            negative_indices = [idx for idx, value in enumerate(candidate_labels) if value == 0]
            if not negative_indices:
                continue
            coverage[target]["captured_sessions"] += 1
            if len(negative_indices) > max_negatives:
                negative_indices = rng.sample(negative_indices, max_negatives)
            keep_indices = positive_indices + negative_indices
            query_id = query_ids[target]
            query_ids[target] += 1
            group_sizes_by_target[target].append(len(keep_indices))
            for idx in keep_indices:
                rows_by_target[target].append(features[idx])
                labels_by_target[target].append(candidate_labels[idx])
                group_ids_by_target[target].append(query_id)

    target_data = {
        target: TargetTrainingData(
            frame=pd.DataFrame(rows_by_target[target], columns=FEATURE_COLUMNS),
            labels=np.asarray(labels_by_target[target], dtype=np.int8),
            group_ids=np.asarray(group_ids_by_target[target], dtype=np.int32),
            group_sizes=np.asarray(group_sizes_by_target[target], dtype=np.int32),
        )
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
    return TrainingArtifacts(by_target=target_data, coverage=coverage_ratio)


def _resolve_predict_threads(num_threads: int) -> int:
    omp_limit = os.environ.get("OMP_NUM_THREADS")
    if omp_limit is None:
        return max(1, num_threads)
    try:
        return max(1, min(num_threads, int(omp_limit)))
    except ValueError:
        return max(1, num_threads)


def compile_prediction_model(
    model: object,
    target: str,
    predict_backend: str,
    cache_dir: Path,
    num_threads: int,
    experiment_name: str,
) -> PredictionModel:
    if predict_backend == "lightgbm":
        return PredictionModel(backend="lightgbm", predictor=model)
    if predict_backend != "tl2cgen":
        raise ValueError(f"unsupported prediction backend={predict_backend}")

    import tl2cgen
    import treelite.frontend

    compile_threads = _resolve_predict_threads(num_threads)
    libpath = cache_dir / f"{experiment_name}_{target}_tl2cgen.so"
    if libpath.exists():
        libpath.unlink()
    tl_model = treelite.frontend.from_lightgbm(model.booster_)
    tl2cgen.export_lib(
        tl_model,
        toolchain="gcc",
        libpath=libpath,
        nthread=compile_threads,
        params={"parallel_comp": max(1, min(4, compile_threads))},
        verbose=False,
    )
    predictor = tl2cgen.Predictor(libpath, nthread=compile_threads, verbose=False)
    return PredictionModel(backend="tl2cgen", predictor=predictor)


def train_models(
    training_artifacts: TrainingArtifacts,
    num_threads: int,
    seed: int,
    predict_backend: str,
    cache_dir: Path,
    experiment_name: str,
) -> Dict[str, ModelEnsemble]:
    os.environ["OMP_NUM_THREADS"] = str(max(1, num_threads))
    import lightgbm as lgb
    from catboost import CatBoostRanker, Pool

    models: Dict[str, ModelEnsemble] = {}
    for offset, target in enumerate(TARGETS):
        data = training_artifacts.by_target[target]
        frame = data.frame
        target_labels = data.labels
        if frame.empty or target_labels.sum() == 0:
            print(f"[train] skip {target}: empty training frame or no positives")
            continue

        train_matrix = frame.to_numpy(dtype=np.float32, copy=False)
        print(
            f"[train] {target}: rows={len(frame):,}, positives={int(target_labels.sum()):,}, "
            f"queries={len(data.group_sizes):,}"
        )
        lgb_model = lgb.LGBMClassifier(
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
        lgb_model.fit(train_matrix, target_labels)

        cat_pool = Pool(
            data=train_matrix,
            label=target_labels.astype(np.float32, copy=False),
            group_id=data.group_ids,
            feature_names=FEATURE_COLUMNS,
        )
        cat_model = CatBoostRanker(
            task_type="GPU",
            devices="0",
            loss_function="PairLogit",
            eval_metric="NDCG:top=20",
            iterations=400,
            learning_rate=0.05,
            depth=8,
            random_seed=seed + offset,
            boosting_type="Plain",
            border_count=32,
            thread_count=max(1, num_threads),
            verbose=False,
        )
        cat_model.fit(cat_pool, verbose=False)

        models[target] = ModelEnsemble(
            lgbm=compile_prediction_model(
                model=lgb_model,
                target=target,
                predict_backend=predict_backend,
                cache_dir=cache_dir,
                num_threads=num_threads,
                experiment_name=experiment_name,
            ),
            catboost=PredictionModel(backend="catboost", predictor=cat_model),
        )
    return models


def rank_predictions(
    candidates: Sequence[int],
    scores: Sequence[float],
    fallback: Sequence[int],
) -> List[int]:
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda item: item[1], reverse=True)

    ranked = [aid for aid, _ in scored]
    seen = set(ranked)
    for aid in fallback:
        if len(ranked) >= 20:
            break
        if aid in seen:
            continue
        seen.add(aid)
        ranked.append(aid)
    return ranked[:20]


def _rrf_component(scores: np.ndarray, weight: float) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    fused = np.zeros_like(scores, dtype=np.float32)
    fused[order] = weight / (RRF_K + np.arange(1, len(order) + 1, dtype=np.float32))
    return fused


def _fuse_scores(lgbm_scores: Optional[np.ndarray], catboost_scores: Optional[np.ndarray]) -> np.ndarray:
    if lgbm_scores is None and catboost_scores is None:
        raise ValueError("at least one score array is required")
    if lgbm_scores is None:
        return catboost_scores.astype(np.float32, copy=False)
    if catboost_scores is None:
        return lgbm_scores.astype(np.float32, copy=False)
    return _rrf_component(lgbm_scores, LGBM_RRF_WEIGHT) + _rrf_component(catboost_scores, CATBOOST_RRF_WEIGHT)


def prepare_record_batch(
    records: Sequence[SessionRecord],
    covis: CovisitationArtifacts,
    popularity: PopularityArtifacts,
    embedding_artifacts: Optional[EmbeddingArtifacts],
) -> List[PreparedRecord]:
    prepared: List[PreparedRecord] = []
    for record in records:
        context = build_session_context(record.events, embedding_artifacts)
        features_by_target: Dict[str, Tuple[List[int], np.ndarray]] = {}
        for target in TARGETS:
            candidates, rows = build_feature_rows_from_context(
                record.events,
                context,
                covis,
                popularity,
                embedding_artifacts,
                target,
            )
            features_by_target[target] = (candidates, np.asarray(rows, dtype=np.float32))
        prepared.append(PreparedRecord(record=record, features_by_target=features_by_target))
    return prepared


def iter_record_chunks(records: Iterable[SessionRecord], batch_size: int) -> Iterator[List[SessionRecord]]:
    iterator = iter(records)
    while True:
        chunk = list(islice(iterator, batch_size))
        if not chunk:
            break
        yield chunk


def predict_prepared_batch(
    prepared_records: Sequence[PreparedRecord],
    popularity: PopularityArtifacts,
    models: Dict[str, ModelEnsemble],
) -> List[Dict[str, List[int]]]:
    predictions: List[Dict[str, List[int]]] = [{} for _ in prepared_records]
    for target in TARGETS:
        row_slices: List[Tuple[int, List[int], int, int]] = []
        stacked_features: List[np.ndarray] = []
        offset = 0
        for record_idx, prepared in enumerate(prepared_records):
            candidates, feature_matrix = prepared.features_by_target[target]
            row_count = int(feature_matrix.shape[0])
            if row_count == 0:
                predictions[record_idx][target] = list(popularity.fallback[target][:20])
                continue
            row_slices.append((record_idx, candidates, offset, row_count))
            offset += row_count
            stacked_features.append(feature_matrix)

        if not stacked_features:
            continue

        batch_matrix = np.vstack(stacked_features)
        ensemble = models.get(target)
        lgbm_scores = None
        catboost_scores = None
        if ensemble is not None and ensemble.lgbm is not None:
            lgbm_scores = ensemble.lgbm.predict_scores(batch_matrix)
        if ensemble is not None and ensemble.catboost is not None:
            catboost_scores = ensemble.catboost.predict_scores(batch_matrix)

        for record_idx, candidates, start, row_count in row_slices:
            if lgbm_scores is None and catboost_scores is None:
                local_scores = batch_matrix[start : start + row_count, 0]
            else:
                local_lgbm = None if lgbm_scores is None else lgbm_scores[start : start + row_count]
                local_catboost = None if catboost_scores is None else catboost_scores[start : start + row_count]
                local_scores = _fuse_scores(local_lgbm, local_catboost)
            predictions[record_idx][target] = rank_predictions(
                candidates,
                local_scores.tolist(),
                popularity.fallback[target],
            )
    return predictions


def evaluate_recall_at_20(
    valid_examples: Sequence[SessionRecord],
    covis: CovisitationArtifacts,
    popularity: PopularityArtifacts,
    embedding_artifacts: Optional[EmbeddingArtifacts],
    models: Dict[str, ModelEnsemble],
    predict_batch_size: int,
) -> Dict[str, float]:
    numerator = {target: 0.0 for target in TARGETS}
    denominator = {target: 0.0 for target in TARGETS}

    progress = tqdm(total=len(valid_examples), desc="evaluate cv")
    for chunk in iter_record_chunks(valid_examples, predict_batch_size):
        prepared = prepare_record_batch(chunk, covis, popularity, embedding_artifacts)
        chunk_predictions = predict_prepared_batch(prepared, popularity, models)
        for prepared_record, record_predictions in zip(prepared, chunk_predictions):
            labels = prepared_record.record.labels or {}
            for target in TARGETS:
                if target not in labels:
                    continue
                predictions = record_predictions[target]
                if target == "clicks":
                    denominator[target] += 1.0
                    numerator[target] += 1.0 if int(labels[target]) in predictions else 0.0  # type: ignore[arg-type]
                else:
                    target_labels = labels[target]  # type: ignore[assignment]
                    hits = sum(1 for aid in target_labels if aid in predictions)
                    numerator[target] += float(hits)
                    denominator[target] += float(min(20, len(target_labels)))
        progress.update(len(chunk))
    progress.close()

    metrics: Dict[str, float] = {}
    total = 0.0
    for target in TARGETS:
        recall = numerator[target] / denominator[target] if denominator[target] else 0.0
        metrics[f"{target}_recall@20"] = recall
        total += TARGET_WEIGHTS[target] * recall
    metrics["weighted_recall@20"] = total
    metrics["valid_sessions"] = float(len(valid_examples))
    return metrics
