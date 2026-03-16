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
BASE_COVIS_TOPK = 150
CLICK2CLICK_TOPK = 100
CLICK2CART_TOPK = 100

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


@dataclass(slots=True)
class AidStats:
    counts_all: Dict[int, int]
    counts_by_target: Dict[str, Dict[int, int]]
    last_ts_by_target: Dict[str, Dict[int, int]]
    mean_behavior_type: Dict[int, float]


@dataclass(slots=True)
class RecallArtifacts:
    base_covis: Dict[int, List[Tuple[int, float]]]
    click2click: Dict[int, List[Tuple[int, float]]]
    click2cart: Dict[int, List[Tuple[int, float]]]
    popularity: PopularityArtifacts
    aid_stats: AidStats


@dataclass(slots=True)
class CandidateBundle:
    candidates: List[int]
    base_score_sum: Dict[int, float]
    base_score_max: Dict[int, float]
    base_best_rank: Dict[int, int]
    click2click_score_sum: Dict[int, float]
    click2click_score_max: Dict[int, float]
    click2click_best_rank: Dict[int, int]
    click2cart_score_sum: Dict[int, float]
    click2cart_score_max: Dict[int, float]
    click2cart_best_rank: Dict[int, int]
    source_counts: Dict[str, int]


def resolve_raw_files(raw_dir: Path) -> Dict[str, Path]:
    required = {
        "train": raw_dir / "train.jsonl",
        "test": raw_dir / "test.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing raw OTTO files: {', '.join(missing)}")
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
    def __init__(self, topk: int = BASE_COVIS_TOPK, prune_topk: int = 300, forward_window: int = 1) -> None:
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
        return {
            aid: sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[: self.topk]
            for aid, neighbors in self.data.items()
        }


class WeightedPairBuilder:
    def __init__(
        self,
        pair_weights: Dict[Tuple[int, int], float],
        topk: int,
        prune_topk: int,
        forward_window: int,
        time_scale_ms: int,
        backward_scale: float,
        pop_alpha: float,
    ) -> None:
        self.pair_weights = pair_weights
        self.topk = topk
        self.prune_topk = prune_topk
        self.forward_window = forward_window
        self.time_scale_ms = float(time_scale_ms)
        self.backward_scale = backward_scale
        self.pop_alpha = pop_alpha
        self.data: Dict[int, Dict[int, float]] = {}

    def add_session(self, events: Sequence[Event]) -> None:
        recent = list(reversed(unique_recent_events(events, limit=30)))
        if len(recent) < 2:
            return
        session_weight = 1.0 / math.log1p(len(recent))
        n_events = len(recent)
        for left_idx in range(n_events - 1):
            src_aid, src_ts, src_type = recent[left_idx]
            max_right = min(n_events, left_idx + 1 + self.forward_window)
            for right_idx in range(left_idx + 1, max_right):
                dst_aid, dst_ts, dst_type = recent[right_idx]
                gap = right_idx - left_idx
                delta_ms = max(0, dst_ts - src_ts)
                position_weight = 1.0 / float(1 + gap)
                time_weight = 1.0 / (1.0 + (delta_ms / self.time_scale_ms))
                total_weight = position_weight * time_weight * session_weight

                forward_base = self.pair_weights.get((src_type, dst_type), 0.0)
                if forward_base > 0.0:
                    self._update(src_aid, dst_aid, forward_base * total_weight)
                    self._update(dst_aid, src_aid, forward_base * total_weight * self.backward_scale)

                backward_base = self.pair_weights.get((dst_type, src_type), 0.0)
                if backward_base > 0.0:
                    self._update(dst_aid, src_aid, backward_base * total_weight)
                    self._update(src_aid, dst_aid, backward_base * total_weight * self.backward_scale)

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

    def finalize(self, counts_all: Counter[int]) -> Dict[int, List[Tuple[int, float]]]:
        finalized: Dict[int, List[Tuple[int, float]]] = {}
        for aid, neighbors in self.data.items():
            rescored = []
            for dst_aid, score in neighbors.items():
                penalty = 1.0 + self.pop_alpha * math.log1p(counts_all.get(dst_aid, 0))
                rescored.append((dst_aid, score / penalty))
            finalized[aid] = sorted(rescored, key=lambda item: item[1], reverse=True)[: self.topk]
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


def build_aid_stats(
    counts_all: Counter[int],
    counts_by_target: Dict[str, Counter[int]],
    last_ts_by_target: Dict[str, Dict[int, int]],
    type_sum: Dict[int, float],
) -> AidStats:
    mean_behavior_type = {
        aid: (type_sum[aid] / counts_all[aid]) if counts_all[aid] else 0.0
        for aid in counts_all
    }
    return AidStats(
        counts_all=dict(counts_all),
        counts_by_target={target: dict(counter) for target, counter in counts_by_target.items()},
        last_ts_by_target=last_ts_by_target,
        mean_behavior_type=mean_behavior_type,
    )


def _get_split_ts(max_ts: int, cv_split_mode: str) -> int:
    if cv_split_mode == "week":
        return max_ts - 7 * MS_PER_DAY
    if cv_split_mode == "2day":
        return max_ts - 2 * MS_PER_DAY
    raise ValueError(f"unsupported cv_split_mode={cv_split_mode}")


def prepare_cv_corpus(
    train_path: Path,
    sample_size: int,
    seed: int,
    cv_split_mode: str = "week",
) -> Tuple[RecallArtifacts, List[SessionRecord], List[SessionRecord], int, set[int]]:
    max_ts = get_max_ts(train_path)
    split_ts = _get_split_ts(max_ts, cv_split_mode)
    known_aids: set[int] = set()

    base_builder = CovisitationBuilder()
    click2click_builder = WeightedPairBuilder(
        pair_weights={(TYPE_TO_ID["clicks"], TYPE_TO_ID["clicks"]): 1.0},
        topk=CLICK2CLICK_TOPK,
        prune_topk=200,
        forward_window=20,
        time_scale_ms=12 * MS_PER_HOUR,
        backward_scale=1.0,
        pop_alpha=0.10,
    )
    click2cart_builder = WeightedPairBuilder(
        pair_weights={
            (TYPE_TO_ID["clicks"], TYPE_TO_ID["carts"]): 1.0,
            (TYPE_TO_ID["clicks"], TYPE_TO_ID["orders"]): 1.2,
        },
        topk=CLICK2CART_TOPK,
        prune_topk=200,
        forward_window=20,
        time_scale_ms=24 * MS_PER_HOUR,
        backward_scale=0.30,
        pop_alpha=0.08,
    )

    counts_all: Counter[int] = Counter()
    counts_by_target: Dict[str, Counter[int]] = {target: Counter() for target in TARGETS}
    last_ts_by_target: Dict[str, Dict[int, int]] = {target: {} for target in TARGETS}
    type_sum: Dict[int, float] = Counter()
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

            base_builder.add_session(trimmed)
            click2click_builder.add_session(trimmed)
            click2cart_builder.add_session(trimmed)

            for aid, ts, event_type in trimmed:
                target = ID_TO_TYPE[event_type]
                known_aids.add(aid)
                counts_all[aid] += 1
                counts_by_target[target][aid] += 1
                last_ts_by_target[target][aid] = ts
                type_sum[aid] += float(event_type)

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
    aid_stats = build_aid_stats(counts_all, counts_by_target, last_ts_by_target, type_sum)
    valid_examples: List[SessionRecord] = []
    split_rng = random.Random(seed)
    for record in tqdm(valid_pool, desc="build cv valid"):
        filtered = keep_known_aids(record.events, known_aids)
        example = random_split_session(SessionRecord(record.session, filtered), split_rng)
        if example is not None:
            valid_examples.append(example)

    recall_artifacts = RecallArtifacts(
        base_covis=base_builder.finalize(),
        click2click=click2click_builder.finalize(counts_all),
        click2cart=click2cart_builder.finalize(counts_all),
        popularity=popularity,
        aid_stats=aid_stats,
    )
    return recall_artifacts, sample_sessions, valid_examples, split_ts, known_aids


def prepare_full_corpus(
    train_path: Path,
    sample_size: int,
    seed: int,
) -> Tuple[RecallArtifacts, List[SessionRecord], set[int]]:
    known_aids: set[int] = set()
    base_builder = CovisitationBuilder()
    click2click_builder = WeightedPairBuilder(
        pair_weights={(TYPE_TO_ID["clicks"], TYPE_TO_ID["clicks"]): 1.0},
        topk=CLICK2CLICK_TOPK,
        prune_topk=200,
        forward_window=20,
        time_scale_ms=12 * MS_PER_HOUR,
        backward_scale=1.0,
        pop_alpha=0.10,
    )
    click2cart_builder = WeightedPairBuilder(
        pair_weights={
            (TYPE_TO_ID["clicks"], TYPE_TO_ID["carts"]): 1.0,
            (TYPE_TO_ID["clicks"], TYPE_TO_ID["orders"]): 1.2,
        },
        topk=CLICK2CART_TOPK,
        prune_topk=200,
        forward_window=20,
        time_scale_ms=24 * MS_PER_HOUR,
        backward_scale=0.30,
        pop_alpha=0.08,
    )

    counts_all: Counter[int] = Counter()
    counts_by_target: Dict[str, Counter[int]] = {target: Counter() for target in TARGETS}
    last_ts_by_target: Dict[str, Dict[int, int]] = {target: {} for target in TARGETS}
    type_sum: Dict[int, float] = Counter()
    sample_sessions: List[SessionRecord] = []
    rng = random.Random(seed)
    sampled_seen = 0

    for session_id, events in tqdm(iter_sessions(train_path), desc="prepare full corpus"):
        if not events:
            continue
        base_builder.add_session(events)
        click2click_builder.add_session(events)
        click2cart_builder.add_session(events)
        for aid, ts, event_type in events:
            target = ID_TO_TYPE[event_type]
            known_aids.add(aid)
            counts_all[aid] += 1
            counts_by_target[target][aid] += 1
            last_ts_by_target[target][aid] = ts
            type_sum[aid] += float(event_type)
        if len(events) >= 2:
            sampled_seen = update_reservoir(
                sample_sessions,
                sampled_seen,
                SessionRecord(session=session_id, events=events),
                sample_size,
                rng,
            )

    recall_artifacts = RecallArtifacts(
        base_covis=base_builder.finalize(),
        click2click=click2click_builder.finalize(counts_all),
        click2cart=click2cart_builder.finalize(counts_all),
        popularity=build_popularity_artifacts(counts_all, counts_by_target),
        aid_stats=build_aid_stats(counts_all, counts_by_target, last_ts_by_target, type_sum),
    )
    return recall_artifacts, sample_sessions, known_aids


def _score_matrix(
    source_events: Sequence[Event],
    matrix: Dict[int, List[Tuple[int, float]]],
    topk: int,
) -> Tuple[List[int], Dict[int, float], Dict[int, float], Dict[int, int]]:
    score_sum: Dict[int, float] = {}
    score_max: Dict[int, float] = {}
    best_rank: Dict[int, int] = {}

    for src_idx, (src_aid, _, src_type) in enumerate(source_events):
        src_weight = EVENT_TYPE_WEIGHTS[src_type] / float(src_idx + 1)
        for neighbor_rank, (dst_aid, neighbor_score) in enumerate(matrix.get(src_aid, ()), start=1):
            total_score = neighbor_score * src_weight
            score_sum[dst_aid] = score_sum.get(dst_aid, 0.0) + total_score
            score_max[dst_aid] = max(score_max.get(dst_aid, 0.0), total_score)
            best_rank[dst_aid] = min(best_rank.get(dst_aid, 10**9), neighbor_rank)

    ordered = [aid for aid, _ in sorted(score_sum.items(), key=lambda item: item[1], reverse=True)[:topk]]
    return ordered, score_sum, score_max, best_rank


def build_candidates(
    events: Sequence[Event],
    recall_artifacts: RecallArtifacts,
    target: str,
) -> CandidateBundle:
    popularity = recall_artifacts.popularity
    if not events:
        fallback = list(popularity.fallback[target][:20])
        return CandidateBundle(
            candidates=fallback,
            base_score_sum={},
            base_score_max={},
            base_best_rank={},
            click2click_score_sum={},
            click2click_score_max={},
            click2click_best_rank={},
            click2cart_score_sum={},
            click2cart_score_max={},
            click2cart_best_rank={},
            source_counts={"history": 0, "base_covis": 0, "click2click": 0, "click2cart": 0, "union": len(fallback)},
        )

    recent = unique_recent_events(events, limit=20)
    history_recent = [aid for aid, _, _ in recent]
    click_recent = [event for event in recent if event[2] == TYPE_TO_ID["clicks"]][:10]

    base_ordered, base_score_sum, base_score_max, base_best_rank = _score_matrix(
        recent[:5], recall_artifacts.base_covis, BASE_COVIS_TOPK
    )
    click2click_ordered, click2click_score_sum, click2click_score_max, click2click_best_rank = _score_matrix(
        click_recent, recall_artifacts.click2click, CLICK2CLICK_TOPK
    )
    click2cart_ordered, click2cart_score_sum, click2cart_score_max, click2cart_best_rank = _score_matrix(
        click_recent, recall_artifacts.click2cart, CLICK2CART_TOPK
    )

    if target == "clicks":
        combined = history_recent + base_ordered + click2click_ordered
    else:
        combined = history_recent + base_ordered + click2cart_ordered

    deduped: List[int] = []
    seen = set()
    for aid in combined:
        if aid in seen:
            continue
        seen.add(aid)
        deduped.append(aid)

    if len(deduped) < 20:
        for aid in popularity.fallback[target]:
            if aid in seen:
                continue
            seen.add(aid)
            deduped.append(aid)
            if len(deduped) >= 20:
                break

    return CandidateBundle(
        candidates=deduped,
        base_score_sum=base_score_sum,
        base_score_max=base_score_max,
        base_best_rank=base_best_rank,
        click2click_score_sum=click2click_score_sum,
        click2click_score_max=click2click_score_max,
        click2click_best_rank=click2click_best_rank,
        click2cart_score_sum=click2cart_score_sum,
        click2cart_score_max=click2cart_score_max,
        click2cart_best_rank=click2cart_best_rank,
        source_counts={
            "history": len(history_recent),
            "base_covis": len(base_ordered),
            "click2click": len(click2click_ordered),
            "click2cart": len(click2cart_ordered),
            "union": len(deduped),
        },
    )


def load_test_records(test_path: Path, known_aids: set[int]) -> Iterator[SessionRecord]:
    for session_id, events in iter_sessions(test_path):
        yield SessionRecord(session=session_id, events=keep_known_aids(events, known_aids))
