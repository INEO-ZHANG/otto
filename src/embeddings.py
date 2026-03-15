from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from itemcf import Event, PopularityArtifacts, unique_recent_events, iter_sessions

W2V_VECTOR_SIZE = 64
W2V_WINDOW = 10
W2V_EPOCHS = 10
PRONE_VECTOR_SIZE = 32
PRONE_RECENT_AIDS = 10
EMBEDDING_RANK_LIMIT = 100_000


@dataclass(slots=True)
class EmbeddingArtifacts:
    w2v_vectors: Dict[int, np.ndarray]
    prone_vectors: Dict[int, np.ndarray]
    cache_hits: Dict[str, bool]

    def get_w2v_vector(self, aid: Optional[int]) -> Optional[np.ndarray]:
        if aid is None:
            return None
        return self.w2v_vectors.get(aid)

    def get_prone_vector(self, aid: Optional[int]) -> Optional[np.ndarray]:
        if aid is None:
            return None
        return self.prone_vectors.get(aid)

    def build_prone_session_vector(self, events: Sequence[Event]) -> Optional[np.ndarray]:
        weighted: Optional[np.ndarray] = None
        total_weight = 0.0
        for idx, (aid, _, _) in enumerate(unique_recent_events(events, limit=PRONE_RECENT_AIDS)):
            vector = self.prone_vectors.get(aid)
            if vector is None:
                continue
            weight = 1.0 / float(idx + 1)
            if weighted is None:
                weighted = np.zeros_like(vector, dtype=np.float32)
            weighted += weight * vector
            total_weight += weight
        if weighted is None or total_weight <= 0.0:
            return None
        weighted /= total_weight
        return _normalize_vector(weighted)

    def w2v_last_aid_cosine(self, last_aid: Optional[int], candidate_aid: int) -> float:
        last_vector = self.get_w2v_vector(last_aid)
        candidate_vector = self.get_w2v_vector(candidate_aid)
        if last_vector is None or candidate_vector is None:
            return 0.0
        return float(np.dot(last_vector, candidate_vector))

    def prone_session_aid_cosine(
        self,
        session_vector: Optional[np.ndarray],
        candidate_aid: int,
    ) -> float:
        if session_vector is None:
            return 0.0
        candidate_vector = self.get_prone_vector(candidate_aid)
        if candidate_vector is None:
            return 0.0
        return float(np.dot(session_vector, candidate_vector))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def _vector_cache_path(cache_dir: Path, prefix: str, name: str) -> Path:
    return cache_dir / f"{prefix}_{name}.npz"


def _corpus_cache_path(cache_dir: Path, prefix: str) -> Path:
    return cache_dir / f"{prefix}_w2v_corpus.txt"


def _load_vector_cache(path: Path) -> Dict[int, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    aids = payload["aids"].astype(np.int64, copy=False)
    vectors = payload["vectors"].astype(np.float32, copy=False)
    return {int(aid): vectors[idx] for idx, aid in enumerate(aids.tolist())}


def _save_vector_cache(path: Path, vectors: Dict[int, np.ndarray], dim: int) -> None:
    if vectors:
        ordered_aids = np.asarray(sorted(vectors.keys()), dtype=np.int64)
        ordered_vectors = np.vstack([vectors[int(aid)] for aid in ordered_aids]).astype(np.float32, copy=False)
    else:
        ordered_aids = np.asarray([], dtype=np.int64)
        ordered_vectors = np.empty((0, dim), dtype=np.float32)
    np.savez_compressed(path, aids=ordered_aids, vectors=ordered_vectors)


def _build_allowed_aids(popularity: PopularityArtifacts) -> set[int]:
    return {
        aid
        for aid, rank in popularity.all_rank.items()
        if rank <= EMBEDDING_RANK_LIMIT
    }


def _iter_filtered_sequences(
    train_path: Path,
    split_ts: Optional[int],
    allowed_aids: set[int],
):
    for _, events in iter_sessions(train_path):
        sequence = []
        last_aid = None
        for aid, ts, _ in events:
            if split_ts is not None and ts >= split_ts:
                break
            if aid not in allowed_aids:
                continue
            if aid == last_aid:
                continue
            sequence.append(str(aid))
            last_aid = aid
        if len(sequence) >= 2:
            yield sequence


def _build_w2v_corpus_file(
    train_path: Path,
    split_ts: Optional[int],
    allowed_aids: set[int],
    corpus_path: Path,
) -> None:
    with corpus_path.open("w", encoding="utf-8") as handle:
        for sequence in _iter_filtered_sequences(train_path, split_ts, allowed_aids):
            handle.write(" ".join(sequence))
            handle.write("\n")


def _train_w2v_vectors(
    train_path: Path,
    split_ts: Optional[int],
    allowed_aids: set[int],
    cache_dir: Path,
    cache_prefix: str,
    num_threads: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence

    corpus_path = _corpus_cache_path(cache_dir, cache_prefix)
    if not corpus_path.exists():
        _build_w2v_corpus_file(train_path, split_ts, allowed_aids, corpus_path)

    sentences = LineSentence(str(corpus_path))
    model = Word2Vec(
        sentences=sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        sg=1,
        negative=10,
        min_count=1,
        epochs=W2V_EPOCHS,
        # gensim 4.4 on Python 3.12 can emit ignored teardown errors from the
        # threaded inner loop; keep W2V single-threaded for stability.
        workers=1,
        seed=seed,
    )
    vectors: Dict[int, np.ndarray] = {}
    for aid in allowed_aids:
        token = str(aid)
        if token not in model.wv:
            continue
        vectors[aid] = _normalize_vector(model.wv[token].astype(np.float32, copy=False))
    return vectors


def _build_sparse_graph(
    pair_counts: Dict[int, Dict[int, float]],
    allowed_aids: set[int],
) -> Tuple[np.ndarray, sparse.csr_matrix]:
    ordered_aids = np.asarray(sorted(allowed_aids), dtype=np.int64)
    index_by_aid = {int(aid): idx for idx, aid in enumerate(ordered_aids.tolist())}
    rows = array("I")
    cols = array("I")
    data = array("f")

    for src_aid, neighbors in pair_counts.items():
        src_idx = index_by_aid.get(src_aid)
        if src_idx is None:
            continue
        for dst_aid, weight in neighbors.items():
            dst_idx = index_by_aid.get(dst_aid)
            if dst_idx is None or dst_idx == src_idx or weight <= 0.0:
                continue
            rows.append(src_idx)
            cols.append(dst_idx)
            data.append(float(weight))

    if not rows:
        return ordered_aids, sparse.csr_matrix((len(ordered_aids), len(ordered_aids)), dtype=np.float32)

    row_idx = np.frombuffer(rows, dtype=np.uint32).astype(np.int32, copy=False)
    col_idx = np.frombuffer(cols, dtype=np.uint32).astype(np.int32, copy=False)
    values = np.frombuffer(data, dtype=np.float32)
    graph = sparse.coo_matrix(
        (values, (row_idx, col_idx)),
        shape=(len(ordered_aids), len(ordered_aids)),
        dtype=np.float32,
    ).tocsr()
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    return ordered_aids, graph


def _train_prone_vectors(
    pair_counts: Dict[int, Dict[int, float]],
    allowed_aids: set[int],
    seed: int,
) -> Dict[int, np.ndarray]:
    ordered_aids, graph = _build_sparse_graph(pair_counts, allowed_aids)
    n_items = int(graph.shape[0])
    if n_items == 0:
        return {}
    if n_items == 1:
        return {int(ordered_aids[0]): np.ones(PRONE_VECTOR_SIZE, dtype=np.float32)}

    degrees = np.asarray(graph.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    inv_degrees = np.zeros_like(degrees)
    valid = degrees > 0.0
    inv_degrees[valid] = 1.0 / degrees[valid]
    normalized = sparse.diags(inv_degrees) @ graph
    normalized = normalized.tocsr()
    if normalized.nnz:
        normalized.data = np.log1p(normalized.data * float(n_items)).astype(np.float32, copy=False)

    n_components = min(PRONE_VECTOR_SIZE, max(1, n_items - 1))
    svd = TruncatedSVD(n_components=n_components, algorithm="randomized", random_state=seed)
    base_vectors = svd.fit_transform(normalized).astype(np.float32, copy=False)

    inv_sqrt = np.zeros_like(degrees)
    inv_sqrt[valid] = 1.0 / np.sqrt(degrees[valid])
    propagation = sparse.diags(inv_sqrt) @ graph @ sparse.diags(inv_sqrt)
    prop1 = propagation @ base_vectors
    prop2 = propagation @ prop1
    enhanced = base_vectors + 0.5 * prop1 + 0.25 * prop2

    if n_components < PRONE_VECTOR_SIZE:
        padded = np.zeros((n_items, PRONE_VECTOR_SIZE), dtype=np.float32)
        padded[:, :n_components] = enhanced
        enhanced = padded

    enhanced = _normalize_rows(enhanced)
    return {int(aid): enhanced[idx] for idx, aid in enumerate(ordered_aids.tolist())}


def prepare_embedding_artifacts(
    train_path: Path,
    cache_dir: Path,
    popularity: PopularityArtifacts,
    pair_counts: Dict[int, Dict[int, float]],
    split_ts: Optional[int],
    num_threads: int,
    seed: int,
) -> EmbeddingArtifacts:
    embedding_dir = cache_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    allowed_aids = _build_allowed_aids(popularity)
    cache_prefix = f"cv_split_{split_ts}" if split_ts is not None else "full_train"
    w2v_path = _vector_cache_path(embedding_dir, cache_prefix, f"w2v_dim{W2V_VECTOR_SIZE}_win{W2V_WINDOW}")
    prone_path = _vector_cache_path(embedding_dir, cache_prefix, f"prone_dim{PRONE_VECTOR_SIZE}_topk120")

    cache_hits = {
        "w2v": w2v_path.exists(),
        "prone": prone_path.exists(),
    }

    if w2v_path.exists():
        w2v_vectors = _load_vector_cache(w2v_path)
    else:
        w2v_vectors = _train_w2v_vectors(
            train_path=train_path,
            split_ts=split_ts,
            allowed_aids=allowed_aids,
            cache_dir=embedding_dir,
            cache_prefix=cache_prefix,
            num_threads=num_threads,
            seed=seed,
        )
        _save_vector_cache(w2v_path, w2v_vectors, W2V_VECTOR_SIZE)

    if prone_path.exists():
        prone_vectors = _load_vector_cache(prone_path)
    else:
        prone_vectors = _train_prone_vectors(
            pair_counts=pair_counts,
            allowed_aids=allowed_aids,
            seed=seed,
        )
        _save_vector_cache(prone_path, prone_vectors, PRONE_VECTOR_SIZE)

    return EmbeddingArtifacts(
        w2v_vectors=w2v_vectors,
        prone_vectors=prone_vectors,
        cache_hits=cache_hits,
    )
