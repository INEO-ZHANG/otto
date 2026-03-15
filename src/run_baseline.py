#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from embeddings import prepare_embedding_artifacts

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

from features import (
    build_training_frames,
    evaluate_recall_at_20,
    iter_record_chunks,
    predict_prepared_batch,
    prepare_record_batch,
    train_models,
)
from itemcf import (
    TARGETS,
    load_test_records,
    prepare_cv_corpus,
    prepare_full_corpus,
    resolve_raw_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OTTO 3-covis + LightGBM baseline")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="3covis",
        help="Experiment tag used in output filenames and metric payloads",
    )
    parser.add_argument(
        "--matrix-mode",
        choices=("target_aware", "one_hot"),
        default="target_aware",
        help="Covisitation matrix weighting mode",
    )
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
    parser.add_argument("--num-threads", type=int, default=25, help="LightGBM threads")
    parser.add_argument(
        "--predict-backend",
        choices=("lightgbm", "tl2cgen"),
        default="lightgbm",
        help="Prediction backend used after training",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=1024,
        help="Number of sessions per batch for feature build and prediction",
    )
    return parser.parse_args()


def ensure_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    raw_dir = base_dir / "data" / "raw"
    cache_dir = base_dir / "cache"
    output_dir = base_dir / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, cache_dir, output_dir


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_submission(
    test_path: Path,
    known_aids: set[int],
    covis,
    popularity,
    embedding_artifacts,
    models: Dict[str, object],
    output_path: Path,
    predict_batch_size: int,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session_type", "labels"])
        record_iter = load_test_records(test_path, known_aids)
        progress = tqdm(desc="write submission")
        for chunk in iter_record_chunks(record_iter, predict_batch_size):
            prepared = prepare_record_batch(chunk, covis, popularity, embedding_artifacts)
            chunk_predictions = predict_prepared_batch(prepared, popularity, models)
            for prepared_record, record_predictions in zip(prepared, chunk_predictions):
                for target in TARGETS:
                    writer.writerow(
                        [
                            f"{prepared_record.record.session}_{target}",
                            " ".join(str(aid) for aid in record_predictions[target]),
                        ]
                    )
            progress.update(len(chunk))
        progress.close()


def run_cv(
    train_path: Path,
    cache_dir: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    num_threads: int,
    experiment_name: str,
    matrix_mode: str,
    predict_backend: str,
    predict_batch_size: int,
) -> Dict[str, float]:
    covis, popularity, sampled_sessions, valid_examples, split_ts, known_aids = prepare_cv_corpus(
        train_path=train_path,
        sample_size=sample_size,
        seed=seed,
        matrix_mode=matrix_mode,
    )
    embedding_artifacts = prepare_embedding_artifacts(
        train_path=train_path,
        cache_dir=cache_dir,
        popularity=popularity,
        pair_counts=covis.pair_counts,
        split_ts=split_ts,
        num_threads=num_threads,
        seed=seed,
    )
    print(
        f"[cv] train_samples={len(sampled_sessions):,} valid_examples={len(valid_examples):,} "
        f"known_aids={len(known_aids):,} split_ts={split_ts}"
    )
    training_artifacts = build_training_frames(
        sampled_sessions=sampled_sessions,
        covis=covis,
        popularity=popularity,
        embedding_artifacts=embedding_artifacts,
        seed=seed,
    )
    for target, ratio in training_artifacts.coverage.items():
        print(f"[cv] {target} candidate coverage={ratio:.4f}")
    models = train_models(
        training_artifacts=training_artifacts,
        num_threads=num_threads,
        seed=seed,
        predict_backend=predict_backend,
        cache_dir=cache_dir,
        experiment_name=f"{experiment_name}_cv",
    )
    metrics = evaluate_recall_at_20(
        valid_examples,
        covis,
        popularity,
        embedding_artifacts,
        models,
        predict_batch_size=predict_batch_size,
    )
    metrics_path = output_dir / f"cv_metrics_{experiment_name}.json"
    write_json(
        metrics_path,
        {
            **metrics,
            "split_ts": split_ts,
            "coverage": training_artifacts.coverage,
            "experiment": experiment_name,
            "matrix_mode": matrix_mode,
            "predict_backend": predict_backend,
            "predict_batch_size": predict_batch_size,
            "embedding_cache_hit": embedding_artifacts.cache_hits,
            "catboost_device": "0",
            "fusion_method": "rrf",
        },
    )
    print(f"[cv] metrics written to {metrics_path}")
    return metrics


def run_submit(
    train_path: Path,
    test_path: Path,
    cache_dir: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    num_threads: int,
    experiment_name: str,
    matrix_mode: str,
    predict_backend: str,
    predict_batch_size: int,
) -> Path:
    covis, popularity, sampled_sessions, known_aids = prepare_full_corpus(
        train_path=train_path,
        sample_size=sample_size,
        seed=seed,
        matrix_mode=matrix_mode,
    )
    embedding_artifacts = prepare_embedding_artifacts(
        train_path=train_path,
        cache_dir=cache_dir,
        popularity=popularity,
        pair_counts=covis.pair_counts,
        split_ts=None,
        num_threads=num_threads,
        seed=seed,
    )
    print(f"[submit] train_samples={len(sampled_sessions):,} known_aids={len(known_aids):,}")
    training_artifacts = build_training_frames(
        sampled_sessions=sampled_sessions,
        covis=covis,
        popularity=popularity,
        embedding_artifacts=embedding_artifacts,
        seed=seed,
    )
    for target, ratio in training_artifacts.coverage.items():
        print(f"[submit] {target} candidate coverage={ratio:.4f}")
    models = train_models(
        training_artifacts=training_artifacts,
        num_threads=num_threads,
        seed=seed,
        predict_backend=predict_backend,
        cache_dir=cache_dir,
        experiment_name=f"{experiment_name}_submit",
    )
    submission_path = output_dir / f"submission_{experiment_name}.csv"
    write_submission(
        test_path=test_path,
        known_aids=known_aids,
        covis=covis,
        popularity=popularity,
        embedding_artifacts=embedding_artifacts,
        models=models,
        output_path=submission_path,
        predict_batch_size=predict_batch_size,
    )
    print(f"[submit] submission written to {submission_path}")
    return submission_path


def main() -> None:
    args = parse_args()
    base_dir = Path.cwd()
    raw_dir, cache_dir, output_dir = ensure_dirs(base_dir)
    extracted = resolve_raw_files(raw_dir)
    train_path = extracted["train"]
    test_path = extracted["test"]

    if args.mode in ("cv", "both"):
        metrics = run_cv(
            train_path=train_path,
            cache_dir=cache_dir,
            output_dir=output_dir,
            sample_size=args.train_sample_sessions,
            seed=args.seed,
            num_threads=args.num_threads,
            experiment_name=args.experiment_name,
            matrix_mode=args.matrix_mode,
            predict_backend=args.predict_backend,
            predict_batch_size=args.predict_batch_size,
        )
        print(f"[cv] weighted_recall@20={metrics['weighted_recall@20']:.6f}")

    if args.mode in ("submit", "both"):
        run_submit(
            train_path=train_path,
            test_path=test_path,
            cache_dir=cache_dir,
            output_dir=output_dir,
            sample_size=args.train_sample_sessions,
            seed=args.seed,
            num_threads=args.num_threads,
            experiment_name=args.experiment_name,
            matrix_mode=args.matrix_mode,
            predict_backend=args.predict_backend,
            predict_batch_size=args.predict_batch_size,
        )


if __name__ == "__main__":
    main()
