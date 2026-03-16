from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from features import (
    build_feature_rows,
    build_training_frames,
    evaluate_recall_at_20,
    prepare_w2v_artifacts,
    score_target,
    train_models,
)
from recall import (
    TARGETS,
    load_test_records,
    prepare_cv_corpus,
    prepare_full_corpus,
    resolve_raw_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple OTTO multi-recall + LGBM baseline")
    parser.add_argument("--mode", choices=("cv", "submit", "both"), default="both")
    parser.add_argument("--train-sample-sessions", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--cv-split", choices=("week", "2day"), default="week")
    return parser.parse_args()


def ensure_dirs(project_root: Path) -> Dict[str, Path]:
    raw_dir = project_root / "data" / "raw"
    cache_dir = project_root / "cache"
    outputs_dir = project_root / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "raw_dir": raw_dir,
        "cache_dir": cache_dir,
        "outputs_dir": outputs_dir,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def write_submission(
    test_path: Path,
    output_path: Path,
    recall_artifacts,
    w2v_artifacts,
    models,
    known_aids: set[int],
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("session_type,labels\n")
        for record in tqdm(load_test_records(test_path, known_aids), desc="write submission"):
            for target in TARGETS:
                candidates, features, _ = build_feature_rows(record, recall_artifacts, w2v_artifacts, target)
                ranked = score_target(
                    candidates,
                    features,
                    target,
                    models,
                    recall_artifacts.popularity.fallback[target],
                )
                labels = " ".join(str(aid) for aid in ranked)
                handle.write(f"{record.session}_{target},{labels}\n")


def run_cv(
    train_path: Path,
    cache_dir: Path,
    outputs_dir: Path,
    train_sample_sessions: int,
    seed: int,
    num_threads: int,
    cv_split_mode: str,
) -> Dict[str, object]:
    recall_artifacts, sampled_sessions, valid_examples, split_ts, known_aids = prepare_cv_corpus(
        train_path=train_path,
        sample_size=train_sample_sessions,
        seed=seed,
        cv_split_mode=cv_split_mode,
    )
    w2v_artifacts = prepare_w2v_artifacts(
        train_path=train_path,
        cache_dir=cache_dir,
        known_aids=known_aids,
        split_ts=split_ts,
        cache_prefix=f"cv_{cv_split_mode}_{split_ts}",
        seed=seed,
    )
    frames, labels, coverage_by_target = build_training_frames(
        sampled_sessions=sampled_sessions,
        recall_artifacts=recall_artifacts,
        w2v_artifacts=w2v_artifacts,
        seed=seed,
    )
    models = train_models(frames=frames, labels=labels, num_threads=num_threads, seed=seed)
    metrics, candidate_source_stats, avg_candidates_by_target = evaluate_recall_at_20(
        valid_examples=valid_examples,
        recall_artifacts=recall_artifacts,
        w2v_artifacts=w2v_artifacts,
        models=models,
    )

    payload: Dict[str, object] = {
        **metrics,
        "avg_candidates_by_target": avg_candidates_by_target,
        "candidate_source_stats": candidate_source_stats,
        "coverage_by_target": coverage_by_target,
        "cv_split_mode": cv_split_mode,
        "sampled_sessions": len(sampled_sessions),
        "split_ts": split_ts,
        "valid_examples": len(valid_examples),
        "w2v_cache_hit": w2v_artifacts.cache_hit,
    }
    write_json(outputs_dir / "cv_metrics.json", payload)
    return payload


def run_submit(
    train_path: Path,
    test_path: Path,
    cache_dir: Path,
    outputs_dir: Path,
    train_sample_sessions: int,
    seed: int,
    num_threads: int,
) -> Path:
    recall_artifacts, sampled_sessions, known_aids = prepare_full_corpus(
        train_path=train_path,
        sample_size=train_sample_sessions,
        seed=seed,
    )
    w2v_artifacts = prepare_w2v_artifacts(
        train_path=train_path,
        cache_dir=cache_dir,
        known_aids=known_aids,
        split_ts=None,
        cache_prefix="submit_full",
        seed=seed,
    )
    frames, labels, _ = build_training_frames(
        sampled_sessions=sampled_sessions,
        recall_artifacts=recall_artifacts,
        w2v_artifacts=w2v_artifacts,
        seed=seed,
    )
    models = train_models(frames=frames, labels=labels, num_threads=num_threads, seed=seed)
    submission_path = outputs_dir / "submission.csv"
    write_submission(
        test_path=test_path,
        output_path=submission_path,
        recall_artifacts=recall_artifacts,
        w2v_artifacts=w2v_artifacts,
        models=models,
        known_aids=known_aids,
    )
    return submission_path


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    dirs = ensure_dirs(project_root)
    raw_files = resolve_raw_files(dirs["raw_dir"])

    if args.mode in {"cv", "both"}:
        metrics = run_cv(
            train_path=raw_files["train"],
            cache_dir=dirs["cache_dir"],
            outputs_dir=dirs["outputs_dir"],
            train_sample_sessions=args.train_sample_sessions,
            seed=args.seed,
            num_threads=args.num_threads,
            cv_split_mode=args.cv_split,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))

    if args.mode in {"submit", "both"}:
        submission_path = run_submit(
            train_path=raw_files["train"],
            test_path=raw_files["test"],
            cache_dir=dirs["cache_dir"],
            outputs_dir=dirs["outputs_dir"],
            train_sample_sessions=args.train_sample_sessions,
            seed=args.seed,
            num_threads=args.num_threads,
        )
        print(f"[submit] wrote {submission_path}")


if __name__ == "__main__":
    main()
