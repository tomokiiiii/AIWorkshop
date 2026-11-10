#!/usr/bin/env python3
"""Train a BDT to rank true K0S -> pi+ pi- candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from data import FEATURE_NAMES, load_candidates
from data.workflow import (
    balanced_sample_weights,
    binary_metrics,
    save_model,
    stratified_split,
    write_json,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(Path(__file__).resolve().parent.parent / "input_file")],
    )
    parser.add_argument("--tree-name", default="DecayTree")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "output_file" / "bdt"),
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_candidates(args.input, tree_name=args.tree_name, require_truth=True)
    n_signal = int(np.sum(data.labels == 1))
    if n_signal < 20:
        print(
            f"WARNING: only {n_signal} true K0S candidates are available. "
            "This is a code test, not a physics-quality training.",
            flush=True,
        )
    train_index, test_index = stratified_split(
        data.labels, args.test_fraction, args.seed
    )
    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        random_state=args.seed,
    )
    model.fit(
        data.features[train_index],
        data.labels[train_index],
        sample_weight=balanced_sample_weights(data.labels[train_index]),
    )
    scores = model.predict_proba(data.features)[:, 1]
    split = np.full(len(data), "train", dtype=object)
    split[test_index] = "test"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
    }
    save_model(output_dir / "bdt_model.joblib", model, args.tree_name, settings)
    write_predictions(output_dir / "predictions.csv", data, scores, split)
    metrics = {
        "all_entries": len(data),
        "signal_entries": n_signal,
        "background_entries": int(np.sum(data.labels == 0)),
        "train": binary_metrics(data.labels[train_index], scores[train_index]),
        "test": binary_metrics(data.labels[test_index], scores[test_index]),
    }
    write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "feature_importance.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("feature", "importance"))
        writer.writerows(
            sorted(
                zip(FEATURE_NAMES, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        )
    print(
        {
            "status": "finished",
            "model": str(output_dir / "bdt_model.joblib"),
            "predictions": str(output_dir / "predictions.csv"),
            "test_metrics": metrics["test"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
