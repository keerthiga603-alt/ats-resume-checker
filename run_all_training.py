"""
Runs the entire training pipeline end-to-end, stage by stage, in one process.
On a normal machine (no sandbox timeout) this just works as a single script —
no need to invoke each stage as a separate command.

Usage (from the ats-system/ directory, with your venv activated):
    python run_all_training.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ml.train import (
    stage_build_pairs,
    stage_compute_features,
    stage_split_and_medians,
    stage_fit_tfidf,
    stage_tfidf_cosine_chunk,
    stage_finalize_matrices,
    stage_train_models,
)


def main():
    print("=== Stage 1/9: build_pairs ===")
    stage_build_pairs()

    print("=== Stage 2-6/9: compute_features (5 chunks of ~5000 pairs) ===")
    stage_compute_features(0, 5000, "_0")
    stage_compute_features(5000, 10000, "_1")
    stage_compute_features(10000, 15000, "_2")
    stage_compute_features(15000, 20000, "_3")
    stage_compute_features(20000, 25002, "_4")

    print("=== Stage 7/9: split_and_medians ===")
    stage_split_and_medians()

    print("=== Stage 8/9: fit_tfidf ===")
    stage_fit_tfidf()

    # Split sizes are printed by stage_split_and_medians (70/15/15 of 25,002
    # usable pairs, stratified) — read them back from split_indices.json so
    # this script doesn't hardcode a count that could drift if the dataset
    # or split ratio changes.
    import json
    with open(os.path.join("data", "processed", "split_indices.json")) as f:
        split_idx = json.load(f)
    n_train = len(split_idx["idx_train"])
    n_val = len(split_idx["idx_val"])
    n_test = len(split_idx["idx_test"])
    print(f"Split sizes -> train: {n_train}, val: {n_val}, test: {n_test}")

    print("=== Stage 9a/9: TF-IDF cosine similarity (batched, real values) ===")
    stage_tfidf_cosine_chunk("train", 0, n_train)
    stage_tfidf_cosine_chunk("val", 0, n_val)
    stage_tfidf_cosine_chunk("test", 0, n_test)

    print("=== Stage 9b/9: finalize_matrices ===")
    stage_finalize_matrices()

    print("=== Stage 9c/9: train_models (train/compare/select/evaluate/save) ===")
    metadata = stage_train_models()

    print()
    print("DONE. Real trained model saved to models/ats_model.pkl")
    print("Selected algorithm:", metadata["algorithm"])
    print("Test metrics:", metadata["test_metrics"])


if __name__ == "__main__":
    main()
