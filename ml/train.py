"""
Full training pipeline for the ATS compatibility model.
Run: python ml/train.py
"""
import sys
import os
import json
import time
import pickle
import platform
import random
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import xgboost
from xgboost import XGBClassifier

from ml.data_loader import build_resume_jd_pairs, OCCUPATION_CLASSES
from ml.preprocessing import pairs_to_feature_rows, compute_medians, rows_to_matrix
from ml.feature_engineering import FEATURE_NAMES

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def log(msg):
    print(f"[train] {msg}", file=sys.stderr)


def stage_build_pairs():
    log("Loading resume_corpus and building resume-JD pairs (real data, real labels)...")
    pairs = build_resume_jd_pairs(seed=SEED, jd_fraction=0.3)
    log(f"Built {len(pairs)} pairs.")
    with open(os.path.join(PROCESSED_DIR, "pairs.pkl"), "wb") as f:
        pickle.dump(pairs, f)
    log("Saved pairs.pkl")


def stage_compute_features(start=0, end=None, out_suffix=""):
    with open(os.path.join(PROCESSED_DIR, "pairs.pkl"), "rb") as f:
        pairs = pickle.load(f)
    subset = pairs[start:end] if end else pairs[start:]
    log(f"Computing features for pairs[{start}:{end}] ({len(subset)} pairs)...")
    X_rows, y, meta, usable_texts = pairs_to_feature_rows_with_texts(subset)
    out_path = os.path.join(PROCESSED_DIR, f"features{out_suffix}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(
            {"X_rows": X_rows, "y": y, "meta": meta, "usable_texts": usable_texts,
             "start": start, "end": end},
            f,
        )
    log(f"Saved {out_path} ({len(X_rows)} usable rows)")


def pairs_to_feature_rows_with_texts(pairs_subset):
    """Same as pairs_to_feature_rows but also returns the (resume_text, jd_text)
    for every row that survived parsing, so downstream stages never need to
    re-parse the corpus just to recover text alignment."""
    from ml.resume_parser import parse_resume, InvalidResume
    from ml.jd_parser import parse_job_description, InvalidJobDescription
    from ml.feature_engineering import compute_features
    X_rows, y, meta, usable_texts = [], [], [], []
    skipped = 0
    t0 = time.time()
    for i, pair in enumerate(pairs_subset):
        try:
            parsed_resume = parse_resume(pair["resume_text"])
        except InvalidResume:
            skipped += 1
            continue
        try:
            parsed_jd = parse_job_description(pair["jd_text"])
        except InvalidJobDescription:
            skipped += 1
            continue
        features = compute_features(parsed_resume, parsed_jd)
        X_rows.append(features)
        y.append(pair["label"])
        meta.append({"resume_class": pair["resume_class"], "jd_class": pair["jd_class"]})
        usable_texts.append({"resume_text": pair["resume_text"], "jd_text": pair["jd_text"]})
        if (i + 1) % 2000 == 0:
            log(f"  processed {i+1}/{len(pairs_subset)} pairs ({time.time()-t0:.1f}s elapsed)")
    log(f"  done. {len(X_rows)} usable rows, {skipped} skipped.")
    return X_rows, y, meta, usable_texts



def stage_split_and_medians():
    """Fast stage: merge feature chunks, split train/val/test, compute medians, save."""
    t0 = time.time()
    log("Merging feature chunks...")
    X_rows, y, meta = [], [], []
    for suffix in ["_0", "_1", "_2", "_3", "_4"]:
        path = os.path.join(PROCESSED_DIR, f"features{suffix}.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
        X_rows.extend(d["X_rows"])
        y.extend(d["y"])
        meta.extend(d["meta"])
    log(f"Merged {len(X_rows)} total feature rows.")

    idx = list(range(len(X_rows)))
    idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    y_temp = [y[i] for i in idx_temp]
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
    log(f"Split sizes -> train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

    train_rows = [X_rows[i] for i in idx_train]
    medians = compute_medians(train_rows)

    def build_matrix(indices):
        rows = [X_rows[i] for i in indices]
        return np.array(rows_to_matrix(rows, medians), dtype=float)

    X_train = build_matrix(idx_train)
    X_val = build_matrix(idx_val)
    X_test = build_matrix(idx_test)
    y_train = np.array([y[i] for i in idx_train])
    y_val = np.array([y[i] for i in idx_val])
    y_test = np.array([y[i] for i in idx_test])

    np.savez(
        os.path.join(PROCESSED_DIR, "matrices_pre_tfidf.npz"),
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )
    with open(os.path.join(PROCESSED_DIR, "medians.json"), "w") as f:
        json.dump(medians, f, indent=2)
    with open(os.path.join(PROCESSED_DIR, "split_indices.json"), "w") as f:
        json.dump({"idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test}, f)
    log(f"stage_split_and_medians done in {time.time()-t0:.1f}s")


def _load_merged_usable_texts():
    """Re-loads the (resume_text, jd_text) pairs that survived parsing, by
    reading the usable_texts each stage_compute_features chunk already saved
    to disk -- in the SAME chunk order stage_split_and_medians used when it
    merged X_rows/y/meta. This keeps the returned list index-aligned with
    split_indices.json without re-parsing the corpus (which is what made the
    old version of this stage take ~9 minutes on a synced OneDrive folder)."""
    usable_pairs = []
    for suffix in ["_0", "_1", "_2", "_3", "_4"]:
        path = os.path.join(PROCESSED_DIR, f"features{suffix}.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
        usable_pairs.extend(d["usable_texts"])
    return usable_pairs


def stage_fit_tfidf():
    """Fits TF-IDF on TRAIN-ONLY text (fast: one fit call, not per-pair)."""
    t0 = time.time()
    usable_pairs = _load_merged_usable_texts()
    with open(os.path.join(PROCESSED_DIR, "usable_pairs.pkl"), "wb") as f:
        pickle.dump(usable_pairs, f)
    log(f"Loaded {len(usable_pairs)} usable pairs from feature chunks (no re-parsing).")

    with open(os.path.join(PROCESSED_DIR, "split_indices.json")) as f:
        split_idx = json.load(f)
    idx_train = split_idx["idx_train"]

    train_texts = []
    for i in idx_train:
        train_texts.append(usable_pairs[i]["resume_text"])
        train_texts.append(usable_pairs[i]["jd_text"])

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
    tfidf_vectorizer.fit(train_texts)
    log(f"TF-IDF vocab size: {len(tfidf_vectorizer.vocabulary_)}")
    with open(os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    log(f"stage_fit_tfidf done in {time.time()-t0:.1f}s")


def stage_tfidf_cosine_chunk(split_name: str, start: int, end: int):
    """
    Computes real TF-IDF cosine similarity for a chunk of one split, using
    BATCHED transform() calls (not one-transform-per-pair) for speed:
    transform all resume texts in the chunk at once, all jd texts at once,
    then take row-wise cosine similarity.
    """
    t0 = time.time()
    with open(os.path.join(PROCESSED_DIR, "usable_pairs.pkl"), "rb") as f:
        usable_pairs = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "split_indices.json")) as f:
        split_idx = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    indices = split_idx[f"idx_{split_name}"][start:end]
    resume_texts = [usable_pairs[i]["resume_text"] for i in indices]
    jd_texts = [usable_pairs[i]["jd_text"] for i in indices]

    R = tfidf_vectorizer.transform(resume_texts)
    J = tfidf_vectorizer.transform(jd_texts)

    import numpy as _np
    from sklearn.preprocessing import normalize
    Rn = normalize(R)
    Jn = normalize(J)
    sims = _np.asarray(Rn.multiply(Jn).sum(axis=1)).ravel()

    out_path = os.path.join(PROCESSED_DIR, f"tfidf_sims_{split_name}_{start}_{end}.npy")
    np.save(out_path, sims)
    log(f"Saved {out_path} ({len(sims)} sims) in {time.time()-t0:.1f}s")


def stage_finalize_matrices():
    """Merges the batched TF-IDF cosine chunks back into matrices_pre_tfidf.npz's
    tfidf_cosine_similarity column and writes the final matrices.npz."""
    t0 = time.time()
    data = np.load(os.path.join(PROCESSED_DIR, "matrices_pre_tfidf.npz"))
    X_train, X_val, X_test = data["X_train"].copy(), data["X_val"].copy(), data["X_test"].copy()
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    tfidf_col_idx = FEATURE_NAMES.index("tfidf_cosine_similarity")

    def load_split_sims(split_name, n):
        import glob
        files = sorted(
            glob.glob(os.path.join(PROCESSED_DIR, f"tfidf_sims_{split_name}_*.npy")),
            key=lambda p: int(p.split("_")[-2])
        )
        parts = [np.load(f) for f in files]
        sims = np.concatenate(parts)
        assert len(sims) == n, f"{split_name}: expected {n} sims, got {len(sims)}"
        return sims

    X_train[:, tfidf_col_idx] = load_split_sims("train", X_train.shape[0])
    X_val[:, tfidf_col_idx] = load_split_sims("val", X_val.shape[0])
    X_test[:, tfidf_col_idx] = load_split_sims("test", X_test.shape[0])

    np.savez(
        os.path.join(PROCESSED_DIR, "matrices.npz"),
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )
    log(f"stage_finalize_matrices done in {time.time()-t0:.1f}s. Final matrices.npz saved.")



def stage_train_models():
    """
    Loads the precomputed matrices.npz (built by stage_finalize_matrices) and
    trains/compares/selects/evaluates the model. This stage is fast (no text
    parsing) since features were already computed in earlier chunked stages.
    """
    t_start = time.time()

    with open(os.path.join(PROCESSED_DIR, "pairs.pkl"), "rb") as f:
        pairs = pickle.load(f)

    data = np.load(os.path.join(PROCESSED_DIR, "matrices.npz"))
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    with open(os.path.join(PROCESSED_DIR, "medians.json")) as f:
        medians = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    log(f"Loaded matrices: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    log("Training candidate models...")
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=SEED, n_jobs=-1
        ),
    }

    val_results = {}
    fitted_models = {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fitted_models[name] = model
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_val, val_pred)),
            "precision": float(precision_score(y_val, val_pred)),
            "recall": float(recall_score(y_val, val_pred)),
            "f1": float(f1_score(y_val, val_pred)),
            "roc_auc": float(roc_auc_score(y_val, val_proba)),
        }
        val_results[name] = metrics
        log(f"  {name}: val={metrics} ({time.time()-t0:.1f}s)")

    best_name = max(val_results, key=lambda n: val_results[n]["roc_auc"])
    best_model = fitted_models[best_name]
    log(f"Selected best model by validation ROC-AUC: {best_name}")

    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred)),
        "recall": float(recall_score(y_test, test_pred)),
        "f1": float(f1_score(y_test, test_pred)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
    }
    cm = confusion_matrix(y_test, test_pred).tolist()
    log(f"TEST metrics for {best_name}: {test_metrics}")
    log(f"Confusion matrix: {cm}")

    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(FEATURE_NAMES, [float(x) for x in best_model.feature_importances_]))
    elif hasattr(best_model, "coef_"):
        feature_importance = dict(zip(FEATURE_NAMES, [float(x) for x in best_model.coef_[0]]))

    model_path = os.path.join(MODELS_DIR, "ats_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    metadata = {
        "model_version": "1.0.0",
        "dataset_name": "florex/resume_corpus (resumes) — see DATASET.md",
        "dataset_source": "https://github.com/florex/resume_corpus",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "number_of_training_samples": int(X_train.shape[0]),
        "number_of_validation_samples": int(X_val.shape[0]),
        "number_of_test_samples": int(X_test.shape[0]),
        "total_pairs_generated": len(pairs),
        "usable_pairs_after_parsing": int(X_train.shape[0] + X_val.shape[0] + X_test.shape[0]),
        "occupation_classes": OCCUPATION_CLASSES,
        "features_used": FEATURE_NAMES,
        "algorithm": best_name,
        "candidate_models_compared": list(candidates.keys()),
        "validation_metrics_by_model": val_results,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["not_compatible(0)", "compatible(1)"],
        "feature_importance": feature_importance,
        "feature_medians_train": medians,
        "random_seed": SEED,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "tfidf_vocab_size": len(tfidf_vectorizer.vocabulary_),
        "training_wall_time_seconds": round(time.time() - t_start, 1),
        "label_definition": (
            "1 = resume occupation_class == job-description occupation_class "
            "(a real, dataset-derived proxy for compatibility — see DATASET.md); "
            "0 = different occupation_class."
        ),
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    metrics_report_path = os.path.join(REPORTS_DIR, "model_metrics.json")
    with open(metrics_report_path, "w") as f:
        json.dump({"validation": val_results, "test": test_metrics, "confusion_matrix": cm}, f, indent=2)

    _save_confusion_matrix_plot(cm, os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    _save_feature_importance_plot(feature_importance, os.path.join(REPORTS_DIR, "feature_importance.png"))
    _save_evaluation_report_md(metadata, os.path.join(REPORTS_DIR, "evaluation_report.md"))

    log(f"Saved model to {model_path}")
    log(f"Saved vectorizer to {vectorizer_path}")
    log(f"Saved metadata to {metadata_path}")
    log(f"Total wall time (this stage): {time.time()-t_start:.1f}s")
    return metadata


def _save_confusion_matrix_plot(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: 0", "Pred: 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True: 0", "True: 1"])
    ax.set_title("Confusion Matrix (held-out test set)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_feature_importance_plot(importance: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not importance:
        return
    items = sorted(importance.items(), key=lambda x: abs(x[1]))
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, vals, color="#4C72B0")
    ax.set_title("Feature Importance (best model)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_evaluation_report_md(metadata, path):
    lines = [
        "# Evaluation Report",
        "",
        f"**Model:** {metadata['algorithm']}",
        f"**Trained:** {metadata['training_date']}",
        f"**Training samples:** {metadata['number_of_training_samples']}",
        f"**Validation samples:** {metadata['number_of_validation_samples']}",
        f"**Test samples:** {metadata['number_of_test_samples']}",
        "",
        "## Validation metrics (all candidate models)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in metadata["validation_metrics_by_model"].items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['roc_auc']:.4f} |"
        )
    lines += [
        "",
        f"**Selected model (highest validation ROC-AUC): {metadata['algorithm']}**",
        "",
        "## Held-out test set metrics (evaluated once, after model selection)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in metadata["test_metrics"].items():
        lines.append(f"| {k} | {v:.4f} |")
    lines += [
        "",
        f"Confusion matrix: {metadata['confusion_matrix']} "
        f"(rows=true, cols=predicted, order={metadata['confusion_matrix_labels']})",
        "",
        "## Feature importance",
        "",
    ]
    for k, v in sorted(metadata["feature_importance"].items(), key=lambda x: -abs(x[1])):
        lines.append(f"- `{k}`: {v:.4f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
