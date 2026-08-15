"""
Converts raw (resume_text, jd_text, label) pairs into the same feature
representation used at inference time (ml/feature_engineering.py), so
there is exactly one feature-computation code path shared by training
and inference — avoiding train/serve skew.
"""
import sys
import time
from ml.resume_parser import parse_resume, InvalidResume
from ml.jd_parser import parse_job_description, InvalidJobDescription
from ml.feature_engineering import compute_features, FEATURE_NAMES


def pairs_to_feature_rows(pairs, progress_every=2000):
    """
    Returns (X_rows: list[dict], y: list[int], meta: list[dict])
    Skips pairs where parsing fails (e.g. text too short after cleaning) —
    logs how many were skipped rather than silently padding with fake rows.
    """
    X_rows, y, meta = [], [], []
    skipped = 0
    t0 = time.time()
    for i, pair in enumerate(pairs):
        try:
            parsed_resume = parse_resume(pair["resume_text"])
        except InvalidResume:
            skipped += 1
            continue

        # Treat jd_text as a raw job description for parsing purposes.
        try:
            parsed_jd = parse_job_description(pair["jd_text"])
        except InvalidJobDescription:
            skipped += 1
            continue

        features = compute_features(parsed_resume, parsed_jd)
        X_rows.append(features)
        y.append(pair["label"])
        meta.append({"resume_class": pair["resume_class"], "jd_class": pair["jd_class"]})

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  processed {i+1}/{len(pairs)} pairs ({elapsed:.1f}s elapsed)", file=sys.stderr)

    print(f"  done. {len(X_rows)} usable rows, {skipped} skipped.", file=sys.stderr)
    return X_rows, y, meta


def compute_medians(X_rows: list) -> dict:
    """Computes per-feature medians over rows where the feature is not None.
    Used to impute missing values consistently at inference time."""
    medians = {}
    for name in FEATURE_NAMES:
        vals = [row[name] for row in X_rows if row.get(name) is not None]
        if vals:
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            mid = n // 2
            medians[name] = (vals_sorted[mid] if n % 2 == 1
                              else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2)
        else:
            medians[name] = 0.0
    return medians


def rows_to_matrix(X_rows: list, medians: dict):
    from ml.feature_engineering import features_to_model_vector
    return [features_to_model_vector(row, medians) for row in X_rows]
