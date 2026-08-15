"""
Loads the two real source datasets (see DATASET.md) and normalizes the
resume_corpus free-text job titles into the 10 canonical occupation
categories used by the source publication (Jiechieu & Tsopze, 2020).

Every mapping rule below is a documented substring match against the ACTUAL
raw label strings observed in the dataset (see the frequency table produced
during dataset inspection). Titles that don't match any rule are dropped —
we do not force an unclear title into a bucket.
"""
import os
import re
from ml.text_extraction import clean_html_fragments

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESUME_CORPUS_PATH = os.path.join(RAW_DIR, "resume_samples.txt")
JD_POOL_PATH = os.path.join(RAW_DIR, "data.csv")

# Ordered rules: (compiled regex over lowercased raw title, canonical category)
# Order matters: more specific rules first.
NORMALIZED_OCCUPATION_MAP = [
    (re.compile(r"security analyst|cyber\s*security|information security"), "Security_Analyst"),
    (re.compile(r"database administrator|dba\b|oracle dba"), "Database_Administrator"),
    (re.compile(r"network administrator|network engineer|systems? engineer"), "Network_Administrator"),
    (re.compile(r"systems? administrator|system admin"), "Systems_Administrator"),
    (re.compile(r"project manager|it manager|program manager"), "Project_Manager"),
    (re.compile(r"front[\s\-]?end (web )?developer|ui developer"), "Front_End_Developer"),
    (re.compile(r"\bjava\b.*developer|java developer"), "Java_Developer"),
    (re.compile(r"\bpython\b.*developer|python developer"), "Python_Developer"),
    (re.compile(r"web developer"), "Web_Developer"),
    (re.compile(r"software (developer|engineer)|full stack"), "Software_Developer"),
]

OCCUPATION_CLASSES = sorted({cat for _, cat in NORMALIZED_OCCUPATION_MAP})


def normalize_occupation(raw_title: str):
    title_lower = raw_title.strip().lower()
    for pattern, category in NORMALIZED_OCCUPATION_MAP:
        if pattern.search(title_lower):
            return category
    return None


def load_resume_corpus(min_text_len: int = 200, max_records: int = None):
    """
    Yields dicts: {id, raw_title, occupation_class, text}
    Only records whose title normalizes to one of the 10 classes AND whose
    cleaned text is long enough are yielded.
    """
    if not os.path.exists(RESUME_CORPUS_PATH):
        raise FileNotFoundError(
            f"{RESUME_CORPUS_PATH} not found. Run data download step first (see DATASET.md)."
        )
    count = 0
    with open(RESUME_CORPUS_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split(":::")
            if len(parts) < 3:
                continue
            rid, labels_raw, text_raw = parts[0], parts[1], parts[2]
            labels = labels_raw.split(";")
            if not labels:
                continue
            raw_title = labels[0].strip()
            occupation_class = normalize_occupation(raw_title)
            if occupation_class is None:
                continue
            clean_text = clean_html_fragments(text_raw)
            if len(clean_text) < min_text_len:
                continue
            yield {
                "id": rid.strip(),
                "raw_title": raw_title,
                "occupation_class": occupation_class,
                "text": clean_text,
            }
            count += 1
            if max_records and count >= max_records:
                return


def load_jd_pool():
    """
    Loads the real job postings (Glassdoor scrape, see DATASET.md) and
    normalizes their `position` field into the same occupation classes.
    NOTE: kept for transparency/documentation only. As explained in
    DATASET.md, only 5/157 postings normalize to our 10 occupation
    classes (this dataset is dominated by non-technical internships),
    so it is NOT used to build training pairs. See build_resume_jd_pairs().
    """
    import pandas as pd
    if not os.path.exists(JD_POOL_PATH):
        raise FileNotFoundError(f"{JD_POOL_PATH} not found. Run data download step first.")
    df = pd.read_csv(JD_POOL_PATH)
    records = []
    for _, row in df.iterrows():
        position = str(row.get("position", ""))
        occupation_class = normalize_occupation(position)
        if occupation_class is None:
            continue
        jd_text = str(row.get("Job Description", "")).strip()
        if len(jd_text) < 80:
            continue
        records.append({
            "company": row.get("company"),
            "position": position,
            "occupation_class": occupation_class,
            "jd_text": jd_text,
        })
    return records


def build_resume_jd_pairs(seed: int = 42, jd_fraction: float = 0.3, max_per_class: int = None):
    """
    Builds the supervised (resume_text, jd_text, label, resume_class, jd_class) pairs
    used for training, entirely from resume_corpus (see DATASET.md for the
    documented rationale for not using data.csv here).

    For each occupation class, a random `jd_fraction` of that class's resumes
    are set aside as "JD-side" documents (real language describing that
    occupation) and the remaining resumes are the "resume-side" documents.
    This avoids ever pairing a document with itself while keeping both
    sides of every pair grounded in real, labeled text.

    label = 1 if resume_class == jd_class (positive/compatible pair)
    label = 0 otherwise (negative/incompatible pair)

    Negative pairs are sampled 1:1 with positives per class to keep the
    label distribution balanced (a real, documented sampling choice — not
    a fabricated label).
    """
    import random
    rng = random.Random(seed)

    by_class = {}
    for rec in load_resume_corpus(max_records=max_per_class * 10 if max_per_class else None):
        by_class.setdefault(rec["occupation_class"], []).append(rec)

    resume_side = {}
    jd_side = {}
    for cls, recs in by_class.items():
        rng.shuffle(recs)
        split_idx = int(len(recs) * jd_fraction)
        jd_side[cls] = recs[:split_idx]
        resume_side[cls] = recs[split_idx:] if max_per_class is None else recs[split_idx:split_idx + max_per_class]

    classes = sorted(by_class.keys())
    pairs = []
    for cls in classes:
        resumes = resume_side[cls]
        jds_same = jd_side[cls]
        if not resumes or not jds_same:
            continue
        # positive pairs: resume matched with a same-class JD-side text
        for r in resumes:
            jd_rec = rng.choice(jds_same)
            pairs.append({
                "resume_text": r["text"],
                "jd_text": jd_rec["text"],
                "resume_class": cls,
                "jd_class": cls,
                "label": 1,
            })
        # negative pairs: resume matched with a different-class JD-side text
        other_classes = [c for c in classes if c != cls and jd_side.get(c)]
        for r in resumes:
            if not other_classes:
                continue
            neg_cls = rng.choice(other_classes)
            jd_rec = rng.choice(jd_side[neg_cls])
            pairs.append({
                "resume_text": r["text"],
                "jd_text": jd_rec["text"],
                "resume_class": cls,
                "jd_class": neg_cls,
                "label": 0,
            })

    rng.shuffle(pairs)
    return pairs
