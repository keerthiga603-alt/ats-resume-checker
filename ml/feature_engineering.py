"""
Generates the numeric feature vector used by the trained ATS model, plus
the human-readable "supporting metrics" shown in the ATS report.
Every feature here is computed from the actual parsed resume + JD — none
are placeholders.
"""
import re
from ml.experience_education import education_alignment_score

FEATURE_NAMES = [
    "skill_overlap",
    "required_skill_coverage",
    "preferred_skill_coverage",
    "keyword_overlap",
    "experience_alignment",
    "education_alignment",
    "job_title_similarity",
    "resume_length",
    "relevant_section_coverage",
    "technical_keyword_density",
    "tfidf_cosine_similarity",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9+#\.]{2,}", text.lower()))


def compute_features(parsed_resume: dict, parsed_jd: dict, tfidf_vectorizer=None) -> dict:
    resume_skills = set(parsed_resume["skills"])
    required_skills = set(parsed_jd["required_skills"])
    preferred_skills = set(parsed_jd["preferred_skills"])
    all_jd_skills = required_skills | preferred_skills

    skill_overlap = _jaccard(resume_skills, all_jd_skills)

    required_skill_coverage = (
        len(resume_skills & required_skills) / len(required_skills) if required_skills else 0.0
    )
    preferred_skill_coverage = (
        len(resume_skills & preferred_skills) / len(preferred_skills) if preferred_skills else 0.0
    )

    resume_tokens = _tokenize(parsed_resume["raw_text"])
    jd_tokens = _tokenize(parsed_jd["raw_text"])
    keyword_overlap = _jaccard(resume_tokens, jd_tokens)

    # experience alignment: actual years vs required years (if both known)
    exp = parsed_resume["experience"]
    required_years = parsed_jd.get("required_years_experience")
    if exp["total_years_experience"] is not None and required_years:
        experience_alignment = min(1.0, exp["total_years_experience"] / required_years)
    elif exp["total_years_experience"] is not None and not required_years:
        # JD didn't specify years; can't score alignment against nothing
        experience_alignment = None
    else:
        experience_alignment = None

    education_alignment = education_alignment_score(
        parsed_resume["education"],
        parsed_jd.get("required_degree"),
        parsed_jd.get("required_fields", []),
    )

    # job title similarity: token overlap between JD title and resume text (crude but real)
    job_title_similarity = None
    if parsed_jd.get("job_title"):
        title_tokens = _tokenize(parsed_jd["job_title"])
        job_title_similarity = _jaccard(title_tokens, resume_tokens)

    resume_length = parsed_resume["resume_length_words"]

    detected_sections = set(parsed_resume["sections"].keys()) - {"full_text", "preamble"}
    expected_sections = {"skills", "experience", "education"}
    relevant_section_coverage = len(detected_sections & expected_sections) / len(expected_sections)

    technical_keyword_density = (
        len(parsed_resume["skills"]) / resume_length if resume_length else 0.0
    )

    tfidf_cosine_similarity = None
    if tfidf_vectorizer is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        vecs = tfidf_vectorizer.transform([parsed_resume["raw_text"], parsed_jd["raw_text"]])
        tfidf_cosine_similarity = float(cosine_similarity(vecs[0], vecs[1])[0][0])

    features = {
        "skill_overlap": skill_overlap,
        "required_skill_coverage": required_skill_coverage,
        "preferred_skill_coverage": preferred_skill_coverage,
        "keyword_overlap": keyword_overlap,
        "experience_alignment": experience_alignment,
        "education_alignment": education_alignment,
        "job_title_similarity": job_title_similarity,
        "resume_length": resume_length,
        "relevant_section_coverage": relevant_section_coverage,
        "technical_keyword_density": technical_keyword_density,
        "tfidf_cosine_similarity": tfidf_cosine_similarity,
    }
    return features


def features_to_model_vector(features: dict, medians: dict) -> list:
    """
    Converts feature dict to an ordered numeric vector for the model.
    Missing/None values are imputed with training-set medians (computed
    once during training and stored in model_metadata.json) — not zero,
    not a guess made at inference time.
    """
    vector = []
    for name in FEATURE_NAMES:
        val = features.get(name)
        if val is None:
            val = medians.get(name, 0.0)
        vector.append(val)
    return vector
