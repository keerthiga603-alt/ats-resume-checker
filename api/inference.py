"""
Inference layer: loads the trained model/vectorizer/medians ONCE at process
start, then scores resume/JD pairs using the exact same parsing + feature
code (ml/resume_parser.py, ml/jd_parser.py, ml/feature_engineering.py) used
during training, so there is no train/serve skew.
"""
import os
import json
import pickle

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")

from ml.resume_parser import parse_resume, InvalidResume
from ml.jd_parser import parse_job_description, InvalidJobDescription
from ml.feature_engineering import compute_features, features_to_model_vector, FEATURE_NAMES

FRIENDLY_NAMES = {
    "skill_overlap": "Overall skill overlap",
    "required_skill_coverage": "Required skills covered",
    "preferred_skill_coverage": "Preferred skills covered",
    "keyword_overlap": "Keyword overlap",
    "experience_alignment": "Experience alignment",
    "education_alignment": "Education alignment",
    "job_title_similarity": "Job title similarity",
    "resume_length": "Resume length (words)",
    "relevant_section_coverage": "Resume section coverage",
    "technical_keyword_density": "Technical keyword density",
    "tfidf_cosine_similarity": "Overall text similarity",
}


class ATSPredictor:
    """Loads model artifacts once; call .predict() as many times as needed."""

    def __init__(self, models_dir=MODELS_DIR):
        model_path = os.path.join(models_dir, "ats_model.pkl")
        vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
        metadata_path = os.path.join(models_dir, "model_metadata.json")

        for p in (model_path, vectorizer_path, metadata_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found. Train the model first: python run_all_training.py"
                )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.medians = self.metadata["feature_medians_train"]

    def predict(self, resume_text: str, jd_text: str) -> dict:
        """
        Returns a dict describing compatibility. Raises InvalidResume /
        InvalidJobDescription (both ValueError-derived via the parser
        modules) if either input can't be reliably parsed -- callers
        (the API layer) turn these into 4xx responses rather than 500s,
        since a too-short resume/JD is a user-input problem, not a bug.
        """
        parsed_resume = parse_resume(resume_text)
        parsed_jd = parse_job_description(jd_text)

        features = compute_features(parsed_resume, parsed_jd, tfidf_vectorizer=self.vectorizer)
        vector = features_to_model_vector(features, self.medians)

        proba = float(self.model.predict_proba([vector])[0][1])
        label = "compatible" if proba >= 0.5 else "not_compatible"

        resume_skills = set(parsed_resume["skills"])
        required_skills = set(parsed_jd["required_skills"])
        preferred_skills = set(parsed_jd["preferred_skills"])

        feature_breakdown = []
        for name in FEATURE_NAMES:
            raw_val = features.get(name)
            was_imputed = raw_val is None
            value = raw_val if raw_val is not None else self.medians.get(name, 0.0)
            feature_breakdown.append({
                "feature": name,
                "label": FRIENDLY_NAMES.get(name, name),
                "value": round(float(value), 4),
                "imputed_with_training_median": was_imputed,
            })

        return {
            "compatibility_score": round(proba * 100, 1),
            "label": label,
            "model_confidence": round(max(proba, 1 - proba) * 100, 1),
            "matched_skills": sorted(resume_skills & (required_skills | preferred_skills)),
            "missing_required_skills": sorted(required_skills - resume_skills),
            "missing_preferred_skills": sorted(preferred_skills - resume_skills),
            "feature_breakdown": feature_breakdown,
            "resume_section_coverage": sorted(
                set(parsed_resume["sections"].keys()) - {"full_text", "preamble"}
            ),
            "jd_required_degree": parsed_jd.get("required_degree"),
            "jd_required_years_experience": parsed_jd.get("required_years_experience"),
            "model_version": self.metadata.get("model_version"),
            "algorithm": self.metadata.get("algorithm"),
        }


_predictor = None


def get_predictor() -> ATSPredictor:
    """Lazy singleton so the model is loaded once per process, not per request."""
    global _predictor
    if _predictor is None:
        _predictor = ATSPredictor()
    return _predictor
