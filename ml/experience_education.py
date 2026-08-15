"""
Real experience and education extraction from resume text.
Uses regex date-range detection; when dates can't be reliably parsed,
returns uncertain=True rather than inventing a number.
"""
import re
from datetime import datetime

MONTHS = ("jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
          "january|february|march|april|june|july|august|september|october|november|december")

# e.g. "Jan 2020 - Present", "2019 to 2022", "03/2018 – 06/2021"
DATE_RANGE_RE = re.compile(
    r"(?:(" + MONTHS + r")\.?\s+)?(\d{4})\s*(?:-|to|–|—)\s*"
    r"(?:(" + MONTHS + r")\.?\s+)?(\d{4}|present|current)",
    re.IGNORECASE,
)

DEGREE_PATTERNS = {
    "phd": [r"ph\.?d\.?", r"doctorate"],
    "master": [r"master'?s?", r"\bm\.?s\.?\b", r"\bm\.?tech\.?\b", r"\bmba\b", r"\bm\.?a\.?\b"],
    "bachelor": [r"bachelor'?s?", r"\bb\.?s\.?\b", r"\bb\.?tech\.?\b", r"\bb\.?a\.?\b", r"\bb\.?e\.?\b"],
    "associate": [r"associate'?s?\s+degree"],
    "diploma": [r"diploma"],
    "high school": [r"high school", r"\bged\b"],
}

DEGREE_RANK = {"phd": 5, "master": 4, "bachelor": 3, "associate": 2, "diploma": 1.5, "high school": 1}

FIELD_KEYWORDS = [
    "computer science", "information technology", "data science", "artificial intelligence",
    "software engineering", "electrical engineering", "electronics", "information systems",
    "computer engineering", "mathematics", "statistics", "business administration",
    "cybersecurity", "network engineering",
]


def extract_experience(text: str) -> dict:
    """
    Detects date ranges in the text and estimates total years of experience
    from the union of the ranges found. Returns uncertain=True if no
    reliable ranges are detected, rather than guessing.
    """
    current_year = datetime.now().year
    ranges = []
    for m in DATE_RANGE_RE.finditer(text):
        _, start_year, _, end_year_raw = m.groups()
        try:
            start_year = int(start_year)
        except (TypeError, ValueError):
            continue
        if end_year_raw and end_year_raw.lower() in ("present", "current"):
            end_year = current_year
        else:
            try:
                end_year = int(end_year_raw)
            except (TypeError, ValueError):
                continue
        if 1950 <= start_year <= current_year and start_year <= end_year <= current_year + 1:
            ranges.append((start_year, end_year))

    if not ranges:
        return {"total_years_experience": None, "uncertain": True, "date_ranges_found": []}

    # merge overlapping ranges to avoid double-counting concurrent roles
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    total_years = sum(end - start for start, end in merged)
    return {
        "total_years_experience": round(total_years, 1),
        "uncertain": False,
        "date_ranges_found": [f"{s}-{e}" for s, e in merged],
    }


def extract_education(text: str) -> dict:
    text_lower = text.lower()
    found_degrees = []
    for degree, patterns in DEGREE_PATTERNS.items():
        for p in patterns:
            if re.search(p, text_lower):
                found_degrees.append(degree)
                break

    highest_degree = None
    if found_degrees:
        highest_degree = max(found_degrees, key=lambda d: DEGREE_RANK[d])

    found_fields = [f for f in FIELD_KEYWORDS if f in text_lower]

    return {
        "degrees_found": found_degrees,
        "highest_degree": highest_degree,
        "highest_degree_rank": DEGREE_RANK.get(highest_degree) if highest_degree else None,
        "fields_found": found_fields,
    }


def education_alignment_score(resume_education: dict, jd_required_degree: str, jd_required_fields: list) -> float:
    """
    Returns a 0-1 alignment score based on ACTUAL extracted degree rank and field overlap.
    Does not simply return 1.0 because "a degree exists".
    """
    score_components = []

    if jd_required_degree and jd_required_degree in DEGREE_RANK:
        required_rank = DEGREE_RANK[jd_required_degree]
        resume_rank = resume_education.get("highest_degree_rank") or 0
        degree_score = min(1.0, resume_rank / required_rank) if required_rank else 0.0
        score_components.append(degree_score)

    if jd_required_fields:
        resume_fields = set(resume_education.get("fields_found", []))
        req_fields = set(jd_required_fields)
        field_score = len(resume_fields & req_fields) / len(req_fields) if req_fields else None
        if field_score is not None:
            score_components.append(field_score)

    if not score_components:
        return None
    return sum(score_components) / len(score_components)
