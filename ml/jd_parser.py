"""
Job description parsing: splits into sections, extracts required vs preferred
skills using the same taxonomy as resumes, extracts degree requirement.
"""
import re
from ml.section_extraction import extract_jd_sections
from ml.skill_extraction import extract_skills
from ml.experience_education import DEGREE_PATTERNS, DEGREE_RANK, FIELD_KEYWORDS

MIN_JD_LENGTH = 80  # characters; below this we consider the JD too short for reliable matching


class InvalidJobDescription(Exception):
    pass


def validate_job_description(text: str):
    text = (text or "").strip()
    if len(text) < MIN_JD_LENGTH:
        raise InvalidJobDescription(
            "The job description is too short for reliable matching. Please provide a real job description."
        )
    # crude sanity check: does it contain at least a few real words, not just repeated junk
    words = re.findall(r"[A-Za-z]{3,}", text)
    distinct_words = set(w.lower() for w in words)
    if len(distinct_words) < 15:
        raise InvalidJobDescription("Please provide a real job description.")
    return text


def parse_job_description(text: str) -> dict:
    text = validate_job_description(text)
    sections = extract_jd_sections(text)

    requirements_text = sections.get("requirements", "")
    preferred_text = sections.get("preferred", "")
    responsibilities_text = sections.get("responsibilities", "")
    full_text = sections.get("full_text", text)

    # If no sections were detected at all, treat the whole text as "requirements"
    # for skill extraction purposes (still 100% derived from actual text).
    required_search_text = requirements_text or full_text or text
    preferred_search_text = preferred_text

    required_skills_result = extract_skills(required_search_text + " " + responsibilities_text)
    preferred_skills_result = extract_skills(preferred_search_text) if preferred_search_text else {"skills": []}

    # If document had no explicit "preferred" section, skills mentioned outside
    # requirements/responsibilities aren't separately tracked as preferred —
    # we do not fabricate a preferred list.
    required_skills = set(required_skills_result["skills"])
    preferred_skills = set(preferred_skills_result["skills"]) - required_skills

    text_lower = text.lower()
    required_degree = None
    for degree, patterns in DEGREE_PATTERNS.items():
        for p in patterns:
            if re.search(p, text_lower):
                required_degree = degree
                break
        if required_degree:
            break

    required_fields = [f for f in FIELD_KEYWORDS if f in text_lower]

    # crude experience requirement extraction: "X+ years" or "X-Y years"
    exp_match = re.search(r"(\d+)\s*\+?\s*(?:-|to)?\s*(\d+)?\s*years?", text_lower)
    required_years = None
    if exp_match:
        try:
            required_years = int(exp_match.group(1))
        except ValueError:
            required_years = None

    job_title = None
    title_match = re.match(r"^\s*([A-Za-z0-9 /\-&]{3,60})\n", text)
    if title_match:
        job_title = title_match.group(1).strip()

    return {
        "raw_text": text,
        "sections": sections,
        "job_title": job_title,
        "required_skills": sorted(required_skills),
        "preferred_skills": sorted(preferred_skills),
        "required_degree": required_degree,
        "required_fields": required_fields,
        "required_years_experience": required_years,
    }
