from ml.section_extraction import extract_resume_sections
from ml.skill_extraction import extract_skills
from ml.experience_education import extract_experience, extract_education

MIN_RESUME_LENGTH = 100


class InvalidResume(Exception):
    pass


def parse_resume(text: str) -> dict:
    text = (text or "").strip()
    if len(text) < MIN_RESUME_LENGTH:
        raise InvalidResume(
            "Unable to extract enough text from this resume for reliable analysis."
        )

    sections = extract_resume_sections(text)
    skills_result = extract_skills(text)
    experience_result = extract_experience(text)
    education_result = extract_education(text)

    return {
        "raw_text": text,
        "sections": sections,
        "skills": skills_result["skills"],
        "skills_by_category": skills_result["by_category"],
        "skill_match_counts": skills_result["raw_matches"],
        "experience": experience_result,
        "education": education_result,
        "resume_length_chars": len(text),
        "resume_length_words": len(text.split()),
    }
