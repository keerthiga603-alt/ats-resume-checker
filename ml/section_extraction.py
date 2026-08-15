"""
Resume / job-description section detection.
Real regex-based section splitting — no invented content.
"""
import re

RESUME_SECTION_HEADERS = {
    "summary": ["summary", "professional summary", "profile", "objective"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "areas of expertise"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history"],
    "education": ["education", "academic background", "educational qualification"],
    "projects": ["projects", "personal projects", "academic projects"],
    "certifications": ["certifications", "certificates", "licenses"],
    "achievements": ["achievements", "awards", "honors"],
    "internships": ["internships", "internship experience"],
}

JD_SECTION_HEADERS = {
    "responsibilities": ["responsibilities", "duties", "what you'll do", "role", "job summary",
                          "essential functions", "essential job functions"],
    "requirements": ["requirements", "qualifications", "required skills", "what you'll need",
                      "minimum qualifications", "basic qualifications"],
    "preferred": ["preferred qualifications", "preferred skills", "nice to have", "bonus points"],
    "education": ["education", "educational requirements"],
    "benefits": ["benefits", "perks", "what we offer"],
}


def _build_header_pattern(header_map):
    all_headers = sorted(
        {h for variants in header_map.values() for h in variants},
        key=len, reverse=True
    )
    escaped = [re.escape(h) for h in all_headers]
    pattern = r"(?im)^\s*(" + "|".join(escaped) + r")\s*:?\s*$"
    return re.compile(pattern)


_RESUME_HEADER_RE = _build_header_pattern(RESUME_SECTION_HEADERS)
_JD_HEADER_RE = _build_header_pattern(JD_SECTION_HEADERS)


def _canonical_section(raw_header: str, header_map: dict) -> str:
    raw_lower = raw_header.strip().lower()
    for canonical, variants in header_map.items():
        if raw_lower in variants:
            return canonical
    return raw_lower


def split_into_sections(text: str, header_map: dict, header_re: re.Pattern) -> dict:
    """
    Splits text on lines that look like section headers.
    Returns {section_name: text}. If no headers are found, returns
    {"full_text": text} — we do NOT invent section boundaries that aren't there.
    """
    matches = list(header_re.finditer(text))
    if not matches:
        return {"full_text": text.strip()}

    sections = {}
    for i, m in enumerate(matches):
        section_name = _canonical_section(m.group(1), header_map)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            if section_name in sections:
                sections[section_name] += "\n" + content
            else:
                sections[section_name] = content

    # preamble before the first detected header (often contains name/summary/contact)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["preamble"] = preamble

    return sections


def extract_resume_sections(text: str) -> dict:
    return split_into_sections(text, RESUME_SECTION_HEADERS, _RESUME_HEADER_RE)


def extract_jd_sections(text: str) -> dict:
    return split_into_sections(text, JD_SECTION_HEADERS, _JD_HEADER_RE)
