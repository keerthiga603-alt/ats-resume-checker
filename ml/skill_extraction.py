"""
Skill extraction against the curated taxonomy in models/skill_taxonomy.json.
Matches whole words/phrases only (word-boundary regex) to avoid false positives
like matching "r" inside "ار" or "go" inside "google".
"""
import json
import re
import os

_TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "skill_taxonomy.json")

_ALIAS_TO_CANONICAL = {}
_CANONICAL_TO_CATEGORY = {}


def _load_taxonomy():
    global _ALIAS_TO_CANONICAL, _CANONICAL_TO_CATEGORY
    if _ALIAS_TO_CANONICAL:
        return
    with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    for category, skills in taxonomy.items():
        if category.startswith("_"):
            continue
        for canonical, aliases in skills.items():
            _CANONICAL_TO_CATEGORY[canonical] = category
            for alias in aliases:
                _ALIAS_TO_CANONICAL[alias.strip().lower()] = canonical


_SINGLE_WORD_ALIASES = None   # {alias: canonical} where alias is exactly one token
_MULTI_WORD_ALIASES = None    # [(alias_tuple, alias_str, canonical), ...] sorted longest-first
_TOKEN_RE = re.compile(r"[A-Za-z0-9+#\.]+")


def _prepare_matchers():
    """
    O(n) matching strategy: tokenize the text once, then look up each token
    (and short n-gram windows for multi-word aliases) against precomputed
    dicts/sets, instead of running a large alternation regex over the whole
    string. Produces identical matches to a naive per-alias scan, but avoids
    catastrophic per-character regex overhead from ~190 alternatives.
    """
    global _SINGLE_WORD_ALIASES, _MULTI_WORD_ALIASES
    if _SINGLE_WORD_ALIASES is not None:
        return
    _load_taxonomy()
    _SINGLE_WORD_ALIASES = {}
    multi = []
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        alias_norm = alias.strip().lower()
        parts = alias_norm.split()
        if len(parts) == 1:
            _SINGLE_WORD_ALIASES[parts[0]] = canonical
        else:
            multi.append((tuple(parts), alias_norm, canonical))
    multi.sort(key=lambda x: len(x[0]), reverse=True)
    _MULTI_WORD_ALIASES = multi
    _MAX_NGRAM = max((len(p[0]) for p in multi), default=1)
    globals()["_MAX_NGRAM"] = _MAX_NGRAM


def extract_skills(text: str) -> dict:
    """
    Returns {
      "skills": [canonical skill names found, deduplicated],
      "by_category": {category: [skills]},
      "raw_matches": {canonical: count of occurrences}
    }
    """
    _prepare_matchers()
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    n = len(tokens)
    found_canonical_counts = {}
    max_ngram = globals().get("_MAX_NGRAM", 1)

    # multi-word aliases first (checked via sliding window, longest first)
    covered = [False] * n
    for length in range(max_ngram, 1, -1):
        if n < length:
            continue
        for i in range(n - length + 1):
            if any(covered[i:i + length]):
                continue
            window = tuple(tokens[i:i + length])
            for alias_tuple, alias_str, canonical in _MULTI_WORD_ALIASES:
                if len(alias_tuple) == length and window == alias_tuple:
                    found_canonical_counts[canonical] = found_canonical_counts.get(canonical, 0) + 1
                    for j in range(i, i + length):
                        covered[j] = True
                    break

    # single-word aliases
    for i, tok in enumerate(tokens):
        if covered[i]:
            continue
        canonical = _SINGLE_WORD_ALIASES.get(tok)
        if canonical:
            found_canonical_counts[canonical] = found_canonical_counts.get(canonical, 0) + 1

    by_category = {}
    for canonical in found_canonical_counts:
        cat = _CANONICAL_TO_CATEGORY.get(canonical, "Other")
        by_category.setdefault(cat, []).append(canonical)

    return {
        "skills": sorted(found_canonical_counts.keys()),
        "by_category": by_category,
        "raw_matches": found_canonical_counts,
    }


def compare_skills(resume_skills: set, jd_required_skills: set, jd_preferred_skills: set = None):
    jd_preferred_skills = jd_preferred_skills or set()
    matched = resume_skills & (jd_required_skills | jd_preferred_skills)
    missing_required = jd_required_skills - resume_skills
    missing_preferred = jd_preferred_skills - resume_skills
    return {
        "matched_skills": sorted(matched),
        "missing_required_skills": sorted(missing_required),
        "missing_preferred_skills": sorted(missing_preferred),
        "required_skill_coverage": (
            len(jd_required_skills & resume_skills) / len(jd_required_skills)
            if jd_required_skills else None
        ),
        "preferred_skill_coverage": (
            len(jd_preferred_skills & resume_skills) / len(jd_preferred_skills)
            if jd_preferred_skills else None
        ),
    }
