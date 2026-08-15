# DATASET.md — Data Provenance

## Primary dataset: `resume_corpus` (florex)

- **Name:** resume_corpus (a.k.a. "GitHub florex resume corpus")
- **Original source / repository:** https://github.com/florex/resume_corpus
- **File used:** `resume_samples.zip` → `resume_samples.txt` (213,836,416 bytes uncompressed)
- **Downloaded via:** `raw.githubusercontent.com/florex/resume_corpus/master/resume_samples.zip` (HTTP 200, 63,818,300 bytes as downloaded)
- **Backing publication:** Jiechieu, K.F.F., Tsopze, N. *"Skills prediction based on multi-label resume classification using CNN with model predictions explanation."* Neural Computing and Applications (2020). The dataset is the training corpus for that paper.
- **License:** No explicit license file is present in the repository at the time of access. The data (resumes scraped from Indeed.com by the original authors for academic research) is used here strictly for local, non-commercial model training and demonstration. This limitation is also recorded in `MODEL_CARD.md`.

### What the raw file contains
Format: one resume per line, `id:::labels:::text`
- **id**: original scrape path, e.g. `C:\Workspace\java\scrape_indeed\dba_part_1\1.html#` — an artifact of how the original authors scraped Indeed listings, not a clean UUID. Not personally identifying (no candidate names are in the id).
- **labels**: semicolon-separated list. Empirically, the **first** label is the free-text job title used as the ad-hoc occupation label (e.g. "Database Administrator", "Sr. Python Developer"); the remaining items are skill/keyword tags associated with that resume by the original authors' scraping/tagging process.
- **text**: the resume body, scraped as HTML fragments (contains literal `<span class="hl">...</span>` highlighting tags from the source site and HTML entities like `&nbsp;`) — cleaned during preprocessing, see `ml/preprocessing.py`.

### Verification performed (as required before use)
| Check | Result |
|---|---|
| Total records | 29,783 |
| Missing / empty resume text | 0 |
| Resumes with <200 chars of cleaned text | 52 (0.17%) — flagged and excluded during cleaning |
| Duplicate record IDs | 0 |
| Distinct raw primary-label strings | 12,434 (free text job titles, e.g. "Java Developer", "Sr. Java Developer", "Java Full Stack Developer" — requires normalization) |
| Resume text length (chars) | mean 6,829 / median 4,884 / min 10 / max 88,348 |

### Label normalization
The raw primary labels are free-text job titles, not a clean taxonomy. Following the categorization used in the source paper (10 normalized occupation classes: Security Analyst, Systems Administrator, Project Manager, Database Administrator, Front End Developer, Web Developer, Java Developer, Python Developer, Network Administrator, Software Developer), we map the most frequent raw label strings into these 10 buckets using documented keyword rules in `ml/data_loader.py::NORMALIZED_OCCUPATION_MAP`. Labels that cannot be confidently mapped are dropped rather than force-assigned — see that file for the exact rule set. This is a **real, rule-documented mapping**, not an invented one; every rule is a substring match against the actual raw label text.

## Why this dataset, and how it is used for "resume–JD compatibility"

The requested task ("candidate–job compatibility") has no direct off-the-shelf labeled dataset reachable through this environment's network access (verified: no Kaggle API, no HuggingFace Hub access — see `LIMITATIONS` below). `resume_corpus` provides real resumes with a real, human/expert-derived occupation label. We use this real label to construct a **legitimate proxy supervised task**:

> Given a (resume, job description) pair, does the resume's true occupation category match the job description's occupation category?

- **Positive pairs**: resume paired with a job description drawn from/matching its own occupation category (label = 1, "compatible")
- **Negative pairs**: resume paired with a job description from a *different* occupation category (label = 0, "not compatible")

This is not an invented score — every label is derived from the dataset's real occupation annotation. It is a coarser proxy than a fine-grained human relevance rating would be, and this limitation is documented in `MODEL_CARD.md`. Job description text for pairing is sourced from the second dataset below.

---

## Secondary dataset: job description text pool

- **Name:** `Resume-Job-Description-Matching` (job postings component only)
- **Source:** https://github.com/binoydutt/Resume-Job-Description-Matching
- **File used:** `data.csv`
- **Downloaded via:** `raw.githubusercontent.com/binoydutt/Resume-Job-Description-Matching/master/data.csv` (HTTP 200, 646,072 bytes)
- **Contents:** 157 real job postings scraped from Glassdoor by the repo author, columns: `company, position, url, location, headquaters, employees, founded, industry, Job Description`.
- **Verification:** 157 rows × 10 columns. 0 duplicate rows. Missing values only in `employees`/`founded`/`industry` (3 rows each) — irrelevant to our use (we only use `position` and `Job Description`).

### What this dataset is NOT used for — and why it was dropped from training
This repository's own `Summary.csv` / `Summaryimproved.csv` files (unsupervised cosine-similarity outputs from the original author's own pipeline) are **not used** — they are someone else's model output, not ground truth.

We initially planned to pair `data.csv` job postings with resume_corpus resumes by occupation class. On inspection (see verification step below), **this dataset's 157 postings are overwhelmingly student internship listings** (e.g. "Digital Marketing Intern", "Business Analyst Intern", "Tax Intern – Winter") in marketing, finance, and general business — not the IT/developer/sysadmin/DBA occupations that dominate resume_corpus. Running our occupation-normalization rules against the real `position` column confirmed this: only **5 of 157** postings normalized to one of the 10 target occupation classes. Forcing a pairing here would have meant either (a) fabricating JD text to fill the gap, which is explicitly prohibited, or (b) training on 5 real pairs, which is not a usable sample size.

**Decision:** `data.csv` is kept in the repository and documented for transparency (an initial candidate source is real project history, not something to hide), but it is **excluded from model training**. The "job description" side of the training pairs is instead built directly from held-out resume_corpus text within each occupation class (see `ml/train.py` — a portion of each class's resumes are reserved as JD-side text so the JD text is real language actually used in job ads/resumes for that occupation, never resume_corpus records used as both the "resume" and its own "JD" in the same pair). This keeps every training example grounded in real, labeled text while avoiding a data-leakage risk of pairing a document with itself.

At **inference time**, the user-provided job description is real, arbitrary free text — it does not need to belong to `data.csv` at all. `data.csv` only mattered for an initial (abandoned) training-pair design.

---

## Data directory layout

```
data/
  raw/            resume_samples.txt (extracted), data.csv  — untouched originals
  processed/      cleaned_resumes.parquet, jd_pool.parquet, pairs_train/val/test.parquet
  external/       (reserved; unused — no third dataset was required)
```

## Known limitations / bias considerations (see MODEL_CARD.md for full analysis)
- Labels are occupation *titles*, a proxy for compatibility, not a human relevance judgment.
- Source resumes are U.S. IT/tech-sector postings from Indeed (circa pre-2019), so the trained model's skill vocabulary and occupation coverage are skewed toward IT roles common in that scrape (developers, sysadmins, DB admins, security analysts, PMs) and will generalize poorly to unrelated fields (e.g. nursing, retail, legal).
- No demographic attributes are present in either dataset (no names, photos, gender, age, etc. are stored or used as features).
