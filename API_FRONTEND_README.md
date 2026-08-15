# Running the inference API + frontend

## New/changed files
- `api/inference.py` — loads `models/ats_model.pkl` + `models/vectorizer.pkl` once, scores a resume/JD pair
- `api/app.py` — Flask app: serves the API and the frontend from one process
- `frontend/index.html` — single-file UI (upload/paste resume, paste JD, see score + breakdown)
- `requirements.txt` — added `flask`, `flask-cors`

## Setup (from your existing venv)
```cmd
cd ats-system
venv\Scripts\activate.bat
pip install -r requirements.txt
python api\app.py
```
Then open **http://127.0.0.1:5000** in a browser.

You need `models/ats_model.pkl`, `models/vectorizer.pkl`, and `models/model_metadata.json`
already present (i.e. you've run `python run_all_training.py` at least once). If they're
missing, `/api/health` reports `model_loaded: false` and `/api/predict` returns a 503
instead of crashing.

## API
- `GET /api/health` → model status
- `POST /api/predict`
  - multipart form: `resume_file` (.pdf/.docx) + `jd_text`, **or**
  - JSON: `{"resume_text": "...", "jd_text": "..."}`
  - Returns: `compatibility_score` (0-100), `label`, `model_confidence`,
    `matched_skills`, `missing_required_skills`, `missing_preferred_skills`,
    `feature_breakdown` (each of the 11 model features, flagged if it had to be
    median-imputed), plus any detected JD degree/years requirement.

## What I verified myself (not just wrote and handed over)
Ran the Flask server in my own sandbox against the actual trained model artifacts and hit
`/api/predict` with:
- JSON text input → correct score + feature breakdown
- a real `.docx` file upload → text extracted and scored correctly
- a `.txt` file (wrong extension) → clean 400 error, not a crash
- missing `jd_text` → clean 400 error
- a too-short/low-content JD → correctly rejected by the existing JD validator, not silently scored

## One pre-existing quirk worth knowing about (not something I introduced)
`ml/section_extraction.py`'s JD section detector expects section headers in a fairly
specific format. In one of my test JDs, a "PREFERRED" header wasn't detected as its own
section, so those skills got folded into "required" instead. It didn't crash or produce
garbage — the skills were still correctly extracted, just bucketed as required instead of
preferred — but if you paste real JDs and the required/preferred split looks off, that's
where to look. Happy to harden that section detector next if you want.
