# ATS Resume Screening — Setup & Run (VS Code / local machine)

## Current state (honest status)

This is a **mid-build, real (not fake) system**. What exists right now:

- ✅ Real dataset sourced, verified, documented (`DATASET.md`)
- ✅ Full NLP pipeline: PDF/DOCX extraction, section detection, skill taxonomy + extraction,
  experience/education extraction, JD parsing (`ml/*.py`)
- ✅ Feature engineering (11 real features, `ml/feature_engineering.py`)
- ✅ Staged training script (`ml/train.py`) — **the training run itself has not finished**;
  it was being executed inside a sandboxed environment with a hard ~2-minute-per-command
  limit, so it's split into resumable stages that checkpoint to disk.
- ❌ No trained model yet (`models/ats_model.pkl` does not exist yet)
- ❌ No API / frontend yet

Running this locally in VS Code removes the timeout problem entirely — each stage below
will just run to completion in one shot instead of needing to be chunked.

## 1. Setup (Windows CMD)

You're on Windows CMD with Python 3.10 — here are the Windows-native equivalents.

**Unzip** (CMD has no `unzip`; either right-click → "Extract All" in File Explorer, or use PowerShell):
```powershell
powershell -Command "Expand-Archive -Path ats-system-code.zip -DestinationPath ."
```

**Create and activate a virtual environment** (CMD uses `activate.bat`, not `source`):
```cmd
cd ats-system
python -m venv venv
venv\Scripts\activate.bat
```
Your prompt should now show `(venv)` at the start of the line — if it doesn't, the venv isn't active and packages will install globally.

**Install requirements:**
```cmd
pip install -r requirements.txt
```
`requirements.txt` now pins `scikit-learn>=1.3,<1.8` so it works on Python 3.10 (scikit-learn 1.8 requires Python 3.11+, which is what caused the earlier error). If `pip install` still fails on `xgboost` or `shap`, run `pip install --upgrade pip` first, then retry.

## 1b. Setup (macOS/Linux, for reference)

```bash
unzip ats-system-code.zip
cd ats-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## 2. Get the data (not included in this zip — too large)

The zip you downloaded from Claude does **not** include the raw dataset (a 213MB text file)
or the large intermediate pickles, to keep it a reasonable download. Re-download the real
source data.

**Windows CMD** (Windows 10/11 ships `curl.exe` and `tar` by default, so this works without installing anything extra):
```cmd
mkdir data\raw
cd data\raw
curl -L -o resume_samples.zip "https://raw.githubusercontent.com/florex/resume_corpus/master/resume_samples.zip"
tar -xf resume_samples.zip
cd ..\..
```
If `curl` or `tar` aren't recognized, you're on an older Windows build — download the URL manually in a browser, save it as `data\raw\resume_samples.zip`, then right-click → "Extract All" into `data\raw\`.

**macOS/Linux:**
```bash
mkdir -p data/raw
cd data/raw
curl -L -o resume_samples.zip \
  "https://raw.githubusercontent.com/florex/resume_corpus/master/resume_samples.zip"
unzip resume_samples.zip
cd ../..
```

Verify: `resume_samples.txt` should be ~213,836,416 bytes, ~29,783 lines.
See `DATASET.md` for full provenance, license notes, and the (rejected) second
dataset candidate and why it wasn't used.

## 3. Run the training pipeline

The pipeline is staged internally (so it could be checkpointed to disk under a
sandboxed timeout), but on your own machine you can just run it all in one go:

```cmd
python run_all_training.py
```

(macOS/Linux: `python3 run_all_training.py`)

This runs all 9 stages back to back — should take well under 10 minutes total on a
normal machine. It prints progress for each stage and ends by printing the real
selected model and test metrics.

If you'd rather run stages individually to inspect intermediate output, see the
stage functions imported at the top of `run_all_training.py` — each one is
independently callable, e.g.:
```cmd
python -c "from ml.train import stage_build_pairs; stage_build_pairs()"
```

## 4. After training completes, you'll have:

```
models/ats_model.pkl
models/vectorizer.pkl
models/model_metadata.json
reports/model_metrics.json
reports/confusion_matrix.png
reports/feature_importance.png
reports/evaluation_report.md
```

Check `models/model_metadata.json` for real training/validation/test metrics — until that
file exists, no model has actually been trained.

## 5. Still to build (not in this zip)

- `ml/predict.py` — inference pipeline that loads the saved model
- `api/` — FastAPI app exposing `POST /api/v1/analyze`
- `frontend/` — upload UI
- `tests/` — automated tests per the original spec
- `MODEL_CARD.md`

## Project layout

```
ats-system/
  DATASET.md              <- dataset provenance, verification, honesty notes
  requirements.txt
  models/
    skill_taxonomy.json   <- hand-curated skill taxonomy (real, used by ml/skill_extraction.py)
  ml/
    text_extraction.py    <- PDF/DOCX -> text
    section_extraction.py <- resume/JD section detection
    skill_extraction.py   <- taxonomy-based skill extraction (optimized, O(n) tokenizer)
    experience_education.py
    jd_parser.py
    resume_parser.py
    feature_engineering.py
    data_loader.py         <- loads real datasets, builds labeled pairs
    preprocessing.py        <- shared feature-computation code path (train == inference)
    train.py                 <- staged training pipeline
  data/
    raw/        (empty in zip — see step 2)
    processed/  (empty in zip — regenerated by running the stages)
```
