"""
Flask API for the ATS compatibility model.

Endpoints:
  GET  /api/health   -> {"status": "ok", "model_loaded": bool, ...}
  POST /api/predict   -> accepts either:
      (a) multipart/form-data: resume_file (.pdf/.docx) + jd_text (text field)
      (b) application/json: {"resume_text": "...", "jd_text": "..."}
    Returns compatibility score + feature breakdown (see api/inference.py).

Also serves the static frontend (../frontend) at "/", so the whole app
runs from one process/port and there's no CORS to configure for local use.

Run:  python api/app.py   (from the project root, with venv active)
Then open http://127.0.0.1:5000 in a browser.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from ml.resume_parser import InvalidResume
from ml.jd_parser import InvalidJobDescription
from ml.text_extraction import extract_resume_text, ExtractionError
from api.inference import get_predictor

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
ALLOWED_RESUME_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    try:
        predictor = get_predictor()
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "algorithm": predictor.metadata.get("algorithm"),
            "model_version": predictor.metadata.get("model_version"),
            "trained_on_samples": predictor.metadata.get("number_of_training_samples"),
        })
    except FileNotFoundError as e:
        return jsonify({"status": "ok", "model_loaded": False, "error": str(e)}), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    resume_text = None
    jd_text = None

    if request.content_type and "multipart/form-data" in request.content_type:
        jd_text = (request.form.get("jd_text") or "").strip()
        resume_file = request.files.get("resume_file")
        if resume_file and resume_file.filename:
            ext = os.path.splitext(resume_file.filename)[1].lower()
            if ext not in ALLOWED_RESUME_EXTENSIONS:
                return jsonify({
                    "error": f"Unsupported resume file type '{ext}'. Use .pdf or .docx."
                }), 400
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                resume_file.save(tmp.name)
                tmp_path = tmp.name
            try:
                resume_text = extract_resume_text(tmp_path)
            except ExtractionError as e:
                return jsonify({"error": str(e)}), 400
            finally:
                os.remove(tmp_path)
        else:
            resume_text = (request.form.get("resume_text") or "").strip()
    else:
        data = request.get_json(silent=True) or {}
        resume_text = (data.get("resume_text") or "").strip()
        jd_text = (data.get("jd_text") or "").strip()

    if not resume_text:
        return jsonify({"error": "No resume provided (upload a .pdf/.docx or send resume_text)."}), 400
    if not jd_text:
        return jsonify({"error": "No job description provided (jd_text is required)."}), 400

    try:
        result = predictor.predict(resume_text, jd_text)
    except InvalidResume as e:
        return jsonify({"error": f"Resume: {e}"}), 400
    except InvalidJobDescription as e:
        return jsonify({"error": f"Job description: {e}"}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
