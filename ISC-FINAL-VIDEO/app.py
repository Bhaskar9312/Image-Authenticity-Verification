"""
Flask Web Application for AI Video Forgery Detection & Content Quantification.
Provides interactive UI to upload or select videos, view Real vs AI proportions,
inspect forensic branches, and analyze second-by-second timelines.
"""

import os
import sys
import uuid
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

from analyze_video import VideoForgeryAnalyzer

app = Flask(__name__)

UPLOAD_FOLDER = Path("web_uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
EVIDENCE_FOLDER = Path("evidence_output")
EVIDENCE_FOLDER.mkdir(exist_ok=True)

DATA_UPLOADS = Path("data/uploads")

# Initialize analyzer (loads best checkpoint if available)
CHECKPOINT_PATH = Path("checkpoints/best_forgery_detector.pt")
fallback_ckpt = Path("C:/Users/Tankala Bhaskar teja/ISC-VIDEO-KOMAL/isc_ensemble_improved_output/checkpoints/full_ensemble.pt")
active_ckpt = str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else (str(fallback_ckpt) if fallback_ckpt.exists() else None)

print(f"[App] Active model checkpoint: {active_ckpt}")
analyzer = VideoForgeryAnalyzer(checkpoint_path=active_ckpt)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Lists available benchmark test videos from data/uploads."""
    samples = []
    if DATA_UPLOADS.exists():
        for f in sorted(DATA_UPLOADS.glob("*.mp4")):
            size_mb = round(f.stat().st_size / (1024 * 1024), 2)
            samples.append({
                "filename": f.name,
                "size_mb": size_mb
            })
    return jsonify({"samples": samples})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Handles video analysis requests via file upload or sample selection."""
    try:
        video_path = None
        
        # Check for uploaded file
        if "video_file" in request.files and request.files["video_file"].filename != "":
            f = request.files["video_file"]
            ext = Path(f.filename).suffix or ".mp4"
            unique_name = f"{uuid.uuid4().hex}{ext}"
            saved_path = UPLOAD_FOLDER / unique_name
            f.save(str(saved_path))
            video_path = saved_path
            
        # Or check for selected benchmark sample
        elif "sample_filename" in request.form and request.form["sample_filename"] != "":
            sample_name = request.form["sample_filename"]
            sample_path = DATA_UPLOADS / sample_name
            if sample_path.exists():
                video_path = sample_path
            else:
                return jsonify({"error": f"Sample video not found: {sample_name}"}), 404
        else:
            return jsonify({"error": "No video file provided."}), 400

        # Run analysis
        report = analyzer.analyze(
            video_path=video_path,
            sample_fps=2.0,
            max_frames=64,
            window_size=4,
            save_evidence_dir=EVIDENCE_FOLDER
        )
        return jsonify(report)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/evidence/<path:filename>")
def serve_evidence(filename):
    """Serves generated keyframe evidence images."""
    return send_from_directory(EVIDENCE_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================")
    print(f" VeriVid AI Forgery Detection Web Server Running")
    print(f" Open in browser: http://localhost:{port}")
    print(f"==================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
