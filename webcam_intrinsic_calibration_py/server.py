#!/usr/bin/env python3
"""Local Flask server for webcam intrinsic calibration (Python OpenCV)."""

from __future__ import annotations

import os

from flask import Flask, jsonify, request, send_from_directory
import numpy as np

import calib

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")

app = Flask(__name__, static_folder=STATIC, static_url_path="")


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


def _board_from_form_or_json():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        cols = int(data.get("cols", 9))
        rows = int(data.get("rows", 6))
        square_mm = float(data.get("squareMm", 25))
        return cols, rows, square_mm, data
    cols = int(request.form.get("cols", 9))
    rows = int(request.form.get("rows", 6))
    square_mm = float(request.form.get("squareMm", 25))
    return cols, rows, square_mm, None


def _image_from_request():
    """Accept multipart file `image` or JSON `image` (data URL / base64)."""
    if request.files.get("image"):
        return calib.decode_image_bytes(request.files["image"].read())
    data = request.get_json(silent=True) or {}
    if data.get("image"):
        return calib.decode_image_b64(data["image"])
    raise ValueError("Missing image (multipart file or JSON base64)")


@app.post("/api/detect")
def api_detect():
    try:
        cols, rows, square_mm, _ = _board_from_form_or_json()
        img = _image_from_request()
        result = calib.detect_chessboard(
            img, cols, rows, square_mm=square_mm
        )
        return jsonify(result)
    except Exception as err:
        return jsonify({"error": str(err)}), 400


@app.post("/api/diversity")
def api_diversity():
    """Check whether a new board pose is diverse enough vs existing captures."""
    try:
        data = request.get_json(force=True)
        cols = int(data["cols"])
        rows = int(data["rows"])
        square_mm = float(data["squareMm"])
        image_width = int(data["imageWidth"])
        image_height = int(data["imageHeight"])
        corners = data.get("corners")
        if not isinstance(corners, list):
            raise ValueError("corners must be a flat list")
        existing = data.get("existingPoses") or []
        new_pose = calib.estimate_pose(
            np.asarray(corners, dtype=np.float32),
            cols,
            rows,
            square_mm,
            image_width,
            image_height,
        )
        if new_pose is None:
            raise ValueError("Could not estimate pose for new corners")
        check = calib.check_pose_diversity(
            new_pose, existing, image_width, image_height
        )
        check["pose"] = new_pose
        return jsonify(check)
    except Exception as err:
        return jsonify({"error": str(err)}), 400


@app.post("/api/calibrate")
def api_calibrate():
    try:
        data = request.get_json(force=True)
        cols = int(data["cols"])
        rows = int(data["rows"])
        square_mm = float(data["squareMm"])
        image_width = int(data["imageWidth"])
        image_height = int(data["imageHeight"])
        captures = data.get("captures") or data.get("imagePointSets")
        if not isinstance(captures, list):
            raise ValueError("captures must be a list of corner arrays")
        result = calib.calibrate_from_captures(
            captures, cols, rows, square_mm, image_width, image_height
        )
        return jsonify(result)
    except Exception as err:
        return jsonify({"error": str(err)}), 400


@app.post("/api/selftest")
def api_selftest():
    try:
        cols, rows, _square, _ = _board_from_form_or_json()
        return jsonify(calib.selftest_detection(cols, rows))
    except Exception as err:
        return jsonify({"error": str(err)}), 400


@app.post("/api/probe")
def api_probe():
    try:
        cols, rows, _square, _ = _board_from_form_or_json()
        img = _image_from_request()
        return jsonify(calib.probe_pattern_sizes(img, cols, rows))
    except Exception as err:
        return jsonify({"error": str(err)}), 400


def main():
    """Dev entrypoint (`python server.py`). Prefer Gunicorn in production."""
    port = int(os.environ.get("PORT", "8767"))
    print(f"Webcam intrinsic calibration (Python OpenCV)")
    print(f"Open http://127.0.0.1:{port}")
    # Deploy / LAN: bind all interfaces (PaaS, Docker, tunnels).
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    # Local-only loopback:
    # app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
