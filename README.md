# Webcam intrinsic calibration (Python OpenCV)

Local web app that estimates a webcam’s **intrinsic matrix** and **horizontal / vertical FOV** from a printed checkerboard of known size.

Browser: webcam capture + guided poses.  
Server: full Python OpenCV (`findChessboardCorners`, `cornerSubPix`, `calibrateCamera`).

1. Enter the board’s **inner corner** counts (cols × rows) and **square size in mm**
2. Start the camera and begin capture
3. Follow on-screen pose prompts (center, near/far, corners, tilts)
4. The app **auto-captures** when the board is detected and held steady in the guide
5. Compute intrinsics and download JSON

## Run

```bash
cd webcam_calibration
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
gunicorn -b 0.0.0.0:${PORT:-8767} -w 2 --threads 4 --timeout 120 server:app
# Local Flask (no Gunicorn): python server.py
# open http://127.0.0.1:8767
```

Override port with `PORT=9000` (works for both Gunicorn and `python server.py`).

## Deploy

Webcam access needs **HTTPS** (or localhost). Use a host that terminates TLS (Railway, Render, Fly.io, etc.), or put Nginx/Caddy in front.

- **Start command** (most PaaS): see `Procfile`, or the `gunicorn` line above with `$PORT`
- **Docker**: `docker build -t webcam-calib . && docker run -p 8080:8080 -e PORT=8080 webcam-calib`
- Root directory / workdir must be this folder so `import calib` resolves

### Detect streaming (bandwidth)

Live board detection posts JPEG frames to `/api/detect`. On **localhost** the client uses snappy settings (≈120 ms interval, max width 1280, JPEG quality 0.85). On a **remote** host it throttles (≈280 ms base interval, max width 640, quality 0.6) and may raise the interval further if detect RTT is high (capped at 700 ms). Calibration (`/api/calibrate`) still uses full-resolution corner lists from captures — only the live detect stream is reduced. Tunables: `DETECT_LOCAL` / `DETECT_REMOTE` in `static/app.js`.

## Board setup

- Use **inner corners**, not squares. Example: a board with **10×7 squares** has **9×6** inner corners.
- Measure one square accurately in millimeters.
- Keep the board flat and rigid; good lighting helps detection.
- Do **not** use a mirrored preview for calibration — this app shows the raw camera image.
- Prefer asymmetric boards (e.g. 9×6), not 8×8.

## Troubleshooting detection

1. Click **Test detector** — runs OpenCV on a perfect synthetic board of your configured size.
   - **Pass** → OpenCV works; your physical board counts / lighting / focus are the issue.
   - **Fail** → report the message shown (unusual with Python OpenCV).
2. With the camera on, click **Probe live frame** to try common corner sizes and auto-fill a match.
3. Keep the whole board visible with a bit of margin; avoid glare.

## Camera auto-framing

Newer webcams often digitally pan/zoom (“Center Stage”, Windows Studio Effects, Logitech/vendor auto-frame). That changes FOV mid-session and breaks calibration.

This app best-effort requests `faceFraming: false` and minimum `zoom` when the browser exposes those constraints. Many platforms still require turning the effect off in **OS / camera settings**.

## Accuracy features

- Prefers OpenCV `findChessboardCornersSB` (falls back to classic + `cornerSubPix`)
- Guided poses include corner **and edge** placements for better principal-point / distortion coverage
- Captures are rejected if the board pose is too similar to an earlier one (`solvePnP` diversity check)
- Calibration drops high-reprojection outlier views and re-runs `calibrateCamera`

Results include:

| Field | Meaning |
|-------|---------|
| `cameraMatrix` / `fx, fy, cx, cy` | Pinhole intrinsics (pixels) |
| `horizontalFovDeg` / `verticalFovDeg` | FOV from fx/fy and image size |
| `distCoeffs` | Radial/tangential distortion |
| `rms` | Reprojection error (pixels) |

Download via **Download JSON**.

## Layout

| File | Role |
|------|------|
| `server.py` | Flask static server + JSON API |
| `calib.py` | Python OpenCV detect / calibrate / FOV |
| `static/` | Browser UI (webcam, poses, results) |
| `requirements.txt` | flask, gunicorn, opencv-python-headless, numpy, Pillow |
| `Procfile` | PaaS start command (Gunicorn) |
| `Dockerfile` | Container image (Gunicorn) |
