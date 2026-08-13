"""Python OpenCV checkerboard detection and camera calibration."""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Pose diversity thresholds (new capture vs any existing)
MIN_ROTATION_DEG = 12.0
MIN_DEPTH_REL = 0.18
MIN_CENTROID_FRAC = 0.15

# Per-view outlier rejection after first calibrate pass
OUTLIER_MEDIAN_FACTOR = 1.5
OUTLIER_ABS_FLOOR_PX = 0.35
MIN_VIEWS_AFTER_REJECT = 4


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to BGR uint8 image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def decode_image_b64(data_url_or_b64: str) -> np.ndarray:
    """Decode a data URL or raw base64 string to BGR."""
    raw = data_url_or_b64
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return decode_image_bytes(base64.b64decode(raw))


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception:
        return cv2.equalizeHist(gray)


def _corners_payload(
    corners: np.ndarray, width: int, height: int, method: str
) -> dict[str, Any]:
    flat = corners.reshape(-1, 2).astype(np.float64)
    xs = flat[:, 0]
    ys = flat[:, 1]
    return {
        "found": True,
        "corners": flat.reshape(-1).tolist(),
        "centroid": {"x": float(xs.mean()), "y": float(ys.mean())},
        "bounds": {
            "x": float(xs.min()),
            "y": float(ys.min()),
            "w": float(xs.max() - xs.min()),
            "h": float(ys.max() - ys.min()),
        },
        "width": int(width),
        "height": int(height),
        "method": method,
    }


def find_corners(gray: np.ndarray, cols: int, rows: int) -> tuple[np.ndarray | None, str]:
    """Prefer findChessboardCornersSB; fall back to classic + cornerSubPix.

    Returns (Nx2 float32 corners or None, method name).
    """
    pattern = (int(cols), int(rows))

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = 0
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            sb_flags |= cv2.CALIB_CB_ACCURACY
        try:
            found, corners = cv2.findChessboardCornersSB(gray, pattern, sb_flags)
            if found and corners is not None and len(corners) == cols * rows:
                return corners.reshape(-1, 2).astype(np.float32), "sb"
        except Exception:
            pass

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found or corners is None:
        return None, "none"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2).astype(np.float32), "classic"


def make_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """Object points in mm, row-major matching OpenCV corner order."""
    pts = np.zeros((cols * rows, 3), dtype=np.float32)
    k = 0
    for r in range(rows):
        for c in range(cols):
            pts[k, 0] = c * square_mm
            pts[k, 1] = r * square_mm
            pts[k, 2] = 0.0
            k += 1
    return pts


def rough_camera_matrix(width: int, height: int) -> np.ndarray:
    f = float(max(width, height))
    return np.array(
        [[f, 0.0, width * 0.5], [0.0, f, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def estimate_pose(
    corners: np.ndarray,
    cols: int,
    rows: int,
    square_mm: float,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """solvePnP pose with a rough K (good enough for diversity checks)."""
    n = cols * rows
    flat = np.asarray(corners, dtype=np.float32).reshape(-1)
    if flat.size != n * 2:
        return None
    obj = make_object_points(cols, rows, square_mm)
    img = flat.reshape(n, 1, 2)
    K = rough_camera_matrix(image_width, image_height)
    dist = np.zeros((5, 1), dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except Exception:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2].astype(np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    depth = float(np.linalg.norm(tvec))
    xs = flat[0::2]
    ys = flat[1::2]
    return {
        "rvec": [float(rvec[0, 0]), float(rvec[1, 0]), float(rvec[2, 0])],
        "tvec": [float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0])],
        "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
        "depth": depth,
        "centroid": {"x": float(xs.mean()), "y": float(ys.mean())},
    }


def rotation_angle_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    Ra, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))
    Rb, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))
    R = Ra.T @ Rb
    cos_a = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cos_a)))


def check_pose_diversity(
    new_pose: dict[str, Any],
    existing_poses: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Return ok=False if new_pose is too similar to any existing capture."""
    if not existing_poses:
        return {"ok": True, "reason": "", "nearest": None}

    diag = math.hypot(image_width, image_height)
    nearest = None
    for i, old in enumerate(existing_poses):
        ang = rotation_angle_deg(new_pose["rvec"], old["rvec"])
        d_new = max(float(new_pose["depth"]), 1e-6)
        d_old = max(float(old["depth"]), 1e-6)
        depth_rel = abs(d_new - d_old) / max(d_new, d_old)
        c0 = new_pose["centroid"]
        c1 = old["centroid"]
        cent_dist = math.hypot(c0["x"] - c1["x"], c0["y"] - c1["y"]) / max(diag, 1.0)
        entry = {
            "index": i,
            "rotationDeg": ang,
            "depthRel": depth_rel,
            "centroidFrac": cent_dist,
        }
        if nearest is None or (
            ang + 40 * depth_rel + 40 * cent_dist
            < nearest["rotationDeg"]
            + 40 * nearest["depthRel"]
            + 40 * nearest["centroidFrac"]
        ):
            nearest = entry

        if (
            ang < MIN_ROTATION_DEG
            and depth_rel < MIN_DEPTH_REL
            and cent_dist < MIN_CENTROID_FRAC
        ):
            return {
                "ok": False,
                "reason": (
                    f"Too similar to capture #{i + 1}: "
                    f"Δrot {ang:.1f}° (need ≥{MIN_ROTATION_DEG:.0f}°), "
                    f"Δdepth {depth_rel:.0%} (need ≥{MIN_DEPTH_REL:.0%}), "
                    f"Δpos {cent_dist:.0%} of diagonal (need ≥{MIN_CENTROID_FRAC:.0%}). "
                    "Move / tilt / change distance more."
                ),
                "nearest": entry,
            }

    return {"ok": True, "reason": "", "nearest": nearest}


def detect_chessboard(
    img_bgr: np.ndarray,
    cols: int,
    rows: int,
    max_detect_width: int = 1280,
    square_mm: float | None = None,
) -> dict[str, Any]:
    """Detect inner corners; scale coordinates back to full image size."""
    h, w = img_bgr.shape[:2]
    empty = {
        "found": False,
        "corners": None,
        "centroid": None,
        "bounds": None,
        "width": int(w),
        "height": int(h),
        "method": "none",
        "pose": None,
    }
    if w < 2 or h < 2 or cols < 2 or rows < 2:
        return empty

    scale = max_detect_width / w if w > max_detect_width else 1.0
    if scale < 1.0:
        dw = int(round(w * scale))
        dh = int(round(h * scale))
        work = cv2.resize(img_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        work = img_bgr
        scale = 1.0

    gray = to_gray(work)
    enhanced = enhance_gray(gray)

    corners = None
    method = "none"
    for mat in (enhanced, gray):
        corners, method = find_corners(mat, cols, rows)
        if corners is not None:
            break

    if corners is None:
        return empty

    if scale != 1.0:
        corners = corners / scale

    payload = _corners_payload(corners, w, h, method)
    if square_mm is not None and square_mm > 0:
        payload["pose"] = estimate_pose(corners, cols, rows, square_mm, w, h)
    else:
        payload["pose"] = None
    return payload


def make_synthetic_board(
    cols: int, rows: int, square_px: int = 48, margin_px: int = 56
) -> np.ndarray:
    """BGR synthetic checkerboard with white margin (inner corners = cols×rows)."""
    sq_cols = cols + 1
    sq_rows = rows + 1
    w = sq_cols * square_px + margin_px * 2
    h = sq_rows * square_px + margin_px * 2
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for r in range(sq_rows):
        for c in range(sq_cols):
            color = 17 if (r + c) % 2 == 0 else 245
            x0 = margin_px + c * square_px
            y0 = margin_px + r * square_px
            img[y0 : y0 + square_px, x0 : x0 + square_px] = color
    return img


def selftest_detection(cols: int, rows: int) -> dict[str, Any]:
    board = make_synthetic_board(cols, rows)
    gray = to_gray(board)
    corners, method = find_corners(gray, cols, rows)
    expected = cols * rows
    found_n = 0 if corners is None else int(corners.shape[0])
    ok = corners is not None and found_n == expected

    rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    preview = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    if ok:
        message = (
            f"Self-test OK ({method}): detected {found_n}/{expected} corners "
            f"on a synthetic {cols}×{rows} board."
        )
    else:
        message = (
            f"Self-test FAILED ({method}): expected {expected} corners for "
            f"{cols}×{rows}, got {found_n}."
        )
    return {
        "ok": ok,
        "expectedCorners": expected,
        "foundCorners": found_n,
        "method": method,
        "message": message,
        "previewDataUrl": preview,
    }


PROBE_CANDIDATES = [
    (9, 6),
    (6, 9),
    (8, 6),
    (6, 8),
    (7, 5),
    (5, 7),
    (10, 7),
    (7, 10),
    (9, 7),
    (7, 9),
    (8, 5),
    (5, 8),
]


def probe_pattern_sizes(
    img_bgr: np.ndarray, preferred_cols: int, preferred_rows: int
) -> dict[str, Any]:
    candidates: list[tuple[int, int]] = []

    def add(c: int, r: int) -> None:
        if c < 2 or r < 2:
            return
        if (c, r) not in candidates:
            candidates.append((c, r))

    add(preferred_cols, preferred_rows)
    add(preferred_rows, preferred_cols)
    for c, r in PROBE_CANDIDATES:
        add(c, r)

    tried = []
    best = None
    for cols, rows in candidates:
        det = detect_chessboard(img_bgr, cols, rows)
        entry = {"cols": cols, "rows": rows, "found": det["found"], "method": det["method"]}
        tried.append(entry)
        if det["found"] and best is None:
            best = entry
    return {"best": best, "tried": tried}


def _per_view_rms(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    err = projected.reshape(-1, 2) - img_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def _run_calibrate(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray,
    flags: int,
) -> tuple[float, np.ndarray, np.ndarray, list, list]:
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=obj_points,
        imagePoints=img_points,
        imageSize=image_size,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=flags,
    )
    return float(rms), K, dist, rvecs, tvecs


def calibrate_from_captures(
    image_point_sets: list[list[float]],
    cols: int,
    rows: int,
    square_mm: float,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    if len(image_point_sets) < 3:
        raise ValueError("Need at least 3 successful board captures")

    n = cols * rows
    obj_template = make_object_points(cols, rows, square_mm)
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    for flat in image_point_sets:
        arr = np.asarray(flat, dtype=np.float32).reshape(-1)
        if arr.size != n * 2:
            raise ValueError(
                f"Expected {n * 2} corner coords ({cols}×{rows}), got {arr.size}"
            )
        obj_points.append(obj_template.copy())
        img_points.append(arr.reshape(n, 1, 2))

    image_size = (int(image_width), int(image_height))
    flags = cv2.CALIB_FIX_K3
    dist0 = np.zeros((5, 1), dtype=np.float64)

    rms1, K1, dist1, rvecs1, tvecs1 = _run_calibrate(
        obj_points, img_points, image_size, None, dist0, flags
    )

    per_view = [
        _per_view_rms(obj_points[i], img_points[i], rvecs1[i], tvecs1[i], K1, dist1)
        for i in range(len(obj_points))
    ]
    median_rms = float(np.median(per_view))
    threshold = max(median_rms * OUTLIER_MEDIAN_FACTOR, median_rms + OUTLIER_ABS_FLOOR_PX)

    keep_idx = [i for i, e in enumerate(per_view) if e <= threshold]
    dropped_idx = [i for i in range(len(per_view)) if i not in keep_idx]

    # Never drop below a usable set; if too aggressive, keep best views by RMS
    if len(keep_idx) < MIN_VIEWS_AFTER_REJECT:
        order = sorted(range(len(per_view)), key=lambda i: per_view[i])
        keep_idx = sorted(order[: max(MIN_VIEWS_AFTER_REJECT, min(3, len(order)))])
        dropped_idx = [i for i in range(len(per_view)) if i not in keep_idx]

    used_outlier_pass = len(dropped_idx) > 0 and len(keep_idx) < len(per_view)

    if used_outlier_pass:
        obj2 = [obj_points[i] for i in keep_idx]
        img2 = [img_points[i] for i in keep_idx]
        try:
            flags2 = flags | cv2.CALIB_USE_INTRINSIC_GUESS
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = _run_calibrate(
                obj2, img2, image_size, K1.copy(), dist1.copy(), flags2
            )
        except cv2.error:
            # OpenCV 5+ rejects out-of-image principal points from a bad first pass
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = _run_calibrate(
                obj2,
                img2,
                image_size,
                None,
                np.zeros((5, 1), dtype=np.float64),
                flags,
            )
        per_view_final = [
            _per_view_rms(obj2[i], img2[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            for i in range(len(obj2))
        ]
    else:
        rms = rms1
        camera_matrix = K1
        dist_coeffs = dist1
        keep_idx = list(range(len(per_view)))
        dropped_idx = []
        per_view_final = per_view
        used_outlier_pass = False
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]
    dist = [float(dist_coeffs[i, 0]) for i in range(dist_coeffs.shape[0])]

    hfov_deg = math.degrees(2.0 * math.atan(image_width / (2.0 * fx)))
    vfov_deg = math.degrees(2.0 * math.atan(image_height / (2.0 * fy)))

    return {
        "rms": float(rms),
        "rmsInitial": float(rms1),
        "cameraMatrix": K,
        "distCoeffs": dist,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "hfovDeg": hfov_deg,
        "vfovDeg": vfov_deg,
        "imageWidth": int(image_width),
        "imageHeight": int(image_height),
        "numImages": len(keep_idx),
        "numImagesInput": len(image_point_sets),
        "keptIndices": keep_idx,
        "droppedIndices": dropped_idx,
        "perViewRms": [float(e) for e in per_view],
        "perViewRmsFinal": [float(e) for e in per_view_final],
        "outlierThreshold": float(threshold),
        "outlierRejectionApplied": used_outlier_pass,
    }
