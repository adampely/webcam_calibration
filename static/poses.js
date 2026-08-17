/**
 * Guided checkerboard poses for intrinsic calibration.
 * Regions are normalized [cx, cy, scale] in image space (0–1).
 * scale ≈ fraction of the shorter image side the board should span.
 */

/** @typedef {{ id: string, title: string, instruction: string, cx: number, cy: number, scale: number, tiltHint?: string }} Pose */

/** @type {Pose[]} */
export const POSES = [
  {
    id: "center",
    title: "Center",
    instruction: "Hold the board flat, facing the camera, in the center of the frame.",
    cx: 0.5,
    cy: 0.5,
    scale: 0.55,
  },
  {
    id: "closer",
    title: "Closer",
    instruction: "Move the board closer so it fills more of the frame (still fully visible).",
    cx: 0.5,
    cy: 0.5,
    scale: 0.72,
  },
  {
    id: "farther",
    title: "Farther",
    instruction: "Move the board farther away, still centered and facing the camera.",
    cx: 0.5,
    cy: 0.5,
    scale: 0.38,
  },
  {
    id: "top-left",
    title: "Top left",
    instruction: "Place the board in the upper-left corner area, facing the camera (keep it fully visible).",
    cx: 0.22,
    cy: 0.22,
    scale: 0.36,
  },
  {
    id: "top-right",
    title: "Top right",
    instruction: "Place the board in the upper-right corner area, facing the camera (keep it fully visible).",
    cx: 0.78,
    cy: 0.22,
    scale: 0.36,
  },
  {
    id: "bottom-left",
    title: "Bottom left",
    instruction: "Place the board in the lower-left corner area, facing the camera (keep it fully visible).",
    cx: 0.22,
    cy: 0.78,
    scale: 0.36,
  },
  {
    id: "bottom-right",
    title: "Bottom right",
    instruction: "Place the board in the lower-right corner area, facing the camera (keep it fully visible).",
    cx: 0.78,
    cy: 0.78,
    scale: 0.36,
  },
  {
    id: "top-edge",
    title: "Top edge",
    instruction: "Place the board near the top edge of the frame, centered horizontally.",
    cx: 0.5,
    cy: 0.18,
    scale: 0.36,
  },
  {
    id: "bottom-edge",
    title: "Bottom edge",
    instruction: "Place the board near the bottom edge of the frame, centered horizontally.",
    cx: 0.5,
    cy: 0.82,
    scale: 0.36,
  },
  {
    id: "left-edge",
    title: "Left edge",
    instruction: "Place the board near the left edge of the frame, centered vertically.",
    cx: 0.18,
    cy: 0.5,
    scale: 0.36,
  },
  {
    id: "right-edge",
    title: "Right edge",
    instruction: "Place the board near the right edge of the frame, centered vertically.",
    cx: 0.82,
    cy: 0.5,
    scale: 0.36,
  },
  {
    id: "tilt-left",
    title: "Tilt left",
    instruction: "Tilt the left edge of the board away from the camera (~30–45°).",
    cx: 0.5,
    cy: 0.5,
    scale: 0.5,
    tiltHint: "left",
  },
  {
    id: "tilt-right",
    title: "Tilt right",
    instruction: "Tilt the right edge of the board away from the camera (~30–45°).",
    cx: 0.5,
    cy: 0.5,
    scale: 0.5,
    tiltHint: "right",
  },
  {
    id: "tilt-up",
    title: "Tilt up",
    instruction: "Tilt the top edge of the board away from the camera (~30–45°).",
    cx: 0.5,
    cy: 0.5,
    scale: 0.5,
    tiltHint: "up",
  },
  {
    id: "tilt-down",
    title: "Tilt down",
    instruction: "Tilt the bottom edge of the board away from the camera (~30–45°).",
    cx: 0.5,
    cy: 0.5,
    scale: 0.5,
    tiltHint: "down",
  },
  {
    id: "center-final",
    title: "Center (final)",
    instruction: "One more centered, flat pose to finish the set.",
    cx: 0.5,
    cy: 0.5,
    scale: 0.55,
  },
];

export const MIN_CAPTURES_TO_COMPUTE = 10;
/** ~5 preview detections while holding ≈ 0.5–1s once aligned; then one full-res capture */
export const STABLE_FRAMES_REQUIRED = 5;
export const CAPTURE_COOLDOWN_MS = 700;
/** Auto-capture when alignment score reaches this (0–100) */
export const ALIGN_SCORE_THRESHOLD = 75;
/**
 * Minimum relative edge foreshortening to count as a tilt.
 * ~0.12 ≈ cos(30°) foreshortening on the far edge.
 */
export const TILT_RATIO_THRESHOLD = 0.12;

/**
 * Guide rectangle in pixel coords for a pose.
 * @param {Pose} pose
 * @param {number} width
 * @param {number} height
 */
export function guideRect(pose, width, height) {
  const shortSide = Math.min(width, height);
  const side = pose.scale * shortSide;
  const aspect = 1.4;
  const w = side * aspect;
  const h = side;
  return {
    x: pose.cx * width - w / 2,
    y: pose.cy * height - h / 2,
    w,
    h,
  };
}

/**
 * Axis-aligned bounds of detected corners.
 * @param {Float32Array | number[]} corners
 */
export function cornersBounds(corners) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < corners.length; i += 2) {
    const x = corners[i];
    const y = corners[i + 1];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Estimate which board edge is farther from the camera via perspective
 * foreshortening of the outer quad (image-space left/right/top/bottom).
 *
 * OpenCV corner order can start at any board corner, so edges are classified
 * by midpoint position relative to the board centroid — not by grid indices.
 *
 * @param {Float32Array | number[]} corners flat [x0,y0,...]
 * @param {number} cols
 * @param {number} rows
 * @returns {{
 *   leftLen: number,
 *   rightLen: number,
 *   topLen: number,
 *   bottomLen: number,
 *   horizontal: number,
 *   vertical: number,
 *   direction: 'left'|'right'|'up'|'down'|'flat'|null,
 * } | null}
 */
export function estimateBoardTilt(corners, cols, rows) {
  const n = (corners.length / 2) | 0;
  if (!cols || !rows || n !== cols * rows) return null;

  const at = (c, r) => ({
    x: corners[(r * cols + c) * 2],
    y: corners[(r * cols + c) * 2 + 1],
  });
  const quad = [at(0, 0), at(cols - 1, 0), at(cols - 1, rows - 1), at(0, rows - 1)];
  const cx = (quad[0].x + quad[1].x + quad[2].x + quad[3].x) / 4;
  const cy = (quad[0].y + quad[1].y + quad[2].y + quad[3].y) / 4;

  /** @type {Record<string, number>} */
  const sideLen = { left: 0, right: 0, top: 0, bottom: 0 };
  /** @type {Record<string, number>} */
  const sideCount = { left: 0, right: 0, top: 0, bottom: 0 };

  for (let i = 0; i < 4; i++) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    const dx = mx - cx;
    const dy = my - cy;
    const side =
      Math.abs(dx) >= Math.abs(dy) ? (dx >= 0 ? "right" : "left") : dy >= 0 ? "bottom" : "top";
    sideLen[side] += dist(a, b);
    sideCount[side] += 1;
  }

  // Need one edge per side; ambiguous quads (strong rotation in-plane) bail out.
  for (const k of ["left", "right", "top", "bottom"]) {
    if (sideCount[k] !== 1) return null;
  }

  const leftLen = sideLen.left;
  const rightLen = sideLen.right;
  const topLen = sideLen.top;
  const bottomLen = sideLen.bottom;
  const hMean = (leftLen + rightLen) / 2;
  const vMean = (topLen + bottomLen) / 2;
  // Positive horizontal ⇒ left edge shorter ⇒ left edge farther from camera
  const horizontal = hMean > 1e-6 ? (rightLen - leftLen) / hMean : 0;
  // Positive vertical ⇒ top edge shorter ⇒ top edge farther from camera
  const vertical = vMean > 1e-6 ? (bottomLen - topLen) / vMean : 0;

  let direction = null;
  if (
    Math.abs(horizontal) < TILT_RATIO_THRESHOLD &&
    Math.abs(vertical) < TILT_RATIO_THRESHOLD
  ) {
    direction = "flat";
  } else if (Math.abs(horizontal) >= Math.abs(vertical)) {
    direction = horizontal > 0 ? "left" : "right";
  } else {
    direction = vertical > 0 ? "up" : "down";
  }

  return {
    leftLen,
    rightLen,
    topLen,
    bottomLen,
    horizontal,
    vertical,
    direction,
  };
}

/**
 * @param {NonNullable<ReturnType<typeof estimateBoardTilt>>} tilt
 * @param {'left'|'right'|'up'|'down'} expected
 */
export function scoreTiltMatch(tilt, expected) {
  const signed =
    expected === "left" || expected === "right" ? tilt.horizontal : tilt.vertical;
  const wantPositive = expected === "left" || expected === "up";
  const correctSign = wantPositive ? signed > 0 : signed < 0;
  const magnitude = Math.abs(signed);
  const primary =
    expected === "left" || expected === "right"
      ? Math.abs(tilt.horizontal) >= Math.abs(tilt.vertical) * 0.85
      : Math.abs(tilt.vertical) >= Math.abs(tilt.horizontal) * 0.85;

  const ok =
    correctSign && primary && magnitude >= TILT_RATIO_THRESHOLD;

  // Map magnitude 0→threshold→~0.35 into 0–100
  const tiltScore = Math.max(
    0,
    Math.min(100, ((magnitude - TILT_RATIO_THRESHOLD * 0.25) / 0.28) * 100)
  );

  let tip = "Hold steady";
  if (!correctSign || tilt.direction === "flat") {
    const labels = {
      left: "left edge away from the camera",
      right: "right edge away from the camera",
      up: "top edge away from the camera",
      down: "bottom edge away from the camera",
    };
    tip =
      tilt.direction && tilt.direction !== "flat" && tilt.direction !== expected
        ? `Wrong tilt (${tilt.direction}) — tilt the ${labels[expected]}`
        : `Tilt the ${labels[expected]} more (~30–45°)`;
  } else if (!primary) {
    tip = `Tilt more ${expected} (reduce the other axis)`;
  } else if (magnitude < TILT_RATIO_THRESHOLD) {
    tip = `Tilt ${expected} a bit more`;
  } else {
    tip = "Good tilt — hold still";
  }

  return { ok, tiltScore: correctSign ? tiltScore : tiltScore * 0.25, tip, magnitude };
}

/**
 * Score how well the detected board matches the current pose guide.
 * @param {{ x: number, y: number }} centroid
 * @param {Float32Array | number[]} corners
 * @param {Pose} pose
 * @param {number} width
 * @param {number} height
 * @param {number} [cols]
 * @param {number} [rows]
 * @returns {{
 *   score: number,
 *   positionScore: number,
 *   sizeScore: number,
 *   tiltScore: number | null,
 *   tiltOk: boolean | null,
 *   tilt: ReturnType<typeof estimateBoardTilt>,
 *   ready: boolean,
 *   tip: string,
 *   guide: ReturnType<typeof guideRect>,
 *   board: ReturnType<typeof cornersBounds>,
 *   dx: number,
 *   dy: number,
 * }}
 */
export function scoreAlignment(centroid, corners, pose, width, height, cols, rows) {
  const guide = guideRect(pose, width, height);
  const board = cornersBounds(corners);
  const gcx = guide.x + guide.w / 2;
  const gcy = guide.y + guide.h / 2;
  const dx = centroid.x - gcx;
  const dy = centroid.y - gcy;

  // Normalize position error by guide size (0 = perfect, 1 ≈ one guide-width away)
  const posErr = Math.hypot(dx / Math.max(guide.w, 1), dy / Math.max(guide.h, 1));
  const positionScore = Math.max(0, Math.min(100, (1 - posErr / 0.85) * 100));

  // Size: compare board diagonal to guide diagonal
  const boardDiag = Math.hypot(board.w, board.h);
  const guideDiag = Math.hypot(guide.w, guide.h);
  const sizeRatio = boardDiag / Math.max(guideDiag, 1);
  // Perfect at 1.0; still decent from ~0.55–1.55
  const sizeScore = Math.max(
    0,
    Math.min(100, (1 - Math.abs(Math.log2(Math.max(sizeRatio, 0.05))) / 1.2) * 100)
  );

  const tilt = pose.tiltHint ? estimateBoardTilt(corners, cols, rows) : null;
  const tiltMatch =
    pose.tiltHint && tilt ? scoreTiltMatch(tilt, pose.tiltHint) : null;

  // Tilt poses: require correct foreshortening; weight tilt heavily
  let score;
  if (pose.tiltHint && tiltMatch) {
    score = 0.35 * positionScore + 0.15 * sizeScore + 0.5 * tiltMatch.tiltScore;
  } else if (pose.tiltHint) {
    score = 0.5 * positionScore + 0.2 * sizeScore; // cannot verify tilt
  } else {
    score = 0.55 * positionScore + 0.45 * sizeScore;
  }

  let tip = "Hold steady";
  if (positionScore < 50) {
    const parts = [];
    if (Math.abs(dx) > guide.w * 0.12) parts.push(dx < 0 ? "right" : "left");
    if (Math.abs(dy) > guide.h * 0.12) parts.push(dy < 0 ? "down" : "up");
    tip = parts.length ? `Move board ${parts.join(" + ")}` : "Center on the guide";
  } else if (pose.tiltHint && tiltMatch && !tiltMatch.ok) {
    tip = tiltMatch.tip;
  } else if (sizeScore < 50) {
    tip = sizeRatio < 1 ? "Move closer" : "Move farther";
  } else if (score >= ALIGN_SCORE_THRESHOLD && (!pose.tiltHint || tiltMatch?.ok)) {
    tip = pose.tiltHint ? "Good tilt — hold still" : "Good — hold still";
  } else {
    tip = "Almost — fine-tune position/size";
  }

  const ready =
    score >= ALIGN_SCORE_THRESHOLD &&
    (!pose.tiltHint || !!tiltMatch?.ok);

  return {
    score,
    positionScore,
    sizeScore,
    tiltScore: tiltMatch ? tiltMatch.tiltScore : null,
    tiltOk: tiltMatch ? tiltMatch.ok : null,
    tilt,
    ready,
    tip,
    guide,
    board,
    dx,
    dy,
  };
}
