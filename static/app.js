/**
 * Webcam intrinsic calibration — guided checkerboard capture via Python OpenCV API.
 */

import {
  POSES,
  MIN_CAPTURES_TO_COMPUTE,
  STABLE_FRAMES_REQUIRED,
  CAPTURE_COOLDOWN_MS,
  ALIGN_SCORE_THRESHOLD,
  guideRect,
  scoreAlignment,
} from "./poses.js";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("start-btn");
const captureBtn = document.getElementById("capture-btn");
const snapBtn = document.getElementById("snap-btn");
const skipBtn = document.getElementById("skip-btn");
const computeBtn = document.getElementById("compute-btn");
const resetBtn = document.getElementById("reset-btn");
const downloadBtn = document.getElementById("download-btn");
const selftestBtn = document.getElementById("selftest-btn");
const probeBtn = document.getElementById("probe-btn");
const selftestPreview = document.getElementById("selftest-preview");
const selftestResult = document.getElementById("selftest-result");

const cornersColsInput = document.getElementById("corners-cols");
const cornersRowsInput = document.getElementById("corners-rows");
const squareMmInput = document.getElementById("square-mm");
const boardInputs = document.getElementById("board-inputs");
const iphoneModelSelect = document.getElementById("iphone-model");
const iphonePresetHint = document.getElementById("iphone-preset-hint");

/**
 * Active display size (mm) from Apple’s rectangular diagonal + resolution.
 * Portrait: width × height.
 */
const IPHONE_SCREENS = {
  "iphone-15": { label: "iPhone 15", widthMm: 65.11, heightMm: 141.16 },
  "iphone-15-plus": { label: "iPhone 15 Plus", widthMm: 71.19, heightMm: 154.3 },
  "iphone-15-pro": { label: "iPhone 15 Pro", widthMm: 65.11, heightMm: 141.16 },
  "iphone-15-pro-max": {
    label: "iPhone 15 Pro Max",
    widthMm: 71.19,
    heightMm: 154.3,
  },
  "iphone-16": { label: "iPhone 16", widthMm: 65.11, heightMm: 141.16 },
  "iphone-16-plus": { label: "iPhone 16 Plus", widthMm: 71.19, heightMm: 154.3 },
  "iphone-16-pro": { label: "iPhone 16 Pro", widthMm: 66.87, heightMm: 145.38 },
  "iphone-16-pro-max": {
    label: "iPhone 16 Pro Max",
    widthMm: 72.85,
    heightMm: 158.28,
  },
  "iphone-17": { label: "iPhone 17", widthMm: 66.87, heightMm: 145.38 },
  "iphone-17-pro": { label: "iPhone 17 Pro", widthMm: 66.87, heightMm: 145.38 },
  "iphone-17-pro-max": {
    label: "iPhone 17 Pro Max",
    widthMm: 73.28,
    heightMm: 159.21,
  },
};

/** Fullscreen phone board: 4×9 inner corners → 5×10 squares. */
const IPHONE_BOARD_CORNER_COLS = 4;
const IPHONE_BOARD_CORNER_ROWS = 9;
const IPHONE_BOARD_SQUARE_COLS = IPHONE_BOARD_CORNER_COLS + 1;
const IPHONE_BOARD_SQUARE_ROWS = IPHONE_BOARD_CORNER_ROWS + 1;

function squareMmFromScreen(widthMm, heightMm) {
  return Math.min(
    widthMm / IPHONE_BOARD_SQUARE_COLS,
    heightMm / IPHONE_BOARD_SQUARE_ROWS
  );
}

function applyIphonePreset(modelId) {
  const screen = IPHONE_SCREENS[modelId];
  if (!screen) {
    iphonePresetHint.textContent =
      "Select a model if the board is a fullscreen checkerboard with 4×9 inner corners on that phone — corners and square size fill in from the display size.";
    return;
  }
  const squareMm = squareMmFromScreen(screen.widthMm, screen.heightMm);
  cornersColsInput.value = String(IPHONE_BOARD_CORNER_COLS);
  cornersRowsInput.value = String(IPHONE_BOARD_CORNER_ROWS);
  squareMmInput.value = String(Math.round(squareMm * 100) / 100);
  iphonePresetHint.textContent = `${screen.label} display ≈ ${screen.widthMm}×${screen.heightMm} mm → ${IPHONE_BOARD_SQUARE_COLS}×${IPHONE_BOARD_SQUARE_ROWS} squares → ${squareMmInput.value} mm/square (inner corners ${IPHONE_BOARD_CORNER_COLS}×${IPHONE_BOARD_CORNER_ROWS}).`;
}

function clearIphonePresetSelection() {
  if (iphoneModelSelect.value) {
    iphoneModelSelect.value = "";
    iphonePresetHint.textContent =
      "Select a model if the board is a fullscreen checkerboard with 4×9 inner corners on that phone — corners and square size fill in from the display size.";
  }
}

const statusEl = document.getElementById("status");
const stageBadge = document.getElementById("stage-badge");
const posePrompt = document.getElementById("pose-prompt");
const thumbsEl = document.getElementById("thumbs");
const resultsPanel = document.getElementById("results-panel");
const matrixKEl = document.getElementById("matrix-k");
const paramsOutEl = document.getElementById("params-out");
const distOutEl = document.getElementById("dist-out");

const alignValue = document.getElementById("align-value");
const alignBar = document.getElementById("align-bar");
const alignTip = document.getElementById("align-tip");

const metricStage = document.getElementById("metric-stage");
const metricPose = document.getElementById("metric-pose");
const metricCaptures = document.getElementById("metric-captures");
const metricDetected = document.getElementById("metric-detected");
const metricMethod = document.getElementById("metric-method");
const metricAlignParts = document.getElementById("metric-align-parts");
const metricHold = document.getElementById("metric-hold");
const metricSize = document.getElementById("metric-size");
const metricRms = document.getElementById("metric-rms");
const metricFov = document.getElementById("metric-fov");

const stepButtons = {
  setup: document.getElementById("step-setup"),
  capture: document.getElementById("step-capture"),
  results: document.getElementById("step-results"),
};

/** @type {'setup'|'capture'|'results'} */
let stage = "setup";
let cameraReady = false;
let capturing = false;
let poseIndex = 0;
let stableCount = 0;
let lastCaptureAt = 0;
let animFrameId = null;
let detecting = false;
let lastDetect = null;
let lastDetectAt = 0;

/**
 * /api/detect JPEG tiers (corners scaled back to full video size).
 * - preview: low-res alignment while moving into pose (capture session only)
 * - standard: default live detect / probe
 * - capture: full camera resolution for saved corners + calibrate
 *
 * Localhost stays snappy; remote hosts throttle interval + standard-tier size.
 * Interval may rise further from measured detect RTT (see adaptDetectFromLatency).
 */
const DETECT_TIERS = {
  preview: { maxWidth: 320, jpegQuality: 0.5 },
  standard: { maxWidth: 1280, jpegQuality: 0.6 },
  capture: { maxWidth: Infinity, jpegQuality: 0.92 },
};
const DETECT_LOCAL = {
  intervalMs: 120,
};
const DETECT_REMOTE = {
  intervalMs: 280,
  maxWidth: 640,
  jpegQuality: 0.6,
  /** Cap when adapting to high RTT (ms). */
  intervalCapMs: 700,
};

function isLocalHost() {
  const h = location.hostname;
  return (
    h === "localhost" ||
    h === "127.0.0.1" ||
    h === "[::1]" ||
    h === "::1"
  );
}

const IS_LOCAL_HOST = isLocalHost();
const DETECT_BASE = IS_LOCAL_HOST ? DETECT_LOCAL : DETECT_REMOTE;

/** Mutable; remote may raise intervalMs from RTT. */
let detectIntervalMs = DETECT_BASE.intervalMs;
/** @type {number | null} EMA of /api/detect round-trip (ms). */
let detectRttEma = null;

function adaptDetectFromLatency(rttMs) {
  if (IS_LOCAL_HOST || !Number.isFinite(rttMs) || rttMs < 0) return;
  detectRttEma =
    detectRttEma == null ? rttMs : detectRttEma * 0.7 + rttMs * 0.3;
  // Keep roughly one in-flight cadence: don't poll much faster than RTT.
  const target = Math.round(detectRttEma * 1.15);
  detectIntervalMs = Math.min(
    DETECT_REMOTE.intervalCapMs,
    Math.max(DETECT_REMOTE.intervalMs, target)
  );
}

/** @type {{ corners: number[], thumbUrl: string, poseId: string, pose?: object }[]} */
let captures = [];
/** @type {object | null} */
let calibration = null;
let imageWidth = 0;
let imageHeight = 0;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function setBadge(text, kind = "idle") {
  stageBadge.textContent = text;
  stageBadge.className = `stage-badge ${kind}`;
}

function boardConfig() {
  const cols = Math.round(Number(cornersColsInput.value));
  const rows = Math.round(Number(cornersRowsInput.value));
  const squareMm = Number(squareMmInput.value);
  if (!Number.isFinite(cols) || cols < 2 || !Number.isFinite(rows) || rows < 2) {
    throw new Error("Inner corners must be integers ≥ 2");
  }
  if (!Number.isFinite(squareMm) || squareMm <= 0) {
    throw new Error("Square size must be a positive number (mm)");
  }
  return { cols, rows, squareMm };
}

function setStage(next) {
  stage = next;
  metricStage.textContent =
    next === "setup" ? "Setup" : next === "capture" ? "Capture" : "Results";
  for (const [name, btn] of Object.entries(stepButtons)) {
    btn.classList.toggle("active", name === next);
  }
  if (next === "capture") stepButtons.capture.disabled = false;
  if (next === "results") stepButtons.results.disabled = false;
  resultsPanel.hidden = next !== "results" && !calibration;
}

function currentPose() {
  return POSES[poseIndex] || null;
}

function updateCaptureMetrics() {
  metricCaptures.textContent = `${captures.length} / ${POSES.length}`;
  computeBtn.disabled = captures.length < MIN_CAPTURES_TO_COMPUTE;
  resetBtn.disabled = captures.length === 0 && !calibration;
  const pose = currentPose();
  metricPose.textContent = pose
    ? `${poseIndex + 1}. ${pose.title}`
    : captures.length >= POSES.length
      ? "Done"
      : "—";
}

function showPosePrompt(pose, kind = "") {
  if (!pose) {
    posePrompt.hidden = true;
    return;
  }
  posePrompt.hidden = false;
  posePrompt.className = `pose-prompt${kind ? ` ${kind}` : ""}`;
  posePrompt.textContent = `${pose.title}: ${pose.instruction}`;
}

function addThumb(dataUrl) {
  const img = document.createElement("img");
  img.src = dataUrl;
  img.alt = `Capture ${captures.length}`;
  thumbsEl.appendChild(img);
}

function clearThumbs() {
  thumbsEl.replaceChildren();
}

function frameThumbUrl() {
  const t = document.createElement("canvas");
  const maxW = 160;
  const scale = Math.min(1, maxW / (video.videoWidth || maxW));
  t.width = Math.round((video.videoWidth || maxW) * scale);
  t.height = Math.round((video.videoHeight || maxW) * scale);
  const tctx = t.getContext("2d");
  tctx.drawImage(video, 0, 0, t.width, t.height);
  return t.toDataURL("image/jpeg", 0.7);
}

/**
 * JPEG encode settings for a detect tier (remote standard tier uses smaller JPEGs).
 * @param {'preview'|'standard'|'capture'} tier
 */
function detectTierOpts(tier) {
  const base = DETECT_TIERS[tier] || DETECT_TIERS.standard;
  if (tier === "capture" || tier === "preview" || IS_LOCAL_HOST) return base;
  if (tier === "standard") {
    return {
      maxWidth: Math.min(base.maxWidth, DETECT_REMOTE.maxWidth),
      jpegQuality: DETECT_REMOTE.jpegQuality,
    };
  }
  return base;
}

/**
 * Capture a JPEG blob of the current video frame for /api/detect or probe.
 * @param {{ maxWidth?: number, jpegQuality?: number }} [opts]
 * @returns {Promise<Blob>}
 */
function frameJpegBlob(opts = DETECT_TIERS.standard) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const limit = opts.maxWidth ?? DETECT_TIERS.standard.maxWidth;
  const jpegQuality = opts.jpegQuality ?? DETECT_TIERS.standard.jpegQuality;
  const maxW = Number.isFinite(limit) ? limit : vw;
  const scale = vw > maxW ? maxW / vw : 1;
  const c = document.createElement("canvas");
  c.width = Math.round(vw * scale);
  c.height = Math.round(vh * scale);
  const cctx = c.getContext("2d", { willReadFrequently: true });
  cctx.drawImage(video, 0, 0, c.width, c.height);
  return new Promise((resolve, reject) => {
    c.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Failed to encode frame"))),
      "image/jpeg",
      jpegQuality
    );
  });
}

/**
 * Scale corner coords from detect-resolution back to full video size.
 * Server already returns full-size coords when it resizes internally;
 * we send a possibly-downscaled JPEG, so scale up by video/jpeg ratio.
 * @param {object} detect
 * @param {number} jpegW
 * @param {number} jpegH
 */
function scaleDetectToVideo(detect, jpegW, jpegH) {
  if (!detect?.found || !detect.corners) return detect;
  const sx = imageWidth / jpegW;
  const sy = imageHeight / jpegH;
  if (Math.abs(sx - 1) < 1e-6 && Math.abs(sy - 1) < 1e-6) {
    return { ...detect, width: imageWidth, height: imageHeight };
  }
  const corners = detect.corners.map((v, i) =>
    i % 2 === 0 ? v * sx : v * sy
  );
  const centroid = detect.centroid
    ? { x: detect.centroid.x * sx, y: detect.centroid.y * sy }
    : null;
  const bounds = detect.bounds
    ? {
        x: detect.bounds.x * sx,
        y: detect.bounds.y * sy,
        w: detect.bounds.w * sx,
        h: detect.bounds.h * sy,
      }
    : null;
  let pose = detect.pose || null;
  if (pose?.centroid) {
    pose = {
      ...pose,
      centroid: { x: pose.centroid.x * sx, y: pose.centroid.y * sy },
      // rvec/tvec/depth estimated in JPEG space with rough K — re-estimate
      // diversity uses full-res corners via /api/diversity at capture time.
    };
  }
  return {
    ...detect,
    corners,
    centroid,
    bounds,
    pose,
    width: imageWidth,
    height: imageHeight,
  };
}

async function apiDetect(cols, rows, squareMm, tier = "standard") {
  const opts = detectTierOpts(tier);
  const blob = await frameJpegBlob(opts);
  const jpegW = Math.min(
    video.videoWidth,
    Number.isFinite(opts.maxWidth) ? opts.maxWidth : video.videoWidth
  );
  const jpegH = Math.round(video.videoHeight * (jpegW / video.videoWidth));

  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  form.append("cols", String(cols));
  form.append("rows", String(rows));
  form.append("squareMm", String(squareMm));
  const t0 = performance.now();
  const res = await fetch("/api/detect", { method: "POST", body: form });
  const data = await res.json();
  adaptDetectFromLatency(performance.now() - t0);
  if (!res.ok) throw new Error(data.error || res.statusText);
  return scaleDetectToVideo(data, jpegW, jpegH);
}

/**
 * Score alignment for the current pose from a detect result.
 * @param {object} detect
 * @param {{cols:number,rows:number,squareMm:number}} cfg
 */
function alignForDetect(detect, cfg) {
  if (!detect.found || !detect.centroid || !detect.corners) return null;
  const pose = capturing ? currentPose() : null;
  if (!pose) return null;
  return scoreAlignment(
    detect.centroid,
    detect.corners,
    pose,
    detect.width,
    detect.height,
    cfg.cols,
    cfg.rows
  );
}

/**
 * @param {number[]} corners
 * @param {{cols:number,rows:number,squareMm:number}} cfg
 */
async function checkDiversity(corners, cfg) {
  const existingPoses = captures
    .map((c) => c.pose)
    .filter(Boolean);
  const res = await fetch("/api/diversity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      cols: cfg.cols,
      rows: cfg.rows,
      squareMm: cfg.squareMm,
      imageWidth,
      imageHeight,
      corners,
      existingPoses,
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function applyCropGuards(track) {
  try {
    const caps = track.getCapabilities?.() || {};
    /** @type {MediaTrackConstraints} */
    const advanced = {};
    if ("zoom" in caps && Array.isArray(caps.zoom) === false && caps.zoom?.min != null) {
      advanced.zoom = caps.zoom.min;
    }
    // @ts-expect-error vendor
    if ("faceFraming" in caps) advanced.faceFraming = false;
    // @ts-expect-error vendor
    if ("pan" in caps && caps.pan?.min != null) advanced.pan = 0;
    // @ts-expect-error vendor
    if ("tilt" in caps && caps.tilt?.min != null) advanced.tilt = 0;
    if (Object.keys(advanced).length) {
      await track.applyConstraints({ advanced: [advanced] });
    }
  } catch (err) {
    console.warn("[camera] crop guards failed", err);
  }
}

async function startCamera() {
  setStatus("Requesting camera…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    video.srcObject = stream;
    await video.play();
    const track = stream.getVideoTracks()[0];
    if (track) await applyCropGuards(track);

    await new Promise((resolve) => {
      if (video.videoWidth) resolve();
      else video.onloadedmetadata = () => resolve();
    });

    imageWidth = video.videoWidth;
    imageHeight = video.videoHeight;
    canvas.width = imageWidth;
    canvas.height = imageHeight;
    metricSize.textContent = `${imageWidth} × ${imageHeight}`;
    cameraReady = true;
    captureBtn.disabled = false;
    probeBtn.disabled = false;
    setStatus(
      "Camera ready. Confirm board settings, then start capture and follow the pose prompts."
    );
    setBadge("Camera ready — start capture", "ok");
    setStage("setup");
    startLoop();
  } catch (err) {
    setStatus(`Camera error: ${err.message || err}`, true);
    setBadge("Camera failed", "err");
  }
}

function startCaptureSession() {
  try {
    boardConfig();
  } catch (err) {
    setStatus(err.message, true);
    return;
  }
  capturing = true;
  poseIndex = 0;
  stableCount = 0;
  lastCaptureAt = 0;
  captureBtn.disabled = true;
  skipBtn.disabled = false;
  snapBtn.disabled = true;
  boardInputs.classList.add("locked");
  iphoneModelSelect.disabled = true;
  setStage("capture");
  showPosePrompt(currentPose());
  setBadge(`Pose ${poseIndex + 1} / ${POSES.length}`, "warn");
  setStatus(
    "Green corners = board detected. Fill the guide until Alignment is high, then hold — or press Capture now."
  );
  updateCaptureMetrics();
  updateAlignUi(null);
}

function skipPose() {
  if (!capturing) return;
  poseIndex += 1;
  stableCount = 0;
  if (poseIndex >= POSES.length) {
    finishCapturePrompts();
    return;
  }
  showPosePrompt(currentPose());
  setBadge(`Pose ${poseIndex + 1} / ${POSES.length}`, "warn");
  updateCaptureMetrics();
}

function finishCapturePrompts() {
  capturing = false;
  skipBtn.disabled = true;
  snapBtn.disabled = true;
  showPosePrompt(null);
  if (captures.length >= MIN_CAPTURES_TO_COMPUTE) {
    setBadge("Enough captures — compute when ready", "ok");
    setStatus(
      `Captured ${captures.length} frames. Click “Compute intrinsics” (or capture more by resetting).`
    );
    computeBtn.disabled = false;
  } else {
    setBadge("Need more captures", "warn");
    setStatus(
      `Only ${captures.length} captures (need ≥ ${MIN_CAPTURES_TO_COMPUTE}). Reset and try again, or skip fewer poses.`
    );
    captureBtn.disabled = false;
  }
  updateCaptureMetrics();
}

/**
 * @param {object} detect
 * @param {object} [pose]
 */
function acceptCapture(detect, pose) {
  const current = currentPose();
  const thumbUrl = frameThumbUrl();
  captures.push({
    corners: detect.corners.slice(),
    thumbUrl,
    poseId: current?.id || `extra-${captures.length}`,
    pose: pose || detect.pose || null,
  });
  addThumb(thumbUrl);
  lastCaptureAt = performance.now();
  stableCount = 0;
  updateCaptureMetrics();

  poseIndex += 1;
  if (poseIndex >= POSES.length) {
    finishCapturePrompts();
    return;
  }
  showPosePrompt(currentPose(), "ready");
  setBadge(`Captured! Next: ${currentPose().title}`, "ok");
  setStatus(`Saved capture ${captures.length}. Move to the next pose.`);
}

/**
 * Accept after diversity check. Returns true if saved.
 * @param {object} detect
 */
async function tryAcceptCapture(detect) {
  let cfg;
  try {
    cfg = boardConfig();
  } catch (err) {
    setStatus(err.message, true);
    return false;
  }
  if (!detect?.found || !detect.corners) {
    setStatus("Board not detected — cannot capture yet.", true);
    return false;
  }
  const poseGuide = currentPose();
  if (poseGuide?.tiltHint && !detect.align?.ready) {
    setStatus(
      detect.align?.tip ||
        "Board is not tilted enough in the required direction yet.",
      true
    );
    return false;
  }

  try {
    const div = await checkDiversity(detect.corners, cfg);
    if (!div.ok) {
      stableCount = 0;
      metricHold.textContent = `0 / ${STABLE_FRAMES_REQUIRED}`;
      setBadge("Need a more different pose", "warn");
      setStatus(div.reason || "Pose too similar to an earlier capture.", true);
      alignTip.textContent = "Change position, distance, or tilt more";
      return false;
    }
    acceptCapture(detect, div.pose);
    return true;
  } catch (err) {
    setStatus(`Diversity check failed: ${err.message || err}`, true);
    return false;
  }
}

function resetCaptures() {
  captures = [];
  calibration = null;
  poseIndex = 0;
  stableCount = 0;
  capturing = false;
  clearThumbs();
  resultsPanel.hidden = true;
  downloadBtn.disabled = true;
  skipBtn.disabled = true;
  snapBtn.disabled = true;
  captureBtn.disabled = !cameraReady;
  boardInputs.classList.remove("locked");
  iphoneModelSelect.disabled = false;
  computeBtn.disabled = true;
  metricHold.textContent = "—";
  updateAlignUi(null);
  metricRms.textContent = "—";
  metricFov.textContent = "—";
  matrixKEl.textContent = "";
  paramsOutEl.textContent = "";
  distOutEl.textContent = "";
  showPosePrompt(null);
  setStage("setup");
  stepButtons.results.disabled = true;
  updateCaptureMetrics();
  setBadge(cameraReady ? "Camera ready — start capture" : "Reset", "idle");
  setStatus("Captures cleared. Start capture again when ready.");
}

function formatMatrix(K) {
  const fmt = (v) => v.toFixed(4).padStart(12);
  return [
    `[ ${fmt(K[0][0])} ${fmt(K[0][1])} ${fmt(K[0][2])} ]`,
    `[ ${fmt(K[1][0])} ${fmt(K[1][1])} ${fmt(K[1][2])} ]`,
    `[ ${fmt(K[2][0])} ${fmt(K[2][1])} ${fmt(K[2][2])} ]`,
  ].join("\n");
}

function showResults(result, cfg) {
  metricRms.textContent = `${result.rms.toFixed(4)} px`;
  metricFov.textContent = `${result.hfovDeg.toFixed(2)}° / ${result.vfovDeg.toFixed(2)}°`;
  matrixKEl.textContent = formatMatrix(result.cameraMatrix);
  const dropped = result.droppedIndices || [];
  const outlierLine = result.outlierRejectionApplied
    ? `outliers dropped = ${dropped.length} (kept ${result.numImages}/${result.numImagesInput})`
    : `outliers dropped = 0 (all ${result.numImagesInput} views kept)`;
  paramsOutEl.textContent = [
    `fx = ${result.fx.toFixed(4)}`,
    `fy = ${result.fy.toFixed(4)}`,
    `cx = ${result.cx.toFixed(4)}`,
    `cy = ${result.cy.toFixed(4)}`,
    ``,
    `HFOV = ${result.hfovDeg.toFixed(3)}°`,
    `VFOV = ${result.vfovDeg.toFixed(3)}°`,
    ``,
    `images = ${result.numImages}`,
    outlierLine,
    result.rmsInitial != null && result.outlierRejectionApplied
      ? `RMS initial → final = ${result.rmsInitial.toFixed(4)} → ${result.rms.toFixed(4)}`
      : null,
    `board = ${cfg.cols}×${cfg.rows} @ ${cfg.squareMm} mm`,
    `size = ${result.imageWidth}×${result.imageHeight}`,
  ]
    .filter((line) => line != null)
    .join("\n");
  distOutEl.textContent = result.distCoeffs
    .map((v, i) => `d[${i}] = ${Number(v).toExponential(6)}`)
    .join("\n");
}

async function runCalibration() {
  let cfg;
  try {
    cfg = boardConfig();
  } catch (err) {
    setStatus(err.message, true);
    return;
  }
  if (captures.length < MIN_CAPTURES_TO_COMPUTE) {
    setStatus(`Need at least ${MIN_CAPTURES_TO_COMPUTE} captures.`, true);
    return;
  }

  setStatus("Running calibrateCamera (Python OpenCV)…");
  setBadge("Computing…", "warn");
  computeBtn.disabled = true;

  try {
    const res = await fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cols: cfg.cols,
        rows: cfg.rows,
        squareMm: cfg.squareMm,
        imageWidth,
        imageHeight,
        captures: captures.map((c) => c.corners),
      }),
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.error || res.statusText);
    calibration = result;
    showResults(result, cfg);
    setStage("results");
    resultsPanel.hidden = false;
    downloadBtn.disabled = false;
    setBadge("Calibration complete", "ok");
    const dropN = (result.droppedIndices || []).length;
    const dropNote = dropN
      ? ` · dropped ${dropN} outlier view(s)`
      : "";
    setStatus(
      `Done. RMS=${result.rms.toFixed(3)} px · HFOV ${result.hfovDeg.toFixed(1)}° · VFOV ${result.vfovDeg.toFixed(1)}°${dropNote}`
    );
  } catch (err) {
    console.error(err);
    setStatus(`Calibration failed: ${err.message || err}`, true);
    setBadge("Calibration failed", "err");
    computeBtn.disabled = false;
  }
}

function downloadJson() {
  if (!calibration) return;
  const cfg = boardConfig();
  const payload = {
    createdAt: new Date().toISOString(),
    board: {
      innerCornersCols: cfg.cols,
      innerCornersRows: cfg.rows,
      squareSizeMm: cfg.squareMm,
    },
    imageSize: {
      width: calibration.imageWidth,
      height: calibration.imageHeight,
    },
    rms: calibration.rms,
    cameraMatrix: calibration.cameraMatrix,
    fx: calibration.fx,
    fy: calibration.fy,
    cx: calibration.cx,
    cy: calibration.cy,
    horizontalFovDeg: calibration.hfovDeg,
    verticalFovDeg: calibration.vfovDeg,
    distCoeffs: calibration.distCoeffs,
    numImages: calibration.numImages,
    numImagesInput: calibration.numImagesInput,
    keptIndices: calibration.keptIndices,
    droppedIndices: calibration.droppedIndices,
    perViewRms: calibration.perViewRms,
    outlierRejectionApplied: calibration.outlierRejectionApplied,
    poseIds: captures.map((c) => c.poseId),
    backend: "python-opencv",
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `webcam_intrinsics_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function updateAlignUi(align, detected = false) {
  if (!detected || !align) {
    alignValue.textContent = detected === false ? "—" : "0%";
    alignBar.style.width = "0%";
    alignBar.className = "align-meter-bar";
    alignTip.textContent = detected
      ? "Board detected — move toward the guide"
      : "Board not detected — check lighting, focus, and inner-corner counts";
    metricAlignParts.textContent = "—";
    return;
  }
  const pct = Math.round(align.score);
  alignValue.textContent = `${pct}%`;
  alignBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  alignBar.className =
    "align-meter-bar" +
    (pct >= ALIGN_SCORE_THRESHOLD ? " good" : pct >= 35 ? " mid" : "");
  alignTip.textContent = align.tip;
  const tiltBit =
    align.tiltOk == null
      ? ""
      : align.tiltOk
        ? ` · tilt ✓`
        : ` · tilt ${align.tilt?.direction || "?"}`;
  metricAlignParts.textContent = `${Math.round(align.positionScore)}% / ${Math.round(align.sizeScore)}%${tiltBit}`;
}

function drawCorners(corners, color) {
  ctx.fillStyle = color;
  for (let i = 0; i < corners.length; i += 2) {
    ctx.beginPath();
    ctx.arc(corners[i], corners[i + 1], 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawTiltHint(c, rect, hint) {
  c.strokeStyle = "rgba(255, 184, 110, 0.9)";
  c.fillStyle = "rgba(255, 184, 110, 0.9)";
  c.lineWidth = 2;
  const cx = rect.x + rect.w / 2;
  const cy = rect.y + rect.h / 2;
  c.beginPath();
  if (hint === "left") {
    c.moveTo(rect.x + rect.w * 0.15, cy);
    c.lineTo(rect.x + rect.w * 0.45, cy - 18);
    c.lineTo(rect.x + rect.w * 0.45, cy + 18);
  } else if (hint === "right") {
    c.moveTo(rect.x + rect.w * 0.85, cy);
    c.lineTo(rect.x + rect.w * 0.55, cy - 18);
    c.lineTo(rect.x + rect.w * 0.55, cy + 18);
  } else if (hint === "up") {
    c.moveTo(cx, rect.y + rect.h * 0.15);
    c.lineTo(cx - 18, rect.y + rect.h * 0.45);
    c.lineTo(cx + 18, rect.y + rect.h * 0.45);
  } else if (hint === "down") {
    c.moveTo(cx, rect.y + rect.h * 0.85);
    c.lineTo(cx - 18, rect.y + rect.h * 0.55);
    c.lineTo(cx + 18, rect.y + rect.h * 0.55);
  }
  c.closePath();
  c.fill();
}

function drawOverlay(detect) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pose = capturing ? currentPose() : null;
  const align = detect?.align || null;

  if (pose) {
    const rect = align?.guide || guideRect(pose, w, h);
    const ready = !!align?.ready;
    ctx.save();
    ctx.strokeStyle = ready
      ? "rgba(110, 231, 168, 0.95)"
      : detect?.found
        ? "rgba(255, 184, 110, 0.95)"
        : "rgba(110, 200, 255, 0.85)";
    ctx.lineWidth = ready ? 3 : 2;
    ctx.setLineDash(ready ? [] : [8, 6]);
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    ctx.setLineDash([]);
    ctx.fillStyle = ready
      ? "rgba(110, 231, 168, 0.12)"
      : "rgba(110, 200, 255, 0.08)";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

    const gcx = rect.x + rect.w / 2;
    const gcy = rect.y + rect.h / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(gcx - 12, gcy);
    ctx.lineTo(gcx + 12, gcy);
    ctx.moveTo(gcx, gcy - 12);
    ctx.lineTo(gcx, gcy + 12);
    ctx.stroke();

    if (pose.tiltHint) drawTiltHint(ctx, rect, pose.tiltHint);
    ctx.restore();
  }

  if (detect?.found && detect.corners) {
    const color = detect.align?.ready ? "#6ee7a8" : "#ffb86e";
    drawCorners(detect.corners, color);

    if (detect.bounds || detect.align?.board) {
      const b = detect.align?.board || detect.bounds;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(b.x, b.y, b.w, b.h);
    }

    if (detect.centroid && pose && detect.align) {
      const g = detect.align.guide;
      const gcx = g.x + g.w / 2;
      const gcy = g.y + g.h / 2;
      ctx.strokeStyle = "rgba(255,255,255,0.75)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(detect.centroid.x, detect.centroid.y);
      ctx.lineTo(gcx, gcy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(detect.centroid.x, detect.centroid.y, 6, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.save();
    ctx.font = "600 18px Segoe UI, system-ui, sans-serif";
    ctx.fillStyle = "rgba(10,12,16,0.65)";
    const label = detect.align
      ? `Detected · align ${Math.round(detect.align.score)}%`
      : "Detected";
    const pad = 8;
    const tw = ctx.measureText(label).width;
    ctx.fillRect(12, 12, tw + pad * 2, 28);
    ctx.fillStyle = color;
    ctx.fillText(label, 12 + pad, 32);
    ctx.restore();
  } else if (cameraReady) {
    ctx.save();
    ctx.font = "600 18px Segoe UI, system-ui, sans-serif";
    ctx.fillStyle = "rgba(10,12,16,0.65)";
    const label = "Looking for checkerboard…";
    const pad = 8;
    const tw = ctx.measureText(label).width;
    ctx.fillRect(12, 12, tw + pad * 2, 28);
    ctx.fillStyle = "#ff9b9b";
    ctx.fillText(label, 12 + pad, 32);
    ctx.restore();
  }
}

async function manualSnap() {
  if (!capturing || !lastDetect?.found || !lastDetect.corners) {
    setStatus("Board not detected — cannot capture yet.", true);
    return;
  }
  snapBtn.disabled = true;
  try {
    const cfg = boardConfig();
    const detect = await apiDetect(cfg.cols, cfg.rows, cfg.squareMm, "capture");
    const align = alignForDetect(detect, cfg);
    await tryAcceptCapture({ ...detect, align });
  } catch (err) {
    setStatus(`Capture failed: ${err.message || err}`, true);
  } finally {
    snapBtn.disabled = !(capturing && lastDetect?.found);
  }
}

async function tick() {
  animFrameId = requestAnimationFrame(tick);
  if (!cameraReady || !video.videoWidth) return;

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    imageWidth = video.videoWidth;
    imageHeight = video.videoHeight;
    canvas.width = imageWidth;
    canvas.height = imageHeight;
    metricSize.textContent = `${imageWidth} × ${imageHeight}`;
  }

  const now = performance.now();
  const dueForDetect = !detecting && now - lastDetectAt >= detectIntervalMs;

  if (!dueForDetect) {
    drawOverlay(lastDetect);
    return;
  }

  let cfg;
  try {
    cfg = boardConfig();
  } catch {
    drawOverlay(null);
    return;
  }

  detecting = true;
  lastDetectAt = now;
  try {
    const detectTier = capturing ? "preview" : "standard";
    const detect = await apiDetect(cfg.cols, cfg.rows, cfg.squareMm, detectTier);

    const align = alignForDetect(detect, cfg);

    lastDetect = { ...detect, align };
    snapBtn.disabled = !(capturing && detect.found);

    if (detect.found) {
      metricDetected.textContent = align?.ready
        ? "Yes · ready"
        : capturing
          ? "Yes · adjust"
          : "Yes";
      metricMethod.textContent = detect.method || "—";
      if (align) updateAlignUi(align, true);
      else {
        updateAlignUi(null, true);
        alignTip.textContent = "Board detected (start capture for alignment guide)";
      }
    } else {
      metricDetected.textContent = "No";
      metricMethod.textContent = detect.method || "none";
      updateAlignUi(null, false);
      snapBtn.disabled = true;
    }

    if (capturing && detect.found && align?.ready) {
      if (now - lastCaptureAt > CAPTURE_COOLDOWN_MS) {
        stableCount += 1;
        const remain = Math.max(0, STABLE_FRAMES_REQUIRED - stableCount);
        metricHold.textContent = `${stableCount} / ${STABLE_FRAMES_REQUIRED}`;
        showPosePrompt(currentPose(), remain <= 2 ? "ready" : "hold");
        setBadge(
          remain > 0 ? `Hold steady… ${remain}` : "Capturing…",
          remain > 0 ? "warn" : "ok"
        );
        if (stableCount >= STABLE_FRAMES_REQUIRED) {
          const captureDetect = await apiDetect(
            cfg.cols,
            cfg.rows,
            cfg.squareMm,
            "capture"
          );
          const captureAlign = alignForDetect(captureDetect, cfg) || align;
          if (!captureDetect.found || !captureDetect.corners) {
            stableCount = 0;
            metricHold.textContent = `0 / ${STABLE_FRAMES_REQUIRED}`;
            setBadge("Full-res detect failed — hold steady", "warn");
            setStatus(
              "Board lost at full resolution — adjust and hold steady again.",
              true
            );
          } else {
            await tryAcceptCapture({ ...captureDetect, align: captureAlign });
          }
        }
      }
    } else if (capturing) {
      stableCount = 0;
      metricHold.textContent = `0 / ${STABLE_FRAMES_REQUIRED}`;
      showPosePrompt(currentPose());
      if (!detect.found) {
        setBadge(`Show board · pose ${poseIndex + 1}/${POSES.length}`, "warn");
      } else if (align) {
        setBadge(align.tip, "warn");
      } else {
        setBadge("Move board into the guide", "warn");
      }
    }

    drawOverlay(lastDetect);
  } catch (err) {
    console.warn("[detect]", err);
    metricDetected.textContent = "Error";
    updateAlignUi(null, false);
  } finally {
    detecting = false;
  }
}

function startLoop() {
  if (animFrameId != null) cancelAnimationFrame(animFrameId);
  animFrameId = requestAnimationFrame(tick);
}

iphoneModelSelect.addEventListener("change", () => {
  applyIphonePreset(iphoneModelSelect.value);
});

for (const el of [cornersColsInput, cornersRowsInput, squareMmInput]) {
  el.addEventListener("input", () => {
    if (!iphoneModelSelect.value) return;
    const screen = IPHONE_SCREENS[iphoneModelSelect.value];
    if (!screen) return;
    const expectedSquare = squareMmFromScreen(screen.widthMm, screen.heightMm);
    const cols = Math.round(Number(cornersColsInput.value));
    const rows = Math.round(Number(cornersRowsInput.value));
    const squareMm = Number(squareMmInput.value);
    const matchesPreset =
      cols === IPHONE_BOARD_CORNER_COLS &&
      rows === IPHONE_BOARD_CORNER_ROWS &&
      Math.abs(squareMm - expectedSquare) < 0.05;
    if (!matchesPreset) clearIphonePresetSelection();
  });
}

startBtn.addEventListener("click", () => {
  startBtn.disabled = true;
  startCamera().finally(() => {
    if (!cameraReady) startBtn.disabled = false;
  });
});
captureBtn.addEventListener("click", startCaptureSession);
snapBtn.addEventListener("click", manualSnap);
skipBtn.addEventListener("click", skipPose);
computeBtn.addEventListener("click", runCalibration);
resetBtn.addEventListener("click", resetCaptures);
downloadBtn.addEventListener("click", () => {
  try {
    downloadJson();
  } catch (err) {
    setStatus(err.message, true);
  }
});

selftestBtn.addEventListener("click", async () => {
  selftestBtn.disabled = true;
  selftestResult.hidden = false;
  selftestResult.className = "hint selftest-result";
  selftestResult.textContent = "Running synthetic-board self-test (Python OpenCV)…";
  try {
    const cfg = boardConfig();
    const res = await fetch("/api/selftest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cols: cfg.cols, rows: cfg.rows }),
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.error || res.statusText);
    selftestPreview.hidden = false;
    selftestPreview.src = result.previewDataUrl;
    selftestPreview.alt = `Synthetic ${cfg.cols}×${cfg.rows} checkerboard`;
    selftestResult.className =
      "hint selftest-result " + (result.ok ? "ok" : "err");
    selftestResult.textContent = result.message;
    setStatus(result.message, !result.ok);
    setBadge(result.ok ? "Detector OK" : "Detector failed", result.ok ? "ok" : "err");
  } catch (err) {
    selftestResult.className = "hint selftest-result err";
    selftestResult.textContent = String(err.message || err);
    setStatus(String(err.message || err), true);
  } finally {
    selftestBtn.disabled = false;
  }
});

probeBtn.addEventListener("click", async () => {
  if (!cameraReady || !video.videoWidth) {
    setStatus("Start the camera first.", true);
    return;
  }
  probeBtn.disabled = true;
  setStatus("Probing common board sizes on the current frame…");
  try {
    const cfg = boardConfig();
    const blob = await frameJpegBlob(detectTierOpts("standard"));
    const form = new FormData();
    form.append("image", blob, "frame.jpg");
    form.append("cols", String(cfg.cols));
    form.append("rows", String(cfg.rows));
    const res = await fetch("/api/probe", { method: "POST", body: form });
    const { best, tried, error } = await res.json();
    if (!res.ok) throw new Error(error || res.statusText);
    const hits = (tried || []).filter((t) => t.found);
    if (best) {
      clearIphonePresetSelection();
      cornersColsInput.value = String(best.cols);
      cornersRowsInput.value = String(best.rows);
      const msg = `Live probe found a board at ${best.cols}×${best.rows} inner corners (${hits.length} matching size(s)). Inputs updated — try capture again.`;
      setStatus(msg);
      setBadge(`Found ${best.cols}×${best.rows}`, "ok");
      selftestResult.hidden = false;
      selftestResult.className = "hint selftest-result ok";
      selftestResult.textContent = msg;
    } else {
      const msg = `Live probe found no checkerboard among ${(tried || []).length} sizes. Run “Test detector” — if that passes, your corner counts/lighting/focus are the issue.`;
      setStatus(msg, true);
      setBadge("Probe: nothing found", "err");
      selftestResult.hidden = false;
      selftestResult.className = "hint selftest-result err";
      selftestResult.textContent = msg;
    }
  } catch (err) {
    setStatus(String(err.message || err), true);
  } finally {
    probeBtn.disabled = !cameraReady;
  }
});

updateCaptureMetrics();
updateAlignUi(null, false);
setStatus(
  "Enter your board’s inner corner grid and square size, then start the camera. Use “Test detector” to verify Python OpenCV."
);
