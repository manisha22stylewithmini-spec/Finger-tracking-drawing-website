import { HandLandmarker, ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const drawCanvas = document.getElementById("draw");
const overlay = document.getElementById("overlay");
const person = document.getElementById("person");
const pctx = person.getContext("2d");
const maskCanvas = document.createElement("canvas");
const mctx = maskCanvas.getContext("2d");
let segmenter = null;
let maskReady = false;
const dctx = drawCanvas.getContext("2d");
const octx = overlay.getContext("2d");
const modeText = document.getElementById("modeText");
const modeDot = document.getElementById("modeDot");
const startBtn = document.getElementById("startBtn");
const startMsg = document.getElementById("startMsg");

const bgImg = new Image();
bgImg.src = "bg.png";

// ---------- state ----------
const state = {
  color: "#ff3b6b",
  size: 10,
  glow: true,
  cutout: true,
  hue: 0,
  mode: "none",        // draw | hover | erase | idle | none
  last: null,          // last drawn point
  smooth: null,        // smoothed fingertip
  pending: { mode: "none", count: 0 },
  handSize: 100,
  mid: null,
  hoverEl: null,
  hoverStart: 0,
};
const undoStack = [];
const MODE_COLORS = { draw: "#3ee07a", hover: "#2ec5ff", erase: "#ffb020", idle: "#a9a9bd", none: "#a9a9bd" };
const MODE_LABELS = { draw: "Drawing", hover: "Moving · hover a button to pick", erase: "Erasing", idle: "Paused", none: "Waiting for hand…" };
const DWELL_MS = 700;
let dpr = 1;
let handLandmarker = null;
let lastVideoTime = -1;

// ---------- canvas sizing ----------
function resize() {
  dpr = 1; // draw at CSS resolution: much lighter, and the 1:1 canvas is stretched by the browser
  const w = window.innerWidth, h = window.innerHeight;
  // keep the existing drawing when the window changes size
  const copy = document.createElement("canvas");
  copy.width = drawCanvas.width; copy.height = drawCanvas.height;
  if (copy.width && copy.height) copy.getContext("2d").drawImage(drawCanvas, 0, 0);
  for (const c of [drawCanvas, overlay, person]) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  if (copy.width && copy.height) dctx.drawImage(copy, 0, 0);
  dctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize);
resize();

// Map a normalized landmark to screen coords, matching the mirrored object-fit: cover video.
function toScreen(lm) {
  const W = window.innerWidth, H = window.innerHeight;
  const vw = video.videoWidth || W, vh = video.videoHeight || H;
  const scale = Math.max(W / vw, H / vh);
  const dw = vw * scale, dh = vh * scale;
  return { x: (W - dw) / 2 + (1 - lm.x) * dw, y: (H - dh) / 2 + lm.y * dh };
}

// ---------- gesture detection ----------
const d2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function fingersUp(lm) {
  // a finger is "up" when its tip is clearly farther from the wrist than its middle joint,
  // which works at any hand angle (not just when the hand points straight up)
  const ext = (tip, pip) => d2(lm[tip], lm[0]) > d2(lm[pip], lm[0]) * 1.12;
  return { index: ext(8, 6), middle: ext(12, 10), ring: ext(16, 14), pinky: ext(20, 18) };
}

function classify(lm) {
  const f = fingersUp(lm);
  if (f.index && f.middle && f.ring && f.pinky) return "erase";
  if (f.index && f.middle && !f.ring && !f.pinky) return "hover";
  if (f.index && !f.middle && !f.ring && !f.pinky) return "draw";
  return "idle";
}

// require a few stable frames before switching modes, so lines don't break from jitter
function stabilize(raw) {
  if (raw === state.mode) { state.pending = { mode: raw, count: 0 }; return state.mode; }
  if (raw === state.pending.mode) state.pending.count++;
  else state.pending = { mode: raw, count: 1 };
  const needed = raw === "draw" ? 2 : raw === "erase" ? 2 : state.mode === "draw" ? 5 : 3;
  return state.pending.count >= needed ? raw : state.mode;
}

function setMode(mode) {
  if (mode === state.mode) return;
  if (state.mode === "draw" || state.mode === "erase") { state.last = null; state.mid = null; }
  if (mode === "draw" || mode === "erase") pushUndo();
  state.mode = mode;
  modeText.textContent = MODE_LABELS[mode];
  modeDot.style.background = MODE_COLORS[mode];
}

// ---------- drawing ----------
function strokeColor() {
  if (state.color !== "rainbow") return state.color;
  state.hue = (state.hue + 3) % 360;
  return `hsl(${state.hue} 100% 60%)`;
}

function drawSegment(from, to, color, size) {
  dctx.save();
  dctx.globalCompositeOperation = "source-over";
  dctx.strokeStyle = color;
  dctx.lineWidth = size;
  dctx.lineCap = "round";
  dctx.lineJoin = "round";
  if (state.glow) {
    dctx.globalAlpha = 0.22;
    dctx.lineWidth = size * 2.4;
    dctx.beginPath(); dctx.moveTo(from.x, from.y); dctx.lineTo(to.x, to.y); dctx.stroke();
    dctx.globalAlpha = 1;
    dctx.lineWidth = size;
  }
  dctx.beginPath();
  dctx.moveTo(from.x, from.y);
  dctx.lineTo(to.x, to.y);
  dctx.stroke();
  dctx.restore();
}

function drawCurve(tip, color, size) {
  const last = state.last;
  const newMid = { x: (last.x + tip.x) / 2, y: (last.y + tip.y) / 2 };
  const from = state.mid || last;
  dctx.save();
  dctx.strokeStyle = color;
  dctx.lineWidth = size;
  dctx.lineCap = "round";
  dctx.lineJoin = "round";
  dctx.beginPath();
  dctx.moveTo(from.x, from.y);
  dctx.quadraticCurveTo(last.x, last.y, newMid.x, newMid.y);
  if (state.glow) {
    dctx.globalAlpha = 0.22;
    dctx.lineWidth = size * 2.4;
    dctx.stroke();
    dctx.globalAlpha = 1;
    dctx.lineWidth = size;
  }
  dctx.stroke();
  dctx.restore();
  state.mid = newMid;
  state.last = tip;
}

// One Euro filter: heavy smoothing when the finger is slow (kills jitter),
// light smoothing when it moves fast (no lag).
class OneEuro {
  constructor(minCutoff = 0.8, beta = 0.012) { this.minCutoff = minCutoff; this.beta = beta; this.x = null; this.dx = 0; }
  static alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  filter(v, dt) {
    if (this.x === null) { this.x = v; return v; }
    const rawD = (v - this.x) / dt;
    this.dx += OneEuro.alpha(1, dt) * (rawD - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (v - this.x);
    return this.x;
  }
}
const fx = new OneEuro(), fy = new OneEuro();
let lastT = 0;

function eraseAt(p, radius) {
  dctx.save();
  dctx.globalCompositeOperation = "destination-out";
  dctx.lineCap = "round";
  dctx.lineWidth = radius * 2;
  dctx.beginPath();
  dctx.moveTo((state.last || p).x, (state.last || p).y);
  dctx.lineTo(p.x, p.y);
  dctx.stroke();
  dctx.restore();
  state.last = p;
}

function pushUndo() {
  undoStack.push(dctx.getImageData(0, 0, drawCanvas.width, drawCanvas.height));
  if (undoStack.length > 10) undoStack.shift();
}

function undo() {
  const img = undoStack.pop();
  if (!img) return;
  dctx.save();
  dctx.setTransform(1, 0, 0, 1, 0, 0);
  dctx.putImageData(img, 0, 0);
  dctx.restore();
}

function clearAll() {
  pushUndo();
  dctx.save();
  dctx.setTransform(1, 0, 0, 1, 0, 0);
  dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  dctx.restore();
}

function save() {
  const out = document.createElement("canvas");
  out.width = drawCanvas.width; out.height = drawCanvas.height;
  const c = out.getContext("2d");
  c.fillStyle = "#fff";
  c.fillRect(0, 0, out.width, out.height);
  if (bgImg.complete && bgImg.naturalWidth) {
    const k = Math.max(out.width / bgImg.naturalWidth, out.height / bgImg.naturalHeight);
    const w = bgImg.naturalWidth * k, h = bgImg.naturalHeight * k;
    c.drawImage(bgImg, (out.width - w) / 2, (out.height - h) / 2, w, h);
  }
  c.drawImage(drawCanvas, 0, 0);
  const a = document.createElement("a");
  a.download = `air-draw-${Date.now()}.png`;
  a.href = out.toDataURL("image/png");
  a.click();
}

// ---------- toolbar ----------
function activate(btn) {
  if (btn.dataset.color) {
    state.color = btn.dataset.color;
    document.querySelectorAll(".swatch").forEach(b => b.classList.toggle("active", b === btn));
  } else if (btn.dataset.size) {
    state.size = Number(btn.dataset.size);
    document.querySelectorAll("[data-size]").forEach(b => b.classList.toggle("active", b === btn));
  } else {
    switch (btn.dataset.action) {
      case "glow": state.glow = !state.glow; btn.classList.toggle("active", state.glow); break;
      case "camera": state.cutout = !state.cutout; btn.classList.toggle("active", state.cutout); break;
      case "undo": undo(); break;
      case "clear": clearAll(); break;
      case "save": save(); break;
    }
  }
}
document.getElementById("toolbar").addEventListener("click", e => {
  const btn = e.target.closest("button");
  if (btn) activate(btn);
});

// Two-finger hover: dwell over a toolbar button to press it.
function handleHover(p, now) {
  const el = document.elementFromPoint(p.x, p.y)?.closest(".toolbar button") || null;
  if (el !== state.hoverEl) {
    state.hoverEl?.classList.remove("hovered");
    el?.classList.add("hovered");
    state.hoverEl = el;
    state.hoverStart = now;
    return 0;
  }
  if (!el) return 0;
  const progress = (now - state.hoverStart) / DWELL_MS;
  if (progress >= 1) {
    activate(el);
    state.hoverStart = now + 600; // cool-down before it can fire again
    return 1;
  }
  return Math.max(0, progress);
}

function clearHover() {
  state.hoverEl?.classList.remove("hovered");
  state.hoverEl = null;
}

// ---------- overlay ----------
function drawOverlay(lm, tip, palm, progress) {
  const W = window.innerWidth, H = window.innerHeight;
  octx.clearRect(0, 0, W, H);
  if (!lm) return;

  // faint hand skeleton
  const pts = lm.map(toScreen);
  const bones = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  octx.strokeStyle = "rgba(255,255,255,0.8)";
  octx.lineWidth = 2;
  octx.beginPath();
  for (const [a, b] of bones) { octx.moveTo(pts[a].x, pts[a].y); octx.lineTo(pts[b].x, pts[b].y); }
  octx.stroke();
  octx.fillStyle = "#fff";
  for (const p of pts) { octx.beginPath(); octx.arc(p.x, p.y, 2.5, 0, Math.PI * 2); octx.fill(); }


  // cursor
  if (state.mode === "erase") {
    octx.strokeStyle = MODE_COLORS.erase;
    octx.lineWidth = 2;
    octx.fillStyle = "rgba(255,176,32,0.15)";
    octx.setLineDash([8, 8]);
    octx.beginPath(); octx.arc(palm.x, palm.y, eraserRadius(), 0, Math.PI * 2); octx.fill(); octx.stroke();
    octx.setLineDash([]);
  } else if (tip) {
    const color = state.color === "rainbow" ? `hsl(${state.hue} 100% 60%)` : state.color;
    octx.fillStyle = state.mode === "draw" ? color : "transparent";
    octx.strokeStyle = state.mode === "draw" ? "#222" : MODE_COLORS[state.mode];
    octx.lineWidth = 2;
    octx.beginPath(); octx.arc(tip.x, tip.y, Math.max(6, state.size / 2 + 2), 0, Math.PI * 2);
    octx.fill(); octx.stroke();
    if (progress > 0) {
      octx.strokeStyle = MODE_COLORS.hover;
      octx.lineWidth = 4;
      octx.beginPath();
      octx.arc(tip.x, tip.y, 20, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
      octx.stroke();
    }
  }
}

const eraserRadius = () => Math.min(170, Math.max(75, state.handSize * 1.3));

// ---------- me, in front of the background ----------
let maskFrame = 0;
let maskInvert = null;
function updateMask(now) {
  // the cut-out only needs refreshing every 3rd frame; the video itself is drawn every frame
  if (maskReady && ++maskFrame % 3) return;
  segmenter.segmentForVideo(video, now, res => {
    const m = res.confidenceMasks?.[0];
    if (!m) return;
    const w = m.width, h = m.height, arr = m.getAsFloat32Array();
    if (maskInvert === null) {
      // which side of the mask is "person"? the centre is usually me, the top corners are not
      let c = 0, cn = 0, e = 0, en = 0;
      for (let y = 0; y < h; y += 8) for (let x = 0; x < w; x += 8) {
        const v = arr[y * w + x];
        if (x > w * 0.3 && x < w * 0.7 && y > h * 0.3 && y < h * 0.8) { c += v; cn++; }
        else if ((x < w * 0.15 || x > w * 0.85) && y < h * 0.4) { e += v; en++; }
      }
      maskInvert = c / cn < e / en;
    }
    if (maskCanvas.width !== w || maskCanvas.height !== h) { maskCanvas.width = w; maskCanvas.height = h; }
    if (!updateMask.img || updateMask.img.width !== w) updateMask.img = mctx.createImageData(w, h);
    const d = updateMask.img.data;
    for (let i = 0, j = 3; i < arr.length; i++, j += 4) {
      const v = maskInvert ? 1 - arr[i] : arr[i];
      d[j] = v <= 0.35 ? 0 : v >= 0.65 ? 255 : (v - 0.35) * 850;
    }
    mctx.putImageData(updateMask.img, 0, 0);
    maskReady = true;
  });
}

function composePerson(now) {
  const W = window.innerWidth, H = window.innerHeight;
  const vw = video.videoWidth || W, vh = video.videoHeight || H;
  const k = Math.max(W / vw, H / vh);
  const dw = vw * k, dh = vh * k, ox = (W - dw) / 2, oy = (H - dh) / 2;
  const cut = state.cutout && segmenter;
  if (cut) updateMask(now);
  pctx.clearRect(0, 0, W, H);
  if (cut && !maskReady) return;
  pctx.save();
  pctx.translate(W, 0);
  pctx.scale(-1, 1); // mirror
  pctx.drawImage(video, ox, oy, dw, dh);
  if (cut) {
    pctx.globalCompositeOperation = "destination-in";
    pctx.drawImage(maskCanvas, ox, oy, dw, dh);
  }
  pctx.restore();
}

// ---------- main loop ----------
function loop() {
  requestAnimationFrame(loop);
  if (!handLandmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const now = performance.now();
  composePerson(now);
  const res = handLandmarker.detectForVideo(video, now);
  const lm = res.landmarks?.[0];

  if (!lm) {
    setMode("none");
    state.smooth = null;
    fx.x = fy.x = null;
    clearHover();
    drawOverlay(null);
    return;
  }

  setMode(stabilize(classify(lm)));

  const dt = Math.min(0.1, Math.max(0.005, (now - lastT) / 1000));
  lastT = now;
  const raw = toScreen(lm[8]);
  state.smooth = { x: fx.filter(raw.x, dt), y: fy.filter(raw.y, dt) };
  const tip = state.smooth;
  const palm = toScreen({ x: (lm[0].x + lm[9].x) / 2, y: (lm[0].y + lm[9].y) / 2 });
  state.handSize += (d2(toScreen(lm[0]), toScreen(lm[9])) - state.handSize) * 0.2;

  let progress = 0;
  if (state.mode === "draw") {
    if (state.last) drawCurve(tip, strokeColor(), state.size);
    else { state.last = tip; state.mid = null; }
    clearHover();
  } else if (state.mode === "erase") {
    eraseAt(palm, eraserRadius());
    clearHover();
  } else if (state.mode === "hover") {
    progress = handleHover(tip, now);
  } else {
    clearHover();
  }

  drawOverlay(lm, tip, palm, progress);
}

// ---------- mouse / touch fallback ----------
let pointerDown = false;
drawCanvas.addEventListener("pointerdown", e => {
  pointerDown = true;
  pushUndo();
  state.last = { x: e.clientX, y: e.clientY };
});
drawCanvas.addEventListener("pointermove", e => {
  if (!pointerDown) return;
  const p = { x: e.clientX, y: e.clientY };
  drawSegment(state.last, p, strokeColor(), state.size);
  state.last = p;
});
window.addEventListener("pointerup", () => { pointerDown = false; state.last = null; });

// ---------- startup ----------
async function loadSegmenter(vision) {
  try {
    segmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    });
  } catch (err) {
    console.warn("Background cut-out unavailable, showing the plain camera", err);
    segmenter = null;
  }
}

async function start() {
  startBtn.disabled = true;
  startMsg.textContent = "Loading hand tracking model…";
  try {
    const [stream, vision] = await Promise.all([
      navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: "user" }, audio: false }),
      FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"),
    ]);
    video.srcObject = stream;
    await video.play();

    const options = {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    };
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, options);
    } catch {
      options.baseOptions.delegate = "CPU"; // some machines have no WebGL GPU support
      handLandmarker = await HandLandmarker.createFromOptions(vision, options);
    }

    document.getElementById("start").classList.add("gone");
    requestAnimationFrame(loop);
    loadSegmenter(vision); // not awaited: drawing starts right away
  } catch (err) {
    console.error(err);
    startBtn.disabled = false;
    startMsg.textContent = err.name === "NotAllowedError"
      ? "Camera permission was blocked. Allow camera access and try again."
      : `Couldn't start: ${err.message}`;
  }
}
startBtn.addEventListener("click", start);
