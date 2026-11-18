// EchoPlane3D — zero-build front end. Every dependency is vendored under ./vendor (no network needed).
// Educational / research prototype: nothing shown here is a clinical measurement.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/* ───────────────────────────── helpers ───────────────────────────── */
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, d = 1) => (v == null || Number.isNaN(Number(v)) ? '–' : Number(v).toFixed(d));
const signed = (v, d = 1) => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(d);
const V3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const CLEAR_COLOR = 0x0b0f19; // opaque viewport background, re-applied on every resize and frame

async function request(url, options) {
  const res = await fetch(url, options);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error body */ }
  if (!res.ok) {
    const detail = body && body.detail;
    const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : res.statusText;
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return body;
}
const api = {
  get: (url) => request(url),
  post: (url, data) => request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
};

function toast(message, tone = 'info') {
  const el = document.createElement('div');
  const tones = { info: 'border-sky-500/50 text-sky-100', error: 'border-rose-500/60 text-rose-100', ok: 'border-emerald-500/50 text-emerald-100', warn: 'border-amber-500/60 text-amber-100' };
  el.className = `hud ${tones[tone] || tones.info} text-[12.5px] shadow-lg max-w-[80vw]`;
  el.textContent = message;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), tone === 'error' || tone === 'warn' ? 7000 : 3200);
}

function setStatus(text, tone = 'ok') {
  $('statusText').textContent = text;
  $('statusDot').className = 'w-2 h-2 rounded-full ' + ({ ok: 'bg-emerald-400', busy: 'bg-amber-400 animate-pulse', error: 'bg-rose-500' }[tone] || 'bg-emerald-400');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
// escape first, then allow only **bold** and line breaks
const richText = (s) => escapeHtml(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');

/* ───────────────────────────── state ───────────────────────────── */
const state = {
  samples: [],
  subjects: [],
  sample: null,
  frames: [],
  frameIdx: 0,
  playing: false,
  speed: 1,
  seg: null,
  busy: false,
  overlays: { mask: true, axis: true, chords: true },
  prompt: { autoSnap: true, multi: false, points: [], labels: [] },
  view: 'A4C',
  camMode: 'anatomic',
  tilt: 0,
  cutaway: true,
  showFan: true,
  showLabels: true,
  beat: true,
  planes: null,
  fore: null,
  quiz: { index: 0, asked: 0, correct: 0 },
  engine: null,
  loadToken: 0,
};
const unitOf = (s) => (s && s.volume_unit === 'index' ? 'idx' : 'mL');

/* ═════════════════════════════ 2D ULTRASOUND ═════════════════════════════ */
const canvas = $('usCanvas');
const ctx = canvas.getContext('2d');

function subjectOf(sample) {
  return state.subjects.find((s) => s.id === (sample.subject_id || sample.id)) || null;
}

async function loadSamples(selectId = null) {
  const data = await api.get('/api/samples');
  state.samples = data.samples;
  state.subjects = data.subjects || data.samples.map((s) => ({ id: s.id, name: s.name, kind: s.kind, views: { [s.view || 'A4C']: s.id } }));
  const sel = $('subjectSelect');
  sel.innerHTML = '';
  const groups = { phantom: 'Calibrated phantoms', camus: 'CAMUS (real, reference-annotated)', upload: 'Your uploads' };
  const made = {};
  for (const subj of state.subjects) {
    const key = subj.kind in groups ? subj.kind : 'camus';
    if (!made[key]) { made[key] = document.createElement('optgroup'); made[key].label = groups[key]; sel.appendChild(made[key]); }
    const opt = document.createElement('option');
    opt.value = subj.id;
    opt.textContent = subj.name;
    made[key].appendChild(opt);
  }
  if (!state.samples.length) throw new Error('No samples found — run scripts/prepare_samples.py');
  await selectSample(selectId || state.samples[0].id);
}

async function selectSubject(subjectId) {
  const subj = state.subjects.find((s) => s.id === subjectId);
  if (!subj) return;
  const want = state.view === 'A2C' && subj.views.A2C ? 'A2C' : (subj.views.A4C ? 'A4C' : Object.keys(subj.views)[0]);
  await selectSample(subj.views[want]);
}

async function selectSample(id, { syncView = true } = {}) {
  const sample = state.samples.find((s) => s.id === id);
  if (!sample) return;
  const token = ++state.loadToken;
  pause();
  state.sample = sample;
  state.seg = null;
  state.fore = null;
  state.frames = [];
  state.frameIdx = 0;
  state.prompt.points = [];
  state.prompt.labels = [];
  updateMultiBar();
  canvas.width = sample.width;
  canvas.height = sample.height;
  const slider = $('frameSlider');
  slider.max = String(sample.n_frames - 1);
  slider.value = '0';
  const subj = subjectOf(sample);
  $('subjectSelect').value = subj ? subj.id : sample.id;
  document.querySelectorAll('#seqButtons button').forEach((b) => {
    b.classList.toggle('active', b.dataset.seq === sample.view);
    b.disabled = !(subj && subj.views[b.dataset.seq]);
  });
  const gt = sample.ground_truth;
  const u = unitOf(sample);
  $('sampleInfo').textContent =
    `${sample.view} · ${sample.width}×${sample.height} px · ${sample.n_frames} frames @ ${fmt(sample.fps, 1)} fps · ` +
    (sample.calibrated === false ? 'uncalibrated (volume index)' : `${fmt(sample.mm_per_pixel, 3)} mm/px`) +
    (sample.image_quality ? ` · quality ${sample.image_quality}` : '') +
    (gt ? ` · reference EDV ${fmt(gt.edv_ml, 0)} / ESV ${fmt(gt.esv_ml, 0)} ${u}, LVEF ${fmt(gt.lvef_pct, 1)} %` : '');
  document.querySelectorAll('.unit').forEach((el) => { el.textContent = u; });
  $('hudVolumeLabel').textContent = sample.calibrated === false ? 'LV volume index' : 'LV volume';
  $('usHint').classList.remove('hidden');
  $('qualityBadge').classList.add('hidden');
  $('btnED').disabled = $('btnES').disabled = true;
  $('runStats').classList.add('hidden');
  setStatus('Loading frames…', 'busy');
  drawFrame();
  updateHud();
  updateEstimates();
  renderPlot();
  refreshForeshortening();
  updateContraction();
  if (syncView && (sample.view === 'A4C' || sample.view === 'A2C') && state.planes && state.view !== sample.view
      && (state.view === 'A4C' || state.view === 'A2C')) {
    setView(sample.view, { sync: false });
  }
  updateSyncBadge();

  const frames = await Promise.all(
    Array.from({ length: sample.n_frames }, (_, i) => new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = sample.frame_url_template.replace('{index}', String(i));
    })),
  );
  if (token !== state.loadToken) return; // user switched sample meanwhile
  state.frames = frames;
  drawFrame();
  setStatus('Ready');

  // show an earlier auto-segmentation instantly if the backend has one cached
  try {
    const cached = await api.post('/api/segment', { sample_id: sample.id, cached_only: true });
    if (token === state.loadToken && cached.available) applySegmentation(cached, { quiet: true });
  } catch { /* cache probe is best-effort */ }
}

function drawPoint(x, y, positive, px) {
  ctx.lineWidth = 2 * px;
  ctx.strokeStyle = '#ffffff';
  ctx.fillStyle = positive ? '#22c55e' : '#f43f5e';
  ctx.beginPath(); ctx.arc(x, y, 6 * px, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.strokeStyle = '#06101f'; ctx.lineWidth = 1.8 * px;
  ctx.beginPath(); ctx.moveTo(x - 3 * px, y); ctx.lineTo(x + 3 * px, y);
  if (positive) { ctx.moveTo(x, y - 3 * px); ctx.lineTo(x, y + 3 * px); }
  ctx.stroke();
}

function drawFrame() {
  const s = state.sample;
  if (!s) return;
  const img = state.frames[state.frameIdx];
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (img) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const px = canvas.width / 512; // keep stroke widths visually constant across resolutions

  const fd = state.seg ? state.seg.frames_data[state.frameIdx] : null;
  if (fd) {
    if (state.overlays.mask && fd.contour && fd.contour.length > 2) {
      ctx.beginPath();
      fd.contour.forEach(([x, y], i) => (i ? ctx.lineTo(x + 0.5, y + 0.5) : ctx.moveTo(x + 0.5, y + 0.5)));
      ctx.closePath();
      ctx.fillStyle = 'rgba(34,197,94,0.28)';
      ctx.fill();
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2 * px;
      ctx.lineJoin = 'round';
      ctx.stroke();
    }
    if (state.overlays.chords && fd.chords) {
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth = 1.3 * px;
      ctx.beginPath();
      for (const [[xa, ya], [xb, yb]] of fd.chords) { ctx.moveTo(xa + 0.5, ya + 0.5); ctx.lineTo(xb + 0.5, yb + 0.5); }
      ctx.stroke();
    }
    if (state.overlays.axis && fd.long_axis) {
      const [[x0, y0], [x1, y1]] = fd.long_axis;
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2.4 * px;
      ctx.beginPath(); ctx.moveTo(x0 + 0.5, y0 + 0.5); ctx.lineTo(x1 + 0.5, y1 + 0.5); ctx.stroke();
      ctx.fillStyle = '#facc15';
      for (const [x, y] of fd.long_axis) { ctx.beginPath(); ctx.arc(x + 0.5, y + 0.5, 3.6 * px, 0, Math.PI * 2); ctx.fill(); }
    }
  }
  if (state.frameIdx === 0) {
    const seg = state.seg;
    const pending = state.prompt.multi && state.prompt.points.length;
    if (seg && !pending) {
      const pts = seg.points || (seg.click ? [seg.click] : []);
      const lbl = seg.labels || pts.map(() => 1);
      if (seg.snap && seg.snap.snapped && seg.click_original) {
        // show where the user clicked and where the prompt was moved to
        const [ox, oy] = seg.click_original;
        const [nx, ny] = pts[0];
        ctx.setLineDash([5 * px, 4 * px]); ctx.strokeStyle = '#c4b5fd'; ctx.lineWidth = 1.8 * px;
        ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(nx, ny); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(ox, oy, 6 * px, 0, Math.PI * 2); ctx.lineWidth = 2 * px; ctx.stroke();
      }
      pts.forEach(([x, y], i) => drawPoint(x, y, lbl[i] === 1, px));
    }
    if (pending) state.prompt.points.forEach(([x, y], i) => drawPoint(x, y, state.prompt.labels[i] === 1, px));
  }
}

function updateHud() {
  const s = state.sample;
  if (!s) return;
  $('hudFrame').textContent = `frame ${state.frameIdx + 1}/${s.n_frames}`;
  $('hudTime').textContent = `${(state.frameIdx / s.fps).toFixed(2)} s`;
  const seg = state.seg;
  const phase = $('hudPhase');
  if (seg) {
    $('hudVolume').textContent = `${fmt(seg.volumes_ml[state.frameIdx], 1)} ${unitOf(s)}`;
    if (state.frameIdx === seg.ed_frame_idx) { phase.textContent = 'ED'; phase.className = 'ml-1 font-bold text-sky-400'; }
    else if (state.frameIdx === seg.es_frame_idx) { phase.textContent = 'ES'; phase.className = 'ml-1 font-bold text-rose-400'; }
    else phase.textContent = '';
  } else {
    $('hudVolume').textContent = '–';
    phase.textContent = '';
  }
}

function setFrame(i, fromSlider = false) {
  const s = state.sample;
  if (!s) return;
  state.frameIdx = ((Math.round(i) % s.n_frames) + s.n_frames) % s.n_frames;
  if (!fromSlider) $('frameSlider').value = String(state.frameIdx);
  drawFrame();
  updateHud();
  updatePlotCursor();
  updateContraction();
}

let lastTick = 0;
let frameDebt = 0;
function tick(now) {
  if (!state.playing) return;
  const s = state.sample;
  const frameMs = 1000 / (s.fps * state.speed);
  frameDebt += now - lastTick;
  lastTick = now;
  if (frameDebt >= frameMs) {
    const steps = Math.min(Math.floor(frameDebt / frameMs), 4);
    frameDebt -= steps * frameMs;
    if (frameDebt > frameMs) frameDebt = 0;
    setFrame(state.frameIdx + steps);
  }
  requestAnimationFrame(tick);
}
function play() {
  if (state.playing || !state.sample) return;
  state.playing = true;
  $('btnPlay').textContent = '⏸';
  lastTick = performance.now();
  frameDebt = 0;
  requestAnimationFrame(tick);
}
function pause() {
  state.playing = false;
  $('btnPlay').textContent = '▶';
}

function setBusy(on, text) {
  state.busy = on;
  const el = $('usBusy');
  el.classList.toggle('hidden', !on);
  el.classList.toggle('flex', on);
  if (text) $('usBusyText').textContent = text;
  for (const id of ['btnAuto', 'subjectSelect', 'btnMultiRun', 'btnUpload']) $(id).disabled = on;
  if (!on) updateMultiBar();
}

// prompt: null (auto-centroid) | { click: [x, y] } | { points, labels }
async function runSegmentation(prompt = null, force = false) {
  const s = state.sample;
  if (!s || state.busy) return;
  pause();
  setFrame(0);
  setBusy(true, `SAM 2.1 is tracking the LV across ${s.n_frames} frames…`);
  setStatus('Segmenting on GPU…', 'busy');
  try {
    const body = { sample_id: s.id, force, auto_snap: state.prompt.autoSnap };
    if (prompt && prompt.points) { body.points = prompt.points; body.labels = prompt.labels; }
    else if (prompt && prompt.click) body.click = prompt.click;
    else if (Array.isArray(prompt)) body.click = prompt;
    const seg = await api.post('/api/segment', body);
    if (state.sample && state.sample.id === s.id) {
      state.prompt.points = [];
      state.prompt.labels = [];
      updateMultiBar();
      applySegmentation(seg);
    }
    setStatus('Ready');
  } catch (err) {
    setStatus('Segmentation failed', 'error');
    toast(`Segmentation failed: ${err.message}`, 'error');
  } finally {
    setBusy(false);
  }
}

function applySegmentation(seg, { quiet = false } = {}) {
  state.seg = seg;
  $('usHint').classList.add('hidden');
  $('btnED').disabled = $('btnES').disabled = false;
  const stats = $('runStats');
  stats.classList.remove('hidden');
  const within = seg.vram_mb < seg.vram_budget_mb;
  stats.innerHTML =
    `SAM 2.1 tiny · ${escapeHtml(seg.autocast_dtype)} · ${escapeHtml(seg.click_source)} · peak VRAM ` +
    `<span class="${within ? 'text-emerald-400' : 'text-rose-400'} font-semibold">${fmt(seg.vram_mb, 0)} MB</span> of ${fmt(seg.vram_budget_mb, 0)} · ` +
    `${fmt(seg.inference_time_sec, 1)} s${seg.cached ? ' · cached' : ''}`;
  stats.title = stats.textContent;
  const chip = $('vramChip');
  chip.classList.remove('hidden');
  chip.textContent = `peak VRAM ${fmt(seg.vram_mb, 0)} MB`;

  const q = seg.quality;
  const badge = $('qualityBadge');
  if (q && q.level === 'warning') {
    badge.classList.remove('hidden');
    $('qualityText').textContent = `Tracking quality: ${q.messages[0] || q.flags.join(', ')}`;
    badge.onclick = () => toast(q.messages.join(' ') + ' Try a click deeper inside the cavity, or Multi-point.', 'warn');
  } else {
    badge.classList.add('hidden');
  }
  if (!quiet) {
    if (seg.snap && seg.snap.snapped) toast(`Your click was on bright tissue — the prompt was moved ${fmt(seg.snap.distance_px, 0)} px into the blood pool (Auto-snap).`, 'info');
    toast(seg.cached ? 'Loaded cached segmentation' : `Tracked ${seg.n_frames} frames in ${fmt(seg.inference_time_sec, 1)} s · peak VRAM ${fmt(seg.vram_mb, 0)} MB`, 'ok');
  }
  drawFrame();
  updateHud();
  updateEstimates();
  renderPlot();
  refreshForeshortening();
  updateContraction();
}

function updateEstimates() {
  const s = state.sample;
  const seg = state.seg;
  const gt = (s && s.ground_truth) || {};
  $('kvView').textContent = s ? s.view : '–';
  $('kvSingle').textContent = seg ? `${fmt(seg.single_plane_lvef ?? seg.lvef_pct, 1)} %` : 'segment first';
  const bp = $('kvBiplane');
  if (seg && seg.biplane) {
    bp.textContent = `${fmt(seg.biplane.lvef_pct, 1)} %` + (seg.biplane.length_warning ? ' ⚠' : '');
    bp.className = '';
    bp.title = `EDV ${fmt(seg.biplane.edv_ml, 1)} / ESV ${fmt(seg.biplane.esv_ml, 1)} · long-axis mismatch ${fmt(seg.biplane.length_mismatch_pct_ed, 0)} % at ED` +
      (seg.biplane.length_warning ? ' (above the 10 % tolerance — one view may be foreshortened)' : '');
  } else {
    const subj = s ? subjectOf(s) : null;
    const hasPair = subj && subj.views.A4C && subj.views.A2C;
    bp.textContent = hasPair ? 'segment the other view too' : 'needs A4C + A2C';
    bp.className = 'text-slate-500';
    bp.title = '';
  }
  const refS = seg ? seg.reference_single_plane_lvef : gt.lvef_pct;
  const refB = seg ? seg.reference_biplane_lvef : gt.reference_biplane_lvef_pct;
  const rep = seg ? seg.dataset_reported_biplane_ef : gt.camus_biplane_ef_pct;
  $('kvRefSingle').textContent = refS != null ? `${fmt(refS, 1)} %` : '–';
  $('kvRefBiplane').textContent = refB != null ? `${fmt(refB, 1)} %` : '–';
  $('kvReported').textContent = rep != null ? `${fmt(rep, 0)} %` : '–';
}

function updateMultiBar() {
  const n = state.prompt.points.length;
  const bar = $('multiBar');
  bar.classList.toggle('hidden', !state.prompt.multi);
  bar.classList.toggle('flex', state.prompt.multi);
  const pos = state.prompt.labels.filter((l) => l === 1).length;
  $('multiCount').textContent = `${pos} cavity (+) · ${n - pos} wall (−)`;
  $('btnMultiRun').disabled = state.busy || pos === 0;
  $('usHint').innerHTML = state.prompt.multi
    ? 'Multi-point: <b>click</b> = cavity (+), <b>Shift+click</b> = wall (−), then <b>Segment points</b>'
    : 'Click inside the LV cavity on frame 1, or press <b>Auto-segment</b>';
  if (state.prompt.multi && !state.seg) $('usHint').classList.remove('hidden');
}

function canvasPoint(ev) {
  const rect = canvas.getBoundingClientRect();
  const x = clamp(((ev.clientX - rect.left) / rect.width) * canvas.width, 0, canvas.width - 1);
  const y = clamp(((ev.clientY - rect.top) / rect.height) * canvas.height, 0, canvas.height - 1);
  return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
}

function onCanvasClick(ev, negative) {
  if (!state.sample || state.busy) return;
  if (state.frameIdx !== 0) {
    pause();
    setFrame(0);
    toast('The prompt applies to frame 1 — jumped there. Click the LV cavity again.');
    return;
  }
  const pt = canvasPoint(ev);
  if (negative && !state.prompt.multi) { // a negative point only makes sense in a multi-point prompt
    state.prompt.multi = true;
    $('tgMulti').checked = true;
  }
  if (state.prompt.multi) {
    state.prompt.points.push(pt);
    state.prompt.labels.push(negative ? 0 : 1);
    $('usHint').classList.remove('hidden');
    updateMultiBar();
    drawFrame();
    return;
  }
  runSegmentation({ click: pt });
}
canvas.addEventListener('click', (ev) => onCanvasClick(ev, ev.shiftKey));
canvas.addEventListener('contextmenu', (ev) => { ev.preventDefault(); onCanvasClick(ev, true); });

/* ───────────────────────────── upload ───────────────────────────── */
function toggleUpload(show) {
  const f = $('uploadForm');
  f.classList.toggle('hidden', !show);
  f.classList.toggle('flex', show);
}

async function submitUpload(ev) {
  ev.preventDefault();
  const file = $('uploadFile').files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  const mm = $('uploadMm').value.trim();
  if (mm) fd.append('mm_per_pixel', mm);
  fd.append('view', $('uploadView').value);
  $('uploadSubmit').disabled = true;
  setStatus('Uploading…', 'busy');
  try {
    const res = await request('/api/upload', { method: 'POST', body: fd });
    toggleUpload(false);
    $('uploadForm').reset();
    await loadSamples(res.sample.id);
    toast(`Uploaded ${res.sample.n_frames} frames` + (res.sample.subsampled ? ' (sub-sampled to protect VRAM)' : '') +
      (res.sample.calibrated ? '' : ' — uncalibrated: volumes are an index, LVEF is unaffected'), 'ok');
    setStatus('Ready');
  } catch (err) {
    setStatus('Upload failed', 'error');
    toast(`Upload failed: ${err.message}`, 'error');
  } finally {
    $('uploadSubmit').disabled = false;
  }
}

/* ═════════════════════════════ PLOT ═════════════════════════════ */
const hasPlotly = typeof window.Plotly !== 'undefined';
let plotReady = false;

function cursorShape() {
  const s = state.sample;
  const t = s ? state.frameIdx / s.fps : 0;
  return { type: 'line', xref: 'x', yref: 'paper', x0: t, x1: t, y0: 0, y1: 1, line: { color: '#e2e8f0', width: 1.2, dash: 'dot' } };
}

function renderPlot() {
  const s = state.sample;
  if (!s) return;
  if (!hasPlotly) { $('plot').innerHTML = '<p class="text-xs text-slate-500 p-4">Plotly could not be loaded.</p>'; return; }
  const u = unitOf(s);
  const time = Array.from({ length: s.n_frames }, (_, i) => i / s.fps);
  const tEnd = time[time.length - 1] || 1;
  const traces = [];
  const shapes = [cursorShape()]; // index 0 is the frame cursor
  const gt = s.ground_truth;
  if (gt && gt.volumes_ml && s.kind === 'phantom') {
    // the phantom's volume curve is analytic, so the whole reference curve is exact
    traces.push({ x: time, y: gt.volumes_ml, name: 'Analytic truth', mode: 'lines', line: { color: '#64748b', width: 1.6, dash: 'dot' }, hovertemplate: `%{y:.1f} ${u}<extra>analytic</extra>` });
  }
  if (gt && gt.edv_ml != null && gt.esv_ml != null) {
    // reference ED / ES benchmarks: horizontal dashed levels + star markers at the exact frames
    for (const v of [gt.edv_ml, gt.esv_ml]) {
      shapes.push({ type: 'line', xref: 'x', yref: 'y', x0: 0, x1: tEnd, y0: v, y1: v, line: { color: '#93c5fd', width: 1.1, dash: 'dash' }, opacity: 0.75 });
    }
    traces.push({
      x: [gt.ed_frame_idx / s.fps, gt.es_frame_idx / s.fps], y: [gt.edv_ml, gt.esv_ml], name: 'Reference ED / ES', mode: 'markers',
      marker: { symbol: 'star', size: 13, color: '#93c5fd', line: { color: '#0b111d', width: 1 } },
      text: ['reference ED', 'reference ES'], hovertemplate: `%{text}: %{y:.1f} ${u}<extra></extra>`,
    });
  }
  const seg = state.seg;
  if (seg) {
    traces.push({ x: time, y: seg.volumes_ml, name: 'SAM 2.1 · Simpson', mode: 'lines+markers', line: { color: '#22c55e', width: 2.4, shape: 'spline', smoothing: 0.5 }, marker: { size: 4, color: '#22c55e' }, hovertemplate: `%{y:.1f} ${u}<extra>measured</extra>` });
    if (state.fore && state.fore.curve_ml && state.tilt > 0) {
      traces.push({ x: time, y: state.fore.curve_ml, name: `Foreshortened ${state.tilt}°`, mode: 'lines', line: { color: '#f59e0b', width: 2, dash: 'dash', shape: 'spline', smoothing: 0.5 }, hovertemplate: `%{y:.1f} ${u}<extra>foreshortened</extra>` });
    }
    traces.push({ x: [seg.ed_frame_idx / s.fps], y: [seg.edv_ml], name: 'ED', mode: 'markers+text', text: ['ED'], textposition: 'top center', textfont: { color: '#38bdf8', size: 11 }, marker: { color: '#38bdf8', size: 9, symbol: 'diamond' }, showlegend: false, hovertemplate: `EDV %{y:.1f} ${u}<extra></extra>` });
    traces.push({ x: [seg.es_frame_idx / s.fps], y: [seg.esv_ml], name: 'ES', mode: 'markers+text', text: ['ES'], textposition: 'bottom center', textfont: { color: '#fb7185', size: 11 }, marker: { color: '#fb7185', size: 9, symbol: 'diamond' }, showlegend: false, hovertemplate: `ESV %{y:.1f} ${u}<extra></extra>` });
  }
  $('plotNote').textContent = seg ? 'click the curve to jump to a frame' : 'segment to measure V(t)';
  const peak = Math.max(1, ...traces.flatMap((t) => t.y));
  const layout = {
    margin: { l: 44, r: 8, t: 4, b: 30 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#94a3b8', size: 10.5, family: 'Inter, ui-sans-serif, system-ui, sans-serif' },
    xaxis: { title: { text: 'time (s)', standoff: 4 }, gridcolor: '#1e293b', zeroline: false, linecolor: '#334155', range: [-0.02 * tEnd, tEnd * 1.02] },
    yaxis: { title: { text: s.calibrated === false ? 'LV volume index' : 'LV volume (mL)', standoff: 4 }, gridcolor: '#1e293b', zeroline: false, range: [0, peak * 1.22], linecolor: '#334155' },
    legend: { orientation: 'h', x: 0, y: 1.2, font: { size: 9.5 }, bgcolor: 'rgba(0,0,0,0)' },
    hoverlabel: { bgcolor: '#0f172a', bordercolor: '#334155', font: { color: '#e2e8f0' } },
    shapes,
  };
  window.Plotly.react('plot', traces, layout, { displayModeBar: false, responsive: true }).then(() => {
    if (!plotReady) {
      plotReady = true;
      $('plot').on('plotly_click', (ev) => {
        if (!state.sample || !ev.points || !ev.points.length) return;
        pause();
        setFrame(Math.round(ev.points[0].x * state.sample.fps));
      });
    }
  });
}

let cursorPending = false;
function updatePlotCursor() {
  if (!hasPlotly || !plotReady || cursorPending) return;
  cursorPending = true;
  requestAnimationFrame(() => {
    cursorPending = false;
    const sh = cursorShape();
    window.Plotly.relayout('plot', { 'shapes[0].x0': sh.x0, 'shapes[0].x1': sh.x1 });
  });
}

/* ═════════════════════════════ FORESHORTENING CARDS ═════════════════════════════ */
function trueVolumes() {
  if (state.seg) return { edv: state.seg.edv_ml, esv: state.seg.esv_ml, source: 'algorithm: SAM 2.1 + single-plane Simpson' };
  const gt = state.sample && state.sample.ground_truth;
  if (gt) return { edv: gt.edv_ml, esv: gt.esv_ml, source: state.sample.kind === 'phantom' ? 'analytic phantom truth' : 'dataset reference masks' };
  return null;
}

let foreSeq = 0;
let foreTimer = null;
function refreshForeshortening() {
  clearTimeout(foreTimer);
  foreTimer = setTimeout(doRefreshForeshortening, 40);
}
async function doRefreshForeshortening() {
  const tv = trueVolumes();
  $('efTilt').textContent = `${state.tilt}°`;
  if (!tv || !(tv.edv > 0) || !(tv.esv > 0) || tv.esv > tv.edv) {
    for (const id of ['efTrue', 'edvTrue', 'esvTrue', 'efFore', 'edvFore', 'esvFore']) $(id).textContent = '–';
    for (const id of ['efDelta', 'edvErr', 'esvErr', 'lenFore', 'efSource']) $(id).textContent = '';
    return;
  }
  const seq = ++foreSeq;
  try {
    const body = { true_edv: tv.edv, true_esv: tv.esv, tilt_deg: state.tilt };
    if (state.seg) body.volumes_ml = state.seg.volumes_ml;
    const r = await api.post('/api/foreshortening', body);
    if (seq !== foreSeq) return; // a newer slider value superseded this one
    state.fore = r;
    $('efTrue').textContent = fmt(r.true.lvef_pct, 1);
    $('edvTrue').textContent = fmt(r.true.edv_ml, 1);
    $('esvTrue').textContent = fmt(r.true.esv_ml, 1);
    $('efSource').textContent = tv.source;
    $('efSource').title = tv.source;
    $('efFore').textContent = fmt(r.foreshortened.lvef_pct, 1);
    $('edvFore').textContent = fmt(r.foreshortened.edv_ml, 1);
    $('esvFore').textContent = fmt(r.foreshortened.esv_ml, 1);
    const tilted = state.tilt > 0;
    $('efDelta').textContent = tilted ? `${signed(r.errors.lvef_abs_pts)} pts` : '';
    $('efDelta').className = 'num text-[11.5px] ml-1 font-semibold ' + (tilted ? 'text-amber-300' : 'text-slate-400');
    $('edvErr').textContent = tilted ? `(${signed(r.errors.edv_pct, 0)} %)` : '';
    $('esvErr').textContent = tilted ? `(${signed(r.errors.esv_pct, 0)} %)` : '';
    $('lenFore').textContent = state.sample && state.sample.calibrated === false
      ? 'model lengths need a calibrated clip'
      : `fitted ED length ${fmt(r.foreshortened.ed_length_mm, 0)} mm` + (tilted ? ` (on axis ${fmt(r.true.ed_length_mm, 0)} mm)` : '');
    renderPlot();
  } catch (err) {
    if (seq === foreSeq) toast(`Foreshortening model: ${err.message}`, 'error');
  }
}

/* ═════════════════════════════ 3D NAVIGATOR ═════════════════════════════ */
const three = {
  ok: false,
  renderer: null, scene: null, camera: null, controls: null,
  clipPlane: new THREE.Plane(new THREE.Vector3(0, 0, -1), 0),
  chambers: {},        // name -> { mesh, cap, stencil, color, label, movers }
  fan: null, fanEdge: null, beamLine: null, probe: null,
  apexMarker: null, falseApex: null, axisLine: null, apparentLine: null,
  tween: null,
  highlight: { names: [], start: null, dur: 0 },
  contextLost: false,
  recoveries: 0,       // times the camera watchdog had to reset a non-finite pose
  resizes: 0,          // accepted canvas resizes (diagnostics)
  size: [0, 0],
  lvScale: null,
};
const LABELS = { Aorta: 'Ao' };

function planePose() {
  const meta = state.planes;
  const p = meta.planes[state.view];
  const origin = V3(p.origin);
  const beam = V3(p.beam).normalize();
  const right = V3(p.right).normalize();
  const normal = V3(p.normal).normalize();
  const pivot = V3(meta.pivot);
  const tilt = p.apical ? state.tilt : 0;
  if (tilt > 0 && p.tilt_axis) {
    // The plane pivots about the mitral-annulus centre: the probe slides off the true apex.
    const q = new THREE.Quaternion().setFromAxisAngle(V3(p.tilt_axis).normalize(), THREE.MathUtils.degToRad(tilt));
    const dist = origin.distanceTo(pivot);
    beam.applyQuaternion(q);
    normal.applyQuaternion(q);
    origin.copy(pivot).addScaledVector(beam, -dist);
  }
  return { origin, beam, right, normal, pivot, depth: p.depth_mm, sector: p.sector_deg, apical: p.apical, tilt, def: p };
}

function makeLabelSprite(text, color) {
  const c = document.createElement('canvas');
  c.width = 192; c.height = 96;
  const g = c.getContext('2d');
  g.font = '700 44px Inter, system-ui, sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.lineWidth = 9; g.strokeStyle = 'rgba(6,10,19,0.9)';
  g.strokeText(text, 96, 50);
  g.fillStyle = color;
  g.fillText(text, 96, 50);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, depthWrite: false, transparent: true }));
  sprite.scale.set(30, 15, 1);
  sprite.renderOrder = 60;
  return sprite;
}

// Resize the drawing buffer to the wrapper. Layout can momentarily report 0 or absurd sizes (hidden pane,
// mid-reflow); those are ignored. Resizing clears the buffer, so the scene is redrawn in the same task —
// otherwise the canvas would show an empty buffer until the next animation frame.
function resizeThree() {
  if (!three.ok) return;
  const wrap = $('threeWrap');
  const w = Math.floor(wrap.clientWidth), h = Math.floor(wrap.clientHeight);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w < 16 || h < 16 || w > 8192 || h > 8192) return;
  if (w === three.size[0] && h === three.size[1]) return;
  three.size = [w, h];
  three.resizes += 1;
  three.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  three.renderer.setSize(w, h, false);
  three.renderer.setClearColor(CLEAR_COLOR, 1);
  three.camera.aspect = w / h;
  three.camera.updateProjectionMatrix();
  if (!three.contextLost) three.renderer.render(three.scene, three.camera);
}

function initThree() {
  const wrap = $('threeWrap');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, stencil: true });
  } catch (err) {
    const fb = $('threeFallback');
    fb.classList.remove('hidden'); fb.classList.add('flex');
    fb.textContent = 'WebGL is not available in this browser, so the 3-D navigator is disabled. The 2-D workbench still works.';
    return;
  }
  renderer.localClippingEnabled = true; // per-material clipping planes
  renderer.setClearColor(CLEAR_COLOR, 1);
  wrap.insertBefore(renderer.domElement, wrap.firstChild);
  renderer.domElement.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;background:#0b0f19';

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 5, 4000);
  camera.position.set(260, 160, 360);
  scene.add(new THREE.HemisphereLight(0xdbeafe, 0x0b1020, 1.15));
  const key = new THREE.DirectionalLight(0xffffff, 1.7);
  key.position.set(200, 320, 260);
  camera.add(key); // headlight that follows the camera
  const fill = new THREE.DirectionalLight(0x93c5fd, 0.6);
  fill.position.set(-260, -80, 120);
  camera.add(fill);
  scene.add(camera);

  Object.assign(three, { renderer, scene, camera, ok: true });

  // GPU memory pressure (SAM 2.1, a local LLM and WebGL share 6 GB) can make the browser drop the
  // WebGL context, which blanks the canvas. Say so, and redraw everything once it is restored.
  renderer.domElement.addEventListener('webglcontextlost', (ev) => {
    ev.preventDefault(); // required for the browser to attempt a restore
    three.contextLost = true;
    const fb = $('threeFallback');
    fb.classList.remove('hidden'); fb.classList.add('flex');
    fb.style.background = 'rgba(6,10,19,0.92)';
    fb.textContent = 'The 3-D view lost its GPU context (usually GPU memory pressure). Restoring… if it does not return, reload the page.';
  });
  renderer.domElement.addEventListener('webglcontextrestored', () => {
    three.contextLost = false;
    const fb = $('threeFallback');
    fb.classList.add('hidden'); fb.classList.remove('flex');
    three.scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
    three.size = [0, 0];
    resizeThree();
    updatePlane();
    toast('3-D view restored', 'ok');
  });

  makeControls(new THREE.Vector3(0, 0, 0));
  new ResizeObserver(resizeThree).observe(wrap);
  window.addEventListener('resize', resizeThree);
  resizeThree();
  renderer.setAnimationLoop(renderLoop);
}

function makeControls(target) {
  if (three.controls) three.controls.dispose();
  const controls = new OrbitControls(three.camera, three.renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 120;
  controls.maxDistance = 1400;
  controls.target.copy(target);
  controls.update();
  three.controls = controls;
}

async function loadHeart() {
  if (!three.ok) return;
  const [meta, gltf] = await Promise.all([
    api.get('models/heart_planes.json'),
    new GLTFLoader().loadAsync('models/heart_chambers.glb'),
  ]);
  state.planes = meta;
  const names = Object.keys(meta.colors);
  const found = {};
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const name = names.find((n) => o.name === n || (o.parent && o.parent.name === n) || o.name.startsWith(n + '_'));
    if (name && !found[name]) {
      const g = o.geometry.clone();
      g.applyMatrix4(o.matrixWorld);
      g.computeVertexNormals();
      found[name] = g;
    }
  });
  const missing = names.filter((n) => !found[n]);
  if (missing.length) throw new Error(`heart_chambers.glb is missing: ${missing.join(', ')}`);

  const legend = $('legend');
  legend.innerHTML = '';
  names.forEach((name, i) => {
    const color = new THREE.Color(meta.colors[name]);
    const geom = found[name];

    // 1) the structure itself, clipped by the echo plane
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.42, metalness: 0.06, side: THREE.DoubleSide, clippingPlanes: [three.clipPlane] });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.name = name;
    mesh.renderOrder = 40; // after every stencil / cap pass

    // 2) stencil pass: count back/front faces so we know where the cut is "inside" the solid
    const stencil = new THREE.Group();
    const base = { depthWrite: false, depthTest: false, colorWrite: false, stencilWrite: true, stencilFunc: THREE.AlwaysStencilFunc, clippingPlanes: [three.clipPlane] };
    const back = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ ...base, side: THREE.BackSide, stencilFail: THREE.IncrementWrapStencilOp, stencilZFail: THREE.IncrementWrapStencilOp, stencilZPass: THREE.IncrementWrapStencilOp }));
    const front = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ ...base, side: THREE.FrontSide, stencilFail: THREE.DecrementWrapStencilOp, stencilZFail: THREE.DecrementWrapStencilOp, stencilZPass: THREE.DecrementWrapStencilOp }));
    back.renderOrder = front.renderOrder = i + 1;
    stencil.add(back, front);

    // 3) the cap: a big quad on the clipping plane, drawn only where the stencil says "inside"
    const cap = new THREE.Mesh(
      new THREE.PlaneGeometry(900, 900),
      new THREE.MeshBasicMaterial({ color: color.clone(), side: THREE.DoubleSide, stencilWrite: true, stencilRef: 0, stencilFunc: THREE.NotEqualStencilFunc, stencilFail: THREE.ReplaceStencilOp, stencilZFail: THREE.ReplaceStencilOp, stencilZPass: THREE.ReplaceStencilOp }),
    );
    cap.renderOrder = i + 1.1;
    cap.onAfterRender = (r) => r.clearStencil();

    const info = meta.chambers[name] || {};
    const label = makeLabelSprite(LABELS[name] || name, '#f8fafc');
    label.position.copy(V3(info.label || info.centroid || [0, 0, 0]));
    three.scene.add(mesh, stencil, cap, label);
    three.chambers[name] = { mesh, cap, stencil, color, label, movers: [mesh, back, front] };

    const item = document.createElement('span');
    item.className = 'flex items-center gap-1 text-[10.5px] text-slate-300';
    item.innerHTML = `<span class="legend-dot" style="background:${escapeHtml(meta.colors[name])}"></span>${escapeHtml(name)}`;
    item.title = info.description || name;
    legend.appendChild(item);
  });

  // true long axis + true apex marker
  const apex = V3(meta.apex), pivot = V3(meta.pivot);
  const axisDir = apex.clone().sub(pivot).normalize();
  const axisGeom = new THREE.BufferGeometry().setFromPoints([pivot.clone().addScaledVector(axisDir, -18), apex.clone().addScaledVector(axisDir, 26)]);
  three.axisLine = new THREE.Line(axisGeom, new THREE.LineDashedMaterial({ color: 0xfacc15, dashSize: 5, gapSize: 3.5, depthTest: false, transparent: true, opacity: 0.95 }));
  three.axisLine.computeLineDistances();
  three.axisLine.renderOrder = 55;
  three.scene.add(three.axisLine);
  three.apexMarker = new THREE.Mesh(new THREE.SphereGeometry(3.4, 20, 14), new THREE.MeshBasicMaterial({ color: 0xfacc15, depthTest: false }));
  three.apexMarker.position.copy(apex);
  three.apexMarker.renderOrder = 56;
  three.scene.add(three.apexMarker);

  // apparent (foreshortened) axis + false apex, only shown when tilted
  three.apparentLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([pivot, apex]), new THREE.LineBasicMaterial({ color: 0xfb923c, depthTest: false }));
  three.apparentLine.renderOrder = 55;
  three.falseApex = new THREE.Mesh(new THREE.SphereGeometry(3.0, 18, 12), new THREE.MeshBasicMaterial({ color: 0xfb923c, depthTest: false }));
  three.falseApex.renderOrder = 56;
  three.scene.add(three.apparentLine, three.falseApex);

  // sector fan (rebuilt per view), its outline, the central beam and the probe
  three.fan = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false }));
  three.fan.renderOrder = 50;
  three.fanEdge = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.95 }));
  three.fanEdge.renderOrder = 51;
  three.beamLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0xbae6fd, dashSize: 4, gapSize: 4, transparent: true, opacity: 0.7 }));
  three.beamLine.renderOrder = 51;
  const probe = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(24, 34, 13), new THREE.MeshStandardMaterial({ color: 0xcbd5e1, roughness: 0.5, metalness: 0.2 }));
  body.position.y = 21;
  const face = new THREE.Mesh(new THREE.BoxGeometry(26, 5, 15), new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.7 }));
  face.position.y = 2.5;
  const cable = new THREE.Mesh(new THREE.CylinderGeometry(3, 3, 22, 12), new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.8 }));
  cable.position.y = 49;
  const marker = new THREE.Mesh(new THREE.SphereGeometry(2.6, 14, 10), new THREE.MeshBasicMaterial({ color: 0x4ade80 }));
  marker.position.set(12.5, 14, 0); // index marker = screen-right side
  probe.add(body, face, cable, marker);
  three.probe = probe;
  for (const o of [three.fan, three.fanEdge, three.beamLine, three.probe]) { o.matrixAutoUpdate = false; three.scene.add(o); }

  applyCutaway();
  const s = state.sample;
  setView(s && (s.view === 'A4C' || s.view === 'A2C') ? s.view : state.view, { animate: false, sync: false });
  updateContraction();
}

function rebuildFanGeometry(depth, sectorDeg) {
  // local frame: +X = screen right, +Y = toward the probe, beam runs along −Y
  const half = THREE.MathUtils.degToRad(sectorDeg / 2);
  const n = 48;
  const pts = [new THREE.Vector3(0, 0, 0)];
  for (let i = 0; i <= n; i++) {
    const a = -half + (2 * half * i) / n;
    pts.push(new THREE.Vector3(Math.sin(a) * depth, -Math.cos(a) * depth, 0));
  }
  const pos = [];
  for (let i = 1; i <= n; i++) pos.push(0, 0, 0, pts[i].x, pts[i].y, 0, pts[i + 1].x, pts[i + 1].y, 0);
  three.fan.geometry.dispose();
  three.fan.geometry = new THREE.BufferGeometry();
  three.fan.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  three.fanEdge.geometry.dispose();
  three.fanEdge.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  three.beamLine.geometry.dispose();
  three.beamLine.geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -depth, 0)]);
  three.beamLine.computeLineDistances();
}

function updatePlane() {
  if (!three.ok || !state.planes || !three.fan) return;
  const P = planePose();
  const up = P.beam.clone().negate();

  // keep the half-space behind the plane; the half facing the viewer (+normal) is cut away
  three.clipPlane.setFromNormalAndCoplanarPoint(P.normal.clone().negate(), P.origin);

  // caps live on the plane, centred under the heart
  const centre = new THREE.Vector3(0, 0, 0);
  const onPlane = centre.clone().addScaledVector(P.normal, -P.normal.dot(centre.clone().sub(P.origin)));
  for (const ch of Object.values(three.chambers)) {
    ch.cap.position.copy(onPlane);
    ch.cap.lookAt(onPlane.clone().add(P.normal));
  }

  // fan / beam / probe share the probe frame (right, toward-probe, normal)
  const frame = new THREE.Matrix4().makeBasis(P.right, up, P.normal);
  const lifted = P.origin.clone().addScaledVector(P.normal, 0.6); // avoid z-fighting with the caps
  for (const o of [three.fan, three.fanEdge, three.beamLine]) { o.matrix.copy(frame).setPosition(lifted); o.matrixWorldNeedsUpdate = true; }
  three.probe.matrix.copy(frame).setPosition(P.origin);
  three.probe.matrixWorldNeedsUpdate = true;

  // foreshortening cues
  const meta = state.planes;
  const tilted = P.apical && P.tilt > 0;
  three.apparentLine.visible = three.falseApex.visible = tilted;
  if (tilted) {
    const th = THREE.MathUtils.degToRad(P.tilt);
    const a = meta.lv_length_mm, b = meta.lv_radius_mm;
    const apparent = 1 / Math.sqrt(Math.cos(th) ** 2 / (a * a) + Math.sin(th) ** 2 / (b * b));
    const tip = P.pivot.clone().addScaledVector(P.beam, -apparent);
    three.apparentLine.geometry.setFromPoints([P.pivot, tip]);
    three.falseApex.position.copy(tip);
    $('tiltBadgeText').textContent = `3-D model LV length ${apparent.toFixed(0)} of ${a.toFixed(0)} mm (${((apparent / a - 1) * 100).toFixed(0)} %)`;
  }
  $('tiltBadge').classList.toggle('hidden', !tilted);
}

function updateLabels() {
  // with the cut-away on, only label the structures this plane actually slices
  const p = state.planes && state.planes.planes[state.view];
  for (const [name, ch] of Object.entries(three.chambers)) {
    const inView = !state.cutaway || !p || p.expected_chambers.includes(name);
    ch.label.visible = state.showLabels && inView;
  }
}

function applyCutaway() {
  for (const ch of Object.values(three.chambers)) {
    const m = ch.mesh.material;
    m.clippingPlanes = state.cutaway ? [three.clipPlane] : [];
    m.transparent = !state.cutaway;
    m.opacity = state.cutaway ? 1 : 0.5;
    m.depthWrite = state.cutaway;
    m.needsUpdate = true;
    ch.cap.visible = state.cutaway;
    ch.stencil.visible = state.cutaway;
  }
  updateLabels();
}

/* ── synchronized LV contraction ──────────────────────────────────────────
   The 3-D LV follows the active clip's V(t): with r = V(t) / EDV the cavity is
   scaled by r^0.25 along the long axis and r^0.375 radially (so its volume
   scales by exactly r), about the mitral-annulus centre, which stays put. */
function volumeRatio() {
  const s = state.sample;
  if (!s) return 1;
  const vols = state.seg ? state.seg.volumes_ml : (s.ground_truth && s.ground_truth.volumes_ml);
  if (!vols || !vols.length) return 1;
  const edv = Math.max(...vols);
  const v = vols[Math.min(state.frameIdx, vols.length - 1)];
  return edv > 0 && v > 0 ? clamp(v / edv, 0.15, 1) : 1;
}

function updateContraction() {
  const lv = three.chambers.LV;
  if (!three.ok || !lv || !state.planes) return;
  const r = state.beat ? volumeRatio() : 1;
  const sl = Math.pow(r, 0.25), sr = Math.pow(r, 0.375);
  const meta = state.planes;
  const pivot = V3(meta.pivot);
  const axis = V3(meta.long_axis).normalize();
  const e1 = new THREE.Vector3(1, 0, 0);
  if (Math.abs(e1.dot(axis)) > 0.9) e1.set(0, 1, 0);
  e1.addScaledVector(axis, -e1.dot(axis)).normalize();
  const e2 = new THREE.Vector3().crossVectors(axis, e1);
  const B = new THREE.Matrix4().makeBasis(e1, axis, e2);
  const M = new THREE.Matrix4().makeTranslation(pivot.x, pivot.y, pivot.z)
    .multiply(B).multiply(new THREE.Matrix4().makeScale(sr, sl, sr)).multiply(B.clone().transpose())
    .multiply(new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z));
  for (const o of lv.movers) { o.matrixAutoUpdate = false; o.matrix.copy(M); o.matrixWorldNeedsUpdate = true; }
  three.lvScale = { ratio: r, longitudinal: sl, radial: sr };
}

/* ── camera safety ──────────────────────────────────────────────────────────
   A camera pose is only usable if position, target and up are finite, the eye
   is not on the target, and `up` is not (nearly) parallel to the viewing
   direction — otherwise lookAt() is singular and the view matrix degenerates. */
const MIN_UP_SINE = 0.12; // ~7 degrees between `up` and the line of sight
const finiteVec = (v) => Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);

function sanitizePose(pose) {
  const target = pose.target && finiteVec(pose.target) ? pose.target.clone() : new THREE.Vector3();
  const offset = pose.pos && finiteVec(pose.pos) ? pose.pos.clone().sub(target) : new THREE.Vector3();
  if (offset.lengthSq() < 1) offset.set(0.35, 0.3, 1).normalize().multiplyScalar(560);
  const sight = offset.clone().normalize();
  let up = pose.up && finiteVec(pose.up) && pose.up.lengthSq() > 1e-8 ? pose.up.clone().normalize() : new THREE.Vector3(0, 1, 0);
  if (up.clone().cross(sight).length() < MIN_UP_SINE) {
    // collinear: fall back to the world axis that is most perpendicular to the line of sight
    const axes = [new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(1, 0, 0)];
    axes.sort((a, b) => Math.abs(a.dot(sight)) - Math.abs(b.dot(sight)));
    up = axes[0].addScaledVector(sight, -axes[0].dot(sight)).normalize();
  }
  return { pos: target.clone().add(offset), target, up };
}

// rotate unit vector a toward unit vector b by fraction t (great-circle interpolation)
function slerpDir(a, b, t) {
  const dot = clamp(a.dot(b), -1, 1);
  if (dot > 0.9995) return a.clone().lerp(b, t).normalize();
  let axis;
  if (dot < -0.9995) {
    // opposite directions: any axis perpendicular to a will do
    axis = Math.abs(a.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
    axis.addScaledVector(a, -axis.dot(a)).normalize();
  } else {
    axis = a.clone().cross(b).normalize();
  }
  return a.clone().applyAxisAngle(axis, Math.acos(dot) * t);
}

// Pose at eased time e. The eye travels on an arc around the target instead of
// a straight chord, so it can never fly through the heart (or the target itself).
function tweenPose(tw, e) {
  const target = tw.fromTarget.clone().lerp(tw.target, e);
  const o0 = tw.fromPos.clone().sub(tw.fromTarget);
  const o1 = tw.pos.clone().sub(tw.target);
  const r1 = o1.length();
  const r0 = o0.length() > 1e-3 ? o0.length() : r1;
  const d1 = o1.clone().normalize();
  const d0 = o0.length() > 1e-3 ? o0.clone().normalize() : d1;
  const dir = slerpDir(d0, d1, e);
  const up = slerpDir(tw.fromUp, tw.up, e);
  return sanitizePose({ pos: target.clone().addScaledVector(dir, THREE.MathUtils.lerp(r0, r1, e)), target, up });
}

function cameraIsFinite() {
  const c = three.camera;
  return finiteVec(c.position) && finiteVec(c.up) && c.matrixWorld.elements.every(Number.isFinite)
    && [c.quaternion.x, c.quaternion.y, c.quaternion.z, c.quaternion.w].every(Number.isFinite);
}

function cameraPose(mode) {
  const P = planePose();
  if (mode === 'echo') {
    // look straight at the cut with the probe at the top: what the echo screen shows
    const target = P.origin.clone().addScaledVector(P.beam, P.depth * 0.48);
    return { pos: target.clone().addScaledVector(P.normal, 520), target, up: P.beam.clone().negate() };
  }
  const up = new THREE.Vector3(0, 1, 0);
  const dir = P.normal.clone().addScaledVector(up, 0.3).addScaledVector(P.right, 0.22).normalize();
  const target = new THREE.Vector3(0, 0, 0);
  return { pos: target.clone().addScaledVector(dir, 560), target, up };
}

function moveCamera(pose, animate = true) {
  if (!three.ok) return;
  pose = sanitizePose(pose);
  // honour the OS "reduce motion" setting: jump straight to the new pose
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) animate = false;
  const from = {
    fromPos: three.camera.position.clone(),
    fromTarget: three.controls ? three.controls.target.clone() : new THREE.Vector3(),
    fromUp: three.camera.up.clone().normalize(),
  };
  if (!finiteVec(from.fromPos) || !finiteVec(from.fromTarget) || !finiteVec(from.fromUp) || from.fromUp.lengthSq() < 1e-8) animate = false;
  if (!animate) {
    three.camera.up.copy(pose.up);
    three.camera.position.copy(pose.pos);
    makeControls(pose.target);
    three.tween = null;
    return;
  }
  three.controls.enabled = false;
  three.tween = { t0: null, dur: 750, ...from, ...pose }; // t0 comes from the render loop's own clock
}

function renderLoop(now) {
  if (three.contextLost) return; // nothing can be drawn until the browser restores the GL context
  const tw = three.tween;
  if (tw) {
    if (tw.t0 === null) tw.t0 = now;
    const k = clamp((now - tw.t0) / tw.dur, 0, 1);
    const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    const p = tweenPose(tw, e);
    three.camera.position.copy(p.pos);
    three.camera.up.copy(p.up);
    three.camera.lookAt(p.target);
    if (k >= 1) {
      three.camera.position.copy(tw.pos);
      three.camera.up.copy(tw.up);
      makeControls(tw.target); // OrbitControls caches camera.up, so rebuild it for the new "up"
      three.tween = null;
    }
  } else if (three.controls) {
    three.controls.update();
  }
  // watchdog: a non-finite camera would blank the viewport, so snap back to the view's home pose
  three.camera.updateMatrixWorld();
  if (!cameraIsFinite()) {
    three.recoveries += 1;
    console.warn('EchoPlane3D: camera pose became non-finite — resetting the 3-D view');
    three.camera.position.set(260, 160, 360);
    three.camera.up.set(0, 1, 0);
    three.camera.quaternion.identity();
    three.tween = null;
    makeControls(new THREE.Vector3(0, 0, 0));
    if (state.planes) moveCamera(cameraPose(state.camMode), false);
    three.camera.updateMatrixWorld(true);
  }
  // chamber highlight pulse requested by the guide
  const hl = three.highlight;
  if (hl.dur > 0 && hl.start === null) hl.start = now;
  for (const [name, ch] of Object.entries(three.chambers)) {
    const on = hl.dur > 0 && now - hl.start < hl.dur && hl.names.includes(name);
    const pulse = on ? 0.35 + 0.35 * Math.sin(now / 130) : 0;
    ch.mesh.material.emissive.copy(ch.color).multiplyScalar(pulse);
  }
  three.renderer.setClearColor(CLEAR_COLOR, 1); // never let another code path leave a transparent / white clear colour
  three.renderer.render(three.scene, three.camera);
}

function updateSyncBadge() {
  const badge = $('syncBadge');
  const s = state.sample;
  if (!s || !state.planes) { badge.classList.add('hidden'); return; }
  let text = '';
  if (state.view === 'PLAX' || state.view === 'PSAX') {
    text = `No ${state.view} clip in the library — the 2D pane keeps showing ${s.view}. Volumes are only measured in apical views.`;
  } else if (s.view !== state.view) {
    text = `This subject has no ${state.view} clip — the 2D pane keeps showing ${s.view}.`;
  }
  badge.textContent = text;
  badge.classList.toggle('hidden', !text);
}

// sync = true: an apical 3-D view also loads the same subject's matching 2-D clip
function setView(view, { animate = true, sync = true } = {}) {
  if (!state.planes || !state.planes.planes[view]) return;
  state.view = view;
  const p = state.planes.planes[view];
  document.querySelectorAll('#viewButtons button').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
  $('viewLabel').textContent = p.label;
  $('viewDesc').textContent = p.description;
  const slider = $('tiltSlider');
  slider.disabled = !p.apical;
  $('tiltNote').textContent = p.apical
    ? (state.tilt > 0 ? 'the fan has slid off the true apex (yellow)' : '0° = plane through the true apex')
    : 'tilt applies to the apical views (A4C / A2C)';
  if (three.ok && three.fan) {
    rebuildFanGeometry(p.depth_mm, p.sector_deg);
    updatePlane();
    updateLabels();
    moveCamera(cameraPose(state.camMode), animate);
  }
  if (sync && (view === 'A4C' || view === 'A2C') && state.sample && state.sample.view !== view && !state.busy) {
    const subj = subjectOf(state.sample);
    if (subj && subj.views[view]) selectSample(subj.views[view], { syncView: false }).catch((err) => toast(err.message, 'error'));
  }
  updateSyncBadge();
}

function setTilt(deg) {
  state.tilt = clamp(Math.round(Number(deg) || 0), 0, 30);
  $('tiltSlider').value = String(state.tilt);
  $('tiltValue').textContent = `${state.tilt}°`;
  const p = state.planes && state.planes.planes[state.view];
  if (p && p.apical) $('tiltNote').textContent = state.tilt > 0 ? 'the fan has slid off the true apex (yellow)' : '0° = plane through the true apex';
  updatePlane();
  refreshForeshortening();
}

function setCamMode(mode) {
  state.camMode = mode;
  document.querySelectorAll('#camButtons button').forEach((b) => b.classList.toggle('active', b.dataset.cam === mode));
  if (state.planes) moveCamera(cameraPose(mode), true);
}

function highlightChambers(names) {
  three.highlight = { names: names || [], start: null, dur: 2600 };
}

function setToggle(id, on) {
  const el = $(id);
  if (!el || el.checked === !!on) return;
  el.checked = !!on;
  el.dispatchEvent(new Event('change'));
}

/* ═════════════════════════════ CLINICAL GUIDE ═════════════════════════════ */
function addBubble(html, who) {
  const el = document.createElement('div');
  el.className = `bubble ${who}`;
  el.innerHTML = html;
  const log = $('guideLog');
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function setEngine(engine) {
  if (!engine) return;
  state.engine = engine;
  const llm = engine.id === 'ollama';
  const label = engine.label || (llm ? 'Ollama Local LLM' : 'Offline Deterministic Tutor');
  for (const id of ['engineChip', 'guideEngine']) {
    const el = $(id);
    el.textContent = label + (llm && engine.model ? ` · ${engine.model}` : '');
    el.classList.toggle('text-emerald-300', !llm);
    el.classList.toggle('text-amber-300', llm);
  }
}

function guideContext() {
  const tv = trueVolumes();
  const s = state.sample;
  return {
    view: state.view,
    tilt_deg: state.tilt,
    quiz_index: state.quiz.index,
    quiz_asked: state.quiz.asked,
    quiz_correct: state.quiz.correct,
    sample_id: s ? s.id : null,
    sample_view: s ? s.view : null,
    sample_kind: s ? s.kind : null,
    calibrated: s ? s.calibrated !== false : true,
    edv_ml: tv ? tv.edv : null,
    esv_ml: tv ? tv.esv : null,
    segmented: !!state.seg,
    biplane_lvef: state.seg && state.seg.biplane ? state.seg.biplane.lvef_pct : null,
    overlays: { ...state.overlays, fan: state.showFan, cutaway: state.cutaway, labels: state.showLabels, beat: state.beat },
  };
}

const OVERLAY_TOGGLES = { mask: 'ovMask', axis: 'ovAxis', chords: 'ovChords', fan: 'tgFan', cutaway: 'tgCutaway', labels: 'tgLabels', beat: 'tgBeat' };

function applyActions(actions) {
  for (const a of actions || []) {
    if (a.type === 'set_view') { if (a.view !== state.view) setView(a.view); }
    else if (a.type === 'set_tilt') setTilt(a.tilt_deg);
    else if (a.type === 'highlight') highlightChambers(a.chambers);
    else if (a.type === 'toggle_overlay' && OVERLAY_TOGGLES[a.overlay]) setToggle(OVERLAY_TOGGLES[a.overlay], a.on);
    else if (a.type === 'select_sample' && state.samples.some((s) => s.id === a.sample_id)) {
      if (!state.busy && (!state.sample || state.sample.id !== a.sample_id)) selectSample(a.sample_id).catch((err) => toast(err.message, 'error'));
    }
  }
}

async function sendGuide({ message = '', preset = null, quizAnswer = null, echo = null }) {
  if (echo) addBubble(escapeHtml(echo), 'user');
  try {
    const r = await api.post('/api/guide', { message, preset, context: guideContext(), quiz_answer: quizAnswer });
    if (r.engine_status) setEngine(r.engine_status);
    const llm = r.engine === 'ollama';
    let html = `<span class="tag ${llm ? 'llm' : 'det'}">${llm ? 'local LLM · explanation only' : 'deterministic'}</span><br>` + richText(r.reply);
    if (r.verified) html += `<div class="verified"><b>Verified calculation</b> (deterministic engine)<br>${richText(r.verified)}</div>`;
    const bubble = addBubble(html, 'bot');
    applyActions(r.actions);
    if (r.quiz_result) {
      state.quiz.asked += 1;
      if (r.quiz_result.correct) state.quiz.correct += 1;
    }
    if (r.quiz) {
      state.quiz.index += 1;
      r.quiz.options.forEach((opt, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'quiz-opt';
        b.textContent = `${String.fromCharCode(65 + i)}. ${opt}`;
        b.addEventListener('click', async () => {
          const buttons = bubble.querySelectorAll('.quiz-opt');
          buttons.forEach((x) => (x.disabled = true));
          const res = await sendGuide({ quizAnswer: { id: r.quiz.id, answer: i }, echo: `${String.fromCharCode(65 + i)}. ${opt}` });
          if (res && res.quiz_result) {
            buttons[res.quiz_result.correct_index].classList.add('right');
            if (!res.quiz_result.correct) b.classList.add('wrong');
          }
        });
        bubble.appendChild(b);
      });
      $('guideLog').scrollTop = $('guideLog').scrollHeight;
    }
    return r;
  } catch (err) {
    addBubble(`⚠️ ${escapeHtml(err.message)}`, 'bot');
    return null;
  }
}

async function initGuide() {
  const info = await api.get('/api/guide');
  setEngine(info.engine);
  const wrap = $('guidePresets');
  wrap.innerHTML = '';
  for (const p of info.presets) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = p.label;
    b.addEventListener('click', () => sendGuide({ preset: p.id, message: p.message, echo: p.message }));
    wrap.appendChild(b);
  }
  addBubble('<span class="tag det">deterministic</span><br>' + richText(info.welcome), 'bot');
}

/* ═════════════════════════════ WIRING ═════════════════════════════ */
function wireUi() {
  $('subjectSelect').addEventListener('change', (e) => selectSubject(e.target.value).catch((err) => toast(err.message, 'error')));
  document.querySelectorAll('#seqButtons button').forEach((b) => b.addEventListener('click', () => {
    const subj = state.sample && subjectOf(state.sample);
    if (!subj || !subj.views[b.dataset.seq] || state.busy) return;
    selectSample(subj.views[b.dataset.seq]).catch((err) => toast(err.message, 'error'));
    if (state.planes && state.view !== b.dataset.seq) setView(b.dataset.seq, { sync: false });
  }));
  $('btnPlay').addEventListener('click', () => (state.playing ? pause() : play()));
  $('btnPrev').addEventListener('click', () => { pause(); setFrame(state.frameIdx - 1); });
  $('btnNext').addEventListener('click', () => { pause(); setFrame(state.frameIdx + 1); });
  $('frameSlider').addEventListener('input', (e) => { pause(); setFrame(Number(e.target.value), true); });
  $('speedSelect').addEventListener('change', (e) => { state.speed = Number(e.target.value) || 1; });
  $('btnAuto').addEventListener('click', () => runSegmentation(null));
  $('btnED').addEventListener('click', () => { if (state.seg) { pause(); setFrame(state.seg.ed_frame_idx); } });
  $('btnES').addEventListener('click', () => { if (state.seg) { pause(); setFrame(state.seg.es_frame_idx); } });
  for (const [id, key] of [['ovMask', 'mask'], ['ovAxis', 'axis'], ['ovChords', 'chords']]) {
    $(id).addEventListener('change', (e) => { state.overlays[key] = e.target.checked; drawFrame(); });
  }
  $('tgSnap').addEventListener('change', (e) => { state.prompt.autoSnap = e.target.checked; });
  $('tgMulti').addEventListener('change', (e) => {
    state.prompt.multi = e.target.checked;
    if (!e.target.checked) { state.prompt.points = []; state.prompt.labels = []; }
    updateMultiBar();
    drawFrame();
  });
  $('btnMultiClear').addEventListener('click', () => { state.prompt.points = []; state.prompt.labels = []; updateMultiBar(); drawFrame(); });
  $('btnMultiRun').addEventListener('click', () => {
    if (state.prompt.points.length) runSegmentation({ points: state.prompt.points.slice(), labels: state.prompt.labels.slice() });
  });

  $('btnUpload').addEventListener('click', () => toggleUpload($('uploadForm').classList.contains('hidden')));
  $('uploadCancel').addEventListener('click', () => toggleUpload(false));
  $('uploadForm').addEventListener('submit', submitUpload);

  document.querySelectorAll('#viewButtons button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
  document.querySelectorAll('#camButtons button').forEach((b) => b.addEventListener('click', () => setCamMode(b.dataset.cam)));
  $('tiltSlider').addEventListener('input', (e) => setTilt(e.target.value));
  $('tgCutaway').addEventListener('change', (e) => { state.cutaway = e.target.checked; applyCutaway(); });
  $('tgFan').addEventListener('change', (e) => {
    state.showFan = e.target.checked;
    for (const o of [three.fan, three.fanEdge, three.beamLine]) if (o) o.visible = state.showFan;
  });
  $('tgLabels').addEventListener('change', (e) => { state.showLabels = e.target.checked; updateLabels(); });
  $('tgBeat').addEventListener('change', (e) => { state.beat = e.target.checked; updateContraction(); });

  $('guideForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const input = $('guideInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    sendGuide({ message: text, echo: text });
  });

  window.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (e.code === 'Escape') { toggleUpload(false); return; }
    if (tag === 'input' || tag === 'select' || tag === 'textarea' || tag === 'button') return;
    if (e.code === 'Space') { e.preventDefault(); state.playing ? pause() : play(); }
    else if (e.code === 'ArrowRight') { pause(); setFrame(state.frameIdx + 1); }
    else if (e.code === 'ArrowLeft') { pause(); setFrame(state.frameIdx - 1); }
  });
}

async function boot() {
  wireUi();
  updateMultiBar();
  initThree();
  api.get('/api/health').then((h) => {
    $('gpuChip').textContent = h.cuda ? `${h.gpu.replace('NVIDIA GeForce ', '')} · budget ${fmt(h.vram_budget_mb / 1000, 1)} GB` : 'CUDA unavailable';
    if (!h.cuda) $('gpuChip').classList.add('text-rose-400');
    if (h.guide_engine) setEngine(h.guide_engine);
  }).catch(() => { $('gpuChip').textContent = 'backend offline'; });

  const results = await Promise.allSettled([loadSamples(), loadHeart(), initGuide()]);
  const names = ['samples', '3-D heart', 'guide'];
  let failed = false;
  results.forEach((r, i) => {
    if (r.status === 'rejected') { failed = true; console.error(names[i], r.reason); toast(`Could not load ${names[i]}: ${r.reason.message}`, 'error'); }
  });
  if (failed) setStatus('Loaded with errors', 'error');
  updateSyncBadge();
  // handy for debugging and for the browser smoke tests
  window.echoplane = { state, three, setView, setTilt, setCamMode, runSegmentation, setFrame, selectSample, moveCamera, cameraPose, sanitizePose, tweenPose, resizeThree, updateContraction, applyActions, THREE };
}

boot();
