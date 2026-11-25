import { makeBoard, drawBoard, verdictFor, matchDefects, DEFECTS, SEV_COLOR, BW, BH } from './pcb.js';
import { BridgePhone } from './sources.js';

// Elements of removed pages (Verify, Insights, About) resolve to a detached stub, so old code paths stay harmless.
const _stubs = {};
const $ = s => document.querySelector(s) || (_stubs[s] ??= Object.assign(document.createElement('div'), { innerHTML: '<b></b><span class="tiny"></span><span></span>' }));
// The live video element is moved between pages, so keep direct references to it.
const FRAME = document.getElementById('video-frame');
const F = s => FRAME.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} },
};

const DEFAULT_THR = 0.5;
// The phone's own (tuned) v3 thresholds, used until the phone reports them in its status message
// (ml/results/phaseB_final.json; same values as Detector.DEFAULT_THRESHOLDS on the phone).
const V3_THR = { missing_hole: 0.50, mouse_bite: 0.35, open: 0.55, short: 0.50, spur: 0.25, spurious_copper: 0.60, missing_pad: 0.45 };
const phoneKey = t => (t === 'open' ? 'open_circuit' : t);
const phoneThr = t => state.status.thresholds?.[phoneKey(t)] ?? V3_THR[t === 'open_circuit' ? 'open' : t] ?? DEFAULT_THR;
const STATUS_LABEL = { passed: 'Passed', awaiting_fix: 'Open', back_to_rework: 'Open', released: 'Fixed', fixed: 'Fixed' };
const isOpen = e => e.status === 'awaiting_fix' || e.status === 'back_to_rework';
const isFixed = e => e.status === 'fixed' || e.status === 'released';

const state = {
  operator: null,
  route: '/',
  source: null,            // active phone source
  bridge: null,
  status: { connected: false },
  frame: null,
  mode: 'inspect',         // which page the phone's results belong to
  current: null,           // board on the Inspect page
  goldenImg: null,         // operator-set golden frame (real phone)
  log: store.get('bc-log', []),
  fb: store.get('bc-fb', []),
  thr: store.get('bc-thr', {}),
  images: {},              // log id -> { clean, golden, thumb, after } (this session only)
  verifySel: null,         // log id selected on Verify
  vres: null,              // current re-scan result
  showWhy: false,
};

// Older log entries (before review and verify existed) get the new fields.
state.log = state.log.filter(e => e && e.items).map(e => ({
  w: 960, h: 640, verify: [], idSource: 'qr', model: null,
  status: e.verdict === 'pass' ? 'passed' : 'awaiting_fix', ...e,
  items: e.items.map(i => ('x' in i ? i : { ...i, x: (i.cx ?? 0.5) * 960 - 20, y: (i.cy ?? 0.5) * 640 - 20, w: 40, h: 40 })),
}));
// Thumbnails (~25 KB each) only on the newest 40 boards, so the log stays under the browser's storage limit.
const saveLog = () => store.set('bc-log', state.log.slice(0, 300).map((e, i) => (i < 40 || !e.thumb ? e : { ...e, thumb: undefined })));
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 3200);
}

/* =================== Start screen (sign-in) =================== */
function initLogin() {
  const saved = store.get('bc-op', null);
  if (saved) enter(saved);
  else { $('#login-form').hidden = false; $('#start-continue').hidden = true; setTimeout(() => $('#op-name').focus(), 50); }
  $('#login-form').addEventListener('submit', e => {
    e.preventDefault();
    const name = $('#op-name').value.trim(), pin = $('#op-pin').value.trim();
    const err = $('#login-error');
    if (!name) { err.textContent = 'Enter your name.'; $('#op-name').focus(); return; }
    if (!/^\d{4,6}$/.test(pin)) { err.textContent = 'PIN must be 4 to 6 digits.'; $('#op-pin').focus(); return; }
    store.set('bc-op', name);
    enter(name); if (location.hash === '#/inspect') route(); else location.hash = '#/inspect';
  });
  ['#op-name', '#op-pin'].forEach(sel => $(sel).addEventListener('input', () => ($('#login-error').textContent = '')));
  refreshAll(); route();
}

function enter(name) {
  state.operator = name;
  $('#hello-op').textContent = `Signed in as ${name}`;
  $('#side-name').textContent = name;
  $('#side-avatar').textContent = name.split(/\s+/).map(w => w[0]).join('').slice(0, 2).toUpperCase();
  $('#login-form').hidden = true; $('#start-continue').hidden = false;
}
document.addEventListener('click', e => { if (e.target.closest('[data-logout]')) { store.del('bc-op'); location.hash = '#/'; location.reload(); } });

/* =================== Routing: website + app screens =================== */
const SCREENS = {
  '/inspect': ['Inspect', 'Inspect'],
  '/log': ['Log', 'Log'],
  '/settings': ['Settings', 'Settings'],
  '/board': ['Log', 'Board'],
};
function route() {
  const h = location.hash.slice(1);
  let p = SCREENS[h] ? h : h.startsWith('/board/') ? '/board' : '/';
  if (p === '/board') state.boardSel = decodeURIComponent(h.slice(7));
  if (!state.operator) p = '/';            // sign in on the start screen first
  state.route = p;
  const inApp = p !== '/';
  $('#site').hidden = inApp; $('#app').hidden = !inApp;
  Object.keys(SCREENS).forEach(k => ($('#view-' + k.slice(1)).hidden = k !== p));
  document.querySelectorAll('.side-nav a').forEach(a => a.classList.toggle('on', a.dataset.route === (p === '/board' ? '/log' : p)));
  if (inApp) $('#tb-title').textContent = SCREENS[p][1];
  $('#kbd-hint').hidden = p !== '/inspect';
  document.body.appendChild(FRAME); FRAME.hidden = true;
  if (p === '/inspect') { $('#inspect-video-host').appendChild(FRAME); FRAME.hidden = false; state.mode = 'inspect'; renderInspect(); }
  else if (p === '/log') renderLog();
  else if (p === '/board') renderBoardDetail();
  else if (p === '/settings') renderSettings();
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', () => { route(); if (pendingScroll) { const s = pendingScroll; pendingScroll = null; setTimeout(() => scrollToSel(s), 80); } });
let pendingScroll = null;
document.addEventListener('click', e => {
  const g = e.target.closest('[data-goto]');
  if (g) { e.preventDefault(); goHome(g.dataset.goto); return; }
  const a = e.target.closest('[data-scroll]');
  if (a) { e.preventDefault(); scrollToSel(a.getAttribute('href')); }
});
$('#nav-status').addEventListener('click', () => { const c = $('#connect'); c.hidden = !c.hidden; });
function goHome() { location.hash = '#/'; }
function scrollToSel(sel) { document.querySelector(sel)?.scrollIntoView({ behavior: 'smooth' }); }

/* =================== Start screen: animated sample board =================== */
// The board slides in, a scan line passes, defect boxes pop in one by one, then Llama's advice types itself out.
function drawHero() {
  const c = $('#hero-canvas'); if (!c) return;
  const board = makeBoard(4242, { minDefects: 3, types: ['short', 'spur', 'missing_hole'] });
  drawBoard(c.getContext('2d'), board, {});
  const host = $('#hero-boxes'); host.innerHTML = '';
  board.defects.slice(0, 3).forEach((d, i) => {
    const col = SEV_COLOR[DEFECTS[d.type].sev];
    host.insertAdjacentHTML('beforeend', `<div class="hbox" style="--d:${1.6 + i * 0.45}s;left:${(d.x / BW) * 100}%;top:${(d.y / BH) * 100}%;width:${(d.w / BW) * 100}%;height:${(d.h / BH) * 100}%;border-color:${col}"><span style="background:${col}">${DEFECTS[d.type].label} ${Math.round(d.conf * 100)}%</span></div>`);
  });
  const p = $('#hero-bubble-text');
  p.innerHTML = p.textContent.split(' ').map((w, i) => `<span class="w" style="--d:${3.3 + i * 0.07}s">${w}</span>`).join(' ');
}

/* =================== Phone =================== */
function setStatus(s) {
  state.status = s;
  const on = !!s.connected;
  $('#nav-status').classList.toggle('on', on);
  $('#nav-status').lastElementChild.textContent = on ? 'Phone connected' : 'No phone';
  ['#conn-state', '#conn-state2'].forEach(sel => {
    $(sel).classList.toggle('on', on);
    $(sel).lastElementChild.textContent = on ? `Connected · ${s.name || 'phone'}` : 'Waiting for phone…';
  });
  const sp = $('#side-phone'); sp.classList.toggle('on', on);
  sp.querySelector('b').textContent = on ? s.name || 'Phone' : 'No phone';
  sp.querySelector('.tiny').textContent = on ? `${s.fps ?? '—'} fps · ${s.battery ?? '—'}% battery` : 'Connect in Settings';
  $('#set-model').innerHTML = `<dt>Vision</dt><dd>${esc(on && s.model?.vision || '—')}</dd><dt>Advisor</dt><dd>${esc(on && s.model?.llm || '—')}</dd>`;
  const v = (k, el) => ($(el).textContent = on && s[k] != null ? s[k] : '—');
  v('fps', '#s-fps'); v('latency', '#s-lat'); v('tps', '#s-tps'); v('battery', '#s-bat'); v('temp', '#s-temp');
  const m = s.model;
  $('#i-model').textContent = on && m ? `${m.vision}  ·  ${m.llm}` : 'No phone connected';
  // Console stats bar and status bar
  const k = (el, val) => ($(el).textContent = on && val != null && val !== '' ? val : '—');
  // One FPS everywhere: live YOLO detections per second on the phone (0 while the phone is idle).
  k('#k-fps', s.idle ? 'Idle' : (s.detFps ?? null));
  $('#k-fps-u').hidden = !!(on && s.idle); k('#k-yolo', s.latency || null); k('#k-llm', s.tps || null); k('#k-bat', s.battery != null ? s.battery + '%' : null);
  $('#k-temp').textContent = on && s.temp != null ? `${s.temp}°C` : '';
  const ab = $('#ab-link'); ab.classList.toggle('on', on);
  ab.lastChild.textContent = on ? `${s.name || 'Phone'} connected` : 'Phone app not connected';
  setVideoEmpty();
  updateBadge();
  renderGolden();
  renderArm();
}
// Live numbers from the phone, under the video: shows the AI is running on the phone's Arm CPU.
const THERMAL = ['not throttled', 'warm', 'warm', 'throttling', 'throttling', 'throttling', 'critical'];
function renderArm() {
  const s = state.status, on = !!s.connected, c = state.current?.detectMs ? state.current : state.chat[state.chat.length - 1];
  const val = (v, u) => (on && v != null && v !== '' && v !== 0 ? `<b>${esc(v)}</b><small>${u}</small>` : '<b>—</b>');
  const hot = on && s.thermal >= 3;
  const chips = [
    ['cpu', 'Runs on', on ? '<b>Arm CPU</b>' : '<b>—</b>', 'ExecuTorch · XNNPACK on the phone CPU'],
    ['scan-search', 'YOLO11n', val(s.latency, 'ms/tile'), 'Defect detector time per 640 px tile'],
    ['timer', 'Verdict', c?.detectMs ? `<b>${(c.detectMs / 1000).toFixed(1)}</b><small>s</small>` : '<b>—</b>', 'Photo to PASS / REWORK / REJECT'],
    ['sparkles', 'Llama 3.2 1B', val(s.tps, 'tok/s'), 'Advice writing speed on the phone'],
    ['thermometer', on && s.thermal != null ? `Temp · ${THERMAL[s.thermal] || 'heat'}` : 'Temp', val(s.temp, '°C'), 'Phone battery temperature and Android thermal state'],
    ['battery-medium', 'Battery', val(s.battery, '%'), 'Phone battery'],
  ];
  $('#arm-strip').classList.toggle('off', !on);
  $('#arm-strip').innerHTML = chips.map(([ic, k, v, tip], i) => `<div class="arm-chip${i === 4 && hot ? ' hot' : ''}" title="${tip}"><i data-lucide="${ic}"></i><div><span>${k}</span><div class="v">${v}</div></div></div>`).join('');
  window.lucide?.createIcons();
}
function updateBadge() {
  const s = state.status, b = F('#live-badge'), live = s.connected && !s.idle && state.source?.running !== false;
  b.classList.toggle('on', !!live);
  b.lastChild.textContent = !s.connected ? 'Offline' : s.idle ? 'Phone idle' : live ? `Live · ${s.detFps ?? 0} fps` : 'Paused';
}
// What the empty video area says: never a demo; tell the operator exactly what to do.
function setVideoEmpty() {
  const s = state.status, el = F('#video-empty');
  el.hidden = !!state.frame && !!s.connected;
  if (el.hidden) return;
  const [t, sub] = !state.bridgeUp ? ['Laptop bridge is not running.', 'Start it with: python3 server/bridge.py']
    : !s.connected ? ['Phone app not connected.', 'Open BoardCheck on the phone. With the USB cable: adb reverse tcp:8765 tcp:8765, then Connect → USB link.']
    : s.idle ? ['Phone is idle (saving heat).', 'Press Inspect board here, or tap Live on the phone.']
    : ['Waiting for the camera…', 'Point the phone at the board.'];
  F('#video-empty-t').textContent = t; F('#video-empty-s').textContent = sub;
}

function handlersFor(src) {
  const mine = () => state.source === src;
  return {
    onStatus: s => {
      if (s.connected && !state.source && src === state.bridge) state.source = src;
      if (mine()) setStatus(s);
    },
    onFrame: f => {
      if (!mine()) return;
      state.frame = f; F('#video-empty').hidden = true;
      if (f.newBoard) {
        if (state.mode === 'verify' && state.vres) { state.vres.items = null; renderVerify(); }
        if (state.mode === 'inspect') { state.current = { analyzing: true }; renderInspect(); }
      }
    },
    onDetections: d => { if (mine()) onDetections(d); },
    onLive: d => { if (mine()) state.liveDet = { ...d, t: performance.now() }; },
    onAdviceToken: t => {
      if (!mine()) return;
      const tgt = state.mode === 'verify' ? state.vres : state.current;
      if (!tgt || tgt.analyzing) return;
      tgt.advice = (tgt.advice || '') + t;
      const el = state.mode === 'verify' ? $('#v-advice') : $('#advice');
      if (el) { el.textContent = tgt.advice; el.classList.add('typing'); }
    },
    onAdviceDone: r => { if (mine()) onAdviceDone(r); },
    // The phone's PCB check found no circuit board: show it, never log it as a pass.
    onNotPcb: r => { if (!mine()) return; F('#live-board').textContent = r.boardId; state.current = { notPcb: true, boardId: r.boardId, prob: r.prob, t: Date.now() }; state.chat.push(state.current); renderInspect(); },
    onGolden: m => { if (mine()) onGolden(m); },
  };
}

async function initConnect() {
  let info = null;
  try { const r = await fetch('/api/info', { cache: 'no-store' }); if (r.ok) info = await r.json(); } catch {}
  const qr = $('#qr');
  if (info) {
    const phoneUrl = `ws://${info.ip}:${info.ws_port}/phone`;
    $('#qr-url').textContent = phoneUrl; $('#qr-url2').textContent = phoneUrl;
    if (window.QRCode) {
      new QRCode(qr, { text: phoneUrl, width: 280, height: 280, correctLevel: 0, colorDark: '#1A1A1A', colorLight: '#ffffff' });
      new QRCode($('#qr2'), { text: phoneUrl, width: 132, height: 132, colorDark: '#1A1A1A', colorLight: '#ffffff' });
    }
    const br = new BridgePhone(`ws://${location.hostname}:${info.ws_port}/ui`, null, ok => {
      state.bridgeUp = ok; setVideoEmpty();
      $('#bridge-note').textContent = ok ? 'Laptop bridge running. Waiting for the phone app.' : 'Laptop bridge offline. Restart server/bridge.py.';
    });
    br.h = handlersFor(br); state.bridge = br; br.connect();
  } else {
    qr.innerHTML = '<p class="qr-empty">Start <code>server/bridge.py</code> to pair a real phone.</p>';
    $('#qr2').innerHTML = '<p class="qr-empty">Bridge not running.</p>';
    $('#bridge-note').textContent = 'Laptop bridge not running. Start it with: python3 server/bridge.py';
  }
}

/* =================== Shared drawing =================== */
function canvasFrom(w, h, fn) { const c = document.createElement('canvas'); c.width = w; c.height = h; fn?.(c.getContext('2d')); return c; }
function mapBox(cw, ch, w, h, b) {
  const s = Math.min(cw / w, ch / h), ox = (cw - w * s) / 2, oy = (ch - h * s) / 2;
  return { x: ox + b.x * s, y: oy + b.y * s, w: b.w * s, h: b.h * s };
}
// "Show why": attention glow centred on what drove the detection.
function heat(ctx, b) {
  const cx = b.x + b.w / 2, cy = b.y + b.h / 2, r = Math.max(b.w, b.h) * 1.25;
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
  g.addColorStop(0, 'rgba(255,60,30,0.65)'); g.addColorStop(0.45, 'rgba(255,180,0,0.35)'); g.addColorStop(1, 'rgba(255,220,0,0)');
  ctx.fillStyle = g; ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
}
const visible = items => (items || []).filter(i => !i.suppressed);
const effective = items => visible(items).filter(i => i.review !== 'false');
const thrOf = t => state.thr[t === 'open_circuit' ? 'open' : t] ?? phoneThr(t);

function drawBoxes(ctx, items, w, h, k = 1) {
  const cw = ctx.canvas.width, ch = ctx.canvas.height;
  visible(items).forEach((it, n) => {
    const b = mapBox(cw, ch, w, h, it);
    if (state.showWhy && it.review !== 'false') heat(ctx, b);
    const isFalse = it.review === 'false';
    const col = isFalse ? '#B4B2A9' : SEV_COLOR[DEFECTS[it.type]?.sev] || '#fff';
    if (state.selId === it.id) {   // selected in the defect list: halo so the operator finds it on the board
      ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,0.95)'; ctx.lineWidth = 9 * k; ctx.shadowColor = col; ctx.shadowBlur = 18 * k;
      ctx.beginPath(); ctx.roundRect(b.x - 6 * k, b.y - 6 * k, b.w + 12 * k, b.h + 12 * k, 12 * k); ctx.stroke(); ctx.restore();
    }
    ctx.strokeStyle = col; ctx.lineWidth = 3 * k; ctx.setLineDash(isFalse ? [6 * k, 5 * k] : []);
    ctx.beginPath(); ctx.roundRect(b.x, b.y, b.w, b.h, 8 * k); ctx.stroke(); ctx.setLineDash([]);
    const label = `${n + 1}  ${DEFECTS[it.type]?.label || it.type}${isFalse ? ' · false call' : ` ${Math.round(it.conf * 100)}%`}`;
    ctx.font = `600 ${14 * k}px DM Sans, sans-serif`;
    const tw = ctx.measureText(label).width + 14 * k;
    ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect(b.x - 1, b.y - 25 * k, tw, 21 * k, 6 * k); ctx.fill();
    ctx.fillStyle = '#1A1A1A'; ctx.fillText(label, b.x + 6 * k, b.y - 10 * k);
  });
}

function crop(dest, src, box, pad, color, why) {
  const ctx = dest.getContext('2d');
  ctx.fillStyle = '#111'; ctx.fillRect(0, 0, dest.width, dest.height);
  if (!src) { ctx.fillStyle = '#888'; ctx.font = '500 22px DM Sans'; ctx.textAlign = 'center'; ctx.fillText('Image not kept', dest.width / 2, dest.height / 2); ctx.textAlign = 'left'; return; }
  const cx = box.x + box.w / 2, cy = box.y + box.h / 2, ar = dest.width / dest.height;
  const h = Math.max(box.h, box.w / ar) * pad, w = h * ar;
  ctx.drawImage(src, cx - w / 2, cy - h / 2, w, h, 0, 0, dest.width, dest.height);
  const s = dest.width / w;
  const b = { x: (box.x - (cx - w / 2)) * s, y: (box.y - (cy - h / 2)) * s, w: box.w * s, h: box.h * s };
  if (why) heat(ctx, b);
  if (color) { ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath(); ctx.roundRect(b.x, b.y, b.w, b.h, 8); ctx.stroke(); }
}

// Whole-board map with a crosshair pointing at the defect.
function pointer(dest, src, box, w, h) {
  const ctx = dest.getContext('2d');
  ctx.fillStyle = '#111'; ctx.fillRect(0, 0, dest.width, dest.height);
  if (src) { const b = mapBox(dest.width, dest.height, w, h, { x: 0, y: 0, w, h }); ctx.globalAlpha = 0.75; ctx.drawImage(src, b.x, b.y, b.w, b.h); ctx.globalAlpha = 1; }
  const p = mapBox(dest.width, dest.height, w, h, box), cx = p.x + p.w / 2, cy = p.y + p.h / 2;
  ctx.strokeStyle = '#EBD9FF'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, dest.height); ctx.moveTo(0, cy); ctx.lineTo(dest.width, cy); ctx.stroke();
  ctx.fillStyle = '#E8603C'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy, 7, 0, 7); ctx.fill(); ctx.stroke();
}

function whereText(it, w) {
  const mm = 100 / w; // board drawn as 100 mm wide
  const x = ((it.x + it.w / 2) * mm).toFixed(1), y = ((it.y + it.h / 2) * mm).toFixed(1);
  let rel = it.near ? `near ${it.near}` : '';
  if (it.near && it.nearDx != null) {
    const dx = it.nearDx * mm, dy = it.nearDy * mm;
    rel = Math.abs(dx) > Math.abs(dy) ? `${Math.round(Math.abs(dx))} mm ${dx < 0 ? 'left' : 'right'} of ${it.near}` : `${Math.round(Math.abs(dy))} mm ${dy < 0 ? 'above' : 'below'} ${it.near}`;
  }
  return { coords: `x ${x} mm · y ${y} mm from top-left`, rel };
}

function renderLoop(t) {
  const c = F('#live-canvas'), ctx = c.getContext('2d'), f = state.frame;
  if (f && !FRAME.hidden) {
    { ctx.fillStyle = '#0d0d0d'; ctx.fillRect(0, 0, c.width, c.height); const b = mapBox(c.width, c.height, f.w, f.h, { x: 0, y: 0, w: f.w, h: f.h }); ctx.drawImage(f.img, b.x, b.y, b.w, b.h); }
    const r = state.mode === 'verify' ? state.vres : state.current;
    const L = state.liveDet && performance.now() - state.liveDet.t < 1200 ? state.liveDet : null;
    if (L) drawLive(ctx, L, mapBox(c.width, c.height, f.w, f.h, { x: 0, y: 0, w: f.w, h: f.h }));
    else if (r && r.items) { ctx.globalAlpha = 0.8 + 0.2 * Math.sin(t / 220); drawBoxes(ctx, r.items, r.w, r.h, 1.3); ctx.globalAlpha = 1; }
    else if (state.source?.running) { ctx.fillStyle = 'rgba(235,217,255,0.4)'; ctx.fillRect(0, (t / 5) % c.height, c.width, 3); }
  }
  requestAnimationFrame(renderLoop);
}

// Live YOLO on the phone's camera stream: dashed zone, boxes, FPS. Coordinates arrive as 0..1 of the frame.
function drawLive(ctx, L, b) {
  const X = v => b.x + v * b.w, Y = v => b.y + v * b.h;
  ctx.save();
  ctx.setLineDash([10, 7]); ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 2;
  ctx.strokeRect(X(L.zone.x), Y(L.zone.y), L.zone.w * b.w, L.zone.h * b.h); ctx.setLineDash([]);
  ctx.font = '600 13px system-ui, sans-serif';
  for (const it of L.items || []) {
    const col = SEV_COLOR[DEFECTS[it.type]?.sev] || '#fff';
    ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.strokeRect(X(it.x), Y(it.y), it.w * b.w, it.h * b.h);
    const lab = (DEFECTS[it.type]?.label || it.type) + ' ' + Math.round(it.conf * 100) + '%';
    const tw = ctx.measureText(lab).width + 10;
    ctx.fillStyle = col; ctx.fillRect(X(it.x), Y(it.y) - 20, tw, 18);
    ctx.fillStyle = '#111'; ctx.fillText(lab, X(it.x) + 5, Y(it.y) - 6);
  }
  const n = (L.items || []).length, msg = `LIVE · ${L.fps} FPS · ${L.ms} ms · ${n} defect${n === 1 ? '' : 's'} in zone`;
  ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(b.x + 10, b.y + 10, ctx.measureText(msg).width + 16, 24);
  ctx.fillStyle = '#fff'; ctx.fillText(msg, b.x + 18, b.y + 27);
  ctx.restore();
}

/* =================== Detections =================== */
function onDetections(d) {
  const clean = canvasFrom(d.w, d.h, d.clean);
  let golden = d.golden ? canvasFrom(d.w, d.h, d.golden) : null;
  if (!golden && state.goldenImg) golden = canvasFrom(d.w, d.h, ctx => ctx.drawImage(state.goldenImg, 0, 0, d.w, d.h));
  const stamp = Date.now();
  const items = d.items.map((it, i) => ({ ...it, id: `${d.boardId}-${stamp}-${i}`, review: null, suppressed: it.conf < thrOf(it.type) }));
  F('#live-board').textContent = d.boardId;
  if (state.mode === 'verify' && state.vres) {
    Object.assign(state.vres, { items, clean, golden, w: d.w, h: d.h, advice: '', done: false });
    renderVerify(); return;
  }
  const override = $('#i-override').value.trim();
  state.current = { boardId: override || d.boardId, idSource: override ? 'manual' : d.idSource || 'qr', items, w: d.w, h: d.h, clean, golden, advice: '', t: stamp, logId: null,
    detectMs: d.detectMs, tiles: d.tiles, msPerTile: d.msPerTile, quality: d.quality, pcbProb: d.pcbProb,
    goldenOn: d.goldenOn, goldenHidden: d.goldenHidden || 0 };
  renderInspect(); renderArm();
}

function onAdviceDone({ ms, advice, source }) {
  if (state.mode === 'verify' && state.vres?.items) {
    state.vres.done = true;
    $('#v-advice')?.classList.remove('typing');
    renderVerify(); return;
  }
  const c = state.current; if (!c || c.analyzing) return;
  $('#advice').classList.remove('typing');
  c.ms = ms; c.adv = advice; c.source = source;
  state.chat.push(c);
  const id = c.boardId + '-' + c.t;
  c.logId = id;
  state.images[id] = { clean: c.clean, golden: c.golden, thumb: thumbOf(c) };
  const thumb = state.images[id].thumb;
  const verdict = verdictFor(effective(c.items));
  state.log.unshift({
    id, boardId: c.boardId, idSource: c.idSource, t: c.t, op: state.operator, ms, w: c.w, h: c.h,
    model: state.status.model || null,
    items: c.items.map(({ id, type, conf, x, y, w, h, near, nearDx, nearDy, review, suppressed }) => ({ id, type, conf, x, y, w, h, near, nearDx, nearDy, review, suppressed })),
    verdict, status: verdict === 'pass' ? 'passed' : 'awaiting_fix', advice: advice ? adviceText(advice) : c.advice, verify: [],
    detectMs: c.detectMs, trend: advice?.trend || null, source: source || null, thumb,
  });
  chainAdd('inspected', state.log[0]);
  saveLog(); renderInspect(); refreshAll();
}

function thumbOf(c) {
  const t = canvasFrom(360, 240, x => { x.fillStyle = '#111'; x.fillRect(0, 0, 360, 240); });
  const x = t.getContext('2d'), b = mapBox(360, 240, c.w, c.h, { x: 0, y: 0, w: c.w, h: c.h });
  x.drawImage(c.clean, b.x, b.y, b.w, b.h);
  const why = state.showWhy; state.showWhy = false; drawBoxes(x, c.items, c.w, c.h, 0.55); state.showWhy = why;
  return t.toDataURL('image/jpeg', 0.75);
}

/* =================== Page 2 · Inspect =================== */
/* =================== Confidence =================== */
// Calibrated on held-out boards (ml/calibrate.py): raw YOLO score -> fraction of such detections that were really correct.
let CAL = null;
fetch('js/calibration.json', { cache: 'no-store' }).then(r => r.ok ? r.json() : null).then(j => { CAL = j; renderInspect(); renderSettings(); }).catch(() => {});
const retakeLevel = () => store.get('bc-retake', 0.6);
const CHECK_LEVEL = 0.7;
function calConf(type, raw) {
  const cv = CAL?.curves?.[type === 'open' ? 'open_circuit' : type];
  if (!cv) return raw;
  const g = CAL.grid;
  if (raw <= g[0]) return cv[0];
  if (raw >= g[g.length - 1]) return cv[cv.length - 1];
  const k = g.findIndex(x => x >= raw), t = (raw - g[k - 1]) / (g[k] - g[k - 1]);
  return cv[k - 1] + t * (cv[k] - cv[k - 1]);
}
// Board confidence = the weakest of: the defects that decide the verdict, the photo, the PCB check.
function boardConf(c) {
  const vis = effective(c.items || []), parts = [];
  if (vis.length) {
    const worst = Math.max(...vis.map(i => SEV_RANK[DEFECTS[i.type]?.sev] || 1));
    const driving = vis.filter(i => (SEV_RANK[DEFECTS[i.type]?.sev] || 1) === worst);
    const pAny = 1 - driving.reduce((m, i) => m * (1 - calConf(i.type, i.conf)), 1);   // chance at least one of them is real
    const name = DEFECTS[driving[0].type]?.label.toLowerCase();
    parts.push([pAny, `not sure the ${name}${driving.length > 1 ? 's are' : ' is'} real`]);
  }
  if (c.quality) parts.push([0.4, `photo ${c.quality.split(',')[0]}`]);
  if (c.pcbProb != null && c.pcbProb < 0.9) parts.push([Math.max(0.3, c.pcbProb), 'board not clearly in view']);
  if (!parts.length) return { p: 0.95, why: '' };
  parts.sort((a, b) => a[0] - b[0]);
  return { p: Math.min(0.99, parts[0][0]), why: parts[0][1] };
}
const confLabel = p => (p >= 0.8 ? ['high', 'High'] : p >= retakeLevel() ? ['mid', 'Medium'] : ['low', 'Low']);

/* Inspect: live video on the left, Llama chat on the right.
   Each inspected board becomes one chat message: verdict, every defect found, then how to fix (streamed). */
state.chat = [];                       // finished messages, newest last
function setVerdict() {}
// Inspect shows only the board being inspected now; every earlier board is in the Log (with its defects and advice).
function renderInspect() {
  const c = state.current, host = $('#chat');
  const shown = c || state.chat[state.chat.length - 1];
  host.innerHTML = shown ? chatMsg(shown, shown === c && !state.chat.includes(c))
    : `<div class="chat-empty"><i data-lucide="message-square-text"></i><p>Place a board under the phone and press <b>Inspect board</b>.</p><span>Llama will list every defect it finds, then tell you how to fix each one. Earlier boards are in the Log.</span></div>`;
  window.lucide?.createIcons();
  host.scrollTop = 0;
}
// Board check = the phone's "is there a circuit board in this photo?" model (PcbGate). A clean PASS needs >= 0.70.
const GATE_PASS = 0.7;
const gateChip = p => p == null ? '' : `<span class="gchip ${p >= GATE_PASS ? 'ok' : 'no'}" title="Phone's board check: is there a circuit board in this photo? A clean board needs ${Math.round(GATE_PASS * 100)}% to pass.">
  <i data-lucide="${p >= GATE_PASS ? 'circuit-board' : 'scan-eye'}"></i>Board check ${Math.round(p * 100)}%</span>`;
function chatMsg(c, isLive) {
  const time = new Date(c.t || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (c.notPcb) return `<div class="msg"><div class="msg-h"><span class="vpill big none">NO BOARD</span>${gateChip(c.prob)}<span class="muted tiny">${time}</span></div>
    <p>I can't see a circuit board in this photo. The board check scored it <b>${Math.round((c.prob || 0) * 100)}%</b>; a board needs at least <b>${Math.round(GATE_PASS * 100)}%</b> before it can pass.</p>
    <p>Hold the board flat under the camera so it fills most of the frame, then inspect again.</p></div>`;
  if (c.analyzing) return `<div class="msg"><div class="msg-h"><span class="vpill none">Checking…</span></div><p class="typing-dots"><i></i><i></i><i></i></p></div>`;
  const vis = visible(c.items), v = verdictFor(effective(c.items));
  const list = vis.length ? `<p class="msg-l">I found <b>${vis.length} defect${vis.length > 1 ? 's' : ''}</b>:</p><ol class="msg-defects">${vis.map(i => {
      const D = DEFECTS[i.type] || { label: i.type, sev: 'low' };
      const p = calConf(i.type, i.conf);
      return `<li><span class="tchip sev-${D.sev}">${cap(SEV_NAME[D.sev])}</span> <b>${D.label}</b> <span class="muted">${esc(whereText(i, c.w).rel)}</span>
        <span class="dconf ${p >= CHECK_LEVEL ? 'ok' : 'chk'}">${Math.round(p * 100)}%${p < CHECK_LEVEL ? ' · check by eye' : ''}</span></li>`; }).join('')}</ol>`
    : '<p>No defects found. This board is good to go to the next station.</p>';
  const a = c.adv;
  const fix = !vis.length ? '' : a ? `<p class="msg-l">How to fix it:</p><ol class="msg-steps">${(a.steps || []).map(x => `<li>${esc(x)}</li>`).join('')}</ol>
      ${a.likely_cause ? `<p class="msg-cause"><b>Likely cause:</b> ${esc(a.likely_cause)}</p>` : ''}
      ${a.trend ? `<p class="msg-alert">⚠ ${esc(a.trend)}. Check the process step, not just this board.</p>` : ''}`
    : `<p class="msg-l">How to fix it:</p><p class="msg-stream" ${isLive ? 'id="advice"' : ''}>${esc(c.advice || '')}<span class="caret"></span></p>`;
  const meta = [c.detectMs && `verdict ${(c.detectMs / 1000).toFixed(1)} s`, c.ms && `advice ${(c.ms / 1000).toFixed(1)} s`].filter(Boolean).join(' · ');
  const q = (c.quality ? `<p class="msg-alert">📷 Photo check: ${esc(c.quality)}.</p>` : '') +
    (c.goldenHidden ? `<p class="tiny muted">${c.goldenHidden} detection${c.goldenHidden > 1 ? 's' : ''} matched the golden board (design features) and ${c.goldenHidden > 1 ? 'were' : 'was'} hidden.</p>` : '');
  const bc = boardConf(c), [lv, lt] = confLabel(bc.p), low = bc.p < retakeLevel();
  const conf = `<span class="bconf ${lv}" title="How sure BoardCheck is about this result">Confidence ${Math.round(bc.p * 100)}% · ${lt}</span>`;
  const retake = low ? `<div class="retake"><span>⚠ <b>Not sure about this result</b>: ${esc(bc.why)}. Check by eye or retake the photo.</span>${isLive || c === state.chat[state.chat.length - 1] ? '<button class="btn btn-sm btn-primary" data-retake>Retake photo</button>' : ''}</div>` : '';
  return `<div class="msg ${v}"><div class="msg-h"><span class="vpill big ${v}">${v.toUpperCase()}</span><b>${esc(c.boardId)}</b>${conf}${gateChip(c.pcbProb)}<span class="muted tiny">${time}${meta ? ' · ' + meta : ''}</span></div>${retake}${q}${list}${fix}</div>`;
}
$('#chat').addEventListener('click', e => { if (e.target.closest('[data-retake]')) $('#btn-next').click(); });
$('#chat-clear').addEventListener('click', () => { state.chat = []; state.current = null; renderInspect(); });

const SEV_NAME = { high: 'critical', medium: 'major', low: 'minor' };
const SRC_NAME = { llama: 'Llama 3.2 1B on phone', cache: 'Llama (cached, same pattern)' };
function adviceText(a) {
  return [a.summary, ...(a.steps || []).map((s, i) => `${i + 1}. ${s}`), a.likely_cause ? `Likely cause: ${a.likely_cause}` : '', a.trend ? `Trend: ${a.trend}` : ''].filter(Boolean).join('\n');
}
// Structured advice once the phone is done; the raw stream while Llama is writing.
function renderAdvice(c) {
  const el = $('#advice'), meta = $('#advice-meta'), a = c.adv;
  $('#advice-state').textContent = c.logId ? 'done' : 'writing…';
  if (!a) {
    if (c.advice) el.textContent = c.advice;
    else el.innerHTML = `<p class="ph">${c.items?.length === 0 ? 'No defects, no repair needed. The board can move to the next station.' : 'Llama is reading the findings…'}</p>`;
    meta.innerHTML = ''; return;
  }
  el.classList.remove('typing');
  el.innerHTML = `<b class="adv-sum">${esc(a.summary)}</b>
    ${a.steps?.length ? `<p class="label">What to do</p><ol class="adv-steps">${a.steps.map(s => `<li>${esc(s)}</li>`).join('')}</ol>` : ''}
    ${a.likely_cause ? `<p class="adv-cause"><span class="label">Likely cause</span> ${esc(a.likely_cause)}</p>` : ''}`;
  const src = c.source || a.source || '';
  meta.innerHTML = `<span class="mchip">${esc(SRC_NAME[src] || src)}</span>` +
    (src.startsWith('llama') || src === 'cache' ? `<span class="mchip ${a.grounded ? 'ok' : 'warn'}">${a.grounded ? '✓ Checked against repair guide' : 'Llama text replaced by repair guide'}</span>` : '');
}
// Why this verdict, how long it took, and any photo / process alerts.
function renderMeta(c) {
  const vis = visible(c.items || []);
  const n = s => vis.filter(i => (DEFECTS[i.type]?.sev) === s).length;
  const parts = [['high', 'critical'], ['medium', 'major'], ['low', 'minor']].map(([k, l]) => n(k) ? `${n(k)} ${l}` : '').filter(Boolean);
  $('#i-reason').innerHTML = vis.length ? `<b>${parts.join(' · ')}</b> → ${n('high') ? 'reject: a critical defect breaks the circuit' : 'rework: repairable defects only'}` : '<b>No defects</b> → pass';
  const t = [];
  if (c.detectMs) t.push(`Verdict in ${(c.detectMs / 1000).toFixed(1)} s`);
  if (c.ms) t.push(`advice in ${(c.ms / 1000).toFixed(1)} s`);
  if (c.tiles) t.push(`${c.tiles} tiles × ${c.msPerTile} ms`);
  $('#i-perf').textContent = t.length ? t.join(' · ') + ' · on phone' : '';
  const al = [];
  if (c.quality) al.push(`<div class="alert warn"><i data-lucide="camera-off"></i><span><b>Photo check:</b> ${esc(c.quality)}. Results may be unreliable.</span></div>`);
  if (c.adv?.trend) al.push(`<div class="alert proc"><i data-lucide="trending-up"></i><span><b>Process alert:</b> ${esc(c.adv.trend)}. Check the process step, not just this board.</span></div>`);
  $('#i-alerts').innerHTML = al.join('');
  if (al.length) window.lucide?.createIcons();
}

// One defect card: what it is, where it is, how to fix it, and the operator's review.
function makeDefectCard(it, n, c, interactive) {
  {
    const D = DEFECTS[it.type] || { label: it.type, sev: 'low', cause: '', action: '' };
    const col = SEV_COLOR[D.sev];
    const wt = whereText(it, c.w);
    const card = document.createElement('article');
    card.className = 'dcard' + (it.review ? ' ' + (it.review === 'false' ? 'false' : 'confirmed') : '') + (state.selId === it.id || !interactive ? ' open' : '');
    const mark = it.review === 'confirmed' ? '<span class="rv ok">✓</span>' : it.review === 'false' ? '<span class="rv no">✕</span>' : '';
    card.innerHTML = `
      <header class="drow"><span class="nr" style="background:${col}">${n + 1}</span><h3>${D.label}</h3><span class="sev sev-${D.sev}">${SEV_NAME[D.sev] || D.sev}</span>
        <span class="dwhere">${esc(wt.rel)}</span><span class="conf">${Math.round(it.conf * 100)}%</span>${mark}<i data-lucide="chevron-down" class="chev"></i></header>
      <div class="dbody"><div class="pair"><figure><canvas width="400" height="300" data-c="this"></canvas><figcaption>This board${state.showWhy ? ' · model attention' : ''}</figcaption></figure>
        <figure><canvas width="400" height="300" data-c="gold"></canvas><figcaption>Golden board, same spot</figcaption></figure></div>
      <div class="where"><canvas width="240" height="160" data-c="map"></canvas><div><p class="label">Where</p><p><b>${esc(wt.rel)}</b></p><p class="tiny muted">${wt.coords}</p></div></div>
      <div class="fix"><p class="label">How to fix</p><p>${cap(D.action)}.</p><p class="tiny muted">Likely cause: ${D.cause}.</p></div>
      <div class="review">
        <button class="btn ${it.review === 'confirmed' ? 'on-confirm' : ''}" data-r="confirmed">✓ Confirm defect</button>
        <button class="btn ${it.review === 'false' ? 'on-false' : ''}" data-r="false">✕ False call</button>
        <span class="review-note">${it.review === 'confirmed' ? 'Confirmed' : it.review === 'false' ? 'Marked as false call' : 'Not reviewed'}</span>
      </div></div>`;
    // Click a row: open it, highlight that box on the photo (one open at a time).
    card.querySelector('.drow').addEventListener('click', () => {
      const open = !card.classList.contains('open');
      card.parentElement?.querySelectorAll('.dcard.open').forEach(x => x.classList.remove('open'));
      card.classList.toggle('open', open); state.selId = open ? it.id : null;
    });
    crop(card.querySelector('[data-c="this"]'), c.clean, it, 3, col, state.showWhy);
    crop(card.querySelector('[data-c="gold"]'), c.golden, it, 3, 'rgba(235,217,255,0.9)', false);
    pointer(card.querySelector('[data-c="map"]'), c.golden || c.clean, it, c.w, c.h);
    if (interactive) card.querySelectorAll('[data-r]').forEach(b => b.addEventListener('click', () => review(it, b.dataset.r)));
    return card;
  }
}

function review(it, val) {
  const c = state.current;
  it.review = it.review === val ? null : val;
  // Feedback sample for retraining.
  if (it.review) {
    const cv = canvasFrom(160, 120); crop(cv, c.clean, it, 2.5, null, false);
    state.fb.unshift({ t: Date.now(), boardId: c.boardId, type: it.type, conf: it.conf, label: it.review === 'false' ? 'false_call' : 'confirmed', box: { x: it.x, y: it.y, w: it.w, h: it.h, frame_w: c.w, frame_h: c.h }, model: state.status.model?.vision || null, op: state.operator, crop: cv.toDataURL('image/jpeg', 0.7) });
    state.fb = state.fb.slice(0, 200); store.set('bc-fb', state.fb);
  }
  // A false call raises that type's threshold on the phone.
  if (it.review === 'false') {
    const before = thrOf(it.type), after = Math.min(0.95, Math.max(before, Math.round((it.conf + 0.03) * 100) / 100));
    if (after > before) {
      state.thr[it.type] = after; store.set('bc-thr', state.thr);
      state.source?.setThresholds?.(state.thr);
      toast(`${DEFECTS[it.type].label} threshold raised to ${after.toFixed(2)} for the next boards.`);
    }
  }
  const e = state.log.find(x => x.id === c.logId);
  if (e) {
    const li = e.items.find(x => x.id === it.id); if (li) li.review = it.review;
    e.verdict = verdictFor(effective(e.items));
    if (e.status === 'passed' || e.status === 'awaiting_fix') e.status = e.verdict === 'pass' ? 'passed' : 'awaiting_fix';
    state.images[e.id].thumb = thumbOf(c);
    chainAdd('reviewed', e);
    saveLog();
  }
  renderInspect(); refreshAll();
}

// Keyboard shortcuts on the Inspect screen: N next board, C confirm, F false call.
document.addEventListener('keydown', e => {
  if (state.route !== '/inspect' || e.metaKey || e.ctrlKey || e.altKey) return;
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === 'n') { e.preventDefault(); $('#btn-next').click(); return; }
  if (k !== 'c' && k !== 'f') return;
  const c = state.current; if (!c || c.analyzing || !c.items) return;
  const it = visible(c.items).find(i => !i.review);
  if (!it) return toast('Every defect on this board is reviewed.');
  e.preventDefault(); review(it, k === 'c' ? 'confirmed' : 'false');
});

$('#btn-next').addEventListener('click', () => {
  if (!state.source || !state.status.connected) return toast('Connect the phone first (Settings → Phone).');
  state.mode = 'inspect'; state.source.next(); updateBadge();
});
$('#btn-pause').addEventListener('click', () => { state.source?.stop(); updateBadge(); });
$('#i-override').addEventListener('change', e => {
  const c = state.current, v = e.target.value.trim(); if (!c || c.analyzing || !v) return;
  c.boardId = v; c.idSource = 'manual';
  const en = state.log.find(x => x.id === c.logId); if (en) { en.boardId = v; en.idSource = 'manual'; chainAdd('board ID changed', en); saveLog(); }
  renderInspect(); refreshAll();
});

/* =================== Page 3 · Verify =================== */
const queue = () => state.log.filter(e => e.status === 'awaiting_fix' || e.status === 'back_to_rework');

function renderVerify() {
  const q = queue();
  if (!q.some(e => e.id === state.verifySel) && !state.log.some(e => e.id === state.verifySel && state.vres)) state.verifySel = q[0]?.id || null;   // never a blank page
  $('#queue').innerHTML = q.length ? q.map(e => `<li><button class="${e.id === state.verifySel ? 'sel' : ''}" data-sel="${e.id}">
      <b>${esc(e.boardId)}</b><span class="tiny muted">${effective(e.items).length} defect${effective(e.items).length === 1 ? '' : 's'} · ${new Date(e.t).toLocaleTimeString()}</span>
      <span><span class="status ${e.status}">${STATUS_LABEL[e.status]}</span></span></button></li>`).join('')
    : '<li class="tiny muted">Nothing waiting. Boards you send to rework show up here.</li>';
  const rel = state.log.filter(e => e.status === 'released').slice(0, 6);
  $('#released').innerHTML = rel.length ? rel.map(e => `<li><span>${esc(e.boardId)}</span><span class="status released">Released</span></li>`).join('') : '<li class="tiny muted">None yet.</li>';
  document.querySelectorAll('[data-sel]').forEach(b => b.addEventListener('click', () => { state.verifySel = b.dataset.sel; state.vres = null; renderVerify(); }));

  document.body.appendChild(FRAME); FRAME.hidden = true; // park it before the panel is rebuilt
  const main = $('#verify-main');
  const e = state.log.find(x => x.id === state.verifySel);
  if (!e) { FRAME.hidden = true; main.innerHTML = `<div class="panel-c big-empty tall"><i data-lucide="shield-check"></i><p>Nothing waiting for rework</p><span>When a board is sent to rework from Inspect, it appears on the left. Re-scan it here after the repair to check the fix.</span></div>`; window.lucide?.createIcons(); return; }

  const before = effective(e.items);
  const vr = state.vres && state.vres.entryId === e.id ? state.vres : null;
  main.innerHTML = `
    <div class="vtop">
      <div id="verify-video-host" class="video-host"></div>
      <div class="vinfo">
        <p class="eyebrow">Board</p>
        <div class="id-row"><span class="serif id-big">${esc(e.boardId)}</span>${isOpen(e) ? `<button class="btn btn-primary btn-fix" data-fixd="${e.id}"><i data-lucide="check"></i>Mark fixed</button>` : `<span class="st ${isFixed(e) ? 'fixed' : 'passed'}">${STATUS_LABEL[e.status]}</span>`}</div>
        <p class="tiny muted">Inspected ${new Date(e.t).toLocaleString()} by ${esc(e.op)} · ${e.verify.length} re-check${e.verify.length === 1 ? '' : 's'} so far</p>
        <p><b>To check:</b> ${before.map(i => `${DEFECTS[i.type].label} ${esc(whereText(i, e.w).rel)}`).join(' · ') || 'nothing'}</p>
        <button class="btn btn-primary" id="btn-rescan">${vr ? 'Re-scan again' : 'Re-scan repaired board'}</button>
        <div class="summary"><div class="advice-head"><span class="bubble-tag">Llama · re-check</span></div><p id="v-advice" class="advice-text">${vr ? esc(vr.advice || '') : 'Place the repaired board under the camera, matching the QR sticker, then re-scan.'}</p></div>
      </div>
    </div>
    <div class="cmp" id="cmp"></div>`;
  const frame = FRAME; $('#verify-video-host').appendChild(frame); frame.hidden = false;
  $('#btn-rescan').addEventListener('click', () => rescan(e));
  if (vr && vr.items) renderCompare(e, vr);
  else if (vr) $('#cmp').innerHTML = '<div class="empty-card"><p class="serif">Checking…</p><p class="tiny muted">The phone is re-scanning the board.</p></div>';
  else {   // before the re-scan: show what was found the first time, so the operator knows what to repair
    const th = state.images[e.id]?.thumb || e.thumb;
    const rows = before.map((i, n) => { const D = DEFECTS[i.type] || { label: i.type, sev: 'low', action: '' };
      return `<tr><td>${n + 1}</td><td><b>${D.label}</b></td><td><span class="tchip sev-${D.sev}">${cap(SEV_NAME[D.sev])}</span></td><td>${esc(whereText(i, e.w).rel)}</td><td class="tiny">${cap(D.action || '')}</td></tr>`; }).join('');
    $('#cmp').innerHTML = `<div class="panel-c vfound"><div class="vf-img">${th ? `<img src="${th}" alt="">` : '<div class="big-empty small"><i data-lucide="image-off"></i><p>No photo saved</p><span>Photos are kept for boards inspected after this update.</span></div>'}</div>
      <div class="vf-list"><p class="chart-title">Found at first inspection</p>${rows ? `<table class="mini-table"><thead><tr><th>#</th><th>Defect</th><th>Severity</th><th>Where</th><th>How to fix</th></tr></thead><tbody>${rows}</tbody></table>` : '<p class="muted">No defects.</p>'}
      <p class="tiny muted vf-note">After repairing, press <b>Re-scan repaired board</b>. Each defect will be marked fixed or still there.</p></div></div>`;
    window.lucide?.createIcons();
  }
}

function rescan(e) {
  if (!state.source || !state.status.connected) return toast('Connect the phone first (Settings → Phone).');
  state.mode = 'verify';
  state.vres = { entryId: e.id, items: null, advice: '' };
  renderVerify();
  if (state.source.rescan) state.source.rescan(e.boardId, effective(e.items));
  else state.source.capture();
}

function renderCompare(e, vr) {
  const imgs = state.images[e.id] || {};
  const m = matchDefects(effective(e.items), effective(vr.items), e.w, e.h);
  vr.match = m;
  const host = $('#cmp'); host.innerHTML = '';
  const row = (title, beforeSrc, beforeBox, afterSrc, afterBox, ok, note) => {
    const r = document.createElement('div'); r.className = 'cmp-row';
    r.innerHTML = `<figure><canvas width="400" height="300"></canvas><figcaption>Before repair</figcaption></figure>
      <figure><canvas width="400" height="300"></canvas><figcaption>After repair</figcaption></figure>
      <div class="res ${ok ? 'ok' : 'bad'}"><p class="label">${esc(title)}</p><p class="serif">${ok ? 'Fixed' : note}</p></div>`;
    const [a, b] = r.querySelectorAll('canvas');
    crop(a, beforeSrc, beforeBox, 3, beforeSrc ? SEV_COLOR[DEFECTS[beforeBox.type]?.sev] : null);
    crop(b, afterSrc, afterBox, 3, ok ? '#1D9E75' : '#E8603C');
    host.appendChild(r);
  };
  m.rows.forEach(r => row(`${DEFECTS[r.before.type].label} · ${whereText(r.before, e.w).rel}`, imgs.clean, r.before, vr.clean, r.after || r.before, r.fixed, 'Still there'));
  m.extra.forEach(x => row(`${DEFECTS[x.type].label} · ${whereText(x, e.w).rel}`, imgs.clean, x, vr.clean, x, false, 'New defect'));
  if (!vr.done) return;
  const fixed = m.rows.filter(r => r.fixed).length;
  const act = document.createElement('div'); act.className = 'vinfo';
  act.innerHTML = `<p class="serif h3" style="margin:0">${fixed} of ${m.rows.length} fixed${m.extra.length ? ` · ${m.extra.length} new` : ''}.</p>
    <div class="vactions">${m.allFixed ? '<button class="btn btn-primary" data-v="release">Release board</button>' : '<button class="btn btn-primary" data-v="back">Send back to rework</button>'}
    ${m.allFixed ? '<button class="btn" data-v="back">Send back anyway</button>' : '<button class="btn" data-v="release">Release anyway (supervisor)</button>'}</div>`;
  act.querySelectorAll('[data-v]').forEach(b => b.addEventListener('click', () => finishVerify(e, b.dataset.v, fixed, m)));
  host.appendChild(act);
}

function finishVerify(e, action, fixed, m) {
  e.verify.push({ t: Date.now(), op: state.operator, fixed, total: m.rows.length, new: m.extra.length, result: action === 'release' ? 'released' : 'back_to_rework', advice: state.vres.advice, override: action === 'release' ? !m.allFixed : m.allFixed });
  e.status = action === 'release' ? 'released' : 'back_to_rework';
  if (state.images[e.id]) state.images[e.id].after = state.vres.clean;
  chainAdd('re-checked', e);
  saveLog();
  toast(action === 'release' ? `${e.boardId} released.` : `${e.boardId} sent back to rework.`);
  state.vres = null;
  if (action === 'release') state.verifySel = queue()[0]?.id || null;
  renderVerify(); refreshAll();
}

/* =================== Insights =================== */
const charts = {};
// Draws each bar's value at its end, so nobody has to read the axis.
const barValues = {
  id: 'barValues',
  afterDatasetsDraw(chart) {
    const { ctx } = chart; ctx.save();
    ctx.font = '600 12px DM Sans, sans-serif'; ctx.fillStyle = '#1A1A1A'; ctx.textBaseline = 'middle';
    chart.getDatasetMeta(0).data.forEach((bar, i) => ctx.fillText(chart.data.datasets[0].data[i], bar.x + 6, bar.y));
    ctx.restore();
  },
};
function initCharts() {
  if (!window.Chart || charts.types) return;
  Chart.defaults.font.family = 'DM Sans, sans-serif'; Chart.defaults.color = '#5F5E5A';
  charts.types = new Chart($('#c-types'), {
    type: 'bar', data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 6, barThickness: 18 }] },
    options: { indexAxis: 'y', maintainAspectRatio: false, animation: { duration: 400 }, layout: { padding: { right: 28 } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` ${c.raw} defect${c.raw === 1 ? '' : 's'}` } } },
      scales: { x: { display: false, beginAtZero: true }, y: { grid: { display: false }, border: { display: false } } } },
    plugins: [barValues],
  });
  charts.sev = new Chart($('#c-sev'), {
    type: 'doughnut', data: { labels: ['Critical', 'Major', 'Minor'], datasets: [{ data: [0, 0, 0], backgroundColor: ['#E8603C', '#EF9F27', '#5DCAA5'], borderColor: '#fff', borderWidth: 3 }] },
    options: { maintainAspectRatio: false, cutout: '68%', animation: { duration: 400 }, plugins: { legend: { display: false } } },
  });
  Object.keys(DEFECTS).forEach(t => $('#f-type').insertAdjacentHTML('beforeend', `<option value="${t}">${DEFECTS[t].label}</option>`));
}

function refreshAll() { renderLog(); const n = state.log.filter(isOpen).length; $('#nav-queue').hidden = !n; $('#nav-queue').textContent = n; }

const pct = (a, b) => (b ? Math.round((a / b) * 100) : 0);

function refreshInsights() {
  const L = [...state.log].reverse(), n = L.length;
  // Process alerts: types on >= 3 of the last 5 boards.
  const last = state.log.slice(0, 5), tc = {};
  last.forEach(e => new Set(visible(e.items).map(i => i.type)).forEach(t => (tc[t] = (tc[t] || 0) + 1)));
  const al = Object.entries(tc).filter(([, k]) => k >= 3).sort((a, b) => b[1] - a[1]);
  document.querySelector('.alerts-card').classList.toggle('calm', !al.length);
  $('#proc-alerts').innerHTML = al.length ? al.map(([t, k]) => `<div class="alert proc"><b>${DEFECTS[t]?.label || t}</b> on ${k} of the last ${last.length} boards. Likely cause: ${esc(DEFECTS[t]?.cause || '')}.</div>`).join('')
    : `<p class="calm-note">✓ ${state.log.length ? 'No repeating defect in the last 5 boards. The process looks stable.' : 'No boards inspected yet.'}</p>`;
  const counts = {}, sev = { high: 0, medium: 0, low: 0 };
  let conf = 0, fals = 0;
  L.forEach(e => e.items.forEach(i => {
    if (i.review === 'confirmed') conf++; if (i.review === 'false') fals++;
    if (i.suppressed || i.review === 'false') return;
    const t = i.type === 'open_circuit' ? 'open' : i.type;
    counts[t] = (counts[t] || 0) + 1; sev[DEFECTS[t]?.sev || 'low']++;
  }));
  const by = v => L.filter(e => e.verdict === v).length;
  const pass = by('pass'), rework = by('rework'), reject = by('reject');
  const verified = L.filter(e => e.verify.length), ftf = verified.filter(e => e.verify[0].result === 'released').length;

  // Tiles: number + one line that says what it means
  $('#t-count').textContent = n;
  $('#t-count-sub').textContent = n ? `${pass} passed · ${rework + reject} need work` : 'No inspections yet';
  $('#t-reject').textContent = `${pct(reject, n)}%`;
  $('#t-reject-sub').textContent = n ? `${reject} of ${n} boards` : 'No boards yet';
  $('#t-false').textContent = conf + fals ? `${pct(fals, conf + fals)}%` : '—';
  $('#t-false-sub').textContent = conf + fals ? `${fals} of ${conf + fals} reviewed` : 'None reviewed yet';
  $('#t-ftf').textContent = verified.length ? `${pct(ftf, verified.length)}%` : '—';
  $('#t-ftf-sub').textContent = verified.length ? `${ftf} of ${verified.length} repaired boards` : 'No re-checks yet';

  // Outcomes
  [['pass', pass], ['rework', rework], ['reject', reject]].forEach(([k, v]) => {
    $(`#o-${k}`).textContent = v; $(`#o-${k}-p`).textContent = n ? `${pct(v, n)}% of boards` : '';
    $(`#o-bar .${k}`).style.width = n ? `${(v / n) * 100}%` : '0';
  });
  $('#o-strip').innerHTML = n ? L.slice(-80).map(e => `<i class="${e.verdict}" title="${esc(e.boardId)} · ${e.verdict} · ${new Date(e.t).toLocaleTimeString()}"></i>`).join('')
    : '<span class="empty">Boards appear here as you inspect them.</span>';

  $('#fb-count').textContent = state.fb.length;
  const tuned = Object.entries(state.thr).filter(([t, v]) => v !== phoneThr(t));
  $('#thr-list').innerHTML = tuned.length ? tuned.map(([t, v]) => `<li>${DEFECTS[t].label} ≥ ${v.toFixed(2)}</li>`).join('') + '<li><button class="link" id="thr-reset">Reset</button></li>' : '<li>All thresholds at the tuned values</li>';
  $('#thr-reset')?.addEventListener('click', () => { state.thr = {}; store.set('bc-thr', {}); state.source?.setThresholds?.({}); refreshInsights(); });

  if (charts.types) {
    const rows = Object.keys(DEFECTS).map(t => [t, counts[t] || 0]).filter(r => r[1] > 0).sort((a, b) => b[1] - a[1]);
    charts.types.data.labels = rows.map(([t]) => DEFECTS[t].label);
    charts.types.data.datasets[0].data = rows.map(([, v]) => v);
    charts.types.data.datasets[0].backgroundColor = rows.map(([t]) => SEV_COLOR[DEFECTS[t].sev]);
    charts.types.update();
    charts.sev.data.datasets[0].data = [sev.high, sev.medium, sev.low]; charts.sev.update();
  }
  const total = sev.high + sev.medium + sev.low;
  $('#sev-total').textContent = total;
  $('#sev-legend').innerHTML = [['Critical', 'high', '#E8603C'], ['Major', 'medium', '#EF9F27'], ['Minor', 'low', '#5DCAA5']]
    .map(([l, k, c]) => `<li><i style="background:${c}"></i><span><b>${sev[k]}</b> ${l}</span></li>`).join('');
  $('#types-empty').hidden = total > 0; $('#c-types').parentElement.hidden = !total;
  $('#sev-empty').hidden = total > 0; document.querySelector('.sev-wrap').hidden = !total;
  drawHeatmap(L);
}

// Board map: the latest inspected board underneath, a 12 x 8 grid on top, cells shaded by defect count.
function drawHeatmap(L) {
  const c = $('#c-heat'), ctx = c.getContext('2d'), W = c.width, H = c.height, gx = 12, gy = 8;
  const g = Array(gx * gy).fill(0);
  L.forEach(e => effective(e.items).forEach(i => {
    const x = Math.min(gx - 1, Math.floor(((i.x + i.w / 2) / e.w) * gx)), y = Math.min(gy - 1, Math.floor(((i.y + i.h / 2) / e.h) * gy));
    g[y * gx + x]++;
  }));
  const max = Math.max(1, ...g), cw = W / gx, ch = H / gy;
  ctx.fillStyle = '#1d5a3c'; ctx.fillRect(0, 0, W, H);
  const latest = [...state.log].find(e => state.images[e.id]?.clean);
  const paint = () => {
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    for (let x = 1; x < gx; x++) ctx.fillRect(x * cw, 0, 1, H);
    for (let y = 1; y < gy; y++) ctx.fillRect(0, y * ch, W, 1);
    g.forEach((v, k) => {
      if (!v) return;
      const x = (k % gx) * cw, y = Math.floor(k / gx) * ch;
      ctx.fillStyle = `rgba(232,96,60,${0.25 + 0.7 * (v / max)})`;
      ctx.beginPath(); ctx.roundRect(x + 2, y + 2, cw - 4, ch - 4, 6); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = '700 16px DM Sans'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(v, x + cw / 2, y + ch / 2); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    });
    if (!L.length) { ctx.fillStyle = 'rgba(255,253,240,0.8)'; ctx.font = '500 18px DM Sans'; ctx.textAlign = 'center'; ctx.fillText('Fills in as you inspect boards', W / 2, H / 2); ctx.textAlign = 'left'; }
  };
  if (latest) { ctx.globalAlpha = 0.45; ctx.drawImage(state.images[latest.id].clean, 0, 0, W, H); ctx.globalAlpha = 1; }
  paint();
  $('#heat-max').textContent = L.length ? `busiest cell: ${max} defect${max === 1 ? '' : 's'}` : '';
}

/* =================== Log =================== */
const SEV_RANK = { high: 3, medium: 2, low: 1 };
const worstSev = e => effective(e.items).reduce((m, i) => Math.max(m, SEV_RANK[DEFECTS[i.type]?.sev] || 0), 0);
function filtered() {
  const sev = $('#f-sev').value, t = $('#f-type').value, st = $('#f-status').value, q = $('#f-q').value.trim().toLowerCase();
  const typeOf = i => (i.type === 'open_circuit' ? 'open' : i.type);
  return state.log.filter(e => {
    const its = effective(e.items);
    if (st === 'open' && !isOpen(e)) return false;
    if (st === 'fixed' && !isFixed(e)) return false;
    if (st === 'passed' && e.status !== 'passed') return false;
    if (sev && !its.some(i => DEFECTS[i.type]?.sev === sev)) return false;
    if (t && !its.some(i => typeOf(i) === t)) return false;
    return !q || String(e.boardId).toLowerCase().includes(q);
  });
}
/* =================== Pareto chart (Log) =================== */
// Classic QC Pareto: defect types sorted by count, plus the running share. The first bars are what to fix first.
function renderPareto() {
  const cnt = {};
  state.log.forEach(e => effective(e.items).forEach(i => { const t = i.type === 'open_circuit' ? 'open' : i.type; cnt[t] = (cnt[t] || 0) + 1; }));
  const rows = Object.entries(cnt).sort((a, b) => b[1] - a[1]), total = rows.reduce((m, r) => m + r[1], 0);
  $('#pareto-empty').hidden = total > 0; $('#c-pareto').style.visibility = total ? 'visible' : 'hidden';
  // Draw only while the Log is on screen: a chart created inside a hidden page gets size 0 and stays blank.
  if (!window.Chart || !total || $('#view-log').hidden) return;
  let run = 0;
  const data = {
    labels: rows.map(([t]) => DEFECTS[t]?.label || t),
    datasets: [
      { data: rows.map(r => r[1]), backgroundColor: rows.map(([t]) => SEV_COLOR[DEFECTS[t]?.sev] || '#B4B2A9'), borderRadius: 6, maxBarThickness: 46, yAxisID: 'y', order: 2 },
      { type: 'line', data: rows.map(r => Math.round(((run += r[1]) / total) * 100)), borderColor: '#1A1A1A', backgroundColor: '#1A1A1A', borderWidth: 2, pointRadius: 3, tension: 0, yAxisID: 'yPct', order: 1 },
    ],
  };
  if (charts.pareto) { charts.pareto.data = data; charts.pareto.resize(); charts.pareto.update(); return; }
  Chart.defaults.font.family = 'DM Sans, sans-serif'; Chart.defaults.color = '#5F5E5A';
  charts.pareto = new Chart($('#c-pareto'), {
    type: 'bar', data, options: {
      maintainAspectRatio: false, animation: { duration: 300 },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => c.datasetIndex === 1 ? ` ${c.raw}% of all defects so far` : ` ${c.raw} defect${c.raw === 1 ? '' : 's'}` } } },
      scales: {
        x: { grid: { display: false }, border: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: 'rgba(26,26,26,0.06)' }, border: { display: false }, title: { display: true, text: 'Defects' } },
        yPct: { axis: 'y', position: 'right', min: 0, max: 100, ticks: { callback: v => v + '%', stepSize: 25 }, grid: { display: false }, border: { display: false } },
      },
    },
  });
}

/* =================== Tamper-evident log (hash chain) =================== */
// Every inspection, and every later change to it (review, board ID, marked fixed), is appended to a chain:
// hash = SHA-256(previous hash + event). Editing or deleting a past record in the browser's storage either breaks
// a hash or no longer matches the record's last chained snapshot, and the Log says which board.
function sha256(str) {
  const K = [], H = [], isP = n => { for (let d = 2; d * d <= n; d++) if (n % d === 0) return false; return true; };
  for (let n = 2, k = 0; k < 64; n++) if (isP(n)) { if (k < 8) H[k] = (Math.pow(n, 1 / 2) * 2 ** 32) | 0; K[k++] = (Math.pow(n, 1 / 3) * 2 ** 32) | 0; }
  const b = new TextEncoder().encode(str), l = b.length, words = new Array(((l + 9 + 63) >> 6) << 4).fill(0);
  for (let i = 0; i < l; i++) words[i >> 2] |= b[i] << (24 - (i % 4) * 8);
  words[l >> 2] |= 0x80 << (24 - (l % 4) * 8); words[words.length - 1] = l * 8;
  const r = (x, n) => (x >>> n) | (x << (32 - n)), W = new Array(64);
  for (let j = 0; j < words.length; j += 16) {
    let [a, bb, c, d, e, f, g, h] = H;
    for (let i = 0; i < 64; i++) {
      W[i] = i < 16 ? words[j + i] : (W[i - 16] + (r(W[i - 15], 7) ^ r(W[i - 15], 18) ^ (W[i - 15] >>> 3)) + W[i - 7] + (r(W[i - 2], 17) ^ r(W[i - 2], 19) ^ (W[i - 2] >>> 10))) | 0;
      const t1 = (h + (r(e, 6) ^ r(e, 11) ^ r(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + W[i]) | 0, t2 = ((r(a, 2) ^ r(a, 13) ^ r(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c))) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = bb; bb = a; a = (t1 + t2) | 0;
    }
    [a, bb, c, d, e, f, g, h].forEach((v, i) => (H[i] = (H[i] + v) | 0));
  }
  return H.map(v => (v >>> 0).toString(16).padStart(8, '0')).join('');
}
const GENESIS = '0'.repeat(64);
const snapOf = e => ({ id: e.id, board: e.boardId, t: e.t, op: e.op || '', verdict: e.verdict, status: e.status,
  items: (e.items || []).map(i => [i.type, Math.round((i.conf || 0) * 1000), Math.round(i.x), Math.round(i.y), Math.round(i.w), Math.round(i.h), i.review || '', i.suppressed ? 1 : 0]) });
const evBody = ev => JSON.stringify([ev.n, ev.t, ev.kind, ev.op, ev.snap]);
state.chain = store.get('bc-chain', []);
function chainAdd(kind, e) {
  const last = state.chain[state.chain.length - 1], prev = last ? last.hash : GENESIS;
  const ev = { n: (last?.n || 0) + 1, t: Date.now(), kind, op: state.operator || '', snap: snapOf(e), prev };
  ev.hash = sha256(prev + evBody(ev));
  state.chain.push(ev);
  if (state.chain.length > 3000) state.chain = state.chain.slice(-3000);   // the first kept event's prev is the anchor
  store.set('bc-chain', state.chain);
}
// Boards logged before the chain existed are chained once, oldest first.
if (!state.chain.length && state.log.length) [...state.log].reverse().forEach(e => chainAdd('baseline', e));
function verifyChain() {
  const C = state.chain; let prev = C[0]?.prev ?? GENESIS;
  for (const ev of C) {
    if (ev.prev !== prev || sha256(prev + evBody(ev)) !== ev.hash) return { ok: false, why: `chain broken at entry ${ev.n} (board ${ev.snap?.board ?? '?'})` };
    prev = ev.hash;
  }
  const last = {}; C.forEach(ev => (last[ev.snap.id] = ev));
  for (const e of state.log) {
    const ev = last[e.id];
    if (!ev) return { ok: false, why: `board ${e.boardId} is not in the chain` };
    if (JSON.stringify(ev.snap) !== JSON.stringify(snapOf(e))) return { ok: false, why: `board ${e.boardId} was changed outside BoardCheck` };
  }
  // A chained board missing from the log was deleted (the log keeps the newest 300, so only check that range).
  const ids = new Set(state.log.map(e => e.id)), oldest = state.log.length >= 300 ? state.log[state.log.length - 1].t : 0;
  const gone = Object.values(last).find(ev => !ids.has(ev.snap.id) && ev.snap.t >= oldest);
  if (gone) return { ok: false, why: `board ${gone.snap.board} was deleted from the log` };
  return { ok: true, n: C.length, head: prev };
}
function renderChain() {
  const v = verifyChain(), el = $('#chain-badge');
  el.className = 'chain ' + (v.ok ? 'ok' : 'bad');
  el.innerHTML = v.ok ? `<i data-lucide="shield-check"></i><b>Log intact</b><span>${v.n} records · SHA-256 ${v.head.slice(0, 8)}</span>`
    : `<i data-lucide="shield-alert"></i><b>Log tampered</b><span>${esc(v.why)}</span>`;
  return v;
}

const logSort = { key: 't', dir: -1 };
const SRC_SHORT = { llama: 'Llama', cache: 'Llama (cache)', 'rules (clean)': 'Rules (clean)' };
function markFixed(id) {
  const e = state.log.find(x => x.id === id); if (!e) return;
  e.status = 'fixed'; e.fixedAt = Date.now(); e.fixedBy = state.operator; chainAdd('marked fixed', e); saveLog(); refreshAll();
  toast(`${e.boardId} marked fixed.`);
  if (state.route === '/board') renderBoardDetail();
}
function renderLog() {
  const val = (e, k) => k === 'sevRank' ? worstSev(e) : e[k] ?? '';
  const rows = filtered().slice().sort((a, b) => { const x = val(a, logSort.key), y = val(b, logSort.key); return (x > y ? 1 : x < y ? -1 : 0) * logSort.dir; }).slice(0, 300);
  document.querySelectorAll('.log-table th[data-sort]').forEach(th => th.dataset.dir = th.dataset.sort === logSort.key ? (logSort.dir > 0 ? 'asc' : 'desc') : '');
  const st = $('#f-status').value;
  $('#log-empty').hidden = rows.length > 0;
  $('#log-empty-t').textContent = st === 'open' ? 'No open problems' : 'No boards match';
  $('#log-empty-s').textContent = st === 'open' ? 'Every board with a defect has been fixed. New problems appear here after inspection.' : 'Try another filter.';
  const SEVL = { 3: ['high', 'Critical'], 2: ['medium', 'Major'], 1: ['low', 'Minor'] };
  $('#log-body').innerHTML = rows.map(e => {
    const vis = effective(e.items), w = SEVL[worstSev(e)];
    const counts = Object.entries(vis.reduce((m, i) => { const t = i.type === 'open_circuit' ? 'open' : i.type; m[t] = (m[t] || 0) + 1; return m; }, {}))
      .map(([t, n]) => `${n}× ${DEFECTS[t]?.label || t}`).join(', ') || '—';
    const action = isOpen(e) ? `<button class="btn btn-sm btn-fix" data-fix="${e.id}"><i data-lucide="check"></i>Mark fixed</button>`
      : isFixed(e) ? `<span class="tiny muted">${e.fixedAt ? 'fixed ' + new Date(e.fixedAt).toLocaleDateString() : ''}</span>` : '';
    return `<tr data-id="${e.id}"><td>${new Date(e.t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}<div class="tiny muted">${new Date(e.t).toLocaleDateString()}</div></td>
      <td><b>${esc(e.boardId)}</b></td><td><span class="vpill ${e.verdict}">${cap(e.verdict)}</span></td>
      <td>${w ? `<span class="tchip sev-${w[0]}">${w[1]}</span>` : '<span class="muted">—</span>'}</td><td class="types">${counts}</td>
      <td><span class="st ${isOpen(e) ? 'open' : isFixed(e) ? 'fixed' : 'passed'}">${STATUS_LABEL[e.status] || e.status}</span></td><td class="act">${action}</td></tr>`;
  }).join('');
  window.lucide?.createIcons();
  const L = state.log, open = L.filter(isOpen);
  $('#log-stats').innerHTML = [['Open problems', open.length, 'open'], ['Critical open', open.filter(e => worstSev(e) === 3).length, 'crit'],
    ['Fixed', L.filter(isFixed).length, ''], ['Passed', L.filter(e => e.status === 'passed').length, ''], ['Total boards', L.length, '']]
    .map(([k, v, c]) => `<div class="lstat ${c}"><span>${k}</span><b>${v}</b></div>`).join('');
  renderChain(); renderPareto();
  window.lucide?.createIcons();
}
document.querySelectorAll('.log-table th[data-sort]').forEach(th => th.addEventListener('click', () => {
  logSort.dir = logSort.key === th.dataset.sort ? -logSort.dir : (th.dataset.sort === 't' || th.dataset.sort === 'sevRank' ? -1 : 1); logSort.key = th.dataset.sort; renderLog();
}));
['#f-sev', '#f-type', '#f-status'].forEach(x => $(x).addEventListener('change', renderLog));
$('#f-q').addEventListener('input', renderLog);
Object.keys(DEFECTS).forEach(t => $('#f-type').insertAdjacentHTML('beforeend', `<option value="${t}">${DEFECTS[t].label}</option>`));
$('#log-body').addEventListener('click', ev => {
  const fx = ev.target.closest('[data-fix]'); if (fx) { ev.stopPropagation(); markFixed(fx.dataset.fix); return; }
  const tr = ev.target.closest('tr'); if (tr?.dataset.id) location.hash = '#/board/' + encodeURIComponent(tr.dataset.id);
});
function renderBoardDetail() {
  const e = state.log.find(x => x.id === state.boardSel), d = $('#board-detail');
  if (!e) { d.innerHTML = '<div class="big-empty"><i data-lucide="file-question"></i><p>Board not found</p><span>It may have been cleared from this laptop.</span></div>'; window.lucide?.createIcons(); return; }
  $('#tb-title').textContent = e.boardId;
  const th = state.images[e.id]?.thumb || e.thumb;
  const vis = visible(e.items);
  const rows = vis.map((i, n) => { const D = DEFECTS[i.type] || { label: i.type, sev: 'low' }; return `<tr><td>${n + 1}</td><td><b>${D.label}</b></td><td><span class="tchip sev-${D.sev}">${cap(SEV_NAME[D.sev])}</span></td>
    <td>${esc(whereText(i, e.w).rel)}</td><td class="nr">${Math.round(i.conf * 100)}%</td><td>${i.review === 'false' ? 'False call' : i.review === 'confirmed' ? 'Confirmed' : '—'}</td></tr>`; }).join('');
  const ver = e.verify.map(v => `<li>${new Date(v.t).toLocaleTimeString()} · ${esc(v.op)} · ${v.fixed}/${v.total} fixed${v.new ? `, ${v.new} new` : ''} → ${STATUS_LABEL[v.result]}</li>`).join('');
  const sec = ms => ms ? (ms / 1000).toFixed(1) + ' s' : '—';
  d.innerHTML = `
    <div class="bd-left panel-c">${th ? `<img src="${th}" alt="Board ${esc(e.boardId)}">` : '<div class="big-empty small"><i data-lucide="image-off"></i><p>No photo saved</p><span>Photos are kept for boards inspected after this update.</span></div>'}</div>
    <div class="bd-right">
      <div class="panel-c bd-head"><span class="vpill big ${e.verdict}">${cap(e.verdict)}</span><div><b>${esc(e.boardId)}</b><span class="muted">${new Date(e.t).toLocaleString()} · ${esc(e.op)}</span></div><span class="status ${e.status}">${STATUS_LABEL[e.status]}</span></div>
      <div class="bd-stats">${[['Defects', vis.length], ['Verdict time', sec(e.detectMs)], ['Total time', sec(e.ms)], ['Advice by', SRC_SHORT[e.source] || e.source || 'Llama']].map(([k, v]) => `<div class="lstat"><span>${k}</span><b>${esc(v)}</b></div>`).join('')}</div>
      <div class="panel-c bd-card"><p class="chart-title">Defects</p>${rows ? `<table class="mini-table"><thead><tr><th>#</th><th>Type</th><th>Severity</th><th>Where</th><th class="nr">Conf.</th><th>Review</th></tr></thead><tbody>${rows}</tbody></table>` : '<p class="muted">No defects. The board passed.</p>'}</div>
      <div class="panel-c bd-card"><p class="chart-title">What to do</p><p class="bd-advice">${esc(e.advice || '—')}</p>${e.trend ? `<div class="alert proc">Trend: ${esc(e.trend)}</div>` : ''}</div>
      ${ver ? `<div class="panel-c bd-card"><p class="chart-title">Re-checks</p><ul>${ver}</ul></div>` : ''}
    </div>`;
  d.querySelector('[data-fixd]')?.addEventListener('click', () => markFixed(e.id));
  window.lucide?.createIcons();
}
function download(name, text, type) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$('#btn-csv').addEventListener('click', () => {
  const q = s => `"${String(s ?? '').replace(/"/g, '""')}"`;
  const lines = [['time', 'board', 'board_id_source', 'operator', 'defects', 'reviews', 'verdict', 'status', 're_checks', 'vision_model', 'advisor_model', 'advice', 'chain_hash'].join(',')];
  const lastHash = {}; state.chain.forEach(ev => (lastHash[ev.snap.id] = ev.hash));
  filtered().forEach(e => lines.push([new Date(e.t).toISOString(), e.boardId, e.idSource, e.op,
    visible(e.items).map(i => `${i.type}:${i.conf}@${whereText(i, e.w).coords.split(' from')[0]}`).join(' | '),
    visible(e.items).map(i => i.review || 'open').join(' | '), e.verdict, e.status,
    e.verify.map(v => `${v.fixed}/${v.total}->${v.result}`).join(' | '), e.model?.vision, e.model?.llm, e.advice, lastHash[e.id] || ''].map(q).join(',')));
  download(`boardcheck-log-${new Date().toISOString().slice(0, 10)}.csv`, lines.join('\n'), 'text/csv');
});
$('#btn-fb').addEventListener('click', () => {
  if (!state.fb.length) return toast('Confirm or reject a few defects first.');
  download(`boardcheck-feedback-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify({ thresholds: state.thr, samples: state.fb }, null, 1), 'application/json');
});

/* =================== Settings =================== */
function renderConfBox() {
  const r = CAL?.report;
  $('#retake-lvl').value = retakeLevel(); $('#retake-v').textContent = Math.round(retakeLevel() * 100) + '%';
  if (!r) { $('#conf-box').innerHTML = '<p class="tiny muted">Calibration not loaded: run <code>python3 ml/calibrate.py</code>.</p>'; return; }
  const rows = r.reliability_held_out_half.map(b => `<tr><td>${b.stated}</td><td class="nr">${b.detections}</td><td class="nr"><b>${Math.round(b.actually_correct * 100)}%</b></td></tr>`).join('');
  $('#conf-box').innerHTML = `<p class="tiny">Tested on <b>${r.boards} unseen boards</b> (${r.detections} detections). When BoardCheck says a confidence in this range, this many were really correct:</p>
    <table class="mini-table"><thead><tr><th>Stated confidence</th><th class="nr">Detections</th><th class="nr">Really correct</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="tiny muted">Average gap between stated and real: <b>${Math.round(r.calibration_error_calibrated * 100)} points</b> (raw YOLO score: ${Math.round(r.calibration_error_raw_score * 100)} points). Measured on dataset photos; real camera photos may differ.</p>`;
}
$('#retake-lvl').addEventListener('input', e => { store.set('bc-retake', +e.target.value); $('#retake-v').textContent = Math.round(e.target.value * 100) + '%'; renderInspect(); });
function renderSettings() {
  renderConfBox();
  $('#thr-sliders').innerHTML = Object.entries(DEFECTS).map(([t, D]) => `
    <label class="thr-row"><span>${D.label} <span class="tchip sev-${D.sev}">${cap(SEV_NAME[D.sev])}</span></span><b id="thr-v-${t}">${thrOf(t).toFixed(2)}</b>
    <input type="range" min="0.2" max="0.95" step="0.01" value="${thrOf(t)}" data-thr="${t}"></label>`).join('');
  renderGolden();
  document.querySelectorAll('[data-thr]').forEach(r => r.addEventListener('input', () => {
    const t = r.dataset.thr, v = Math.round(parseFloat(r.value) * 100) / 100;
    state.thr[t] = v; $('#thr-v-' + t).textContent = v.toFixed(2);
    store.set('bc-thr', state.thr); state.source?.setThresholds?.(state.thr);
  }));
}
$('#btn-csv2').addEventListener('click', () => $('#btn-csv').click());
$('#thr-reset2').addEventListener('click', () => { state.thr = {}; store.set('bc-thr', {}); state.source?.setThresholds?.({}); renderSettings(); toast('Thresholds reset to the tuned values.'); });

/* =================== Golden board (reference) =================== */
// The phone compares each detection with a known-good board of the same design and hides design features
// (e.g. trace junctions read as shorts). The reference must be defect-free; the phone warns otherwise.
function renderGolden() {
  const on = !!state.status.connected, set = !!state.status.golden;
  $('#golden-state').textContent = !on ? 'Phone not connected' : set ? 'Reference set on the phone' : 'No reference set';
  $('#golden-state').className = 'tiny ' + (set ? 'ok' : 'muted');
  ['#btn-golden-set', '#btn-golden-clear'].forEach(s => ($(s).disabled = !on));
  $('#btn-golden-clear').hidden = !set;
}
$('#btn-golden-set').addEventListener('click', () => {
  if (!state.status.connected) return toast('Connect the phone first.');
  state.source?.send?.('golden_set', state.goldenForce ? { force: true } : {});
  state.goldenForce = false;
});
$('#btn-golden-clear').addEventListener('click', () => state.source?.send?.('golden_clear'));
function onGolden(m) {
  if (m.warning) {
    state.goldenForce = true;   // a second click within the phone's 10 s window saves anyway
    $('#golden-warn').hidden = false; $('#golden-warn').textContent = m.warning + ' Click "Use last board" again to save anyway.';
    toast('That board had findings. A golden reference must be a known-good board.');
  } else {
    $('#golden-warn').hidden = true; state.goldenForce = false;
    toast(m.set ? 'Golden reference saved on the phone.' : 'Golden reference cleared.');
  }
  state.status = { ...state.status, golden: !!m.set };
  renderGolden();
}
$('#btn-fb2').addEventListener('click', () => $('#btn-fb').click());
$('#btn-clear').addEventListener('click', () => {
  if (!confirm('Delete the whole log, feedback samples and thresholds on this laptop? This cannot be undone.')) return;
  ['bc-log', 'bc-fb', 'bc-thr', 'bc-chain'].forEach(store.del);
  state.chain = []; state.log = []; state.fb = []; state.thr = {}; state.images = {}; state.verifySel = null; state.vres = null;
  refreshAll(); renderSettings(); toast('All data on this laptop cleared.');
});

/* =================== Boot =================== */
window.lucide?.createIcons();
drawHero(); document.fonts?.ready.then(drawHero);
initConnect();
renderArm();
requestAnimationFrame(renderLoop);
initLogin();
