// Procedural PCB renderer used by the hero image on the start screen, plus the defect list and verdict rules.
// Board coordinates are 960 x 640; drawing scales to the canvas.

export const BW = 960, BH = 640;

export const DEFECTS = {
  short:           { label: 'Short',           sev: 'high',
    cause: 'excess solder or under-etched copper bridging adjacent traces',
    action: 'quarantine the board, remove the bridge and re-test continuity; check etch time on this panel' },
  open:            { label: 'Open circuit',    sev: 'high',
    cause: 'over-etching or a handling scratch broke the trace',
    action: 'reject the board; check etchant concentration and handling fixtures' },
  missing_hole:    { label: 'Missing hole',    sev: 'medium',
    cause: 'the drill skipped this pad or a bit broke',
    action: 're-drill if the pad is intact; check the drill bit wear log' },
  mouse_bite:      { label: 'Mouse bite',      sev: 'medium',
    cause: 'uneven etching left notches along the trace edge',
    action: 'rework or verify current capacity of the trace; check resist adhesion' },
  spur:            { label: 'Spur',            sev: 'low',
    cause: 'resist residue left a copper protrusion',
    action: 'usually acceptable; trim it if it is within 0.2 mm of a neighbouring trace' },
  spurious_copper: { label: 'Spurious copper', sev: 'low',
    cause: 'incomplete copper removal during etching',
    action: 'scrape it off; check developer and etch uniformity across the panel' },
  // v3 detector class 6 (same severity and repair guide as the phone's QcAdvisor.kt)
  missing_pad:     { label: 'Missing pad',     sev: 'high',
    cause: 'over-etching or poor resist adhesion lifted the copper pad',
    action: 'reject the board, or rebuild the pad with a repair kit only if the customer allows it; check resist adhesion and etch time' },
};

export const SEV_RANK = { low: 1, medium: 2, high: 3 };
// The phone app uses open_circuit for the same defect.
Object.defineProperty(DEFECTS, 'open_circuit', { value: DEFECTS.open, enumerable: false });   // same defect, not listed twice
export const SEV_COLOR = { low: '#5DCAA5', medium: '#EF9F27', high: '#E8603C' };

export function verdictFor(items) {
  if (!items.length) return 'pass';
  const worst = Math.max(...items.map(d => SEV_RANK[DEFECTS[d.type]?.sev || 'low']));
  return worst >= 3 ? 'reject' : 'rework';
}

export function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeBoard(seed, opts = {}) {
  const r = rng(seed);
  const pick = a => a[Math.floor(r() * a.length)];
  const chips = [];
  const nChips = 3 + Math.floor(r() * 3);
  for (let i = 0; i < nChips * 6 && chips.length < nChips; i++) {
    const w = 90 + r() * 90, h = 60 + r() * 70;
    const x = 70 + r() * (BW - 140 - w), y = 70 + r() * (BH - 140 - h);
    if (chips.some(c => x < c.x + c.w + 60 && x + w + 60 > c.x && y < c.y + c.h + 60 && y + h + 60 > c.y)) continue;
    chips.push({ x, y, w, h, label: 'U' + (chips.length + 1) });
  }
  const pads = [];
  const traces = [];
  const nTraces = 14 + Math.floor(r() * 8);
  for (let i = 0; i < nTraces; i++) {
    const a = { x: 50 + r() * (BW - 100), y: 50 + r() * (BH - 100) };
    const b = { x: 50 + r() * (BW - 100), y: 50 + r() * (BH - 100) };
    const mid = r() < 0.5 ? { x: b.x, y: a.y } : { x: a.x, y: b.y };
    traces.push([a, mid, b]);
    pads.push({ x: a.x, y: a.y, hole: r() < 0.7 }, { x: b.x, y: b.y, hole: r() < 0.7 });
  }
  const vias = Array.from({ length: 10 }, () => ({ x: 40 + r() * (BW - 80), y: 40 + r() * (BH - 80) }));

  const nearest = (x, y) => {
    let best = chips[0], bd = Infinity;
    for (const c of chips) {
      const d = Math.hypot(c.x + c.w / 2 - x, c.y + c.h / 2 - y);
      if (d < bd) { bd = d; best = c; }
    }
    if (!best) return { label: 'edge', dx: 0, dy: 0 };
    return { label: best.label, dx: x - (best.x + best.w / 2), dy: y - (best.y + best.h / 2) };
  };
  const onTrace = () => {
    const t = pick(traces);
    const k = r() < 0.5 ? 0 : 1;
    const p = t[k], q = t[k + 1];
    const f = 0.3 + r() * 0.4;
    const x = p.x + (q.x - p.x) * f, y = p.y + (q.y - p.y) * f;
    const horiz = Math.abs(q.x - p.x) > Math.abs(q.y - p.y);
    return { x, y, horiz };
  };

  let n = r() < 0.3 ? 0 : 1 + Math.floor(r() * 3);
  if (opts.minDefects) n = Math.max(n, opts.minDefects);
  if (opts.noDefects) n = 0;
  const types = Object.keys(DEFECTS);
  const defects = [];
  for (let i = 0; i < n; i++) {
    const type = opts.types ? opts.types[i % opts.types.length] : pick(types);
    let d;
    if (type === 'missing_hole') {
      const p = pick(pads.filter(p => p.hole).concat(pads)); p.hole = false; p.missing = true;
      d = { x: p.x, y: p.y, size: 34, pad: p };
    } else if (type === 'spurious_copper') {
      d = { x: 80 + r() * (BW - 160), y: 80 + r() * (BH - 160), size: 44, blob: Array.from({ length: 6 }, () => r()) };
    } else {
      const p = onTrace();
      d = { ...p, size: 42, side: r() < 0.5 ? -1 : 1 };
    }
    const s = d.size;
    const nr = nearest(d.x, d.y);
    defects.push({
      type, x: d.x - s / 2, y: d.y - s / 2, w: s, h: s,
      conf: Math.round((0.62 + r() * 0.36) * 100) / 100,
      near: nr.label, nearDx: nr.dx, nearDy: nr.dy,
      _r: d,
    });
  }
  const serial = 'BC-' + String(seed).padStart(6, '0');
  const qr = Array.from({ length: 81 }, () => r() < 0.5);
  return { seed, serial, qr, chips, pads, traces, vias, defects };
}

// Simulate a repair: each defect is fixed with probability fixProb.
export function repairBoard(b, fixProb, r = Math.random) {
  b.defects = b.defects.filter(d => {
    if (r() >= fixProb) return true;
    if (d._r.pad) { d._r.pad.hole = true; d._r.pad.missing = false; }
    return false;
  });
  return b;
}

// Board is drawn as 100 mm wide.
export const MM_PER_UNIT = 100 / BW;

// Match defects before and after a repair by type and position.
export function matchDefects(before, after, w, h) {
  const used = new Set();
  const tol = Math.max(w, h) * 0.06;
  const rows = before.map(b => {
    const bx = b.x + b.w / 2, by = b.y + b.h / 2;
    let best = null, bd = Infinity;
    after.forEach((a, i) => {
      if (used.has(i) || a.type !== b.type) return;
      const d = Math.hypot(a.x + a.w / 2 - bx, a.y + a.h / 2 - by);
      if (d < tol && d < bd) { bd = d; best = i; }
    });
    if (best != null) used.add(best);
    return { before: b, after: best != null ? after[best] : null, fixed: best == null };
  });
  const extra = after.filter((_, i) => !used.has(i));
  return { rows, extra, allFixed: rows.every(r => r.fixed) && !extra.length };
}

const MASK = '#1d5a3c', MASK_DARK = '#164a31', COPPER = '#d4ae5a', SILK = '#f1efe8';

export function drawBoard(ctx, b, o = {}) {
  const cw = ctx.canvas.width, ch = ctx.canvas.height;
  ctx.save();
  ctx.fillStyle = '#121212';
  ctx.fillRect(0, 0, cw, ch);
  const s = Math.min(cw / BW, ch / BH) * (o.zoom || 1);
  ctx.translate(cw / 2, ch / 2);
  if (o.jitter) {
    const t = (o.t || 0) / 1000;
    ctx.translate(Math.sin(t * 0.9) * 6 * s, Math.cos(t * 0.7) * 4 * s);
    ctx.rotate(Math.sin(t * 0.4) * 0.006);
  }
  ctx.scale(s, s);
  ctx.translate(-BW / 2, -BH / 2);

  ctx.fillStyle = MASK;
  roundRect(ctx, 0, 0, BW, BH, 18); ctx.fill();
  ctx.strokeStyle = MASK_DARK; ctx.lineWidth = 10; ctx.stroke();

  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.strokeStyle = COPPER; ctx.lineWidth = 7;
  for (const t of b.traces) {
    ctx.beginPath(); ctx.moveTo(t[0].x, t[0].y);
    for (let i = 1; i < t.length; i++) ctx.lineTo(t[i].x, t[i].y);
    ctx.stroke();
  }
  for (const v of b.vias) {
    ctx.fillStyle = COPPER; ctx.beginPath(); ctx.arc(v.x, v.y, 6, 0, 7); ctx.fill();
    ctx.fillStyle = '#0d0d0d'; ctx.beginPath(); ctx.arc(v.x, v.y, 2.5, 0, 7); ctx.fill();
  }
  for (const p of b.pads) {
    ctx.fillStyle = COPPER; ctx.beginPath(); ctx.arc(p.x, p.y, 10, 0, 7); ctx.fill();
    if (p.hole) { ctx.fillStyle = '#0d0d0d'; ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 7); ctx.fill(); }
  }
  for (const c of b.chips) {
    ctx.fillStyle = '#cfcfcf';
    const pins = Math.max(4, Math.floor(c.w / 16));
    for (let i = 0; i < pins; i++) {
      const px = c.x + 10 + (i * (c.w - 20)) / (pins - 1);
      ctx.fillRect(px - 3, c.y - 9, 6, 9); ctx.fillRect(px - 3, c.y + c.h, 6, 9);
    }
    ctx.fillStyle = '#1b1b1b'; roundRect(ctx, c.x, c.y, c.w, c.h, 5); ctx.fill();
    ctx.fillStyle = '#2c2c2c'; ctx.beginPath(); ctx.arc(c.x + 12, c.y + 12, 4, 0, 7); ctx.fill();
    ctx.fillStyle = SILK; ctx.font = '600 15px DM Sans, sans-serif';
    ctx.fillText(c.label, c.x, c.y - 16);
  }
  ctx.fillStyle = SILK; ctx.font = '600 13px DM Sans, sans-serif';
  ctx.fillText('BOARDCHECK REV ' + (b.seed % 9 + 1), 24, BH - 22);

  // Serial sticker with a QR-style code the phone reads for traceability.
  if (b.qr) {
    const sx = BW - 132, sy = BH - 124, m = 9;
    ctx.fillStyle = '#ffffff'; roundRect(ctx, sx, sy, 108, 104, 6); ctx.fill();
    ctx.fillStyle = '#111';
    b.qr.forEach((on, i) => { if (on) ctx.fillRect(sx + 14 + (i % 9) * m, sy + 8 + Math.floor(i / 9) * m, m, m); });
    [[0, 0], [6, 0], [0, 6]].forEach(([cx, cy]) => {
      ctx.fillStyle = '#111'; ctx.fillRect(sx + 14 + cx * m, sy + 8 + cy * m, 3 * m, 3 * m);
      ctx.fillStyle = '#fff'; ctx.fillRect(sx + 14 + cx * m + m * 0.6, sy + 8 + cy * m + m * 0.6, m * 1.8, m * 1.8);
      ctx.fillStyle = '#111'; ctx.fillRect(sx + 14 + cx * m + m, sy + 8 + cy * m + m, m, m);
    });
    ctx.fillStyle = '#111'; ctx.font = '600 11px DM Sans, sans-serif';
    ctx.fillText(b.serial, sx + 14, sy + 98);
  }

  for (const d of b.defects) drawDefect(ctx, d);
  ctx.restore();

  if (o.camera) {
    const g = ctx.createRadialGradient(cw / 2, ch / 2, ch * 0.35, cw / 2, ch / 2, ch * 0.85);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,0.45)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, cw, ch);
  }
}

function drawDefect(ctx, d) {
  const p = d._r;
  switch (d.type) {
    case 'open':
      ctx.fillStyle = MASK;
      if (p.horiz) ctx.fillRect(p.x - 6, p.y - 6, 12, 12); else ctx.fillRect(p.x - 6, p.y - 6, 12, 12);
      break;
    case 'mouse_bite':
      ctx.fillStyle = MASK;
      for (let i = -1; i <= 1; i++) {
        ctx.beginPath();
        if (p.horiz) ctx.arc(p.x + i * 9, p.y + p.side * 3.5, 3.2, 0, 7);
        else ctx.arc(p.x + p.side * 3.5, p.y + i * 9, 3.2, 0, 7);
        ctx.fill();
      }
      break;
    case 'spur':
      ctx.strokeStyle = COPPER; ctx.lineWidth = 3; ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      if (p.horiz) ctx.lineTo(p.x + 5, p.y + p.side * 15); else ctx.lineTo(p.x + p.side * 15, p.y + 5);
      ctx.stroke();
      break;
    case 'short':
      ctx.fillStyle = '#e2c27a';
      if (p.horiz) roundRect(ctx, p.x - 5, p.side < 0 ? p.y - 26 : p.y, 10, 26, 4);
      else roundRect(ctx, p.side < 0 ? p.x - 26 : p.x, p.y - 5, 26, 10, 4);
      ctx.fill();
      ctx.beginPath(); ctx.arc(p.x + (p.horiz ? 0 : p.side * 26), p.y + (p.horiz ? p.side * 26 : 0), 7, 0, 7); ctx.fill();
      break;
    case 'spurious_copper':
      ctx.fillStyle = COPPER; ctx.beginPath();
      p.blob.forEach((v, i) => {
        const a = (i / p.blob.length) * Math.PI * 2, rad = 7 + v * 9;
        const x = p.x + Math.cos(a) * rad, y = p.y + Math.sin(a) * rad;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.closePath(); ctx.fill();
      break;
    case 'missing_hole':
      ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 7); ctx.stroke();
      break;
  }
}

export function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Map a box from board space into canvas space for a given draw transform (no jitter).
export function boardToCanvas(cw, ch, box, zoom = 1) {
  const s = Math.min(cw / BW, ch / BH) * zoom;
  const ox = cw / 2 - (BW / 2) * s, oy = ch / 2 - (BH / 2) * s;
  return { x: ox + box.x * s, y: oy + box.y * s, w: box.w * s, h: box.h * s };
}
