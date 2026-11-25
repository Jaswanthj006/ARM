// The real phone, through the laptop bridge (server/bridge.py). Event interface used by app.js:
//   onStatus({connected, name, fps, detFps, latency, tps, battery, temp, model, thresholds, golden, idle})
//   onFrame({kind:'img', w, h, img})
//   onDetections({boardId, idSource, w, h, items:[{type,x,y,w,h,conf,near}], clean?(ctx), goldenOn, goldenHidden})
//   onAdviceToken(text)
//   onAdviceDone({verdict, ms})
//   onGolden({set, warning?})
// All AI runs on the phone; the website only shows results. (The simulated demo phone was removed.)

// The phone app connects to ws://<laptop>:8765/phone, this page connects to ws://<laptop>:8765/ui.
export class BridgePhone {
  constructor(url, h, onBridge) { this.url = url; this.h = h; this.onBridge = onBridge; this.lastImg = null; }

  connect() {
    try { this.ws = new WebSocket(this.url); } catch { this.onBridge(false); return; }
    this.ws.binaryType = 'blob';
    this.ws.onopen = () => this.onBridge(true);
    // Bridge gone (e.g. restarted): report it and keep trying, so the page recovers on its own.
    this.ws.onclose = () => { this.onBridge(false); this.h.onStatus({ connected: false }); setTimeout(() => this.connect(), 2000); };
    this.ws.onerror = () => {};
    this.ws.onmessage = async e => {
      if (e.data instanceof Blob) { this.showImage(await createImageBitmap(e.data)); return; }
      let m; try { m = JSON.parse(e.data); } catch { return; }
      switch (m.type) {
        case 'bridge': if (!m.phone) this.h.onStatus({ connected: false }); break;
        case 'status': this.h.onStatus({ connected: true, name: m.name || 'Phone', fps: m.fps, detFps: m.det_fps, liveMs: m.live_ms,
          latency: m.latency_ms, tps: m.llm_tps, battery: m.battery, temp: m.temp, thermal: m.thermal ?? null, model: m.model,
          thresholds: m.thresholds || null, golden: !!m.golden, idle: !!m.idle }); break;
        case 'frame': {
          const img = new Image();
          img.onload = () => this.showImage(img);
          img.src = 'data:image/jpeg;base64,' + m.jpeg;
          break;
        }
        case 'detections': {
          const snap = this.lastImg;
          this.h.onDetections({ boardId: m.board_id, idSource: m.board_id_source || 'qr', w: m.width, h: m.height, items: m.items || [],
            detectMs: m.detect_ms, tiles: m.tiles, msPerTile: m.ms_per_tile, quality: m.quality || null, pcbProb: m.pcb_prob ?? null,
            goldenOn: !!m.golden, goldenHidden: m.golden_hidden || 0,
            clean: snap ? ctx => ctx.drawImage(snap, 0, 0, ctx.canvas.width, ctx.canvas.height) : null });
          break;
        }
        case 'live': this.h.onLive?.(m); break;
        case 'not_pcb': this.h.onNotPcb?.({ boardId: m.board_id, prob: m.pcb_prob }); break;
        case 'golden': this.h.onGolden?.(m); break;
        case 'advice_token': this.h.onAdviceToken(m.text); break;
        case 'advice_done': this.h.onAdviceDone({ verdict: m.verdict, ms: m.ms, detectMs: m.detect_ms, advice: m.advice || null, source: m.source }); break;
      }
    };
  }
  showImage(img) { this.lastImg = img; this.h.onFrame({ kind: 'img', img, w: img.width, h: img.height }); }
  send(cmd, extra = {}) { if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ type: 'command', cmd, ...extra })); }
  start() { this.send('start'); }
  stop() { this.send('stop'); }
  next() { this.send('capture'); }
  capture() { this.send('capture'); }
  rescan(serial) { this.send('capture', { mode: 'verify', board_id: serial }); }
  // The website calls open circuit 'open'; the phone calls it 'open_circuit'.
  setThresholds(thresholds) {
    const t = {}; Object.entries(thresholds).forEach(([k, v]) => (t[k === 'open' ? 'open_circuit' : k] = v));
    this.send('thresholds', { thresholds: t });
  }
  disconnect() { this.send('stop'); }
}
