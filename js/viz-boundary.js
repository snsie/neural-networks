/* Canvas drawing: decision boundary, loss sparkline, activation plot, dataset thumbnails. */
(function () {
  'use strict';
  const { mix, css } = Palette;

  function fitCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    return dpr;
  }

  /* ---------------- decision boundary ---------------- */
  class BoundaryViz {
    constructor(canvas, { ext, onPick }) {
      this.canvas = canvas;
      this.ext = ext;
      this.onPick = onPick;
      this.off = document.createElement('canvas');
      this.points = [];
      this.showTest = true;
      canvas.addEventListener('click', e => { const p = this._nearest(e); if (p && this.onPick) this.onPick(p); });
      canvas.addEventListener('mousemove', e => { canvas.style.cursor = this._nearest(e) ? 'pointer' : 'crosshair'; });
      new ResizeObserver(() => this.draw()).observe(canvas);
    }

    _toPx(x, y) {
      const S = this.canvas.width, e = this.ext;
      return [((x + e) / (2 * e)) * S, ((e - y) / (2 * e)) * S];
    }

    _nearest(ev) {
      const r = this.canvas.getBoundingClientRect();
      if (!r.width) return null;
      const k = this.canvas.width / r.width;
      const mx = (ev.clientX - r.left) * k, my = (ev.clientY - r.top) * k;
      let best = null, bd = (14 * k) ** 2;
      for (const p of this.points) {
        if (p.test && !this.showTest) continue;
        const [px, py] = this._toPx(p.x, p.y);
        const d = (px - mx) ** 2 + (py - my) ** 2;
        if (d < bd) { bd = d; best = p; }
      }
      return best;
    }

    update(state) { Object.assign(this, state); this.draw(); }

    draw() {
      const { canvas, grid, pal } = this;
      if (!grid || !pal) return;
      const w = canvas.clientWidth;
      if (!w) return;
      const dpr = fitCanvas(canvas, w, w);
      const ctx = canvas.getContext('2d');
      const S = canvas.width;

      // probability heatmap, rendered small and smoothly upscaled
      const R = grid.R, K = grid.K;
      this.off.width = R; this.off.height = R;
      const octx = this.off.getContext('2d');
      const img = octx.createImageData(R, R);
      const d = img.data;
      for (let i = 0; i < R * R; i++) {
        let r = 0, g = 0, b = 0, maxp = 0;
        for (let k = 0; k < K; k++) {
          const p = grid.probs[i * K + k];
          const c = pal.c[k];
          r += p * c[0]; g += p * c[1]; b += p * c[2];
          if (p > maxp) maxp = p;
        }
        const conf = Math.max(0, (maxp - 1 / K) / (1 - 1 / K));
        const a = 0.08 + 0.6 * Math.pow(conf, 0.85);
        const col = mix(pal.surface, [r, g, b], a);
        d[i * 4] = col[0]; d[i * 4 + 1] = col[1]; d[i * 4 + 2] = col[2]; d[i * 4 + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(this.off, 0, 0, S, S);

      // light grid
      ctx.lineWidth = dpr;
      for (const v of [-1, -0.5, 0, 0.5, 1]) {
        ctx.strokeStyle = css(pal.text, v === 0 ? 0.16 : 0.06);
        const [x] = this._toPx(v, 0), [, y] = this._toPx(0, v);
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, S); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(S, y); ctx.stroke();
      }
      ctx.fillStyle = css(pal.muted, 0.9);
      ctx.font = `500 ${11 * dpr}px "JetBrains Mono", monospace`;
      ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('x₁', S - 6 * dpr, this._toPx(0, 0)[1] - 3 * dpr);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('x₂', this._toPx(0, 0)[0] + 5 * dpr, 5 * dpr);

      // points: training filled, test hollow
      const rad = 3.7 * dpr;
      const draw = (p) => {
        const [x, y] = this._toPx(p.x, p.y);
        const c = pal.c[p.label];
        ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2);
        if (!p.test) {
          ctx.fillStyle = css(c); ctx.fill();
          ctx.lineWidth = 1.2 * dpr; ctx.strokeStyle = css(pal.surface, 0.95); ctx.stroke();
        } else {
          ctx.fillStyle = css(pal.surface, 0.9); ctx.fill();
          ctx.lineWidth = 1.9 * dpr; ctx.strokeStyle = css(c); ctx.stroke();
        }
      };
      for (const p of this.points) if (!p.test) draw(p);
      if (this.showTest) for (const p of this.points) if (p.test) draw(p);

      // selected sample
      const s = this.selected;
      if (s) {
        const [x, y] = this._toPx(s.x, s.y);
        ctx.lineWidth = 2.2 * dpr;
        ctx.strokeStyle = css(pal.text, 0.95);
        ctx.beginPath(); ctx.arc(x, y, rad * 2.5, 0, Math.PI * 2); ctx.stroke();
        ctx.lineWidth = 1.4 * dpr;
        ctx.beginPath();
        const a = rad * 2.5, b = rad * 4;
        ctx.moveTo(x - b, y); ctx.lineTo(x - a, y); ctx.moveTo(x + a, y); ctx.lineTo(x + b, y);
        ctx.moveTo(x, y - b); ctx.lineTo(x, y - a); ctx.moveTo(x, y + a); ctx.lineTo(x, y + b);
        ctx.stroke();
      }
    }
  }

  /* ---------------- loss sparkline ---------------- */
  function drawSparkline(canvas, series, pal) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = fitCanvas(canvas, w, h);
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const pad = 4 * dpr, top = 18 * dpr;
    ctx.strokeStyle = css(pal.text, 0.07); ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(0, H - pad); ctx.lineTo(W, H - pad); ctx.stroke();
    let max = 0, n = 0;
    for (const s of series) { n = Math.max(n, s.data.length); for (const v of s.data) if (isFinite(v) && v > max) max = v; }
    if (n < 2 || max <= 0) return;
    for (const s of series) {
      ctx.beginPath();
      s.data.forEach((v, i) => {
        const x = (i / (n - 1)) * (W - 2 * pad) + pad;
        const y = H - pad - (Math.min(v, max) / max) * (H - pad - top);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.setLineDash(s.dash ? [4 * dpr, 3 * dpr] : []);
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.8 * dpr; ctx.lineJoin = 'round';
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  /* ---------------- activation plot ---------------- */
  function drawActivationPlot(canvas, act, pal) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = fitCanvas(canvas, w, h);
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const zMin = -3, zMax = 3, N = 240;
    const fs = [], ds = [];
    for (let i = 0; i <= N; i++) { const z = zMin + (i / N) * (zMax - zMin); fs.push(act.f(z)); ds.push(act.df(z)); }
    let lo = Math.min(-0.25, ...fs, ...ds), hi = Math.max(1.1, ...fs, ...ds);
    const span = hi - lo; lo -= span * 0.08; hi += span * 0.12;
    const padX = 10 * dpr, padTop = 26 * dpr, padBot = 10 * dpr;
    const X = z => padX + ((z - zMin) / (zMax - zMin)) * (W - 2 * padX);
    const Y = v => padTop + (1 - (v - lo) / (hi - lo)) * (H - padTop - padBot);

    ctx.lineWidth = dpr;
    ctx.strokeStyle = css(pal.text, 0.15);
    ctx.beginPath(); ctx.moveTo(X(zMin), Y(0)); ctx.lineTo(X(zMax), Y(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(0), padTop - 8 * dpr); ctx.lineTo(X(0), H - padBot); ctx.stroke();
    ctx.strokeStyle = css(pal.text, 0.06);
    for (const v of [-1, 1, 2, 3]) {
      if (v < lo || v > hi) continue;
      ctx.beginPath(); ctx.moveTo(X(zMin), Y(v)); ctx.lineTo(X(zMax), Y(v)); ctx.stroke();
    }
    ctx.fillStyle = css(pal.muted);
    ctx.font = `500 ${10 * dpr}px "JetBrains Mono", monospace`;
    ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillText('z', W - 4 * dpr, Y(0) + 3 * dpr);
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    if (1 <= hi) ctx.fillText('1', X(0) + 4 * dpr, Y(1));

    const line = (vals, color, dash) => {
      ctx.beginPath();
      vals.forEach((v, i) => { const x = X(zMin + (i / N) * (zMax - zMin)); i ? ctx.lineTo(x, Y(v)) : ctx.moveTo(x, Y(v)); });
      ctx.setLineDash(dash ? [5 * dpr, 4 * dpr] : []);
      ctx.strokeStyle = css(color); ctx.lineWidth = 2.4 * dpr; ctx.lineJoin = 'round';
      ctx.stroke();
    };
    line(ds, pal.bwd, true);
    line(fs, pal.fwd, false);
    ctx.setLineDash([]);
  }

  /* ---------------- dataset thumbnail ---------------- */
  function drawThumb(canvas, points, pal, ext) {
    const w = canvas.clientWidth || 64;
    const dpr = fitCanvas(canvas, w, w);
    const ctx = canvas.getContext('2d');
    const S = canvas.width;
    ctx.fillStyle = css(pal.surface2); ctx.fillRect(0, 0, S, S);
    const r = Math.max(1.3, S / 48);
    for (const p of points) {
      const x = ((p.x + ext) / (2 * ext)) * S, y = ((ext - p.y) / (2 * ext)) * S;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = css(pal.c[p.label], 0.9); ctx.fill();
    }
    return dpr;
  }

  window.BoundaryViz = BoundaryViz;
  window.Draw = { sparkline: drawSparkline, activation: drawActivationPlot, thumb: drawThumb, fitCanvas };
})();
