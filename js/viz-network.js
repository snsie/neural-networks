/* Network diagram: SVG edges + HTML neuron tiles (each tile is a tiny heatmap). */
(function () {
  'use strict';
  const SVGNS = 'http://www.w3.org/2000/svg';
  const SUBS = '₀₁₂₃₄₅₆₇₈₉';
  const sub = n => String(n).split('').map(d => SUBS[+d]).join('');
  const { mix } = Palette;

  const el = (tag, cls, parent) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  };
  const sv = (tag, attrs, parent) => {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };

  class NetworkViz {
    /** opts.info(id) -> {title, sub, rows: [[label, value]]} | null */
    constructor(root, opts = {}) {
      this.root = root;
      this.opts = opts;
      root.classList.add('netviz');
      this.svg = sv('svg', { class: 'netviz-svg' }, root);
      this.gEdges = sv('g', {}, this.svg);
      this.gPulse = sv('g', {}, this.svg);
      this.gHit = sv('g', {}, this.svg);
      this.labelLayer = el('div', 'netviz-labels', root);
      this.nodeLayer = el('div', 'netviz-nodes', root);
      this.tip = el('div', 'netviz-tip', root);
      this.nodes = new Map();
      this.edges = new Map();
      this.off = document.createElement('canvas');
      new ResizeObserver(() => this.layout()).observe(root);
    }

    build(cfg) {
      // cfg: {H, L, O, K, actName, outMode}
      this.cfg = cfg;
      const L = cfg.L || 1;
      this.gEdges.innerHTML = ''; this.gPulse.innerHTML = ''; this.gHit.innerHTML = '';
      this.nodeLayer.innerHTML = ''; this.labelLayer.innerHTML = '';
      this.nodes.clear(); this.edges.clear();
      this._hideTip();

      this.colKeys = ['in', ...Array.from({ length: L }, (_, l) => `h${l}`), 'out', 'loss'];
      const cols = Object.fromEntries(this.colKeys.map(k => [k, []]));
      let d = 0;
      const delay = () => `${(d++ * 0.035).toFixed(3)}s`;
      const hidLabel = (l, j) => (L === 1 ? `h${sub(j + 1)}` : `h${sub(l + 1)},${sub(j + 1)}`);
      for (let i = 0; i < 2; i++) cols.in.push(this._node(`in${i}`, 'in', 'in', `x${sub(i + 1)}`, null, delay()));
      for (let l = 0; l < L; l++) {
        for (let j = 0; j < cfg.H; j++) cols[`h${l}`].push(this._node(`h${l}-${j}`, 'hid', `h${l}`, hidLabel(l, j), null, delay()));
      }
      for (let k = 0; k < cfg.O; k++) {
        cols.out.push(this._node(`o${k}`, 'out', 'out', cfg.O === 1 ? 'ŷ' : `ŷ${sub(k + 1)}`, cfg.O === 1 ? null : k, delay()));
      }
      cols.loss.push(this._node('loss', 'loss', 'loss', 'loss', null, delay()));
      this.cols = cols;

      // Edge group `e${l}` holds the weights W[l] feeding layer l (l = L is the output layer).
      for (let l = 0; l <= L; l++) {
        const src = cols[this.colKeys[l]], dst = cols[this.colKeys[l + 1]];
        dst.forEach((b, j) => src.forEach((a, i) => this._edge(`e${l}-${j}-${i}`, a.id, b.id, `e${l}`, { l, j, i })));
        dst.forEach((target, j) => this._bias(l, j, target));
      }
      for (let k = 0; k < cfg.O; k++) this._edge(`l-${k}`, `o${k}`, 'loss', 'el', { k });

      // `short` is swapped in by layout() when neighbouring headers would collide.
      const lab = (title, s, short = s) => {
        const e = el('div', 'layer-label', this.labelLayer);
        el('b', '', e).textContent = title;
        const span = el('span', '', e);
        span.textContent = s;
        e._sub = { span, full: s, short };
        return e;
      };
      this.labels = { in: lab('Input', '2') };
      for (let l = 0; l < L; l++) {
        this.labels[`h${l}`] = lab(L === 1 ? 'Hidden' : `Hidden ${l + 1}`, `${cfg.H} · ${cfg.actName}`, `${cfg.H}`);
      }
      this.labels.out = lab('Output', `${cfg.O} · ${cfg.outMode}`, `${cfg.O}`);
      this.labels.loss = lab('Loss', cfg.outMode === 'sigmoid' ? 'BCE' : 'CE');

      this.layout();
    }

    _node(id, group, key, label, classIdx, delay) {
      const n = el('div', `node node--${group}`, this.nodeLayer);
      n.style.setProperty('--d', delay);
      let canvas = null;
      if (group === 'loss') el('span', 'node-glyph', n).textContent = 'L';
      else canvas = el('canvas', '', n);
      if (classIdx != null) n.style.setProperty('--ring', `var(--c${classIdx})`);
      el('div', 'node-label', n).textContent = label;
      const val = el('div', 'node-val', n);
      const rec = { id, group, key, el: n, canvas, val, x: 0, y: 0 };
      n.addEventListener('mouseenter', () => this._showTip(id, rec.x + 30, rec.y - 16));
      n.addEventListener('mouseleave', () => this._hideTip());
      this.nodes.set(id, rec);
      return rec;
    }

    _bias(l, j, target) {
      const button = el('button', 'bias-node', target.el);
      target.bias = button;
      button.type = 'button';
      button.title = 'Show bias';
      button.setAttribute('aria-label', `Show bias for ${l === this.cfg.L ? 'output' : `hidden layer ${l + 1},`} neuron ${j + 1}`);
      button.setAttribute('aria-haspopup', 'dialog');
      button.setAttribute('aria-controls', 'biasDialog');
      button.addEventListener('click', () => {
        this._hideTip();
        this.opts.onBias(l, j, button);
      });
    }

    _edge(id, from, to, group, idx) {
      const base = sv('path', { class: `edge edge--${group}` }, this.gEdges);
      const pulse = sv('path', { class: 'edge-pulse' }, this.gPulse);
      const rec = { id, from, to, group, base, pulse, ...idx };
      if (group !== 'el') {
        rec.hit = sv('path', { class: 'edge-hit' }, this.gHit);
        rec.hit.addEventListener('mousemove', e => {
          const r = this.root.getBoundingClientRect();
          this._showTip(id, e.clientX - r.left + 14, e.clientY - r.top + 12);
        });
        rec.hit.addEventListener('mouseleave', () => this._hideTip());
      }
      this.edges.set(id, rec);
    }

    layout() {
      if (!this.cfg) return;
      const W = this.root.clientWidth, Hh = this.root.clientHeight;
      if (!W || !Hh) return;
      this.svg.setAttribute('width', W);
      this.svg.setAttribute('height', Hh);
      this.svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`);
      const top = 43, bottom = 20;
      const avail = Hh - top - bottom, mid = top + avail / 2;
      const spacing = avail / Math.max(this.cfg.H, this.cfg.O);
      const colGap = (W * 0.85) / (this.colKeys.length - 1);
      const biasSize = Math.min(11, Math.max(7, spacing * 0.16), colGap * 0.23);
      const size = Math.min(42, Math.min(60, spacing) - biasSize - 3, W * 0.1, colGap * 0.45);
      this.root.style.setProperty('--bias-size', `${biasSize}px`);
      this.root.style.setProperty('--node-size', `${size}px`);
      const xs = Object.fromEntries(this.colKeys.map((k, c) => [k, W * 0.08 + c * colGap]));
      for (const g of this.colKeys) {
        const list = this.cols[g], n = list.length;
        const sp = Math.min(g === 'in' ? 90 : 60, avail / n);
        list.forEach((rec, i) => {
          rec.x = xs[g];
          rec.y = mid + (i - (n - 1) / 2) * sp;
          rec.el.style.left = `${rec.x}px`;
          rec.el.style.top = `${rec.y}px`;
        });
        this.labels[g].style.left = `${xs[g]}px`;
      }
      this._fitLabels(xs);
      this._size = size;
      this._biasSize = biasSize;
      const radius = size / 2;
      for (const e of this.edges.values()) {
        const a = this.nodes.get(e.from), b = this.nodes.get(e.to);
        const x1 = a.x + radius, y1 = a.y, x2 = b.x - radius, y2 = b.y;
        const mx = (x1 + x2) / 2;
        const dAttr = `M${x1.toFixed(1)},${y1.toFixed(1)} C${mx.toFixed(1)},${y1.toFixed(1)} ${mx.toFixed(1)},${y2.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`;
        e.base.setAttribute('d', dAttr);
        e.pulse.setAttribute('d', dAttr);
        if (e.hit) e.hit.setAttribute('d', dAttr);
      }
      this._placeValues();
    }

    /** Use the short layer subtitles when any two neighbouring headers would touch. */
    _fitLabels(xs) {
      const labels = this.colKeys.map(k => this.labels[k]);
      const set = short => labels.forEach(e => { e._sub.span.textContent = short ? e._sub.short : e._sub.full; });
      const collides = () => labels.some((e, i) => i > 0 &&
        xs[this.colKeys[i]] - xs[this.colKeys[i - 1]] < (e.offsetWidth + labels[i - 1].offsetWidth) / 2 + 6);
      set(false);
      if (collides()) set(true);
    }

    /**
     * Value badges default to the right of their node, but with several hidden layers the
     * columns are too close for that. Try a few spots per badge and keep the one that
     * overlaps the least (other badges and nodes count most; crossing an edge is fine).
     */
    _placeValues() {
      const size = this._size;
      if (!size) return;
      const W = this.root.clientWidth, Hh = this.root.clientHeight;
      const s = size / 2, gap = 3, bias = this._biasSize;
      const obstacles = [];
      for (const g of this.colKeys) {
        const label = this.labels[g];
        obstacles.push({ x: label.offsetLeft - label.offsetWidth / 2, y: label.offsetTop, w: label.offsetWidth, h: label.offsetHeight, k: 4 });
      }
      for (const rec of this.nodes.values()) {
        const id = rec.id;
        obstacles.push({ id, x: rec.x - s, y: rec.y - s, w: size, h: size, k: 3 });
        const lab = rec.el.querySelector('.node-label');
        const lw = lab.offsetWidth, lh = lab.offsetHeight;
        obstacles.push(rec.group === 'hid'
          ? { id, x: rec.x - s - 4 - lw, y: rec.y - lh / 2, w: lw, h: lh, k: 3 }
          : { id, x: rec.x - lw / 2, y: rec.y + s + 5, w: lw, h: lh, k: 3 });
        if (rec.bias) obstacles.push({ id, x: rec.x - s - 2, y: rec.y + s + 1, w: bias, h: bias, k: 2 });
      }
      const overlap = (a, b) =>
        Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) *
        Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));

      for (const key of this.colKeys) {
        for (const rec of this.cols[key]) {
          const badge = rec.val;
          const shown = badge.classList.contains('show');
          rec.el.style.zIndex = shown ? 2 : '';
          if (!shown) continue;
          const w = badge.offsetWidth, h = badge.offsetHeight, x = rec.x, y = rec.y;
          const right = { x: x + s + gap, y: y - h / 2 };
          const above = { x: x - w / 2, y: y - s - gap - h };
          // A badge may tuck onto its own node's edge to clear the next column's labels.
          const tucked = [0.25, 0.5].map(f => ({ x: right.x - size * f, y: right.y }));
          const spots = [
            ...(rec.group === 'loss' ? [above, right] : [right, ...tucked, above]),
            { x: x + s * 0.4, y: y - s - gap - h },
            { x: x - w / 2, y: y + s + gap },
            { x: x + s * 0.4, y: y + s + gap },
            { x: x - s - gap - w, y: y - h / 2 },
          ];
          let best = null;
          spots.forEach((p, order) => {
            const box = { x: p.x, y: p.y, w, h };
            const inside = { x: 0, y: 0, w: W, h: Hh };
            let cost = (w * h - overlap(box, inside)) * 4 + order * 0.5;
            // Pad neighbours by 2px so a badge that merely touches them still looks crowded.
            for (const o of obstacles) {
              cost += o.id === rec.id ? overlap(box, o) * 0.5
                : overlap(box, { x: o.x - 2, y: o.y - 2, w: o.w + 4, h: o.h + 4 }) * o.k;
            }
            if (!best || cost < best.cost) best = { ...box, cost };
          });
          obstacles.push({ x: best.x, y: best.y, w, h, k: 6 });
          badge.style.left = `${best.x - (x - s)}px`;
          badge.style.top = `${best.y - (y - s)}px`;
        }
      }
    }

    updateWeights(net) {
      for (const e of this.edges.values()) {
        if (e.group === 'el') continue;
        const w = net.W[e.l][e.j][e.i];
        const m = Math.abs(w);
        e.base.setAttribute('stroke-width', (0.6 + 4.4 * Math.tanh(m / 1.5)).toFixed(2));
        e.base.setAttribute('stroke-opacity', (0.3 + 0.65 * Math.tanh(m)).toFixed(2));
        e.base.classList.toggle('neg', w < 0);
      }
    }

    _paint(rec, fill) {
      const c = rec.canvas;
      if (!c) return;
      const R = this._R;
      const size = Math.round((c.clientWidth || 42) * (window.devicePixelRatio || 1));
      if (c.width !== size) { c.width = size; c.height = size; }
      const octx = this.off.getContext('2d');
      const img = octx.createImageData(R, R);
      const d = img.data;
      for (let i = 0; i < R * R; i++) {
        const col = fill(i);
        d[i * 4] = col[0]; d[i * 4 + 1] = col[1]; d[i * 4 + 2] = col[2]; d[i * 4 + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      const ctx = c.getContext('2d');
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(this.off, 0, 0, size, size);
    }

    /** Paint each neuron's response over the input plane. */
    drawNeurons(grid, pal) {
      if (!this.cfg) return;
      const R = grid.R, H = grid.H, K = grid.K, ext = grid.ext, L = grid.L;
      this._R = R;
      this.off.width = R; this.off.height = R;
      const white = pal.surface;
      const div = v => (v >= 0 ? mix(white, pal.pos, Math.min(1, v)) : mix(white, pal.neg, Math.min(1, -v)));

      for (let i = 0; i < 2; i++) {
        this._paint(this.nodes.get(`in${i}`), idx => {
          const r = Math.floor(idx / R), c = idx % R;
          const v = i === 0 ? (-ext + (c + 0.5) * 2 * ext / R) : (ext - (r + 0.5) * 2 * ext / R);
          return div(v / ext);
        });
      }
      for (let l = 0; l < L; l++) {
        const hl = grid.hidden[l];
        for (let j = 0; j < H; j++) {
          let m = 1e-6;
          for (let i = 0; i < R * R; i++) m = Math.max(m, Math.abs(hl[i * H + j]));
          this._paint(this.nodes.get(`h${l}-${j}`), idx => div(hl[idx * H + j] / m));
        }
      }
      if (this.cfg.O === 1) {
        this._paint(this.nodes.get('o0'), idx => {
          const p = grid.probs[idx * K + 1];
          return mix(white, mix(pal.c[0], pal.c[1], p), 0.25 + 0.75 * Math.abs(2 * p - 1));
        });
      } else {
        for (let k = 0; k < this.cfg.O; k++) {
          this._paint(this.nodes.get(`o${k}`), idx => mix(white, pal.c[k], grid.probs[idx * K + k]));
        }
      }
    }

    /** vals: {nodeId: {text, label, kind: 'fwd'|'bwd'}} */
    setValues(vals) {
      for (const rec of this.nodes.values()) {
        const v = vals && vals[rec.id];
        if (v) {
          rec.val.innerHTML = `<i>${v.label}</i>${v.text}`;
          rec.val.className = `node-val show ${v.kind || ''}`;
        } else {
          rec.val.className = 'node-val';
        }
      }
      this._placeValues();
    }

    /** spec: {nodeIds, edgeIds, dir: 'fwd'|'bwd'} or null */
    highlight(spec) {
      this.root.classList.toggle('focus', !!spec);
      this.root.classList.toggle('dir-bwd', !!spec && spec.dir === 'bwd');
      for (const rec of this.nodes.values()) {
        rec.el.classList.toggle('on', !!spec && spec.nodeIds.includes(rec.id));
        rec.bias?.classList.toggle('on', !!spec && spec.nodeIds.includes(`bias-${rec.id}`));
      }
      for (const e of this.edges.values()) {
        const on = !!spec && spec.edgeIds.includes(e.id);
        e.base.classList.toggle('on', on);
        e.pulse.classList.toggle('on', on);
      }
    }

    _showTip(id, x, y) {
      const info = this.opts.info && this.opts.info(id);
      if (!info) return;
      let html = `<div class="tip-title">${info.title}</div>`;
      if (info.sub) html += `<div class="tip-sub">${info.sub}</div>`;
      for (const [k, v] of info.rows) html += `<div class="tip-row"><span>${k}</span><b>${v}</b></div>`;
      this.tip.innerHTML = html;
      const W = this.root.clientWidth;
      const tw = this.tip.offsetWidth || 170;
      this.tip.style.left = `${Math.max(4, Math.min(x, W - tw - 4))}px`;
      this.tip.style.top = `${Math.max(0, y)}px`;
      this.tip.classList.add('show');
    }

    _hideTip() { this.tip.classList.remove('show'); }
  }

  window.NetworkViz = NetworkViz;
})();
