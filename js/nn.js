/* A tiny dense network: 2 inputs -> L hidden layers of H neurons -> O outputs.
 * Every hidden layer uses the same activation function.
 * 2 classes: 1 sigmoid output + binary cross-entropy.
 * 3 classes: 3 softmax outputs + categorical cross-entropy.
 * In both cases dL/dz(out) = yhat - y, which keeps the backward pass clean.
 *
 * Indexing: layer l = 0..L, where l < L are hidden layers and l = L is the output.
 * W[l] has shape sizes[l+1] x sizes[l], with sizes = [2, H, ..., H, O]. */
(function () {
  'use strict';

  const zeros = n => new Array(n).fill(0);
  const zeros2 = (r, c) => Array.from({ length: r }, () => zeros(c));
  const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));

  class Net {
    constructor({ hidden, layers = 1, classes, act, seed = 1 }) {
      this.H = hidden;
      this.L = layers;
      this.K = classes;
      this.O = classes === 2 ? 1 : classes;
      this.outMode = classes === 2 ? 'sigmoid' : 'softmax';
      this.act = act;
      this.adamSteps = 0;
      this.sizes = [2, ...Array(layers).fill(hidden), this.O];
      const rng = Datasets.makeRng(seed);
      this.W = []; this.b = [];
      for (let l = 0; l <= layers; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], isOut = l === layers;
        const s = Math.sqrt((!isOut && act.init === 'he' ? 2 : 1) / nIn) * 1.25;
        this.W.push(zeros2(nOut, nIn).map(r => r.map(() => rng.normal() * s)));
        this.b.push(zeros(nOut).map(() => (!isOut && act.init === 'he' ? 0.05 : 0)));
      }
      this.m = this._zerosLike();
      this.v = this._zerosLike();
    }

    _zerosLike() {
      return { W: this.W.map(M => zeros2(M.length, M[0].length)), b: this.b.map(v => zeros(v.length)) };
    }

    forward(x1, x2) {
      const { L, W, b, act } = this;
      const x = [x1, x2];
      const z = [], a = [];
      let prev = x;
      for (let l = 0; l <= L; l++) {
        const Wl = W[l], bl = b[l], n = Wl.length;
        const zl = new Array(n);
        for (let j = 0; j < n; j++) {
          let s = bl[j];
          const row = Wl[j];
          for (let i = 0; i < prev.length; i++) s += row[i] * prev[i];
          zl[j] = s;
        }
        z.push(zl);
        if (l < L) { prev = zl.map(v => act.f(v)); a.push(prev); }
      }
      const zo = z[L];
      let yhat;
      if (this.outMode === 'sigmoid') {
        yhat = [sigmoid(zo[0])];
      } else {
        const m = Math.max(...zo);
        const e = zo.map(v => Math.exp(v - m));
        const s = e.reduce((p, q) => p + q, 0);
        yhat = e.map(v => v / s);
      }
      return { x, z, a, yhat };
    }

    /** Input to layer l (x for l = 0, otherwise the previous hidden activation). */
    input(c, l) { return l === 0 ? c.x : c.a[l - 1]; }

    /** Class probabilities (length K) from a forward cache. */
    probs(c) { return this.outMode === 'sigmoid' ? [1 - c.yhat[0], c.yhat[0]] : c.yhat; }

    predict(c) {
      const p = this.probs(c);
      let best = 0;
      for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
      return best;
    }

    target(label) {
      return this.outMode === 'sigmoid' ? [label] : Array.from({ length: this.O }, (_, k) => (k === label ? 1 : 0));
    }

    loss(c, label) {
      const scores = c.z[this.L];
      if (this.outMode === 'sigmoid') {
        const margin = label === 1 ? -scores[0] : scores[0];
        return Math.max(0, margin) + Math.log1p(Math.exp(-Math.abs(margin)));
      }
      const peak = scores.indexOf(Math.max(...scores));
      const shift = scores[peak];
      const rest = scores.reduce((sum, score, k) => k === peak ? sum : sum + Math.exp(score - shift), 0);
      return (shift - scores[label]) + Math.log1p(rest);
    }

    /** Per-layer arrays: dz[l], dW[l], db[l] for l = 0..L; da[h], dadz[h] for hidden layers h = 0..L-1. */
    backward(c, label) {
      const { L, W, act } = this;
      const t = this.target(label);
      const dz = new Array(L + 1), dW = new Array(L + 1), db = new Array(L + 1);
      const da = new Array(L), dadz = new Array(L);
      // Use the opposite tail for the true class to preserve small gradients.
      dz[L] = this.outMode === 'sigmoid'
        ? [label === 1 ? -sigmoid(-c.z[L][0]) : c.yhat[0]]
        : c.yhat.map((p, k) => k === label
          ? -c.yhat.reduce((sum, prob, j) => j === k ? sum : sum + prob, 0)
          : p);
      for (let l = L; l >= 0; l--) {
        const inp = this.input(c, l);
        dW[l] = dz[l].map(d => inp.map(v => d * v));
        db[l] = dz[l].slice();
        if (l === 0) break;
        const h = l - 1, n = W[l][0].length;
        da[h] = new Array(n); dadz[h] = new Array(n);
        const dzh = new Array(n);
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let k = 0; k < dz[l].length; k++) s += W[l][k][j] * dz[l][k];
          da[h][j] = s;
          dadz[h][j] = act.df(c.z[h][j]);
          dzh[j] = s * dadz[h][j];
        }
        dz[h] = dzh;
      }
      return { t, dz, dW, db, da, dadz };
    }

    trainEpoch(points, { lr, optimizer = 'adam', batchSize = 10 }) {
      const idx = points.map((_, i) => i);
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
      }
      for (let s = 0; s < idx.length; s += batchSize) {
        const g = this._zerosLike();
        const end = Math.min(idx.length, s + batchSize);
        for (let bi = s; bi < end; bi++) {
          const p = points[idx[bi]];
          const gr = this.backward(this.forward(p.x, p.y), p.label);
          for (let l = 0; l <= this.L; l++) {
            const GW = g.W[l], Gb = g.b[l], dW = gr.dW[l], db = gr.db[l];
            for (let j = 0; j < GW.length; j++) {
              Gb[j] += db[j];
              for (let i = 0; i < GW[j].length; i++) GW[j][i] += dW[j][i];
            }
          }
        }
        this.applyGrads(g, 1 / (end - s), lr, optimizer);
      }
    }

    applyGrads(g, scale, lr, optimizer) {
      if (optimizer === 'adam') this.adamSteps++;
      const B1 = 0.9, B2 = 0.999, EPS = 1e-8;
      const bc1 = 1 - Math.pow(B1, this.adamSteps), bc2 = 1 - Math.pow(B2, this.adamSteps);
      const upd = (P, G, M, V) => {
        for (let i = 0; i < P.length; i++) {
          const gi = G[i] * scale;
          if (optimizer === 'adam') {
            M[i] = B1 * M[i] + (1 - B1) * gi;
            V[i] = B2 * V[i] + (1 - B2) * gi * gi;
            P[i] -= lr * (M[i] / bc1) / (Math.sqrt(V[i] / bc2) + EPS);
          } else {
            P[i] -= lr * gi;
          }
        }
      };
      for (let l = 0; l <= this.L; l++) {
        for (let j = 0; j < this.W[l].length; j++) upd(this.W[l][j], g.W[l][j], this.m.W[l][j], this.v.W[l][j]);
        upd(this.b[l], g.b[l], this.m.b[l], this.v.b[l]);
      }
    }

    evaluate(points) {
      if (!points.length) return { loss: NaN, acc: NaN };
      let loss = 0, correct = 0;
      for (const p of points) {
        const c = this.forward(p.x, p.y);
        loss += this.loss(c, p.label);
        if (this.predict(c) === p.label) correct++;
      }
      return { loss: loss / points.length, acc: correct / points.length };
    }

    /** Evaluate an R x R grid over [-ext, ext]^2 (row 0 = top). hidden[l] holds hidden layer l's activations. */
    evalGrid(R, ext) {
      const { H, K, L } = this;
      const hidden = Array.from({ length: L }, () => new Float32Array(R * R * H));
      const probs = new Float32Array(R * R * K);
      const step = (2 * ext) / R;
      for (let r = 0; r < R; r++) {
        const y = ext - (r + 0.5) * step;
        for (let c = 0; c < R; c++) {
          const x = -ext + (c + 0.5) * step;
          const cache = this.forward(x, y);
          const idx = r * R + c;
          for (let l = 0; l < L; l++) for (let j = 0; j < H; j++) hidden[l][idx * H + j] = cache.a[l][j];
          const p = this.probs(cache);
          for (let k = 0; k < K; k++) probs[idx * K + k] = p[k];
        }
      }
      return { R, ext, H, K, L, hidden, probs };
    }
  }

  /** Compare backprop gradients against central finite differences. */
  function gradCheck({ verbose = true } = {}) {
    const rows = [];
    for (const act of Activations.list) {
      for (const K of [2, 3]) {
        for (const layers of [1, 3]) {
          const net = new Net({ hidden: 4, layers, classes: K, act, seed: 11 });
          net.b.forEach(v => v.forEach((_, j) => { v[j] = 0.1 * (j - 1.5) + 0.03; }));
          const x1 = 0.37, x2 = -0.62, label = K === 3 ? 2 : 1;
          const g = net.backward(net.forward(x1, x2), label);
          const Lf = () => net.loss(net.forward(x1, x2), label);
          const h = 1e-5;
          let maxRel = 0;
          const check = (arr, i, analytic) => {
            const o = arr[i];
            arr[i] = o + h; const lp = Lf();
            arr[i] = o - h; const lm = Lf();
            arr[i] = o;
            const num = (lp - lm) / (2 * h);
            const rel = Math.abs(analytic - num) / Math.max(1e-8, Math.abs(analytic) + Math.abs(num));
            if (rel > maxRel) maxRel = rel;
          };
          for (let l = 0; l <= net.L; l++) {
            for (let j = 0; j < net.W[l].length; j++) {
              for (let i = 0; i < net.W[l][j].length; i++) check(net.W[l][j], i, g.dW[l][j][i]);
              check(net.b[l], j, g.db[l][j]);
            }
          }
          rows.push({ activation: act.name, classes: K, layers, output: net.outMode, maxRelError: +maxRel.toExponential(2), pass: maxRel < 1e-5 });
        }
      }
    }
    if (verbose && typeof console !== 'undefined' && console.table) console.table(rows);
    return rows;
  }

  window.NN = { Net, gradCheck };
})();
