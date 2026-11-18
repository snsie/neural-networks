/* A tiny dense network: 2 inputs -> H hidden -> O outputs.
 * 2 classes: 1 sigmoid output + binary cross-entropy.
 * 3 classes: 3 softmax outputs + categorical cross-entropy.
 * In both cases dL/dz2 = yhat - y, which keeps the backward pass clean. */
(function () {
  'use strict';

  const zeros = n => new Array(n).fill(0);
  const zeros2 = (r, c) => Array.from({ length: r }, () => zeros(c));
  const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));

  class Net {
    constructor({ hidden, classes, act, seed = 1 }) {
      this.H = hidden;
      this.K = classes;
      this.O = classes === 2 ? 1 : classes;
      this.outMode = classes === 2 ? 'sigmoid' : 'softmax';
      this.act = act;
      this.t = 0;
      const rng = Datasets.makeRng(seed);
      const s1 = Math.sqrt((act.init === 'he' ? 2 : 1) / 2) * 1.25;
      const s2 = Math.sqrt(1 / hidden) * 1.25;
      this.W1 = zeros2(hidden, 2).map(r => r.map(() => rng.normal() * s1));
      this.b1 = zeros(hidden).map(() => (act.init === 'he' ? 0.05 : 0));
      this.W2 = zeros2(this.O, hidden).map(r => r.map(() => rng.normal() * s2));
      this.b2 = zeros(this.O);
      this.m = this._zerosLike();
      this.v = this._zerosLike();
    }

    _zerosLike() {
      return { W1: zeros2(this.H, 2), b1: zeros(this.H), W2: zeros2(this.O, this.H), b2: zeros(this.O) };
    }

    forward(x1, x2) {
      const { H, O, W1, b1, W2, b2, act } = this;
      const z1 = new Array(H), a1 = new Array(H), z2 = new Array(O);
      for (let j = 0; j < H; j++) {
        z1[j] = W1[j][0] * x1 + W1[j][1] * x2 + b1[j];
        a1[j] = act.f(z1[j]);
      }
      for (let k = 0; k < O; k++) {
        let s = b2[k];
        for (let j = 0; j < H; j++) s += W2[k][j] * a1[j];
        z2[k] = s;
      }
      let yhat;
      if (this.outMode === 'sigmoid') {
        yhat = [sigmoid(z2[0])];
      } else {
        const m = Math.max(...z2);
        const e = z2.map(z => Math.exp(z - m));
        const s = e.reduce((a, b) => a + b, 0);
        yhat = e.map(v => v / s);
      }
      return { x: [x1, x2], z1, a1, z2, yhat };
    }

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
      const eps = 1e-12;
      if (this.outMode === 'sigmoid') {
        const p = c.yhat[0];
        return label === 1 ? -Math.log(Math.max(p, eps)) : -Math.log(Math.max(1 - p, eps));
      }
      return -Math.log(Math.max(c.yhat[label], eps));
    }

    backward(c, label) {
      const { H, O, W2, act } = this;
      const t = this.target(label);
      const dz2 = c.yhat.map((p, k) => p - t[k]);
      const dW2 = dz2.map(d => c.a1.map(a => d * a));
      const db2 = dz2.slice();
      const da1 = new Array(H), dadz = new Array(H), dz1 = new Array(H);
      for (let j = 0; j < H; j++) {
        let s = 0;
        for (let k = 0; k < O; k++) s += W2[k][j] * dz2[k];
        da1[j] = s;
        dadz[j] = act.df(c.z1[j]);
        dz1[j] = s * dadz[j];
      }
      const dW1 = dz1.map(d => [d * c.x[0], d * c.x[1]]);
      return { t, dz2, dW2, db2, da1, dadz, dz1, dW1, db1: dz1.slice() };
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
        for (let b = s; b < end; b++) {
          const p = points[idx[b]];
          const gr = this.backward(this.forward(p.x, p.y), p.label);
          for (let j = 0; j < this.H; j++) {
            g.W1[j][0] += gr.dW1[j][0];
            g.W1[j][1] += gr.dW1[j][1];
            g.b1[j] += gr.db1[j];
          }
          for (let k = 0; k < this.O; k++) {
            g.b2[k] += gr.db2[k];
            for (let j = 0; j < this.H; j++) g.W2[k][j] += gr.dW2[k][j];
          }
        }
        this.applyGrads(g, 1 / (end - s), lr, optimizer);
      }
    }

    applyGrads(g, scale, lr, optimizer) {
      this.t++;
      const B1 = 0.9, B2 = 0.999, EPS = 1e-8;
      const bc1 = 1 - Math.pow(B1, this.t), bc2 = 1 - Math.pow(B2, this.t);
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
      for (let j = 0; j < this.H; j++) upd(this.W1[j], g.W1[j], this.m.W1[j], this.v.W1[j]);
      upd(this.b1, g.b1, this.m.b1, this.v.b1);
      for (let k = 0; k < this.O; k++) upd(this.W2[k], g.W2[k], this.m.W2[k], this.v.W2[k]);
      upd(this.b2, g.b2, this.m.b2, this.v.b2);
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

    /** Evaluate an R x R grid over [-ext, ext]^2 (row 0 = top). */
    evalGrid(R, ext) {
      const { H, K } = this;
      const hidden = new Float32Array(R * R * H);
      const probs = new Float32Array(R * R * K);
      const step = (2 * ext) / R;
      for (let r = 0; r < R; r++) {
        const y = ext - (r + 0.5) * step;
        for (let c = 0; c < R; c++) {
          const x = -ext + (c + 0.5) * step;
          const cache = this.forward(x, y);
          const idx = r * R + c;
          for (let j = 0; j < H; j++) hidden[idx * H + j] = cache.a1[j];
          const p = this.probs(cache);
          for (let k = 0; k < K; k++) probs[idx * K + k] = p[k];
        }
      }
      return { R, ext, H, K, hidden, probs };
    }
  }

  /** Compare backprop gradients against central finite differences. */
  function gradCheck({ verbose = true } = {}) {
    const rows = [];
    for (const act of Activations.list) {
      for (const K of [2, 3]) {
        const net = new Net({ hidden: 4, classes: K, act, seed: 11 });
        net.b1 = net.b1.map((_, j) => 0.1 * (j - 1.5) + 0.03);
        net.b2 = net.b2.map((_, k) => 0.05 * (k + 1));
        const x1 = 0.37, x2 = -0.62, label = K === 3 ? 2 : 1;
        const g = net.backward(net.forward(x1, x2), label);
        const L = () => net.loss(net.forward(x1, x2), label);
        const h = 1e-5;
        let maxRel = 0;
        const check = (arr, i, analytic) => {
          const o = arr[i];
          arr[i] = o + h; const lp = L();
          arr[i] = o - h; const lm = L();
          arr[i] = o;
          const num = (lp - lm) / (2 * h);
          const rel = Math.abs(analytic - num) / Math.max(1e-8, Math.abs(analytic) + Math.abs(num));
          if (rel > maxRel) maxRel = rel;
        };
        for (let j = 0; j < net.H; j++) {
          check(net.W1[j], 0, g.dW1[j][0]);
          check(net.W1[j], 1, g.dW1[j][1]);
          check(net.b1, j, g.db1[j]);
        }
        for (let k = 0; k < net.O; k++) {
          for (let j = 0; j < net.H; j++) check(net.W2[k], j, g.dW2[k][j]);
          check(net.b2, k, g.db2[k]);
        }
        rows.push({ activation: act.name, classes: K, output: net.outMode, maxRelError: +maxRel.toExponential(2), pass: maxRel < 1e-5 });
      }
    }
    if (verbose && typeof console !== 'undefined' && console.table) console.table(rows);
    return rows;
  }

  window.NN = { Net, gradCheck };
})();
