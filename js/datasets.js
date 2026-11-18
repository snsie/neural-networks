/* Seeded 2-D toy datasets. Coordinates live roughly in [-1, 1] x [-1, 1]. */
(function () {
  'use strict';

  const TAU = Math.PI * 2;
  const EXTENT = 1.15; // plot range is [-EXTENT, EXTENT] on both axes

  // mulberry32: small, fast, deterministic PRNG
  function makeRng(seed) {
    let a = (seed >>> 0) || 1;
    const rng = function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    rng.range = (lo, hi) => lo + (hi - lo) * rng();
    rng.int = n => Math.floor(rng() * n);
    rng.normal = () => {
      let u = 0;
      while (u === 0) u = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * rng());
    };
    return rng;
  }

  const jitter = (p, s, rng) => { p.x += rng.normal() * s; p.y += rng.normal() * s; return p; };

  const gens = {
    circle(n, noise, rng) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const label = i % 2;
        const r = label === 1 ? 0.45 * Math.sqrt(rng()) : rng.range(0.66, 0.95);
        const t = rng() * TAU;
        pts.push(jitter({ x: r * Math.cos(t), y: r * Math.sin(t), label }, noise * 0.22, rng));
      }
      return pts;
    },

    xor(n, noise, rng) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        let x = rng.range(-0.9, 0.9), y = rng.range(-0.9, 0.9);
        x += x > 0 ? 0.08 : -0.08;
        y += y > 0 ? 0.08 : -0.08;
        const label = x * y > 0 ? 1 : 0;
        pts.push(jitter({ x, y, label }, noise * 0.2, rng));
      }
      return pts;
    },

    gaussian(n, noise, rng) {
      const pts = [];
      const centers = [[-0.45, 0.42], [0.45, -0.42]];
      const s = 0.14 + noise * 0.35;
      for (let i = 0; i < n; i++) {
        const label = i % 2;
        const [cx, cy] = centers[label];
        pts.push({ x: cx + rng.normal() * s, y: cy + rng.normal() * s, label });
      }
      return pts;
    },

    moons(n, noise, rng) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const label = i % 2;
        const t = rng() * Math.PI;
        let x, y;
        if (label === 0) { x = Math.cos(t); y = Math.sin(t); }
        else { x = 1 - Math.cos(t); y = 0.5 - Math.sin(t); }
        pts.push(jitter({ x: (x - 0.5) * 0.62, y: (y - 0.25) * 0.62, label }, 0.03 + noise * 0.22, rng));
      }
      return pts;
    },

    spiral(n, noise, rng) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const label = i % 2;
        const t = rng();
        const r = 0.08 + 0.87 * t;
        const a = t * 0.95 * TAU + label * Math.PI;
        pts.push(jitter({ x: r * Math.cos(a), y: r * Math.sin(a), label }, 0.02 + noise * 0.15, rng));
      }
      return pts;
    },

    blobs3(n, noise, rng) {
      const pts = [];
      const s = 0.12 + noise * 0.3;
      for (let i = 0; i < n; i++) {
        const label = i % 3;
        const a = Math.PI / 2 + label * TAU / 3;
        pts.push({ x: 0.52 * Math.cos(a) + rng.normal() * s, y: 0.52 * Math.sin(a) + rng.normal() * s, label });
      }
      return pts;
    },

    pinwheel3(n, noise, rng) {
      const pts = [];
      const sector = TAU / 3;
      let guard = 0;
      while (pts.length < n && guard++ < n * 60) {
        const r = 0.95 * Math.sqrt(rng());
        if (r < 0.06) continue;
        const th = rng() * TAU;
        const u = (((th - 2.4 * r) % TAU) + TAU) % TAU;
        const label = Math.floor(u / sector) % 3;
        const off = u - label * sector;
        if (r * Math.min(off, sector - off) < 0.035) continue; // keep clean gaps between blades
        pts.push(jitter({ x: r * Math.cos(th), y: r * Math.sin(th), label }, noise * 0.2, rng));
      }
      return pts;
    },

    spiral3(n, noise, rng) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const label = i % 3;
        const t = rng();
        const r = 0.08 + 0.87 * t;
        const a = t * 0.75 * TAU + label * TAU / 3;
        pts.push(jitter({ x: r * Math.cos(a), y: r * Math.sin(a), label }, 0.02 + noise * 0.15, rng));
      }
      return pts;
    },
  };

  const LIST = [
    { id: 'circle', name: 'Circle', classes: 2 },
    { id: 'xor', name: 'XOR', classes: 2 },
    { id: 'gaussian', name: 'Gaussian', classes: 2 },
    { id: 'moons', name: 'Moons', classes: 2 },
    { id: 'spiral', name: 'Spiral', classes: 2 },
    { id: 'blobs3', name: '3 Blobs', classes: 3 },
    { id: 'pinwheel3', name: 'Pinwheel', classes: 3 },
    { id: 'spiral3', name: '3 Spirals', classes: 3 },
  ];

  function hash(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  /**
   * Generate a dataset.
   * @returns {{points: Array<{id,x,y,label,test}>, classes: number, meta: object}}
   */
  function generate(id, { n = 300, noise = 0.1, seed = 1, testFrac = 0.2 } = {}) {
    const meta = LIST.find(d => d.id === id) || LIST[0];
    const rng = makeRng((seed * 9973) ^ hash(meta.id));
    const pts = gens[meta.id](n, noise, rng);
    for (let i = pts.length - 1; i > 0; i--) {
      const j = rng.int(i + 1);
      const t = pts[i]; pts[i] = pts[j]; pts[j] = t;
    }
    const nTest = Math.round(pts.length * testFrac);
    const lim = EXTENT - 0.03;
    pts.forEach((p, i) => {
      p.id = i;
      p.test = i < nTest;
      p.x = Math.max(-lim, Math.min(lim, p.x));
      p.y = Math.max(-lim, Math.min(lim, p.y));
    });
    return { points: pts, classes: meta.classes, meta };
  }

  window.Datasets = { list: LIST, generate, makeRng, EXTENT };
})();
