/* A rotatable 3D view of a surface L(w, b), drawn on a canvas with no 3D library.
   Shows the shaded surface (same colors as the contour map), the two slice planes
   through the selected point with their curves, the tangent plane, and the
   downhill direction. Drag to rotate; click the surface to pick a point. */
import { fmt } from './calculus-math.js';

const MESH = 48;          // surface quads per side
const PICK = 60;          // samples per side used to find the clicked point
const SLICE_STEPS = 48;   // points along each slice curve
const Z_HALF = 0.62;      // half-height of the box, in the same units as w and b (−1…1)
const TANGENT_HALF = 0.34;
const ARROW_LEN = 0.42;

export const VIEWS = {
  default: { az: -0.62, el: 0.5, label: 'Default' },
  top: { az: 0, el: Math.PI / 2 - 0.001, label: 'Top (like the map)' },
  alongB: { az: 0, el: 0.06, label: 'Side: L vs w' },
  alongW: { az: -Math.PI / 2, el: 0.06, label: 'Side: L vs b' },
};

const hexToRgb = hex => {
  let h = hex.trim().replace('#', '');
  if (h.length === 3) h = [...h].map(c => c + c).join('');
  const n = parseInt(h, 16) || 0;
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
const rgba = (c, a = 1) => `rgba(${c.map(Math.round).join(',')},${a})`;

/**
 * host: element for the canvas.
 * options: wRange, bRange, onSelect(w, b)
 * Returns { setModel({ fn, rank, zRange }), setPoint({ w, b, L, dw, db }), setView(name), showTangent(bool) }.
 */
export function createSurfacePlot(host, { wRange, bRange, onSelect }) {
  host.innerHTML = `
    <canvas class="calc-surface-canvas" tabindex="0" role="img"
      aria-label="3D surface of L over w and b. Drag to rotate, click the surface to choose a point, or use the arrow keys to rotate."></canvas>`;
  const canvas = host.querySelector('canvas');
  const ctx = canvas.getContext('2d');

  let model = null;
  let point = null;
  let az = VIEWS.default.az, el = VIEWS.default.el;
  let tangentOn = true;
  let picks = [];
  let colors = null;
  let anim = 0;
  let cssW = 0, cssH = 0;

  /* ---------- coordinates ---------- */
  // Data → box units: w and b to −1…1, L to −Z_HALF…Z_HALF (clamped).
  const nx = w => ((w - wRange[0]) / (wRange[1] - wRange[0])) * 2 - 1;
  const ny = b => ((b - bRange[0]) / (bRange[1] - bRange[0])) * 2 - 1;
  const nz = L => {
    const [z0, z1] = model.zRange;
    return ((Math.min(z1, Math.max(z0, L)) - z0) / (z1 - z0)) * 2 * Z_HALF - Z_HALF;
  };
  const W = u => wRange[0] + ((u + 1) / 2) * (wRange[1] - wRange[0]);
  const B = v => bRange[0] + ((v + 1) / 2) * (bRange[1] - bRange[0]);

  // Orthographic camera: rotate about the vertical axis (az), then tilt (el).
  function project(X, Y, Z) {
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
    const x1 = X * ca - Y * sa;
    const y1 = X * sa + Y * ca;
    const scale = Math.min(cssW / 3.4, cssH / 2.85);
    return {
      x: cssW * 0.54 + x1 * scale,
      y: cssH * 0.53 - (Z * ce + y1 * se) * scale,
      depth: y1 * ce - Z * se, // larger is farther away
    };
  }
  // Direction to the viewer, for lighting and for choosing which box edges are in front.
  function viewSpace(X, Y, Z) {
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
    const x1 = X * ca - Y * sa;
    const y1 = X * sa + Y * ca;
    return [x1, Z * ce + y1 * se, y1 * ce - Z * se];
  }

  /* ---------- drawing ---------- */
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const get = name => hexToRgb(cs.getPropertyValue(name));
    const surface = get('--surface');
    colors = {
      surface, text: get('--text'), muted: get('--muted'), border: get('--border'), faint: get('--faint'),
      high: mix(surface, get('--accent'), 0.78), fwd: get('--fwd'), bwd: get('--bwd'), accent: get('--accent'),
      dark: (cs.getPropertyValue('color-scheme') || '').includes('dark') || surface.reduce((a, b) => a + b, 0) < 300,
    };
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return false;
    cssW = rect.width;
    cssH = rect.height;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return true;
  }

  function line(a, b, color, width = 1, dash = []) {
    ctx.beginPath();
    ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function label(text, p, color, align = 'center', font = '12px') {
    ctx.font = `${font} Inter, system-ui, sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 3;
    ctx.strokeStyle = rgba(colors.surface, 0.85);
    ctx.strokeText(text, p.x, p.y);
    ctx.fillStyle = color;
    ctx.fillText(text, p.x, p.y);
  }

  function drawBox() {
    const zf = -Z_HALF;
    // Floor grid at the lowest L, one line per unit of w and b.
    for (let w = Math.ceil(wRange[0]); w <= wRange[1]; w++) line(project(nx(w), -1, zf), project(nx(w), 1, zf), rgba(colors.border), 1);
    for (let b = Math.ceil(bRange[0]); b <= bRange[1]; b++) line(project(-1, ny(b), zf), project(1, ny(b), zf), rgba(colors.border), 1);

    // Tick labels on the floor edges nearest the viewer.
    const nearB = viewSpace(0, -1, 0)[2] < viewSpace(0, 1, 0)[2] ? -1 : 1;
    const nearW = viewSpace(-1, 0, 0)[2] < viewSpace(1, 0, 0)[2] ? -1 : 1;
    const out = 0.16;
    const sideView = el < 0.2;
    if (Math.abs(Math.cos(az)) > 0.25 || !sideView) {
      for (let w = Math.ceil(wRange[0]); w <= wRange[1]; w++) label(fmt(w), project(nx(w), nearB * (1 + out), zf), rgba(colors.muted));
      label('w', project(0, nearB * (1 + out * 2.2), zf), rgba(colors.text), 'center', 'italic 600 14px');
    }
    if (Math.abs(Math.sin(az)) > 0.25 || !sideView) {
      for (let b = Math.ceil(bRange[0]); b <= bRange[1]; b++) label(fmt(b), project(nearW * (1 + out), ny(b), zf), rgba(colors.muted));
      label('b', project(nearW * (1 + out * 2.2), 0, zf), rgba(colors.text), 'center', 'italic 600 14px');
    }

    // Vertical L axis at the floor corner farthest to the left on screen.
    if (el < 1.45) {
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
      const [cx, cy] = corners.reduce((best, c) => (project(c[0], c[1], 0).x < project(best[0], best[1], 0).x ? c : best));
      line(project(cx, cy, -Z_HALF), project(cx, cy, Z_HALF), rgba(colors.muted), 1.2);
      const [z0, z1] = model.zRange;
      const step = niceStep(z1 - z0, 4);
      for (let v = Math.ceil(z0 / step) * step; v <= z1 + 1e-9; v += step) {
        if (v - z0 < (z1 - z0) * 0.06) continue; // leave the floor corner to the w and b labels
        const p = project(cx, cy, nz(v));
        line(p, { x: p.x - 5, y: p.y }, rgba(colors.muted), 1.2);
        label(fmt(v, 1), { x: p.x - 9, y: p.y }, rgba(colors.muted), 'right');
      }
      const top = project(cx, cy, Z_HALF);
      label('L', { x: top.x, y: top.y - 12 }, rgba(colors.text), 'center', 'italic 600 14px');
    }
  }

  function niceStep(span, target) {
    const raw = span / target;
    const p = 10 ** Math.floor(Math.log10(raw));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  }

  function draw() {
    if (!model || !colors || !cssW) return;
    ctx.clearRect(0, 0, cssW, cssH);
    drawBox();

    const light = normalise([-0.35, 0.75, -0.55]);
    const polys = [];

    // Surface quads, colored by rank like the contour map, shaded by the light.
    const grid = [];
    for (let j = 0; j <= MESH; j++) {
      const row = [];
      for (let i = 0; i <= MESH; i++) {
        const X = (i / MESH) * 2 - 1, Y = (j / MESH) * 2 - 1;
        const L = model.fn(W(X), B(Y));
        row.push(Number.isFinite(L) ? { X, Y, Z: nz(L), L } : null);
      }
      grid.push(row);
    }
    for (let j = 0; j < MESH; j++) {
      for (let i = 0; i < MESH; i++) {
        const q = [grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]];
        if (q.some(v => !v)) continue;
        const pts = q.map(v => project(v.X, v.Y, v.Z));
        const a = viewSpace(q[1].X - q[0].X, q[1].Y - q[0].Y, q[1].Z - q[0].Z);
        const c = viewSpace(q[3].X - q[0].X, q[3].Y - q[0].Y, q[3].Z - q[0].Z);
        const n = normalise(cross(a, c));
        const lambert = Math.abs(n[0] * light[0] + n[1] * light[1] + n[2] * light[2]);
        const meanL = (q[0].L + q[1].L + q[2].L + q[3].L) / 4;
        const base = mix(colors.surface, colors.high, model.rank(meanL));
        const shade = colors.dark ? 0.55 + 0.6 * lambert : 0.7 + 0.32 * lambert;
        polys.push({
          pts, depth: pts.reduce((s, p) => s + p.depth, 0) / 4,
          fill: rgba(base.map(v => Math.min(255, v * shade))),
          edges: [j % 6 === 0, i % 6 === 5 || i === MESH - 1, j % 6 === 5 || j === MESH - 1, i % 6 === 0],
        });
      }
    }

    // Slice planes through the point, split into strips so they sort against the surface.
    if (point) {
      const X0 = nx(point.w), Y0 = ny(point.b);
      const across = 16, up = 8;
      const planeFill = rgba(colors.fwd, colors.dark ? 0.09 : 0.07);
      for (let k = 0; k < across; k++) {
        const t0 = (k / across) * 2 - 1, t1 = ((k + 1) / across) * 2 - 1;
        for (let m = 0; m < up; m++) {
          const z0 = -Z_HALF + (m / up) * 2 * Z_HALF, z1 = -Z_HALF + ((m + 1) / up) * 2 * Z_HALF;
          for (const [ax, ay, bx, by] of [[t0, Y0, t1, Y0], [X0, t0, X0, t1]]) {
            const pts = [project(ax, ay, z0), project(bx, by, z0), project(bx, by, z1), project(ax, ay, z1)];
            polys.push({ pts, depth: pts.reduce((s, p) => s + p.depth, 0) / 4, fill: planeFill, plane: true });
          }
        }
      }
    }

    // Painter's algorithm: far to near.
    polys.sort((a, b) => b.depth - a.depth);
    const meshLine = rgba(colors.text, colors.dark ? 0.16 : 0.13);
    for (const p of polys) {
      ctx.beginPath();
      p.pts.forEach((v, k) => (k ? ctx.lineTo(v.x, v.y) : ctx.moveTo(v.x, v.y)));
      ctx.closePath();
      ctx.fillStyle = p.fill;
      ctx.fill();
      if (p.plane) continue;
      // Hairline in the fill color hides seams between neighbouring quads.
      ctx.strokeStyle = p.fill;
      ctx.lineWidth = 0.6;
      ctx.stroke();
      ctx.strokeStyle = meshLine;
      ctx.lineWidth = 1;
      p.edges.forEach((on, k) => {
        if (!on) return;
        const a = p.pts[k], b = p.pts[(k + 1) % 4];
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      });
    }

    if (point) {
      if (tangentOn) drawTangentPlane();
      drawPointOverlay();
    }

    // Points used to map a click back onto the surface.
    picks = [];
    for (let j = 0; j <= PICK; j++) {
      for (let i = 0; i <= PICK; i++) {
        const w = wRange[0] + (i / PICK) * (wRange[1] - wRange[0]);
        const b = bRange[0] + (j / PICK) * (bRange[1] - bRange[0]);
        const L = model.fn(w, b);
        if (Number.isFinite(L)) picks.push({ w, b, ...project(nx(w), ny(b), nz(L)) });
      }
    }
  }

  // Tangent plane L ≈ L₀ + ∂L/∂w (w − w₀) + ∂L/∂b (b − b₀): one translucent patch,
  // kept inside the box and drawn over the surface so it is never hidden.
  function drawTangentPlane() {
    const { w, b, L, dw, db } = point;
    if (![L, dw, db].every(Number.isFinite)) return;
    const dW = TANGENT_HALF * (wRange[1] - wRange[0]) / 2, dB = TANGENT_HALF * (bRange[1] - bRange[0]) / 2;
    const w0 = Math.max(wRange[0], w - dW), w1 = Math.min(wRange[1], w + dW);
    const b0 = Math.max(bRange[0], b - dB), b1 = Math.min(bRange[1], b + dB);
    const at = (wv, bv) => project(nx(wv), ny(bv), rawZ(L + dw * (wv - w) + db * (bv - b)));
    const pts = [at(w0, b0), at(w1, b0), at(w1, b1), at(w0, b1)];
    ctx.beginPath();
    pts.forEach((v, k) => (k ? ctx.lineTo(v.x, v.y) : ctx.moveTo(v.x, v.y)));
    ctx.closePath();
    ctx.fillStyle = rgba(colors.text, colors.dark ? 0.2 : 0.16);
    ctx.fill();
    ctx.strokeStyle = rgba(colors.text, 0.5);
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  // Z for the tangent plane, not clamped, so the patch keeps its true slope.
  function rawZ(L) {
    const [z0, z1] = model.zRange;
    return ((L - z0) / (z1 - z0)) * 2 * Z_HALF - Z_HALF;
  }

  function polyline(points, color, width, dash = []) {
    ctx.beginPath();
    ctx.setLineDash(dash);
    let pen = false;
    for (const p of points) {
      if (!p) { pen = false; continue; }
      if (pen) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
      pen = true;
    }
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = rgba(colors.surface, 0.9);
    ctx.lineWidth = width + 3;
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawPointOverlay() {
    const { w, b, L, dw, db } = point;
    const X0 = nx(w), Y0 = ny(b);
    const onSurface = (wv, bv) => {
      const v = model.fn(wv, bv);
      return Number.isFinite(v) ? project(nx(wv), ny(bv), nz(v)) : null;
    };
    // The two slice curves: exactly what the 2D slice graphs show.
    const alongW = Array.from({ length: SLICE_STEPS + 1 }, (_, k) => onSurface(wRange[0] + (k / SLICE_STEPS) * (wRange[1] - wRange[0]), b));
    const alongB = Array.from({ length: SLICE_STEPS + 1 }, (_, k) => onSurface(w, bRange[0] + (k / SLICE_STEPS) * (bRange[1] - bRange[0])));
    polyline(alongW, rgba(colors.fwd), 2.4);
    polyline(alongB, rgba(colors.fwd), 2.4);
    if (!Number.isFinite(L)) return;

    // Drop line to the floor, where the point sits on the contour map.
    const P = project(X0, Y0, nz(L));
    const F = project(X0, Y0, -Z_HALF);
    line(P, F, rgba(colors.text, 0.6), 1.2, [4, 4]);
    ctx.beginPath();
    ctx.arc(F.x, F.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = rgba(colors.text, 0.6);
    ctx.fill();

    if (Number.isFinite(dw) && Number.isFinite(db)) {
      // Tangent lines along w and along b: the slopes from the two slice graphs.
      const h = 0.24;
      const dW = h * (wRange[1] - wRange[0]) / 2, dB = h * (bRange[1] - bRange[0]) / 2;
      const tw = [project(nx(w - dW), Y0, rawZ(L - dw * dW)), project(nx(w + dW), Y0, rawZ(L + dw * dW))];
      const tb = [project(X0, ny(b - dB), rawZ(L - db * dB)), project(X0, ny(b + dB), rawZ(L + db * dB))];
      polyline(tw, rgba(colors.text), 2.2);
      polyline(tb, rgba(colors.text), 2.2);

      // Steepest descent along the tangent plane, scaled to a fixed length on screen.
      const g = Math.hypot(dw, db);
      if (g > 1e-6) {
        const sw = -dw / g, sb = -db / g; // unit step in (w, b) per unit of box length
        const stepW = sw * (wRange[1] - wRange[0]) / 2, stepB = sb * (bRange[1] - bRange[0]) / 2;
        const dz = rawZ(L + dw * stepW + db * stepB) - rawZ(L);
        const len = Math.hypot(sw, sb, dz);
        const k = ARROW_LEN / len;
        const E = project(X0 + sw * k, Y0 + sb * k, nz(L) + dz * k);
        arrow(P, E);
      }
    }
    ctx.beginPath();
    ctx.arc(P.x, P.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = rgba(colors.fwd);
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = rgba(colors.surface);
    ctx.stroke();
  }

  function arrow(a, b) {
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    if (Math.hypot(b.x - a.x, b.y - a.y) < 6) return;
    polyline([a, b], rgba(colors.text), 2.6);
    ctx.beginPath();
    ctx.moveTo(b.x + Math.cos(ang) * 4, b.y + Math.sin(ang) * 4);
    ctx.lineTo(b.x - Math.cos(ang - 0.5) * 11, b.y - Math.sin(ang - 0.5) * 11);
    ctx.lineTo(b.x - Math.cos(ang + 0.5) * 11, b.y - Math.sin(ang + 0.5) * 11);
    ctx.closePath();
    ctx.fillStyle = rgba(colors.text);
    ctx.fill();
  }

  /* ---------- interaction ---------- */
  let drag = null;
  canvas.addEventListener('pointerdown', event => {
    event.preventDefault(); // no text selection while dragging
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add('pointer', 'dragging');
    drag = { x: event.clientX, y: event.clientY, az, el, moved: false };
  });
  canvas.addEventListener('pointermove', event => {
    if (!drag || !canvas.hasPointerCapture(event.pointerId)) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    drag.moved = true;
    cancelAnimationFrame(anim);
    az = drag.az - dx * 0.011;
    el = Math.min(Math.PI / 2 - 0.001, Math.max(0, drag.el + dy * 0.009));
    draw();
  });
  const endDrag = event => {
    if (!drag) return;
    canvas.classList.remove('dragging');
    if (!drag.moved && event.type === 'pointerup') pickAt(event);
    drag = null;
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('blur', () => canvas.classList.remove('pointer'));

  function pickAt(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    // Nearest visible sample: among points close to the click, take the one nearest the viewer.
    let best = null;
    for (const p of picks) {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d > 14) continue;
      if (!best || p.depth < best.depth - 0.05 || (Math.abs(p.depth - best.depth) <= 0.05 && d < best.d)) best = { ...p, d };
    }
    if (best) onSelect?.(Math.round(best.w * 100) / 100, Math.round(best.b * 100) / 100);
  }

  canvas.addEventListener('keydown', event => {
    const turn = { ArrowLeft: [0.12, 0], ArrowRight: [-0.12, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] }[event.key];
    if (!turn) return;
    event.preventDefault();
    canvas.classList.remove('pointer');
    cancelAnimationFrame(anim);
    az += turn[0];
    el = Math.min(Math.PI / 2 - 0.001, Math.max(0, el + turn[1]));
    draw();
  });

  // Redraw on resize (the canvas follows its CSS size).
  const observer = new ResizeObserver(() => {
    if (!host.isConnected) { observer.disconnect(); return; }
    if (resize()) draw();
  });
  observer.observe(canvas);

  return {
    setModel(next) {
      model = next;
      readColors();
      if (!cssW) resize();
      draw();
    },
    setPoint(next) {
      point = next;
      draw();
    },
    showTangent(on) {
      tangentOn = on;
      draw();
    },
    refreshColors() {
      readColors();
      draw();
    },
    // Turn smoothly to a preset view (instantly when reduced motion is preferred).
    setView(name) {
      const target = VIEWS[name];
      if (!target) return;
      cancelAnimationFrame(anim);
      const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
      // Take the short way round.
      let toAz = target.az;
      while (toAz - az > Math.PI) toAz -= 2 * Math.PI;
      while (az - toAz > Math.PI) toAz += 2 * Math.PI;
      const from = { az, el }, start = performance.now(), dur = reduce ? 0 : 450;
      const step = now => {
        const t = dur ? Math.min(1, (now - start) / dur) : 1;
        const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
        az = from.az + (toAz - from.az) * e;
        el = from.el + (target.el - from.el) * e;
        draw();
        if (t < 1) anim = requestAnimationFrame(step);
      };
      anim = requestAnimationFrame(step);
    },
  };
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalise(v) {
  const n = Math.hypot(...v) || 1;
  return v.map(x => x / n);
}
