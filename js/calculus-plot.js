/* A function plot shared by the calculus lessons: a function (solid), its
   derivative (dashed), and at a selected point the two values and a short
   tangent line. Click, drag or use the arrow keys to move the point. */
import { fmt } from './calculus-math.js';

const SAMPLES = 480;
const TANGENT_HALF = 46;
let uid = 0;

function niceStep(span, target) {
  const raw = span / target;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

function quantile(sorted, q) {
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

// Smooth curves use their full range; curves with an asymptote (1/x, ln x, tan x)
// are trimmed to their middle 90%.
function robustRange(values) {
  const vals = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!vals.length) return null;
  const lo = quantile(vals, 0.05), hi = quantile(vals, 0.95), span = hi - lo;
  const full = vals[vals.length - 1] - vals[0];
  if (full <= span * 3 + 1e-9) return { lo: vals[0], hi: vals[vals.length - 1], heavy: false };
  return { lo: lo - span * 0.25, hi: hi + span * 0.25, heavy: true };
}

/* The function's values set the view; a derivative that blows up only widens it so far.
   Takes arrays of sampled values so callers can pool samples (e.g. across a whole surface). */
export function valueRange(fValues, dValues = []) {
  const fr = robustRange(fValues) ?? { lo: -1, hi: 1 };
  let lo = fr.lo, hi = fr.hi;
  const dr = robustRange(dValues);
  if (dr) {
    const reach = Math.max(fr.hi - fr.lo, 2);
    lo = Math.min(lo, dr.heavy ? Math.max(dr.lo, fr.lo - reach) : dr.lo);
    hi = Math.max(hi, dr.heavy ? Math.min(dr.hi, fr.hi + reach) : dr.hi);
  }
  lo = Math.min(lo, 0);
  hi = Math.max(hi, 0);
  if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

export const sample = (fn, [a, b], n = SAMPLES) =>
  Array.from({ length: n + 1 }, (_, i) => fn(a + (i / n) * (b - a)));

/**
 * host: element to render into.
 * options: xName, fName, dName (legend text), caption, width, height,
 *          interactive (default true), onSelect(x), compact (larger text for small plots).
 * Returns { setFunctions(f, df, { xRange, yRange }), setX(x), setCaption(html), x }.
 */
export function createFunctionPlot(host, options) {
  const {
    xName = 'x', fName = 'y', dName = 'dy/dx', caption = '',
    width = 560, height = 300, interactive = true, onSelect, compact = false,
  } = options;
  const id = `calcClip${++uid}`;
  const W = width, H = height;
  const M = { l: compact ? 52 : 46, r: 12, t: 12, b: compact ? 36 : 30 };
  const PW = W - M.l - M.r, PH = H - M.t - M.b;

  host.innerHTML = `
    <figure class="calc-plot${compact ? ' compact' : ''}">
      ${caption !== null ? `<figcaption>${caption}</figcaption>` : ''}
      <div class="calc-legend" aria-hidden="true">
        <span class="lg lg-f">${fName}</span>
        <span class="lg lg-df">${dName}</span>
        <span class="lg lg-tan">tangent</span>
      </div>
      <svg viewBox="0 0 ${W} ${H}" ${interactive ? `tabindex="0" role="slider" aria-label="Point on the ${fName} curve. Use the arrow keys to move it."` : 'role="img"'} class="${interactive ? 'interactive' : ''}"></svg>
    </figure>`;
  const svg = host.querySelector('svg');
  const captionEl = host.querySelector('figcaption');

  let f = () => NaN, df = () => NaN;
  let xRange = [-3, 3], ys = [-1, 1];
  let px = 0;
  let staticLayer = '';

  const xs = x => M.l + ((x - xRange[0]) / (xRange[1] - xRange[0])) * PW;
  const yPos = y => M.t + PH - ((y - ys[0]) / (ys[1] - ys[0])) * PH;

  function curvePath(fn) {
    const span = ys[1] - ys[0];
    let d = '', pen = false, prev = null;
    for (let i = 0; i <= SAMPLES; i++) {
      const x = xRange[0] + (i / SAMPLES) * (xRange[1] - xRange[0]);
      const y = fn(x);
      // Lift the pen on undefined values and on jumps across an asymptote.
      if (!Number.isFinite(y)) { pen = false; prev = null; continue; }
      if (prev !== null && Math.abs(y - prev) > span * 3) pen = false;
      const Y = Math.min(H * 6, Math.max(-H * 5, yPos(y)));
      d += `${pen ? 'L' : 'M'}${xs(x).toFixed(1)},${Y.toFixed(1)}`;
      pen = true;
      prev = y;
    }
    return d;
  }

  function buildStatic() {
    const [x0, x1] = xRange;
    const xStep = niceStep(x1 - x0, compact ? 4 : 6);
    const yStep = niceStep(ys[1] - ys[0], compact ? 4 : 5);
    let grid = '', labels = '';
    for (let v = Math.ceil(x0 / xStep - 1e-9) * xStep; v <= x1 + 1e-9; v += xStep) {
      const X = xs(v).toFixed(1);
      grid += `<line class="calc-grid${Math.abs(v) < xStep * 1e-6 ? ' calc-zero' : ''}" x1="${X}" y1="${M.t}" x2="${X}" y2="${M.t + PH}"/>`;
      labels += `<text x="${X}" y="${H - (compact ? 12 : 10)}" text-anchor="middle">${fmt(v)}</text>`;
    }
    for (let v = Math.ceil(ys[0] / yStep - 1e-9) * yStep; v <= ys[1] + 1e-9; v += yStep) {
      const Y = yPos(v).toFixed(1);
      grid += `<line class="calc-grid${Math.abs(v) < yStep * 1e-6 ? ' calc-zero' : ''}" x1="${M.l}" y1="${Y}" x2="${M.l + PW}" y2="${Y}"/>`;
      labels += `<text x="${M.l - 6}" y="${(+Y + 4).toFixed(1)}" text-anchor="end">${fmt(v, 1)}</text>`;
    }
    staticLayer = `
      <defs><clipPath id="${id}"><rect x="${M.l}" y="${M.t}" width="${PW}" height="${PH}"/></clipPath></defs>
      <rect class="calc-frame" x="${M.l}" y="${M.t}" width="${PW}" height="${PH}" rx="4"/>
      ${grid}${labels}
      <text class="calc-axis-name" x="${M.l + PW - 5}" y="${M.t + PH - 7}" text-anchor="end">${xName}</text>
      <g clip-path="url(#${id})">
        <path class="calc-df" d="${curvePath(df)}"/>
        <path class="calc-f" d="${curvePath(f)}"/>
      </g>`;
  }

  function draw() {
    const y = f(px), m = df(px), X = xs(px);
    let marks = `<line class="calc-guide" x1="${X.toFixed(1)}" y1="${M.t}" x2="${X.toFixed(1)}" y2="${M.t + PH}"/>`;
    if (Number.isFinite(y) && Number.isFinite(m)) {
      // Fixed on-screen length, so the tangent stays short however steep it is.
      const sx = PW / (xRange[1] - xRange[0]);
      const sy = PH / (ys[1] - ys[0]);
      const len = Math.hypot(sx, m * sy);
      const ux = (sx / len) * TANGENT_HALF, uy = (-m * sy / len) * TANGENT_HALF;
      const Y = yPos(y);
      marks += `<line class="calc-tan" x1="${(X - ux).toFixed(1)}" y1="${(Y - uy).toFixed(1)}" x2="${(X + ux).toFixed(1)}" y2="${(Y + uy).toFixed(1)}"/>`;
    }
    if (Number.isFinite(m)) marks += `<circle class="calc-pt-df" cx="${X.toFixed(1)}" cy="${yPos(m).toFixed(1)}" r="4.5"/>`;
    if (Number.isFinite(y)) marks += `<circle class="calc-pt-f" cx="${X.toFixed(1)}" cy="${yPos(y).toFixed(1)}" r="5.5"/>`;
    svg.innerHTML = `${staticLayer}<g clip-path="url(#${id})">${marks}</g>`;
    if (interactive) {
      svg.setAttribute('aria-valuemin', xRange[0]);
      svg.setAttribute('aria-valuemax', xRange[1]);
      svg.setAttribute('aria-valuenow', px);
      svg.setAttribute('aria-valuetext', `${xName} = ${fmt(px)}, ${fName} = ${fmt(y)}, ${dName} = ${fmt(m)}`);
    } else {
      svg.setAttribute('aria-label', `${fName} versus ${xName}. At ${xName} = ${fmt(px)}, ${fName} = ${fmt(y)} and ${dName} = ${fmt(m)}.`);
    }
  }

  const clamp = x => Math.round(Math.min(xRange[1], Math.max(xRange[0], x)) * 100) / 100;
  function select(x) {
    px = clamp(x);
    draw();
    onSelect?.(px);
  }

  if (interactive) {
    const fromPointer = event => {
      const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
      select(xRange[0] + ((pt.x - M.l) / PW) * (xRange[1] - xRange[0]));
    };
    svg.addEventListener('pointerdown', event => {
      svg.classList.add('pointer');
      svg.setPointerCapture(event.pointerId);
      fromPointer(event);
    });
    svg.addEventListener('pointermove', event => {
      if (svg.hasPointerCapture(event.pointerId)) fromPointer(event);
    });
    svg.addEventListener('blur', () => svg.classList.remove('pointer'));
    svg.addEventListener('keydown', event => {
      const step = (event.shiftKey ? 0.125 : 0.0125) * (xRange[1] - xRange[0]);
      const moves = { ArrowRight: step, ArrowUp: step, ArrowLeft: -step, ArrowDown: -step };
      let x = px;
      if (event.key in moves) x += moves[event.key];
      else if (event.key === 'Home') x = xRange[0];
      else if (event.key === 'End') x = xRange[1];
      else return;
      event.preventDefault();
      svg.classList.remove('pointer');
      select(x);
    });
  }

  return {
    setFunctions(nextF, nextDf, opts = {}) {
      f = nextF;
      df = nextDf;
      if (opts.xRange) xRange = opts.xRange;
      ys = opts.yRange ?? valueRange(sample(f, xRange), sample(df, xRange));
      buildStatic();
      draw();
    },
    setX(x) {
      px = x;
      draw();
    },
    setCaption(html) {
      if (captionEl) captionEl.innerHTML = html;
    },
    get x() { return px; },
  };
}
