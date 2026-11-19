/* Derivative lesson: an editable function, its symbolic derivative, and an
   overlaid plot of y and dy/dx with a tangent line at a clicked point.
   mathjs is loaded lazily so it only downloads when this lesson opens. */
import {
  create,
  parseDependencies,
  compileDependencies,
  derivativeDependencies,
  simplifyDependencies,
  addDependencies,
  subtractDependencies,
  multiplyDependencies,
  divideDependencies,
  powDependencies,
  unaryMinusDependencies,
  unaryPlusDependencies,
  eDependencies,
  piDependencies,
  sinDependencies,
  cosDependencies,
  tanDependencies,
  secDependencies,
  cscDependencies,
  cotDependencies,
  asinDependencies,
  acosDependencies,
  atanDependencies,
  sinhDependencies,
  coshDependencies,
  tanhDependencies,
  sechDependencies,
  cschDependencies,
  cothDependencies,
  expDependencies,
  logDependencies,
  sqrtDependencies,
  cbrtDependencies,
  absDependencies,
} from 'mathjs';

const math = create({
  parseDependencies,
  compileDependencies,
  derivativeDependencies,
  simplifyDependencies,
  addDependencies,
  subtractDependencies,
  multiplyDependencies,
  divideDependencies,
  powDependencies,
  unaryMinusDependencies,
  unaryPlusDependencies,
  eDependencies,
  piDependencies,
  sinDependencies,
  cosDependencies,
  tanDependencies,
  secDependencies,
  cscDependencies,
  cotDependencies,
  asinDependencies,
  acosDependencies,
  atanDependencies,
  sinhDependencies,
  coshDependencies,
  tanhDependencies,
  sechDependencies,
  cschDependencies,
  cothDependencies,
  expDependencies,
  logDependencies,
  sqrtDependencies,
  cbrtDependencies,
  absDependencies,
}, { predictable: true });

const DEFAULT_FN = 'x^3';
const X_RANGE = [-3, 3];
const SAMPLES = 480;
const EXAMPLES = ['x^2 - 2x', 'sin(x)', 'e^x', 'ln(x)', 'tanh(x)'];
const FUNCTIONS = new Set([
  'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'asin', 'acos', 'atan',
  'sinh', 'cosh', 'tanh', 'sech', 'csch', 'coth', 'exp', 'log', 'sqrt', 'cbrt', 'abs',
]);
const CONSTANTS = new Set(['e', 'pi']);

// Plot geometry in SVG units.
const W = 560, H = 300;
const M = { l: 46, r: 12, t: 12, b: 30 };
const PW = W - M.l - M.r, PH = H - M.t - M.b;
const TANGENT_HALF = 46;

const rulesKeepingConstants = math.simplify.rules.filter(r => !(typeof r === 'function' && r.name === 'simplifyConstant'));

/* ---------- formatting ---------- */
function fmt(v, digits = 2) {
  if (!Number.isFinite(v)) return 'undefined';
  if (Math.abs(v) >= 1e4 || (Math.abs(v) > 0 && Math.abs(v) < 1e-3)) {
    const [m, e] = v.toExponential(2).split('e');
    return `${m.replace('-', '−')}×10^${+e}`;
  }
  const r = parseFloat(v.toFixed(digits));
  return String(Object.is(r, -0) ? 0 : r).replace('-', '−');
}
const texNum = v => {
  const s = fmt(v);
  return s.includes('×10^') ? s.replace('×10^', '\\times 10^{') + '}' : s;
};

// mathjs writes implicit products as "~" (a space in TeX). Drop it, except between two numbers.
const tidyTex = tex => tex.replace(/~(?=\s*[\d.])/g, '\\cdot ').replace(/~/g, '');

function renderTex(el, tex) {
  if (window.katex) {
    try {
      window.katex.render(tex, el, { throwOnError: false, displayMode: false });
      return;
    } catch { /* fall through */ }
  }
  el.textContent = tex;
}

/* ---------- symbolic work ---------- */
function normalise(input) {
  return input
    .replace(/\*\*/g, '^')
    .replace(/[×·]/g, '*')
    .replace(/÷/g, '/')
    .replace(/[−–]/g, '-')
    .replace(/π/g, 'pi')
    .replace(/²/g, '^2')
    .replace(/³/g, '^3')
    .replace(/\bln\s*\(/g, 'log(');
}

function checkSymbols(node) {
  let problem = null;
  node.traverse((n, path, parent) => {
    if (problem) return;
    if (n.isFunctionNode) {
      const name = n.fn.name;
      if (!FUNCTIONS.has(name)) problem = `“${name}” isn’t a function this lesson knows. Try sin, cos, tan, exp, ln, sqrt or abs.`;
      else if (n.args.length !== 1) problem = `${name === 'log' ? 'ln' : name}( ) takes one input, like ${name === 'log' ? 'ln' : name}(x).`;
      return;
    }
    if (n.isSymbolNode && !(parent && parent.isFunctionNode && parent.fn === n)) {
      if (n.name === 'x' || CONSTANTS.has(n.name)) return;
      const fn = [...FUNCTIONS].find(f => n.name.startsWith(f) && n.name.length > f.length);
      problem = fn
        ? `Put brackets around the input: ${fn === 'log' ? 'ln' : fn}(${n.name.slice(fn.length)}).`
        : `Use x as the variable (found “${n.name}”). Write products with a space or *, like 2x or x*x.`;
    }
    if (!n.isSymbolNode && !n.isFunctionNode && !n.isConstantNode && !n.isOperatorNode && !n.isParenthesisNode) {
      problem = 'Use a single expression in x, like x^3 - 2x.';
    }
  });
  return problem;
}

const isConst = n => n.isConstantNode || (n.isOperatorNode && n.fn === 'unaryMinus' && n.args[0].isConstantNode);

// Tidy a derivative tree for display: 3·x² → 3x², (1/2)/√x → 1/(2√x).
function forDisplay(node) {
  return node.transform(n => {
    if (n.isOperatorNode && n.fn === 'divide' && n.args[0].isOperatorNode && n.args[0].fn === 'divide' && n.args[0].args.every(isConst)) {
      const [a, b] = n.args[0].args;
      const product = new math.OperatorNode('*', 'multiply', [b, n.args[1]], true);
      return new math.OperatorNode('/', 'divide', [a, product]);
    }
    if (n.isOperatorNode && n.fn === 'multiply' && n.args.length === 2 && isConst(n.args[0]) && !isConst(n.args[1])) {
      return new math.OperatorNode('*', 'multiply', n.args, true);
    }
    return n;
  });
}

// Long decimals (ln 2 ≈ 0.6931…) appear when mathjs folds constants. Keep them symbolic instead.
function hasLongDecimal(node, inputText) {
  let found = false;
  node.traverse(n => {
    if (n.isConstantNode && typeof n.value === 'number' && !Number.isInteger(n.value)) {
      const s = String(n.value);
      if (s.length > 6 && !inputText.includes(s)) found = true;
    }
  });
  return found;
}

function analyse(input) {
  const text = normalise(input.trim());
  if (!text) return { error: 'Type a function of x, like x^3.' };
  let node;
  try {
    node = math.parse(text);
  } catch {
    return { error: 'That isn’t a complete expression yet. Check for a missing bracket or operator.' };
  }
  const problem = checkSymbols(node);
  if (problem) return { error: problem };

  let d;
  try {
    d = math.derivative(node, 'x');
  } catch {
    return { error: 'This lesson can’t differentiate that expression. Try a simpler one.' };
  }
  let shown = d;
  if (hasLongDecimal(d, text)) {
    try {
      shown = math.simplify(math.derivative(node, 'x', { simplify: false }), rulesKeepingConstants);
    } catch { /* keep numeric form */ }
  }

  const f = node.compile();
  const df = d.compile();
  const safe = fn => x => {
    try {
      const v = fn.evaluate({ x });
      return typeof v === 'number' ? v : NaN;
    } catch {
      return NaN;
    }
  };
  return {
    f: safe(f),
    df: safe(df),
    texF: tidyTex(node.toTex({ implicit: 'hide', parenthesis: 'auto' })),
    texDf: tidyTex(forDisplay(shown).toTex({ implicit: 'hide', parenthesis: 'auto' })),
  };
}

/* ---------- plotting ---------- */
function niceStep(span, target = 6) {
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

// Range of one function over the plot. Smooth curves use their full range; curves
// with an asymptote (1/x, ln x, tan x) are trimmed to their middle 90%.
function robustRange(fn) {
  const vals = [];
  for (let i = 0; i <= SAMPLES; i++) {
    const v = fn(X_RANGE[0] + (i / SAMPLES) * (X_RANGE[1] - X_RANGE[0]));
    if (Number.isFinite(v)) vals.push(v);
  }
  if (!vals.length) return null;
  vals.sort((a, b) => a - b);
  const lo = quantile(vals, 0.05), hi = quantile(vals, 0.95), span = hi - lo;
  const full = vals[vals.length - 1] - vals[0];
  if (full <= span * 3 + 1e-9) return { lo: vals[0], hi: vals[vals.length - 1], heavy: false };
  return { lo: lo - span * 0.25, hi: hi + span * 0.25, heavy: true };
}

// The function sets the view. A derivative that blows up only widens it so far.
function yRange(f, df) {
  const fr = robustRange(f) ?? { lo: -1, hi: 1 };
  let lo = fr.lo, hi = fr.hi;
  const dr = robustRange(df);
  if (dr) {
    const reach = Math.max(fr.hi - fr.lo, 2);
    const dlo = dr.heavy ? Math.max(dr.lo, fr.lo - reach) : dr.lo;
    const dhi = dr.heavy ? Math.min(dr.hi, fr.hi + reach) : dr.hi;
    lo = Math.min(lo, dlo);
    hi = Math.max(hi, dhi);
  }
  lo = Math.min(lo, 0);
  hi = Math.max(hi, 0);
  if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function curvePath(fn, xs, ys) {
  const [y0, y1] = ys;
  const span = y1 - y0;
  let d = '';
  let pen = false;
  let prev = null;
  for (let i = 0; i <= SAMPLES; i++) {
    const x = X_RANGE[0] + (i / SAMPLES) * (X_RANGE[1] - X_RANGE[0]);
    const y = fn(x);
    // Lift the pen on undefined values and on jumps across an asymptote.
    if (!Number.isFinite(y)) {
      pen = false;
      prev = null;
      continue;
    }
    if (prev !== null && Math.abs(y - prev) > span * 3) pen = false;
    const Y = Math.min(H * 6, Math.max(-H * 5, M.t + PH - ((y - y0) / span) * PH));
    d += `${pen ? 'L' : 'M'}${xs(x).toFixed(1)},${Y.toFixed(1)}`;
    pen = true;
    prev = y;
  }
  return d;
}

/* ---------- lesson ---------- */
export function mountDerivativeLesson(host) {
  host.innerHTML = `
    <p>The derivative <var>dy/dx</var> gives the slope of <var>y</var> at every <var>x</var>. Edit the function and the derivative updates. Click or drag along the curve to inspect a point.</p>
    <div class="deriv-editor">
      <label class="deriv-input"><span class="deriv-lhs">y =</span>
        <input type="text" spellcheck="false" autocomplete="off" autocapitalize="off" value="${DEFAULT_FN}" aria-describedby="derivError derivHint">
      </label>
      <p id="derivHint" class="deriv-hint">Try ${EXAMPLES.map(e => `<button type="button" class="deriv-example mono">${e}</button>`).join(' ')}</p>
      <p id="derivError" class="deriv-error" role="alert" hidden></p>
      <div class="deriv-math" aria-live="polite">
        <div class="deriv-eq deriv-eq-f"></div>
        <div class="deriv-eq deriv-eq-df"></div>
      </div>
    </div>
    <figure class="deriv-plot">
      <div class="deriv-legend" aria-hidden="true">
        <span class="lg lg-f">y</span>
        <span class="lg lg-df">dy/dx</span>
        <span class="lg lg-tan">tangent</span>
      </div>
      <svg viewBox="0 0 ${W} ${H}" tabindex="0" role="slider" aria-label="Point on the curve. Use the left and right arrow keys to move it."></svg>
    </figure>
    <div class="deriv-readout" role="status">
      <p class="deriv-values"></p>
      <p class="deriv-tangent"><span>Tangent line:</span> <span class="deriv-tangent-eq"></span></p>
      <p class="deriv-meaning"></p>
    </div>
    <p>In a network, <var>y</var> is the loss and <var>x</var> is one weight. The sign of the derivative says which way to nudge the weight, and its size says how strongly the loss responds.</p>`;

  const input = host.querySelector('.deriv-input input');
  const error = host.querySelector('.deriv-error');
  const eqF = host.querySelector('.deriv-eq-f');
  const eqDf = host.querySelector('.deriv-eq-df');
  const svg = host.querySelector('svg');
  const values = host.querySelector('.deriv-values');
  const tangent = host.querySelector('.deriv-tangent');
  const tangentEq = host.querySelector('.deriv-tangent-eq');
  const meaning = host.querySelector('.deriv-meaning');

  let model = null;
  let ys = [-1, 1];
  let px = 1;
  let staticLayer = '';

  const xs = x => M.l + ((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])) * PW;
  const yPos = y => M.t + PH - ((y - ys[0]) / (ys[1] - ys[0])) * PH;

  function buildStatic() {
    const [x0, x1] = X_RANGE;
    const xStep = niceStep(x1 - x0, 6);
    const yStep = niceStep(ys[1] - ys[0], 5);
    let grid = '', labels = '';
    for (let v = Math.ceil(x0 / xStep) * xStep; v <= x1 + 1e-9; v += xStep) {
      const X = xs(v).toFixed(1);
      grid += `<line class="deriv-grid${Math.abs(v) < 1e-9 ? ' deriv-zero' : ''}" x1="${X}" y1="${M.t}" x2="${X}" y2="${M.t + PH}"/>`;
      labels += `<text x="${X}" y="${H - 10}" text-anchor="middle">${fmt(v)}</text>`;
    }
    for (let v = Math.ceil(ys[0] / yStep) * yStep; v <= ys[1] + 1e-9; v += yStep) {
      const Y = yPos(v).toFixed(1);
      grid += `<line class="deriv-grid${Math.abs(v) < yStep * 1e-6 ? ' deriv-zero' : ''}" x1="${M.l}" y1="${Y}" x2="${M.l + PW}" y2="${Y}"/>`;
      labels += `<text x="${M.l - 6}" y="${(+Y + 4).toFixed(1)}" text-anchor="end">${fmt(v, 1)}</text>`;
    }
    staticLayer = `
      <defs><clipPath id="derivClip"><rect x="${M.l}" y="${M.t}" width="${PW}" height="${PH}"/></clipPath></defs>
      <rect class="deriv-frame" x="${M.l}" y="${M.t}" width="${PW}" height="${PH}" rx="4"/>
      ${grid}${labels}
      <text class="deriv-axis-name" x="${M.l + PW - 4}" y="${M.t + PH - 6}" text-anchor="end">x</text>
      <g clip-path="url(#derivClip)">
        <path class="deriv-df" d="${curvePath(model.df, xs, ys)}"/>
        <path class="deriv-f" d="${curvePath(model.f, xs, ys)}"/>
      </g>`;
  }

  function draw() {
    const y = model.f(px);
    const m = model.df(px);
    const X = xs(px);
    let marks = `<line class="deriv-guide" x1="${X.toFixed(1)}" y1="${M.t}" x2="${X.toFixed(1)}" y2="${M.t + PH}"/>`;
    if (Number.isFinite(y) && Number.isFinite(m)) {
      // Fixed on-screen length, so the tangent stays short however steep it is.
      const sx = PW / (X_RANGE[1] - X_RANGE[0]);
      const sy = PH / (ys[1] - ys[0]);
      const len = Math.hypot(sx, m * sy);
      const ux = (sx / len) * TANGENT_HALF, uy = (-m * sy / len) * TANGENT_HALF;
      const Y = yPos(y);
      marks += `<line class="deriv-tan" x1="${(X - ux).toFixed(1)}" y1="${(Y - uy).toFixed(1)}" x2="${(X + ux).toFixed(1)}" y2="${(Y + uy).toFixed(1)}"/>`;
    }
    if (Number.isFinite(m)) marks += `<circle class="deriv-pt-df" cx="${X.toFixed(1)}" cy="${yPos(m).toFixed(1)}" r="4.5"/>`;
    if (Number.isFinite(y)) marks += `<circle class="deriv-pt-f" cx="${X.toFixed(1)}" cy="${yPos(y).toFixed(1)}" r="5.5"/>`;
    svg.innerHTML = `${staticLayer}<g clip-path="url(#derivClip)">${marks}</g>`;

    svg.setAttribute('aria-valuemin', X_RANGE[0]);
    svg.setAttribute('aria-valuemax', X_RANGE[1]);
    svg.setAttribute('aria-valuenow', px);
    svg.setAttribute('aria-valuetext', `x = ${fmt(px)}, y = ${fmt(y)}, dy/dx = ${fmt(m)}`);

    values.innerHTML = `At <var>x</var> = <strong>${fmt(px)}</strong>: <span class="deriv-val-f"><var>y</var> = <strong>${fmt(y)}</strong></span>, <span class="deriv-val-df"><var>dy/dx</var> = <strong>${fmt(m)}</strong></span>`;
    if (Number.isFinite(y) && Number.isFinite(m)) {
      const shift = px === 0 ? 'x' : `(x ${px > 0 ? '-' : '+'} ${texNum(Math.abs(px))})`;
      const flat = Math.abs(m) < 5e-3;
      const coef = fmt(Math.abs(m)) === '1' ? '' : `${texNum(Math.abs(m))}\\,`;
      const intercept = fmt(y) === '0' && !flat ? '' : texNum(y);
      const sign = m < 0 ? '-' : intercept ? '+' : '';
      tangent.hidden = false;
      renderTex(tangentEq, `y = ${intercept} ${flat ? '' : `${sign} ${coef}${shift}`}`);
      const step = 0.1;
      meaning.textContent = Math.abs(m) < 5e-3
        ? 'The slope is zero: the curve is flat here, so a small step in x barely changes y.'
        : `The slope is ${m > 0 ? 'positive' : 'negative'}: stepping x right by ${step} ${m > 0 ? 'raises' : 'lowers'} y by about ${fmt(Math.abs(m * step), 3)}.`;
    } else {
      tangent.hidden = true;
      meaning.textContent = Number.isFinite(y)
        ? 'The derivative is undefined here, so there is no single tangent line.'
        : 'The function is undefined at this x. Pick a point where the curve is drawn.';
    }
  }

  function update() {
    const result = analyse(input.value);
    if (result.error) {
      error.textContent = result.error;
      error.hidden = false;
      input.setAttribute('aria-invalid', 'true');
      host.querySelector('.deriv-math').classList.add('stale');
      return;
    }
    error.hidden = true;
    input.removeAttribute('aria-invalid');
    host.querySelector('.deriv-math').classList.remove('stale');
    model = result;
    renderTex(eqF, `y = ${model.texF}`);
    renderTex(eqDf, `\\dfrac{dy}{dx} = ${model.texDf}`);
    ys = yRange(model.f, model.df);
    buildStatic();
    draw();
  }

  function setFromPointer(event) {
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const x = X_RANGE[0] + ((pt.x - M.l) / PW) * (X_RANGE[1] - X_RANGE[0]);
    px = Math.round(Math.min(X_RANGE[1], Math.max(X_RANGE[0], x)) * 100) / 100;
    draw();
  }

  svg.addEventListener('pointerdown', event => {
    if (!model) return;
    svg.classList.add('pointer');
    svg.setPointerCapture(event.pointerId);
    setFromPointer(event);
  });
  svg.addEventListener('pointermove', event => {
    if (model && svg.hasPointerCapture(event.pointerId)) setFromPointer(event);
  });
  svg.addEventListener('keydown', event => {
    if (!model) return;
    const step = event.shiftKey ? 0.5 : 0.05;
    const moves = { ArrowRight: step, ArrowUp: step, ArrowLeft: -step, ArrowDown: -step };
    if (event.key in moves) px += moves[event.key];
    else if (event.key === 'Home') px = X_RANGE[0];
    else if (event.key === 'End') px = X_RANGE[1];
    else return;
    event.preventDefault();
    svg.classList.remove('pointer');
    px = Math.round(Math.min(X_RANGE[1], Math.max(X_RANGE[0], px)) * 100) / 100;
    draw();
  });

  svg.addEventListener('blur', () => svg.classList.remove('pointer'));
  input.addEventListener('input', update);
  host.querySelectorAll('.deriv-example').forEach(button => button.addEventListener('click', () => {
    input.value = button.textContent;
    update();
    input.focus();
  }));

  update();
}

// Exposed for quick checks from the console or tests.
export { analyse };
