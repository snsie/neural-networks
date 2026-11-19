import { analyse, differentiate, substitute, toTex, editorHTML, lessonHelpHTML, wireEditor, renderTex, fmt } from './calculus-math.js';
import { createFunctionPlot, sample } from './calculus-plot.js';

const X_RANGE = [-1, 3];
const STAGE_MS = 750;

// Bracket a factor only when it is a sum, difference or negation.
const unwrap = node => (node.isParenthesisNode ? unwrap(node.content) : node);
const factor = node => {
  const n = unwrap(node);
  const tex = toTex(n);
  return n.isOperatorNode && ['add', 'subtract', 'unaryMinus'].includes(n.fn) ? `\\left(${tex}\\right)` : tex;
};
const pn = v => (v < 0 ? `(${fmt(v)})` : fmt(v));

function graphHTML() {
  return `
    <figure class="calc-graph" data-stage="0">
      <figcaption><strong>Composition diagram</strong></figcaption>
      <div class="calc-graph-grid">
        <p class="calc-graph-row-label">Function values</p>
        <div class="calc-node calc-node-fwd"><span class="calc-node-name">x</span><span class="calc-node-val" data-f="x"></span></div>
        <div class="calc-edge calc-edge-fwd"><span class="calc-edge-fn">y(x)</span><span class="calc-edge-arrow" aria-hidden="true"></span><span class="calc-edge-slope">slope <span data-f="a"></span></span></div>
        <div class="calc-node calc-node-fwd"><span class="calc-node-name">y</span><span class="calc-node-val" data-f="y"></span></div>
        <div class="calc-edge calc-edge-fwd"><span class="calc-edge-fn">z(y)</span><span class="calc-edge-arrow" aria-hidden="true"></span><span class="calc-edge-slope">slope <span data-f="c"></span></span></div>
        <div class="calc-node calc-node-fwd"><span class="calc-node-name">z</span><span class="calc-node-val" data-f="z"></span></div>

        <p class="calc-graph-row-label">Derivative steps</p>
        <div class="calc-node calc-node-bwd" data-stage="3"><span class="calc-node-name">dz/dx</span><span class="calc-node-calc" data-b="x-calc"></span><span class="calc-node-val" data-b="x"></span></div>
        <div class="calc-edge calc-edge-bwd" data-stage="3"><span class="calc-edge-arrow" aria-hidden="true"></span><span class="calc-edge-slope">× <span data-b="a"></span></span></div>
        <div class="calc-node calc-node-bwd" data-stage="2"><span class="calc-node-name">dz/dy</span><span class="calc-node-calc" data-b="y-calc"></span><span class="calc-node-val" data-b="y"></span></div>
        <div class="calc-edge calc-edge-bwd" data-stage="2"><span class="calc-edge-arrow" aria-hidden="true"></span><span class="calc-edge-slope">× <span data-b="c"></span></span></div>
        <div class="calc-node calc-node-bwd" data-stage="1"><span class="calc-node-name">dz/dz</span><span class="calc-node-calc">start</span><span class="calc-node-val">1</span></div>
      </div>
      <div class="calc-graph-foot">
        <button type="button" class="nav-btn primary calc-run">Trace the derivative</button>
        <p class="calc-graph-note" role="status"></p>
      </div>
    </figure>`;
}

export function mount(host) {
  host.innerHTML = `
    ${lessonHelpHTML('Chain rule', `
      <p>The value of <var>x</var> determines <var>y</var>, which determines <var>z</var>. Substituting the first function into the second gives the full composition. The chain rule gives its derivative by multiplying the slope of each step at the corresponding input.</p>
      <p>Select <strong>Trace the derivative</strong> to start at <var>dz/dz</var> = 1 and multiply by each slope as the trace moves left. The final value is the slope of the composed curve.</p>
      <p>Select a point in the <var>y</var> vs. <var>x</var> or <var>z</var> vs. <var>x</var> plot to change <var>x</var>.</p>
    `)}
    <div class="calc-chain-editors">
      ${editorHTML({ lhs: 'y(x)', value: '2x + 1', label: 'Inner step: y as a function of x', examples: ['x^2', 'sin(x)', 'tanh(x)'] })}
      ${editorHTML({ lhs: 'z(y)', value: 'y^2', label: 'Outer step: z as a function of y', examples: ['(y - 3)^2', 'ln(y)', 'e^y'] })}
    </div>
    <div class="calc-math" aria-live="polite">
      <div class="calc-eq-f" data-equation="composition"></div>
      <div class="calc-chain-slopes">
        <div class="calc-eq-df" data-equation="inner"></div>
        <div class="calc-eq-df" data-equation="outer"></div>
      </div>
      <div class="calc-eq-chain" data-equation="rule"></div>
      <div class="calc-eq-chain" data-equation="result"></div>
    </div>
    ${graphHTML()}
    <div class="calc-chain-c"></div>
    <div class="calc-chain-grid">
      <div class="calc-chain-y"></div>
      <div class="calc-chain-z"></div>
    </div>`;

  const mathBox = host.querySelector('.calc-math');
  const graph = host.querySelector('.calc-graph');
  const runButton = graph.querySelector('.calc-run');
  const note = graph.querySelector('.calc-graph-note');
  const [innerBlock, outerBlock] = host.querySelectorAll('.calc-editor');

  let inner = null, outer = null;
  let innerOK = false, outerOK = false;
  let x = 1;
  let stage = 0;      // 0: backward pass not run; 1–3: how far it has reached
  let timers = [];

  const plotY = createFunctionPlot(host.querySelector('.calc-chain-y'), {
    xName: 'x', fName: 'y', dName: 'dy/dx', width: 420, height: 230, compact: true,
    caption: '<strong>y vs. x</strong>',
    onSelect: v => { x = v; refresh(); },
  });
  const plotZ = createFunctionPlot(host.querySelector('.calc-chain-z'), {
    xName: 'y', fName: 'z', dName: 'dz/dy', width: 420, height: 230, compact: true, interactive: false,
    caption: '<strong>z vs. y</strong>',
  });
  const plotC = createFunctionPlot(host.querySelector('.calc-chain-c'), {
    xName: 'x', fName: 'z', dName: 'dz/dx', width: 640, height: 240,
    caption: '<strong>z vs. x · Composed function</strong>',
    onSelect: v => { x = v; refresh(); },
  });

  const y = v => inner.fn(v);
  const dydx = v => inner.d.x.fn(v);
  const z = v => outer.fn(y(v));
  const dzdy = yv => outer.d.y.fn(yv);
  const dzdx = v => dzdy(y(v)) * dydx(v);

  /* ---------- backward pass animation ---------- */
  function setStage(next) {
    stage = next;
    graph.dataset.stage = String(next);
    graph.querySelectorAll('[data-stage]').forEach(el => {
      if (el === graph) return;
      const s = +el.dataset.stage;
      el.classList.toggle('reached', s <= stage);
      el.classList.toggle('current', s === stage && stage < 3);
    });
    fill();
  }

  function runBackward() {
    timers.forEach(clearTimeout);
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduce) { setStage(3); return; }
    setStage(1);
    timers = [setTimeout(() => setStage(2), STAGE_MS), setTimeout(() => setStage(3), STAGE_MS * 2)];
  }
  runButton.addEventListener('click', runBackward);

  // Write the current numbers into the diagram.
  function fill() {
    if (!(innerOK && outerOK)) return;
    const yv = y(x), zv = z(x), a = dydx(x), c = dzdy(yv), m = a * c;
    const set = (sel, text) => graph.querySelectorAll(sel).forEach(el => { el.textContent = text; });
    set('[data-f="x"]', fmt(x));
    set('[data-f="y"]', fmt(yv));
    set('[data-f="z"]', fmt(zv));
    set('[data-f="a"]', fmt(a));
    set('[data-f="c"]', fmt(c));
    set('[data-b="a"]', pn(a));
    set('[data-b="c"]', pn(c));
    set('[data-b="y-calc"]', stage >= 2 ? `1 × ${pn(c)}` : '');
    set('[data-b="y"]', stage >= 2 ? fmt(c) : '?');
    set('[data-b="x-calc"]', stage >= 3 ? `${pn(c)} × ${pn(a)}` : '');
    set('[data-b="x"]', stage >= 3 ? fmt(m) : '?');

    runButton.textContent = stage === 3 ? 'Replay the trace' : 'Trace the derivative';
    note.textContent = ['', 'dz/dz = 1', `dz/dy = ${fmt(c)}`, `dz/dx = ${fmt(m)}`][stage];
  }

  /* ---------- symbolic work and plots ---------- */
  function rebuild() {
    mathBox.classList.toggle('stale', !(innerOK && outerOK));
    if (!(innerOK && outerOK)) return;
    const equation = (name, tex) => renderTex(mathBox.querySelector(`[data-equation="${name}"]`), tex);
    const composition = substitute(outer.node, 'y', inner.node);
    equation('composition', `z = z(y(x)) = ${toTex(composition)}`);
    equation('inner', `\\dfrac{dy}{dx} = ${inner.d.x.tex}`);
    equation('outer', `\\dfrac{dz}{dy} = ${outer.d.y.tex}`);
    equation('rule', '\\dfrac{dz}{dx} = \\dfrac{dz}{dy}\\cdot\\dfrac{dy}{dx}');

    const outerAtInner = substitute(outer.d.y.shown, 'y', inner.node);
    const productTex = `${factor(outerAtInner)}\\cdot ${factor(inner.d.x.shown)}`;
    let result = `\\dfrac{dz}{dx} = ${productTex}`;
    try {
      const simplified = toTex(differentiate(composition, 'x').shown);
      if (simplified.replace(/\s/g, '') !== productTex.replace(/\s/g, '')) result += ` = ${simplified}`;
    } catch { /* show the product unsimplified */ }
    equation('result', result);

    // The middle plot spans the values of y reached over the displayed x range.
    const ys = sample(y, X_RANGE, 240).filter(Number.isFinite);
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (!ys.length) { lo = -1; hi = 1; }
    if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.1;
    plotY.setFunctions(y, dydx, { xRange: X_RANGE });
    plotZ.setFunctions(outer.fn, dzdy, { xRange: [lo - pad, hi + pad] });
    plotC.setFunctions(z, dzdx, { xRange: X_RANGE });
    refresh();
  }

  function refresh() {
    if (!(innerOK && outerOK)) return;
    plotY.setX(x);
    plotZ.setX(y(x));
    plotC.setX(x);
    fill();
  }

  const innerEditor = wireEditor(innerBlock, text => {
    const result = analyse(text, ['x']);
    innerOK = !result.error;
    if (innerOK) inner = result;
    rebuild();
    return result.error;
  });
  const outerEditor = wireEditor(outerBlock, text => {
    const result = analyse(text, ['y']);
    outerOK = !result.error;
    if (outerOK) outer = result;
    rebuild();
    return result.error;
  });
  innerEditor.run();
  outerEditor.run();
  setStage(0);
}
