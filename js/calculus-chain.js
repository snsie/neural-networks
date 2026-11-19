/* Chain rule lesson: an editable inner step u(w) and outer step L(u).
   - A chain diagram shows the forward pass (values, left to right) and a
     step-by-step backward pass (slopes of L, right to left), as in backpropagation.
   - A nudge Δw travels through the chain: Δw → Δu ≈ (du/dw)Δw → ΔL ≈ (dL/du)Δu,
     drawn on all three linked plots, so students see why the slopes multiply. */
import { analyse, differentiate, substitute, toTex, editorHTML, wireEditor, renderTex, fmt } from './calculus-math.js';
import { createFunctionPlot, sample } from './calculus-plot.js';

const W_RANGE = [-1, 3];
const STAGE_MS = 750;

// Bracket a factor only when it is a sum, difference or negation.
const unwrap = node => (node.isParenthesisNode ? unwrap(node.content) : node);
const factor = node => {
  const n = unwrap(node);
  const tex = toTex(n);
  return n.isOperatorNode && ['add', 'subtract', 'unaryMinus'].includes(n.fn) ? `\\left(${tex}\\right)` : tex;
};
const pn = v => (v < 0 ? `(${fmt(v)})` : fmt(v));
const tag = (kind, text) => `<span class="calc-tag calc-tag-${kind}">${text}</span>`;

function graphHTML() {
  return `
    <figure class="calc-graph" data-stage="0">
      <figcaption><strong>The chain, as a network would compute it</strong></figcaption>
      <div class="calc-graph-grid">
        <p class="calc-graph-row-label">Forward pass: compute each value, left to right</p>
        <div class="calc-node calc-node-fwd"><span class="calc-node-name">w</span><span class="calc-node-val" data-f="w"></span></div>
        <div class="calc-edge calc-edge-fwd"><span class="calc-edge-fn">u(w)</span><span class="calc-edge-arrow" aria-hidden="true"></span><span class="calc-edge-slope">slope <span data-f="a"></span></span></div>
        <div class="calc-node calc-node-fwd"><span class="calc-node-name">u</span><span class="calc-node-val" data-f="u"></span></div>
        <div class="calc-edge calc-edge-fwd"><span class="calc-edge-fn">L(u)</span><span class="calc-edge-arrow" aria-hidden="true"></span><span class="calc-edge-slope">slope <span data-f="c"></span></span></div>
        <div class="calc-node calc-node-fwd"><span class="calc-node-name">L</span><span class="calc-node-val" data-f="L"></span></div>

        <p class="calc-graph-row-label">Backward pass: start at L and multiply by each slope, right to left</p>
        <div class="calc-node calc-node-bwd" data-stage="3"><span class="calc-node-name">∂L/∂w</span><span class="calc-node-calc" data-b="w-calc"></span><span class="calc-node-val" data-b="w"></span></div>
        <div class="calc-edge calc-edge-bwd" data-stage="3"><span class="calc-edge-arrow" aria-hidden="true"></span><span class="calc-edge-slope">× <span data-b="a"></span></span></div>
        <div class="calc-node calc-node-bwd" data-stage="2"><span class="calc-node-name">∂L/∂u</span><span class="calc-node-calc" data-b="u-calc"></span><span class="calc-node-val" data-b="u"></span></div>
        <div class="calc-edge calc-edge-bwd" data-stage="2"><span class="calc-edge-arrow" aria-hidden="true"></span><span class="calc-edge-slope">× <span data-b="c"></span></span></div>
        <div class="calc-node calc-node-bwd" data-stage="1"><span class="calc-node-name">∂L/∂L</span><span class="calc-node-calc">start</span><span class="calc-node-val">1</span></div>
      </div>
      <div class="calc-graph-foot">
        <button type="button" class="nav-btn primary calc-run">Run backward pass</button>
        <p class="calc-graph-note" role="status">What is ∂L/∂w? Guess its sign, then run the backward pass.</p>
      </div>
    </figure>`;
}

export function mount(host) {
  host.innerHTML = `
    <p>When one quantity feeds another, the <strong>chain rule</strong> multiplies their slopes. Here the weight <var>w</var> sets the prediction <var>u</var> = 2<var>w</var> (a neuron with input 2), and the prediction sets the loss <var>L</var> = (<var>u</var> − 3)² (target 3). Edit either step.</p>
    <div class="calc-chain-editors">
      ${editorHTML({ lhs: 'u(w)', value: '2w', label: 'Inner step: u as a function of w', examples: ['w^2', 'sin(w)', 'tanh(w)'] })}
      ${editorHTML({ lhs: 'L(u)', value: '(u - 3)^2', label: 'Outer step: L as a function of u', examples: ['u^2', 'ln(u)', 'e^u'] })}
    </div>
    <div class="calc-math" aria-live="polite">
      <div class="calc-eq-df"></div>
      <div class="calc-eq-df"></div>
      <div class="calc-eq-chain"></div>
    </div>
    ${graphHTML()}
    <section class="calc-nudge-panel" aria-label="Follow a nudge through the chain">
      <h4>Why the slopes multiply: follow a nudge</h4>
      <label class="calc-nudge-control">
        <span>Nudge ${tag('w', 'Δw')} = <output>0.25</output></span>
        <input type="range" min="0.05" max="1" step="0.05" value="0.25" aria-label="Size of the nudge Δw">
      </label>
      <div class="calc-nudge-grid">
        <div class="calc-nudge-step"><div class="calc-mini-u"></div><p></p></div>
        <div class="calc-nudge-step"><div class="calc-mini-l"></div><p></p></div>
        <div class="calc-nudge-step"><div class="calc-mini-c"></div><p></p></div>
      </div>
      <p class="calc-nudge-sum"></p>
    </section>
    <p>The full curves: click or drag in graph 1 or graph 3 to choose <var>w</var>. Graph 2 follows the prediction <var>u</var> that <var>w</var> produces.</p>
    <div class="calc-chain-grid">
      <div class="calc-chain-u"></div>
      <div class="calc-chain-l"></div>
      <div class="calc-chain-c"></div>
    </div>
    <p>Backpropagation is the chain rule applied step by step. It starts at the loss and multiplies slopes backward through each layer until it reaches every weight.</p>`;

  const mathBox = host.querySelector('.calc-math');
  const graph = host.querySelector('.calc-graph');
  const runButton = graph.querySelector('.calc-run');
  const note = graph.querySelector('.calc-graph-note');
  const slider = host.querySelector('.calc-nudge-control input');
  const sliderOut = host.querySelector('.calc-nudge-control output');
  const steps = host.querySelectorAll('.calc-nudge-step p');
  const sum = host.querySelector('.calc-nudge-sum');
  const [innerBlock, outerBlock] = host.querySelectorAll('.calc-editor');

  let inner = null, outer = null;
  let innerOK = false, outerOK = false;
  let w = 1;
  let stage = 0;      // 0: backward pass not run; 1–3: how far it has reached
  let timers = [];

  const plotU = createFunctionPlot(host.querySelector('.calc-chain-u'), {
    xName: 'w', fName: 'u', dName: 'du/dw', width: 420, height: 230, compact: true,
    caption: '<strong>1. u vs w</strong><span>Δw moves u by Δu ≈ (du/dw) × Δw</span>',
    onSelect: v => { w = v; refresh(); },
  });
  const plotL = createFunctionPlot(host.querySelector('.calc-chain-l'), {
    xName: 'u', fName: 'L', dName: 'dL/du', width: 420, height: 230, compact: true, interactive: false,
    caption: '<strong>2. L vs u</strong><span>That Δu moves L by ΔL ≈ (dL/du) × Δu</span>',
  });
  const plotC = createFunctionPlot(host.querySelector('.calc-chain-c'), {
    xName: 'w', fName: 'L', dName: 'dL/dw', width: 640, height: 240,
    caption: '<strong>3. L vs w: both steps at once</strong><span>The same ΔL, now from Δw directly: ΔL ≈ (dL/dw) × Δw</span>',
    onSelect: v => { w = v; refresh(); },
  });

  // Zoomed views for the nudge: each follows the point and frames its own triangle.
  const mini = (sel, xName, fName, caption) => createFunctionPlot(host.querySelector(sel), {
    xName, fName, width: 320, height: 230, compact: true, interactive: false,
    showDerivative: false, legend: false, caption,
  });
  const miniU = mini('.calc-mini-u', 'w', 'u', '<strong>1. Through u(w)</strong>');
  const miniL = mini('.calc-mini-l', 'u', 'L', '<strong>2. Through L(u)</strong>');
  const miniC = mini('.calc-mini-c', 'w', 'L', '<strong>3. Both at once</strong>');

  // Frame a nudge from x0 by dx (rise dy) with a little of the curve on each side.
  function frame(plot, fn, dfn, x0, dx, dy, nudge) {
    const half = Math.max(Math.abs(dx) * 0.95, 0.08);
    const mid = x0 + dx / 2;
    const xRange = [mid - half, mid + half];
    const y0 = fn(x0);
    const ys = sample(fn, xRange, 60).concat([y0, y0 + dy]).filter(Number.isFinite);
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.22;
    plot.setFunctions(fn, dfn, { xRange, yRange: [lo - pad, hi + pad] });
    plot.setX(x0);
    plot.setNudge(nudge);
  }

  const u = v => inner.fn(v);
  const dudw = v => inner.d.w.fn(v);
  const L = v => outer.fn(u(v));
  const dLdu = uv => outer.d.u.fn(uv);
  const dLdw = v => dLdu(u(v)) * dudw(v);

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

  // Write the current numbers into the diagram, the nudge steps and the plots.
  function fill() {
    if (!(innerOK && outerOK)) return;
    const uv = u(w), Lv = L(w), a = dudw(w), c = dLdu(uv), m = a * c;
    const set = (sel, text) => graph.querySelectorAll(sel).forEach(el => { el.textContent = text; });
    set('[data-f="w"]', fmt(w));
    set('[data-f="u"]', fmt(uv));
    set('[data-f="L"]', fmt(Lv));
    set('[data-f="a"]', fmt(a));
    set('[data-f="c"]', fmt(c));
    set('[data-b="a"]', pn(a));
    set('[data-b="c"]', pn(c));
    set('[data-b="u-calc"]', stage >= 2 ? `1 × ${pn(c)}` : '');
    set('[data-b="u"]', stage >= 2 ? fmt(c) : '?');
    set('[data-b="w-calc"]', stage >= 3 ? `${pn(c)} × ${pn(a)}` : '');
    set('[data-b="w"]', stage >= 3 ? fmt(m) : '?');

    runButton.textContent = stage === 3 ? 'Replay backward pass' : 'Run backward pass';
    if (stage === 0) note.textContent = 'What is ∂L/∂w? Guess its sign, then run the backward pass.';
    else if (stage === 1) note.textContent = 'Start at L: nudging L changes L by the same amount, so ∂L/∂L = 1.';
    else if (stage === 2) note.textContent = `Step back through L(u): multiply by its slope, ${fmt(c)}.`;
    else note.textContent = `Step back through u(w): multiply by its slope, ${fmt(a)}. So ∂L/∂w = ${fmt(m)}, the slope of graph 3.`;

    // The nudge, step by step.
    const dw = slider.valueAsNumber;
    sliderOut.value = fmt(dw);
    const du = a * dw, dL = c * du, actual = L(w + dw) - Lv;
    const defined = [uv, Lv, a, c].every(Number.isFinite);
    if (!defined) {
      steps.forEach(p => { p.innerHTML = ''; });
      sum.textContent = 'Something in the chain is undefined at this w. Pick a point where all three graphs are drawn.';
      return;
    }
    steps[0].innerHTML = `${tag('w', 'Δw')} = ${fmt(dw)} moves <var>u</var> by<br>${tag('u', 'Δu')} ≈ <var>du/dw</var> × Δw = ${pn(a)} × ${fmt(dw)} = <strong>${fmt(du, 3)}</strong>`;
    steps[1].innerHTML = `${tag('u', 'Δu')} = ${fmt(du, 3)} moves <var>L</var> by<br>${tag('L', 'ΔL')} ≈ <var>dL/du</var> × Δu = ${pn(c)} × ${pn(du)} = <strong>${fmt(dL, 3)}</strong>`;
    steps[2].innerHTML = `So ${tag('w', 'Δw')} moves <var>L</var> by<br>${tag('L', 'ΔL')} ≈ (<var>dL/du</var> × <var>du/dw</var>) × Δw = ${pn(m)} × ${fmt(dw)} = <strong>${fmt(m * dw, 3)}</strong>`;
    sum.innerHTML = `The slopes multiply because each step scales the nudge it receives: first by <var>du/dw</var> = ${fmt(a)}, then by <var>dL/du</var> = ${fmt(c)}. ` +
      `Each estimate follows the tangent line (the dashed line). The hollow dot is where the curve really goes` +
      (Number.isFinite(actual)
        ? `: the real change in L is ${fmt(actual, 3)}.${Math.abs(actual - dL) > 5e-4 ? ' Make Δw smaller and the estimate gets closer.' : ''}`
        : '.');

    frame(miniU, u, dudw, w, dw, du, { dx: dw, dy: du, xKind: 'w', yKind: 'u', xLabel: 'Δw', yLabel: 'Δu' });
    frame(miniL, outer.fn, dLdu, uv, du, dL, { dx: du, dy: dL, xKind: 'u', yKind: 'L', xLabel: 'Δu', yLabel: 'ΔL' });
    frame(miniC, L, dLdw, w, dw, dL, { dx: dw, dy: dL, xKind: 'w', yKind: 'L', xLabel: 'Δw', yLabel: 'ΔL' });
  }
  slider.addEventListener('input', fill);

  /* ---------- symbolic work and plots ---------- */
  function rebuild() {
    mathBox.classList.toggle('stale', !(innerOK && outerOK));
    if (!(innerOK && outerOK)) return;
    renderTex(mathBox.children[0], `\\dfrac{du}{dw} = ${inner.d.w.tex}`);
    renderTex(mathBox.children[1], `\\dfrac{dL}{du} = ${outer.d.u.tex}`);

    // dL/du with u replaced by u(w), times du/dw, then the simplified result.
    const outerAtInner = substitute(outer.d.u.shown, 'u', inner.node);
    const productTex = `${factor(outerAtInner)}\\cdot ${factor(inner.d.w.shown)}`;
    let chain = `\\dfrac{dL}{dw} = \\dfrac{dL}{du}\\cdot\\dfrac{du}{dw} = ${productTex}`;
    try {
      const simplified = toTex(differentiate(substitute(outer.node, 'u', inner.node), 'w').shown);
      if (simplified.replace(/\s/g, '') !== productTex.replace(/\s/g, '')) chain += ` = ${simplified}`;
    } catch { /* show the product unsimplified */ }
    renderTex(mathBox.children[2], chain);

    // Graph 2's horizontal axis covers every prediction that w can produce.
    const us = sample(u, W_RANGE, 240).filter(Number.isFinite);
    let lo = Math.min(...us), hi = Math.max(...us);
    if (!us.length) { lo = -1; hi = 1; }
    if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.1;
    plotU.setFunctions(u, dudw, { xRange: W_RANGE });
    plotL.setFunctions(outer.fn, dLdu, { xRange: [lo - pad, hi + pad] });
    plotC.setFunctions(L, dLdw, { xRange: W_RANGE });
    refresh();
  }

  function refresh() {
    if (!(innerOK && outerOK)) return;
    plotU.setX(w);
    plotL.setX(u(w));
    plotC.setX(w);
    fill();
  }

  const innerEditor = wireEditor(innerBlock, text => {
    const result = analyse(text, ['w']);
    innerOK = !result.error;
    if (innerOK) inner = result;
    rebuild();
    return result.error;
  });
  const outerEditor = wireEditor(outerBlock, text => {
    const result = analyse(text, ['u']);
    outerOK = !result.error;
    if (outerOK) outer = result;
    rebuild();
    return result.error;
  });
  innerEditor.run();
  outerEditor.run();
  setStage(0);
}
