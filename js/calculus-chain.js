/* Chain rule lesson: an editable inner step u(w) and outer step L(u). Shows each
   derivative symbolically, their product dL/dw, and three linked plots that share
   one selected w: u vs w, L vs u, and L vs w. */
import { analyse, differentiate, substitute, toTex, editorHTML, wireEditor, renderTex, fmt, texNum } from './calculus-math.js';
import { createFunctionPlot, sample } from './calculus-plot.js';

const W_RANGE = [-1, 3];

// Bracket a factor only when it is a sum, difference or negation.
const unwrap = node => (node.isParenthesisNode ? unwrap(node.content) : node);
const factor = node => {
  const n = unwrap(node);
  const tex = toTex(n);
  return n.isOperatorNode && ['add', 'subtract', 'unaryMinus'].includes(n.fn) ? `\\left(${tex}\\right)` : tex;
};
const pnum = v => (v < 0 ? `(${texNum(v)})` : texNum(v));
const verb = (m, up = 'raises', down = 'lowers') => (m > 0 ? up : down);

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
    <p>Click or drag in the first or last graph to choose <var>w</var>. The middle graph follows the prediction <var>u</var> that <var>w</var> produces.</p>
    <div class="calc-chain-grid">
      <div class="calc-chain-u"></div>
      <div class="calc-chain-l"></div>
      <div class="calc-chain-c"></div>
    </div>
    <div class="calc-readout" role="status">
      <p class="calc-values"></p>
      <p class="calc-product"><span class="calc-tex"></span></p>
      <p class="calc-meaning"></p>
    </div>
    <p>Backpropagation is the chain rule applied step by step. It starts at the loss and multiplies slopes backward through each layer until it reaches every weight.</p>`;

  const mathBox = host.querySelector('.calc-math');
  const values = host.querySelector('.calc-values');
  const product = host.querySelector('.calc-product');
  const meaning = host.querySelector('.calc-meaning');
  const [innerBlock, outerBlock] = host.querySelectorAll('.calc-editor');

  let inner = null, outer = null;
  let innerOK = false, outerOK = false;
  let w = 1;

  const plotU = createFunctionPlot(host.querySelector('.calc-chain-u'), {
    xName: 'w', fName: 'u', dName: 'du/dw', width: 420, height: 230, compact: true,
    caption: '<strong>Step 1: u vs w</strong><span>Slope du/dw</span>',
    onSelect: v => { w = v; refresh(); },
  });
  const plotL = createFunctionPlot(host.querySelector('.calc-chain-l'), {
    xName: 'u', fName: 'L', dName: 'dL/du', width: 420, height: 230, compact: true, interactive: false,
    caption: '<strong>Step 2: L vs u</strong><span>Slope dL/du</span>',
  });
  const plotC = createFunctionPlot(host.querySelector('.calc-chain-c'), {
    xName: 'w', fName: 'L', dName: 'dL/dw', width: 640, height: 240,
    caption: '<strong>Both steps: L vs w</strong><span>Slope dL/dw = dL/du × du/dw</span>',
    onSelect: v => { w = v; refresh(); },
  });

  const u = v => inner.fn(v);
  const dudw = v => inner.d.w.fn(v);
  const L = v => outer.fn(u(v));
  const dLdu = uv => outer.d.u.fn(uv);
  const dLdw = v => dLdu(u(v)) * dudw(v);

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

    // Step 2's horizontal axis covers every prediction that w can produce.
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
    const uv = u(w), Lv = L(w), a = dudw(w), c = dLdu(uv), m = a * c;
    plotU.setX(w);
    plotL.setX(uv);
    plotC.setX(w);

    values.innerHTML = `At <var>w</var> = <strong>${fmt(w)}</strong>: <var>u</var> = <strong>${fmt(uv)}</strong>, <span class="calc-val-f"><var>L</var> = <strong>${fmt(Lv)}</strong></span>`;
    const defined = [uv, Lv, a, c].every(Number.isFinite);
    product.hidden = !defined;
    if (!defined) {
      meaning.textContent = 'Something in the chain is undefined at this w. Pick a point where all three curves are drawn.';
      return;
    }
    renderTex(product.querySelector('.calc-tex'),
      `\\dfrac{dL}{dw} = \\dfrac{dL}{du}\\cdot\\dfrac{du}{dw} = ${pnum(c)} \\times ${pnum(a)} = ${texNum(m)}`);
    if (Math.abs(a) < 5e-3 || Math.abs(c) < 5e-3) {
      meaning.textContent = `${Math.abs(a) < 5e-3 ? 'u is flat in w here' : 'L is flat in u here'}, so the product is zero: a small change in w barely moves L.`;
    } else {
      meaning.textContent = `Raising w a little ${verb(a)} u, and raising u ${verb(c)} L, so raising w ${verb(m)} L. ` +
        `Each step scales the change: a step of 0.1 in w moves u by about ${fmt(Math.abs(a) * 0.1, 3)} and L by about ${fmt(Math.abs(m) * 0.1, 3)}.`;
    }
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
}
