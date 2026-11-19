/* Derivative lesson: an editable function y(x), its symbolic derivative, and an
   overlaid plot of y and dy/dx with a tangent line at a selected point. */
import { analyse, editorHTML, lessonHelpHTML, wireEditor, renderTex, fmt, texNum } from './calculus-math.js';
import { createFunctionPlot } from './calculus-plot.js';

const X_RANGE = [-3, 3];

// "y = 2 + 3(x − 1)" with tidy signs, ones and zeros. Shared with the other lessons.
export function tangentTex(lhs, name, a, y, m) {
  const flat = Math.abs(m) < 5e-3;
  const shift = Math.abs(a) < 5e-3 ? name : `(${name} ${a > 0 ? '-' : '+'} ${texNum(Math.abs(a))})`;
  const coef = fmt(Math.abs(m)) === '1' ? '' : `${texNum(Math.abs(m))}\\,`;
  const intercept = fmt(y) === '0' && !flat ? '' : texNum(y);
  const sign = m < 0 ? '-' : intercept ? '+' : '';
  return `${lhs} = ${intercept} ${flat ? '' : `${sign} ${coef}${shift}`}`;
}

export function slopeSentence(m, x, y, step = 0.1) {
  if (Math.abs(m) < 5e-3) return `The slope is zero: the curve is flat here, so a small step in ${x} barely changes ${y}.`;
  return `The slope is ${m > 0 ? 'positive' : 'negative'}: stepping ${x} right by ${step} ${m > 0 ? 'raises' : 'lowers'} ${y} by about ${fmt(Math.abs(m * step), 3)}.`;
}

export function mount(host) {
  host.innerHTML = `
    ${lessonHelpHTML('Derivative', `
      <p>The derivative <var>dy/dx</var> gives the slope of <var>y</var> at every <var>x</var>. Edit the function and the derivative updates. Click or drag along the curve to inspect a point.</p>
      <p>In a network, <var>y</var> is the loss and <var>x</var> is one weight. The sign of the derivative says which way to nudge the weight, and its size says how strongly the loss responds.</p>
    `)}
    ${editorHTML({ lhs: 'y', value: 'x^3', label: 'Function of x', examples: ['x^2 - 2x', 'sin(x)', 'e^x', 'ln(x)', 'tanh(x)'] })}
    <div class="calc-math" aria-live="polite">
      <div class="calc-eq-f"></div>
      <div class="calc-eq-df"></div>
    </div>
    <div class="calc-plot-host"></div>
    <div class="calc-readout" role="status">
      <p class="calc-values"></p>
      <p class="calc-tangent"><span>Tangent line:</span> <span class="calc-tex"></span></p>
      <p class="calc-meaning"></p>
    </div>`;

  const mathBox = host.querySelector('.calc-math');
  const values = host.querySelector('.calc-values');
  const tangent = host.querySelector('.calc-tangent');
  const meaning = host.querySelector('.calc-meaning');
  let model = null;

  function readout(x) {
    const y = model.fn(x), m = model.d.x.fn(x);
    values.innerHTML = `At <var>x</var> = <strong>${fmt(x)}</strong>: <span class="calc-val-f"><var>y</var> = <strong>${fmt(y)}</strong></span>, <span class="calc-val-df"><var>dy/dx</var> = <strong>${fmt(m)}</strong></span>`;
    const defined = Number.isFinite(y) && Number.isFinite(m);
    tangent.hidden = !defined;
    if (defined) {
      renderTex(tangent.querySelector('.calc-tex'), tangentTex('y', 'x', x, y, m));
      meaning.textContent = slopeSentence(m, 'x', 'y');
    } else {
      meaning.textContent = Number.isFinite(y)
        ? 'The derivative is undefined here, so there is no single tangent line.'
        : 'The function is undefined at this x. Pick a point where the curve is drawn.';
    }
  }

  const plot = createFunctionPlot(host.querySelector('.calc-plot-host'), {
    xName: 'x', fName: 'y', dName: 'dy/dx', caption: null, onSelect: readout,
  });
  plot.setX(1);

  const editor = wireEditor(host.querySelector('.calc-editor'), text => {
    const result = analyse(text, ['x']);
    if (result.error) return result.error;
    model = result;
    renderTex(mathBox.children[0], `y = ${model.tex}`);
    renderTex(mathBox.children[1], `\\dfrac{dy}{dx} = ${model.d.x.tex}`);
    plot.setFunctions(model.fn, model.d.x.fn, { xRange: X_RANGE });
    readout(plot.x);
    return null;
  }, mathBox);
  editor.run();
}
