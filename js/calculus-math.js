import { renderMath } from './math-tips.js';
/* Shared symbolic math for the calculus lessons: parsing student input,
   differentiating it with mathjs, typesetting it, and fast evaluation.
   This module (and mathjs) loads only when a calculus lesson opens. */
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

export const math = create({
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

// JavaScript for each function a student may type. mathjs's log(x) is the natural log.
const JS_FUNCTIONS = {
  sin: 'Math.sin', cos: 'Math.cos', tan: 'Math.tan',
  asin: 'Math.asin', acos: 'Math.acos', atan: 'Math.atan',
  sinh: 'Math.sinh', cosh: 'Math.cosh', tanh: 'Math.tanh',
  exp: 'Math.exp', log: 'Math.log', sqrt: 'Math.sqrt', cbrt: 'Math.cbrt', abs: 'Math.abs',
  sec: '1/Math.cos', csc: '1/Math.sin', cot: '1/Math.tan',
  sech: '1/Math.cosh', csch: '1/Math.sinh', coth: '1/Math.tanh',
};
const FUNCTIONS = new Set(Object.keys(JS_FUNCTIONS));
const CONSTANTS = { e: 'Math.E', pi: 'Math.PI' };
const shownName = name => (name === 'log' ? 'ln' : name);

/* ---------- formatting ---------- */
export function fmt(v, digits = 2) {
  if (!Number.isFinite(v)) return 'undefined';
  if (Math.abs(v) >= 1e4 || (Math.abs(v) > 0 && Math.abs(v) < 1e-3)) {
    const [m, e] = v.toExponential(2).split('e');
    return `${m.replace('-', '−')}×10^${+e}`;
  }
  const r = parseFloat(v.toFixed(digits));
  return String(Object.is(r, -0) ? 0 : r).replace('-', '−');
}

export const texNum = v => {
  const s = fmt(v).replace('−', '-');
  return s.includes('×10^') ? s.replace('×10^', '\\times 10^{') + '}' : s;
};

export function renderTex(el, tex) {
  renderMath(el, tex, false, el.closest('[data-lesson]').dataset.lesson);
}

/* ---------- display ---------- */
const isConst = n => n.isConstantNode || (n.isOperatorNode && n.fn === 'unaryMinus' && n.args[0].isConstantNode);

// Tidy a tree for display: 3·x² → 3x², (1/2)/√x → 1/(2√x).
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

// mathjs writes implicit products as "~" (a space in TeX). Drop it, except between two numbers.
// Single-letter variables that mathjs knows as units (b for bytes) come out upright; set them in italics.
const tidyTex = tex => tex
  .replace(/~(?=\s*[\d.])/g, '\\cdot ')
  .replace(/~/g, '')
  .replace(/\\mathrm\{([a-zA-Z])\}/g, '$1');

export const toTex = node => tidyTex(forDisplay(node).toTex({ implicit: 'hide', parenthesis: 'auto' }));

/* ---------- evaluation ---------- */
function toJS(node, vars) {
  if (node.isParenthesisNode) return `(${toJS(node.content, vars)})`;
  if (node.isConstantNode) {
    if (typeof node.value !== 'number') throw new Error('non-numeric constant');
    return `(${node.value})`;
  }
  if (node.isSymbolNode) {
    if (vars.includes(node.name)) return node.name;
    if (node.name in CONSTANTS) return CONSTANTS[node.name];
    throw new Error(`unknown symbol ${node.name}`);
  }
  if (node.isFunctionNode && node.args.length === 1 && JS_FUNCTIONS[node.fn.name]) {
    return `(${JS_FUNCTIONS[node.fn.name]}(${toJS(node.args[0], vars)}))`;
  }
  if (node.isOperatorNode) {
    const [a, b] = node.args.map(arg => toJS(arg, vars));
    switch (node.fn) {
      case 'add': return `(${a}+${b})`;
      case 'subtract': return `(${a}-${b})`;
      case 'multiply': return `(${a}*${b})`;
      case 'divide': return `(${a}/${b})`;
      case 'pow': return `Math.pow(${a},${b})`;
      case 'unaryMinus': return `(-${a})`;
      case 'unaryPlus': return a;
    }
  }
  throw new Error('unsupported node');
}

// Compile to a plain JavaScript function of the variables, in order. The tree has
// already been checked to hold only numbers, known functions and allowed symbols.
export function compile(node, vars) {
  let fn;
  try {
    fn = new Function(...vars, `return ${toJS(node, vars)};`);
  } catch {
    const compiled = node.compile();
    fn = (...args) => compiled.evaluate(Object.fromEntries(vars.map((v, i) => [v, args[i]])));
  }
  return (...args) => {
    try {
      const v = fn(...args);
      return typeof v === 'number' ? v : NaN;
    } catch {
      return NaN;
    }
  };
}

/* ---------- parsing ---------- */
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

const listVars = vars => (vars.length === 1 ? vars[0] : `${vars.slice(0, -1).join(', ')} and ${vars[vars.length - 1]}`);

function checkSymbols(node, vars) {
  let problem = null;
  node.traverse((n, path, parent) => {
    if (problem) return;
    if (n.isFunctionNode) {
      const name = n.fn.name;
      if (!FUNCTIONS.has(name)) problem = `“${name}” isn’t a function these lessons know. Try sin, cos, tan, exp, ln, sqrt or abs.`;
      else if (n.args.length !== 1) problem = `${shownName(name)}( ) takes one input, like ${shownName(name)}(${vars[0]}).`;
      return;
    }
    if (n.isSymbolNode && !(parent && parent.isFunctionNode && parent.fn === n)) {
      if (vars.includes(n.name) || n.name in CONSTANTS) return;
      const fn = [...FUNCTIONS].find(f => n.name.startsWith(f) && n.name.length > f.length);
      problem = fn
        ? `Put brackets around the input: ${shownName(fn)}(${n.name.slice(fn.length)}).`
        : `Use only ${listVars(vars)} as ${vars.length === 1 ? 'the variable' : 'variables'} (found “${n.name}”). Separate products with a space or *, like ${vars[0]}*${vars[vars.length - 1]}.`;
      return;
    }
    if (!n.isSymbolNode && !n.isFunctionNode && !n.isConstantNode && !n.isOperatorNode && !n.isParenthesisNode) {
      problem = `Use a single expression in ${listVars(vars)}.`;
    }
  });
  return problem;
}

// Long decimals (ln 2 ≈ 0.6931…) appear when mathjs folds constants. Keep those symbolic.
const keepConstants = math.simplify.rules.filter(r => !(typeof r === 'function' && r.name === 'simplifyConstant'));
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

export function differentiate(node, v, inputText = '') {
  const d = math.derivative(node, v);
  let shown = d;
  if (hasLongDecimal(d, inputText)) {
    try {
      shown = math.simplify(math.derivative(node, v, { simplify: false }), keepConstants);
    } catch { /* keep the numeric form */ }
  }
  return { node: d, shown };
}

const unwrap = node => (node.isParenthesisNode ? unwrap(node.content) : node);

// Replace a variable with another expression, e.g. u → 2w.
export function substitute(node, name, replacement) {
  const inner = unwrap(replacement);
  return node.transform(n => (n.isSymbolNode && n.name === name ? inner : n));
}

/* Parse a student's expression in the given variables and differentiate it with
   respect to each. Returns { error } or { node, tex, fn, d: { [v]: { node, shown, tex, fn } } }. */
export function analyse(input, vars) {
  const text = normalise(input.trim());
  if (!text) return { error: `Type a function of ${listVars(vars)}.` };
  let node;
  try {
    node = math.parse(text);
  } catch {
    return { error: 'That isn’t a complete expression yet. Check for a missing bracket or operator.' };
  }
  const problem = checkSymbols(node, vars);
  if (problem) return { error: problem };

  const d = {};
  try {
    for (const v of vars) {
      const { node: dn, shown } = differentiate(node, v, text);
      d[v] = { node: dn, shown, tex: toTex(shown), fn: compile(dn, vars) };
    }
  } catch {
    return { error: 'This lesson can’t differentiate that expression. Try a simpler one.' };
  }
  return { node, tex: toTex(node), fn: compile(node, vars), d };
}

/* ---------- shared editor markup ---------- */
export function lessonHelpHTML(title, content) {
  return `<details class="calc-help">
    <summary aria-label="${title} help">?</summary>
    <div class="calc-help-body">${content}</div>
  </details>`;
}

export function editorHTML({ lhs, value, examples = [], label }) {
  return `
    <div class="calc-editor">
      <label class="calc-input"><span class="calc-lhs">${lhs.replace(/[A-Za-z]/g, symbol => `<var>${symbol}</var>`)} =</span>
        <input type="text" spellcheck="false" autocomplete="off" autocapitalize="off" value="${value}" aria-label="${label}">
      </label>
      ${examples.length ? `<p class="calc-hint">Try ${examples.map(e => `<button type="button" class="calc-example mono">${e}</button>`).join(' ')}</p>` : ''}
      <p class="calc-error" role="alert" hidden></p>
    </div>`;
}

/* Wire an editor block: calls check(text) on each edit; check returns an error
   message or nothing. Example buttons fill the input. */
export function wireEditor(block, check, mathBox) {
  const input = block.querySelector('input');
  const error = block.querySelector('.calc-error');
  function run() {
    const message = check(input.value);
    error.textContent = message || '';
    error.hidden = !message;
    if (message) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
    mathBox?.classList.toggle('stale', Boolean(message));
    return !message;
  }
  input.addEventListener('input', run);
  block.querySelectorAll('.calc-example').forEach(button => button.addEventListener('click', () => {
    input.value = button.textContent;
    run();
    input.focus();
  }));
  return { input, run };
}
