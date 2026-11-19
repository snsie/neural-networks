const network = {
  x: 'Input coordinate of the selected data point.',
  a: 'Activation, the value a neuron sends to the next layer.',
  z: 'Neuron score before applying its activation. At the output layer this is called a logit.',
  w: 'Connection weight from the source neuron to the destination neuron.',
  W: 'Matrix of connection weights. Each row contains the weights entering one neuron.',
  b: 'Bias added to the weighted sum. A bold b represents all biases in the layer.',
  y: 'True target. In binary classification, 1 denotes class 1; with a class index, 1 marks the true class and all other entries are 0.',
  'ŷ': 'Predicted probability of class 1 in binary classification, or of the indicated class in multiclass classification. A bold ŷ represents all class probabilities.',
  L: 'Cross-entropy loss for the selected sample. Lower loss means the model gave more probability to the true class.',
  p: 'Predicted probability assigned to the true class.',
  f: 'Activation function applied to a neuron score.',
  "f'": 'Slope of the activation function at a neuron score.',
  '\\sigma': 'Sigmoid function, which maps a score to a value between 0 and 1.',
  '\\delta': 'Gradient of the loss with respect to a neuron’s score z. It measures how sensitive the loss is to that score.',
  '\\theta': 'The weight or bias being updated.',
  '\\Delta': 'Change in the parameter. Δθ is the amount added to the old parameter value.',
  '\\eta': 'Learning rate, which scales the parameter update.',
  '\\epsilon': 'Small positive constant that prevents division by zero in Adam.',
  'm̂': 'Bias-corrected running average of the batch gradient in Adam.',
  'v̂': 'Bias-corrected running average of the squared batch gradient in Adam.',
  m: 'Largest output score, subtracted from each score before exponentiation for numerical stability.',
  D: 'Softmax denominator: the sum of the exponentials of all shifted class scores.',
  R: 'Sum of the exponentials of the shifted scores other than the maximum.',
  s: 'Signed output score used to compute binary cross-entropy stably.',
  g: 'Gradient for one sample, or the average gradient across the mini-batch when subscripted “batch”.',
  B: 'Number of training samples in this mini-batch.',
  i: 'Index that selects an entry in the sum.',
  j: 'Neuron index within the indicated layer.',
  k: 'Index of a class or neuron in the indicated layer.',
  e: 'Euler’s number, the base of the natural exponential.',
  '\\pi': 'Pi, the ratio of a circle’s circumference to its diameter.',
};

const lessonSymbols = {
  lossComparison: {
    y: 'True binary label, either 0 or 1.',
    p: 'Model’s predicted probability of class 1.',
  },
  derivative: {
    x: 'Independent input of the function. In the network analogy, this is a weight.',
    y: 'Output of the function at x. In the network analogy, this is the loss.',
    'dy/dx': 'Slope of y with respect to x at the selected point.',
  },
  partial: {
    L: 'The editable loss function. Its default is the squared error of one neuron.',
    w: 'The neuron’s weight, the first input to the loss function.',
    b: 'The neuron’s bias, the second input to the loss function.',
    '\\nabla': 'Gradient operator. ∇L collects the partial derivatives of L with respect to w and b.',
  },
  chain: {
    x: 'Original input to the inner function.',
    y: 'Output of the inner function and input to the outer function.',
    z: 'Output of the outer function. It depends on x through y.',
    'dz/dz': 'The derivative of z with respect to itself, which equals 1.',
  },
};

function definition(key, source, lesson) {
  if (lesson) return lessonSymbols[lesson][key] ?? ({ e: network.e, '\\pi': network['\\pi'] })[key];
  if (key === 'm' && /y_m|m=k/.test(source)) return 'Class index used to trace how changing one score affects each class probability.';
  if (key === 'x' && /_\{?1\}?$/.test(source)) return 'First input coordinate of the selected data point.';
  if (key === 'x' && /_\{?2\}?$/.test(source)) return 'Second input coordinate of the selected data point.';
  return network[key];
}

// Match the notation produced by the equation builders. Treat text, function
// names, and environment names as opaque, and keep indices with their base.
const tokens = /\\(?:text|operatorname|begin|end|phantom|mathrm)\{[^{}]*\}|(?<symbol>\\hat\s*(?:\{\\mathbf\s+[a-z]\}|[a-z])|\\(?:mathbf|boldsymbol)\{(?:[a-z]|\\delta)\}|\\(?:delta|Delta|theta|eta|epsilon|sigma|nabla|pi)(?![a-zA-Z])|f'|[A-Za-z])(?<scripts>(?:\^\{\(\d+\)\}|_(?:\{(?:[^{}]|\{[^{}]*\})*\}|[a-zA-Z0-9]))*)|\\[a-zA-Z]+|\\./g;
const symbolKey = symbol => {
  if (symbol.startsWith('\\hat')) {
    const letter = symbol.at(-1) === '}' ? symbol.at(-2) : symbol.at(-1);
    return letter + '\u0302';
  }
  return symbol.replace(/^\\(?:mathbf|boldsymbol)\{(.+)\}$/, '$1');
};

function annotate(source, lesson, tips) {
  return source.replace(tokens, (match, symbol, scripts, offset) => {
    if (!symbol) return match;
    const key = symbolKey(symbol).normalize();
    let tip = definition(key, key === 'x' ? match : source, lesson);
    if (!tip) return match;
    const layer = scripts.match(/\^\{\((\d+)\)\}/)?.[1];
    if (layer) tip += ` This refers to layer ${layer}.`;
    if (key === 'w' && /_\{\d+,\d+\}/.test(scripts)) tip += ' The first index is the destination and the second is the source.';
    if (key === 'w' && /_\{?ji\}?/.test(scripts)) tip += ' Index i is the source and j is the destination.';
    if (key === 'w' && /_\{?kj\}?/.test(scripts)) tip += ' Index j is the source and k is the destination.';
    if (/^[ayz]$/.test(key) || key === '\\delta' || key === 'ŷ') {
      const index = scripts.match(/_(?:\{([ijkm]|\d+)\}|([ijkm]|\d))/);
      if (index) tip += ` The subscript ${index[1] ?? index[2]} selects the ${/y/.test(key) || key === 'ŷ' ? 'class' : 'neuron'}.`;
    }
    const marked = `\\htmlClass{math-symbol-${tips.push(tip) - 1}}{${match}}`;
    return /[_^]$/.test(source.slice(0, offset).trimEnd()) ? `{${marked}}` : marked;
  });
}

function enableTip(el, tip, focusable = true, label = el.textContent) {
  el.dataset.mathTip = tip;
  if (focusable) {
    el.tabIndex = 0;
    el.setAttribute('role', 'img');
    el.setAttribute('aria-label', `${label}. ${tip}`);
  }
}

export function addPlainMathTips(el, lesson) {
  for (const variable of el.querySelectorAll('var')) {
    const key = variable.textContent;
    const tip = definition(key, key, lesson);
    if (tip) enableTip(variable, tip);
  }
}

// Keep the original MathML available to screen readers. Expose only the
// focusable definition targets from KaTeX's parallel visual tree.
function exposeTips(el) {
  if (el.hasAttribute('data-math-tip')) return;
  if (!el.querySelector('[data-math-tip]')) {
    el.setAttribute('aria-hidden', 'true');
    return;
  }
  el.removeAttribute('aria-hidden');
  for (const child of el.children) exposeTips(child);
}

export function renderMath(el, source, display = false, lesson = null) {
  const tips = [];
  if (active && el.contains(active)) dismiss();
  if (window.katex) {
    try {
      window.katex.render(annotate(source, lesson, tips), el, {
        displayMode: display,
        strict: 'ignore',
        trust: context => context.command === '\\htmlClass' && /^math-symbol-\d+$/.test(context.class),
      });
      el.classList.remove('eq-fallback');
      const annotation = el.querySelector('.katex-mathml annotation');
      if (annotation) annotation.textContent = source;
      const button = el.closest('button');
      const definitions = new Set();
      for (const symbol of el.querySelectorAll('.katex-html [class*="math-symbol-"]')) {
        const id = [...symbol.classList].find(name => name.startsWith('math-symbol-'));
        const tip = tips[Number(id.slice('math-symbol-'.length))];
        if (!tip) continue;
        enableTip(symbol, tip, !button);
        definitions.add(`${symbol.textContent}: ${tip}`);
      }
      if (definitions.size) {
        if (button) {
          button.dataset.mathTip = [...definitions].join('\n');
          button.setAttribute('aria-description', [button.getAttribute('aria-description'), button.dataset.mathTip].filter(Boolean).join('. '));
        } else {
          exposeTips(el.querySelector('.katex-html'));
        }
      }
      return;
    } catch { /* fall back to the plain equation */ }
  }
  el.textContent = source;
  el.classList.add('eq-fallback');
}

let active;
let tooltip;

function dismiss() {
  active = null;
  if (tooltip) tooltip.hidden = true;
}

function show(target) {
  active = target;
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.className = 'math-tip';
    tooltip.setAttribute('aria-hidden', 'true');
  }
  const host = target.closest('dialog[open]') ?? document.body;
  if (tooltip.parentElement !== host) host.append(tooltip);
  tooltip.textContent = target.dataset.mathTip;
  tooltip.hidden = false;
  const box = target.getBoundingClientRect();
  const { width, height } = tooltip.getBoundingClientRect();
  const edge = 8;
  const left = Math.max(edge, Math.min(box.left + (box.width - width) / 2, window.innerWidth - width - edge));
  const top = box.bottom + height + edge <= window.innerHeight ? box.bottom + edge : box.top - height - edge;
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${Math.max(edge, top)}px`;
}

export function wireMathTips() {
  const target = el => el instanceof Element ? el.closest('[data-math-tip]') : null;
  document.addEventListener('pointerover', event => {
    if (event.pointerType === 'touch') return;
    const symbol = target(event.target);
    // A clickable term gets a summary on keyboard focus. Pointer hover
    // describes the individual symbol and leaves the click action intact.
    if (symbol && !symbol.matches('button') && symbol !== active) show(symbol);
  });
  document.addEventListener('pointerout', event => {
    if (target(event.target) === active && target(event.relatedTarget) !== active) dismiss();
  });
  document.addEventListener('focusin', event => {
    const symbol = target(event.target);
    if (symbol) show(symbol);
    else dismiss();
  });
  document.addEventListener('focusout', dismiss);
  document.addEventListener('pointerdown', event => {
    const symbol = target(event.target);
    if (event.pointerType === 'touch' && symbol && !symbol.closest('button')) show(symbol);
    else if (!symbol) dismiss();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && active) {
      dismiss();
      event.preventDefault();
    }
  });
  document.addEventListener('scroll', dismiss, true);
  document.addEventListener('close', dismiss, true);
  window.addEventListener('resize', dismiss);
}
