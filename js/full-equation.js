const T = String.raw;
const layerNumber = n => T`^{(${n})}`;
const nodeValue = (symbol, layer, neuron) => T`${symbol}${layerNumber(layer)}_{${neuron}}`;
const weight = (layer, to, from) => T`w${layerNumber(layer)}_{${to},${from}}`;

export function forwardTerms({ L, O, act }) {
  const out = layerNumber(L + 1);
  const activation = T`\operatorname{${act.name.replaceAll(' ', T`\,`)}}`;
  const terms = [[O === 1 ? T`\hat y=` : T`\hat{\mathbf y}=`],
    [O === 1 ? T`\sigma` : T`\operatorname{softmax}`, 'yhat'],
    [T`\big[\,W${out}`, 'zout']];
  for (let n = L; n >= 1; n--) {
    terms.push([activation, `a${n}`], [T`\big(\,W${layerNumber(n)}`, `z${n}`]);
  }
  terms.push([T`\begin{bmatrix}x_1\\x_2\end{bmatrix}`, 'input']);
  for (let n = 1; n <= L; n++) {
    terms.push([T`+b${layerNumber(n)}\,\big)`, `z${n}`]);
  }
  terms.push([T`+b${out}\,\big]`, 'zout']);
  return terms;
}

export function backwardTerms({ L, sizes, O }, { l, j, i }) {
  const layer = l + 1;
  const terms = [[T`\dfrac{\partial L}{\partial ${weight(layer, j + 1, i + 1)}}=`]];
  let index = j + 1;
  for (let n = layer; n <= L; n++) {
    const next = sizes[n + 1] === 1 ? 1 : `k_{${n + 1}}`;
    terms.push([T`f'\big(${nodeValue('z', n, index)}\big)`, `dz${n}`]);
    if (next === 1) terms.push([T`\,\big[`]);
    else terms.push([T`\,\Big[\displaystyle\sum_{${next}=1}^{${sizes[n + 1]}}`, `da${n}`]);
    terms.push([weight(n + 1, next, index), `da${n}`]);
    index = next;
  }
  terms.push([O === 1 ? T`(\hat y-y)` : T`(\hat y_{${index}}-y_{${index}})`, 'dzout']);
  if (l < L) terms.push([T`\Big]`.repeat(L - l)]);
  terms.push([T`\,\cdot\,`], [
    l === 0 ? T`x_{${i + 1}}` : nodeValue('a', l, i + 1), `dW${layer}`,
  ]);
  return terms;
}

export function forwardSampleTerms(net, cache) {
  const { column, matrix, number } = Equations;
  const { L, O, W, b } = net;
  const binary = O === 1;
  const output = binary ? T`\sigma` : T`\operatorname{softmax}`;
  const activation = T`\operatorname{${net.act.name.replaceAll(' ', T`\,`)}}`;
  const terms = [[binary ? T`\hat y\approx` : T`\hat{\mathbf y}\approx`], [output, 'yhat'],
    [T`\Big[${matrix(W[L])}`, 'zout']];
  for (let l = L - 1; l >= 0; l--) {
    terms.push([activation, `a${l + 1}`], [T`\Big(${matrix(W[l])}`, `z${l + 1}`]);
  }
  terms.push([column(cache.x), 'input']);
  for (let l = 0; l < L; l++) {
    terms.push([T`+${column(b[l])}\Big)`, `z${l + 1}`]);
  }
  terms.push([T`+${column(b[L])}\Big]`, 'zout'],
    [T`\approx ${binary ? number(cache.yhat[0]) : column(cache.yhat)}`]);
  return terms;
}

export function backwardSampleTerms(net, cache, grads, { l, j, i }) {
  const { column, matrix, number } = Equations;
  const { L, O, W } = net;
  const factor = value => T`\big(${number(value)}\big)`;
  const terms = [[T`\dfrac{\partial L}{\partial ${weight(l + 1, j + 1, i + 1)}}\approx`]];

  if (l < L) {
    terms.push([factor(grads.dadz[l][j]), `dz${l + 1}`]);
    const outgoing = W[l + 1].map(row => row[j]);
    terms.push([outgoing.length === 1 ? factor(outgoing[0]) : matrix([outgoing]), `da${l + 1}`]);
    for (let h = l + 1; h < L; h++) {
      const slopes = grads.dadz[h];
      terms.push([slopes.length === 1 ? factor(slopes[0]) : T`\operatorname{diag}\Big(${column(slopes)}\Big)`, `dz${h + 1}`]);
      const next = W[h + 1];
      terms.push([next.length === 1 && slopes.length === 1 ? factor(next[0][0]) : T`{${matrix(next)}}^{\!\top}`, `da${h + 1}`]);
    }
  }

  const outputIndex = O === 1 ? 0 : j;
  const residual = l < L && O > 1
    ? T`\Big(${column(cache.yhat)}-${column(grads.t)}\Big)`
    : T`\big(${number(cache.yhat[outputIndex])}-${grads.t[outputIndex]}\big)`;
  terms.push([residual, 'dzout'], [T`\cdot`], [factor(net.input(cache, l)[i]), `dW${l + 1}`],
    [T`\approx ${number(grads.dW[l][j][i])}`]);
  return terms;
}

export function wireFullEquation({ tex, goToStep, getSteps }) {
  const dialog = document.querySelector('#fullEquationDialog');
  const chip = document.querySelector('#fullEquationChip');
  const tabs = [...dialog.querySelectorAll('[role=tab]')];
  const panels = [...dialog.querySelectorAll('[role=tabpanel]')];
  const [layerSelect, sourceSelect, destinationSelect] = dialog.querySelectorAll('select');
  const prediction = dialog.querySelector('#fullPrediction');
  const gradient = dialog.querySelector('#fullGradient');
  const note = dialog.querySelector('#fullGradientNote');
  const predictionSample = dialog.querySelector('#fullPredictionSample');
  const gradientSample = dialog.querySelector('#fullGradientSample');
  const sampleNote = dialog.querySelector('#fullGradientSampleNote');
  let net, cache, grads, selectedWeight, trigger, destinationStep;

  function setPass(pass, focus = true) {
    for (const [index, tab] of tabs.entries()) {
      const active = tab.dataset.pass === pass;
      tab.classList.toggle('on', active);
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      panels[index].hidden = !active;
      if (active && focus) tab.focus();
    }
  }

  function renderTerms(container, terms) {
    const labels = new Map(getSteps().map(({ id, label }) => [id, label]));
    container.replaceChildren(...terms.map(([math, step]) => {
      const element = document.createElement(step ? 'button' : 'span');
      if (step) {
        element.type = 'button';
        element.className = 'formula-term';
        element.title = labels.get(step);
        // Keep the math in the accessible name and announce where the button leads.
        element.setAttribute('aria-description', `Open walkthrough step: ${labels.get(step)}`);
        element.addEventListener('click', () => {
          destinationStep = { step, weight: { ...selectedWeight } };
          dialog.close();
        });
      }
      tex(element, math, false);
      return element;
    }));
  }

  function renderWeightOptions() {
    const { l, j, i } = selectedWeight;
    sourceSelect.replaceChildren(...Array.from({ length: net.sizes[l] }, (_, n) =>
      new Option(l === 0 ? `Input x${n + 1}` : `Neuron ${n + 1}`, n)));
    destinationSelect.replaceChildren(...Array.from({ length: net.sizes[l + 1] }, (_, n) =>
      new Option(`Neuron ${n + 1}`, n)));
    sourceSelect.value = i;
    destinationSelect.value = j;
  }

  function renderGradient() {
    const hasBranches = net.sizes.slice(selectedWeight.l + 2).some(size => size > 1);
    note.textContent = hasBranches
      ? 'Each sum adds contributions from the next layer. Select a term to see how that step works.'
      : 'Multiply the factors along this path to get the gradient. Select a term to see how that step works.';
    sampleNote.hidden = !hasBranches;
    const hasDiagonal = grads.dadz.slice(selectedWeight.l + 1).some(slopes => slopes.length > 1);
    sampleNote.textContent = 'Matrix products add the same contributions as the sums above.'
      + (hasDiagonal ? ' The diag function places the activation slopes on a diagonal.' : '');
    renderTerms(gradient, backwardTerms(net, selectedWeight));
    renderTerms(gradientSample, backwardSampleTerms(net, cache, grads, selectedWeight));
  }

  layerSelect.addEventListener('change', () => {
    selectedWeight = { l: Number(layerSelect.value), i: 0, j: 0 };
    renderWeightOptions();
    renderGradient();
  });
  sourceSelect.addEventListener('change', () => {
    selectedWeight.i = Number(sourceSelect.value);
    renderGradient();
  });
  destinationSelect.addEventListener('change', () => {
    selectedWeight.j = Number(destinationSelect.value);
    renderGradient();
  });

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => setPass(tab.dataset.pass));
    tab.addEventListener('keydown', event => {
      const next = { ArrowLeft: 1 - index, ArrowRight: 1 - index, Home: 0, End: 1 }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      setPass(tabs[next].dataset.pass);
    });
  });

  dialog.addEventListener('close', () => {
    if (destinationStep) {
      const { step, weight } = destinationStep;
      destinationStep = null;
      goToStep(step, weight);
    } else {
      trigger?.focus();
    }
  });

  return ({ model, cache: sampleCache, grads: sampleGrads, pass, weight: initialWeight, opener = chip }) => {
    net = model;
    cache = sampleCache;
    grads = sampleGrads;
    selectedWeight = initialWeight || { l: net.L, j: 0, i: 0 };
    trigger = opener;
    layerSelect.replaceChildren(...Array.from({ length: net.L + 1 }, (_, l) =>
      new Option(l === net.L ? 'Output layer' : `Hidden layer ${l + 1}`, l)));
    layerSelect.value = selectedWeight.l;
    renderWeightOptions();
    renderTerms(prediction, forwardTerms(net));
    renderTerms(predictionSample, forwardSampleTerms(net, cache));
    renderGradient();
    dialog.showModal();
    dialog.scrollTop = 0;
    setPass(pass);
  };
}
