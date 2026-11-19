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
  terms.push([O === 1 ? T`(\hat y-y)` : T`(\hat y_{${index}}-y_{${index}})`, 'dzout', { l: h, j: k }]);
  if (l < L) terms.push([T`\Big]`.repeat(L - l)]);
  terms.push([T`\,\cdot\,`], [
    l === 0 ? T`x_{${i + 1}}` : nodeValue('a', l, i + 1), `dW${layer}`,
  ]);
  return terms;
}

const factor = value => {
  const formatted = Equations.number(value);
  return value < 0 ? T`(${formatted})` : formatted;
};
const product = (left, right) => T`${factor(left)}\times${factor(right)}`;

export function forwardSampleRows(net, cache) {
  const { number } = Equations;
  const { L, O, W, b, act } = net;
  const rows = [[cache.x.map((x, i) => T`x_{${i + 1}}=${number(x)}`).join(T`,\quad `), 'input']];
  const activation = T`\operatorname{${act.name.replaceAll(' ', T`\,`)}}`;

  for (let l = 0; l <= L; l++) {
    const inputs = net.input(cache, l);
    const scoreStep = l === L ? 'zout' : `z${l + 1}`;
    rows.push(...W[l].map((weights, j) => {
      const terms = weights.map((w, i) => product(w, inputs[i]));
      terms.push(product(b[l][j], 1));
      return [T`${nodeValue('z', l + 1, j + 1)}\approx${terms.join(' + ')}\approx${number(cache.z[l][j])}`, scoreStep, { l, j }];
    }));
    if (l < L) {
      rows.push(...cache.a[l].map((a, j) => [
        T`${nodeValue('a', l + 1, j + 1)}\approx${activation}(${number(cache.z[l][j])})\approx${number(a)}`,
        `a${l + 1}`, { l, j },
      ]));
    }
  }

  const scores = cache.z[L];
  if (O === 1) {
    rows.push([T`\hat y\approx\frac{1}{1+e^{-${factor(scores[0])}}}\approx${number(cache.yhat[0])}`, 'yhat']);
  } else {
    const shift = Math.max(...scores);
    rows.push([T`m\approx\max\{${scores.map(number).join(', ')}\}\approx${number(shift)}`, 'yhat']);
    const exponentials = scores.map(score => T`e^{${number(score)}-${factor(shift)}}`);
    rows.push(...cache.yhat.map((p, j) => [
      T`\hat y_{${j + 1}}\approx\frac{${exponentials[j]}}{${exponentials.join(' + ')}}\approx${number(p)}`,
      'yhat', { l: L, j },
    ]));
  }
  return rows;
}

export function backwardSampleRows(net, cache, grads, { l, j, i }) {
  const { number } = Equations;
  const { L, W } = net;
  const rows = [];

  for (let h = L; h >= l; h--) {
    const destinations = h === l ? [j] : grads.dz[h].map((_, k) => k);
    for (const k of destinations) {
      const delta = nodeValue(T`\delta`, h + 1, k + 1);
      const result = number(grads.dz[h][k]);
      if (h === L) {
        rows.push([T`${delta}\approx${number(cache.yhat[k])}-${grads.t[k]}\approx${result}`, 'dzout', { l: h, j: k }]);
      } else {
        const contributions = W[h + 1].map((weights, next) => product(weights[k], grads.dz[h + 1][next]));
        const incoming = T`\frac{\partial L}{\partial ${nodeValue('a', h + 1, k + 1)}}`;
        rows.push(
          [T`${incoming}\approx${contributions.join(' + ')}\approx${number(grads.da[h][k])}`, `da${h + 1}`, { l: h, j: k }],
          [T`${delta}\approx${product(grads.dadz[h][k], grads.da[h][k])}\approx${result}`, `dz${h + 1}`, { l: h, j: k }],
        );
      }
    }
  }

  const gradient = T`\dfrac{\partial L}{\partial ${weight(l + 1, j + 1, i + 1)}}`;
  rows.push([T`${gradient}\approx${product(grads.dz[l][j], net.input(cache, l)[i])}\approx${number(grads.dW[l][j][i])}`, `dW${l + 1}`]);
  return rows;
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
    container.replaceChildren(...terms.map(([math, step, selection]) => {
      const element = document.createElement(step ? 'button' : 'span');
      if (step) {
        element.type = 'button';
        element.className = 'formula-term';
        element.title = labels.get(step);
        // Keep the math in the accessible name and announce where the button leads.
        element.setAttribute('aria-description', `Open walkthrough step: ${labels.get(step)}`);
        element.addEventListener('click', () => {
          destinationStep = { step, weight: { ...selectedWeight, ...selection }, hasRowSelection: Boolean(selection) };
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
    renderTerms(gradient, backwardTerms(net, selectedWeight));
    renderTerms(gradientSample, backwardSampleRows(net, cache, grads, selectedWeight));
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
      const { step, weight, hasRowSelection } = destinationStep;
      destinationStep = null;
      goToStep(step, weight, hasRowSelection);
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
    renderTerms(predictionSample, forwardSampleRows(net, cache));
    renderGradient();
    dialog.showModal();
    dialog.scrollTop = 0;
    setPass(pass);
  };
}
