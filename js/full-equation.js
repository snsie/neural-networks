const T = String.raw;
const layerNumber = n => T`^{(${n})}`;
const nodeValue = (symbol, layer, neuron) => T`${symbol}${layerNumber(layer)}_{${neuron}}`;
const weight = (layer, to, from) => T`w${layerNumber(layer)}_{${to},${from}}`;
const partial = (result, input) => T`\dfrac{\partial ${result}}{\partial ${input}}`;
const score = (layer, neuron) => nodeValue('z', layer, neuron);
const activation = (layer, neuron) => nodeValue('a', layer, neuron);
const times = T`\cdot`;

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

export function backwardRows({ L, sizes, O }, { l, j, i }) {
  const layer = l + 1;
  const neuron = j + 1;
  const z = score(layer, neuron);
  const w = weight(layer, neuron, i + 1);
  const localWeight = [partial(z, w), `dW${layer}`];
  const rows = [];

  if (l === L && O === 1) {
    rows.push([
      [T`${partial('L', w)}=`],
      [partial('L', T`\hat y`), 'dzout'], [times],
      [partial(T`\hat y`, z), 'dzout'], [times], localWeight,
    ]);
  } else if (l === L) {
    rows.push([[T`${partial('L', w)}=`], [partial('L', z), 'dzout', { l, j }], [times], localWeight]);
  } else {
    const a = activation(layer, neuron);
    rows.push([
      [T`${partial('L', w)}=`],
      [partial('L', a), `da${layer}`, { l, j }], [times],
      [partial(a, z), `dz${layer}`, { l, j }], [times], localWeight,
    ]);
  }

  const index = n => sizes[n] === 1 ? 1 : T`k_{${n}}`;
  for (let h = l; h < L; h++) {
    const current = h + 1;
    const next = h + 2;
    const currentNeuron = h === l ? neuron : index(current);
    const a = activation(current, currentNeuron);
    const nextNeuron = index(next);
    const nextScore = score(next, nextNeuron);
    const branches = sizes[next] > 1;
    const sum = branches ? T`\displaystyle\sum_{${nextNeuron}=1}^{${sizes[next]}}\bigg[` : '';
    const selection = h === l ? { l: h, j } : undefined;
    rows.push([
      [T`${partial('L', a)}=${sum}`],
      [partial('L', nextScore), h + 1 === L ? 'dzout' : `dz${next}`], [times],
      [partial(nextScore, a), `da${current}`, selection],
      ...(branches ? [[T`\bigg]`]] : []),
    ]);
    if (h + 1 < L) {
      const hiddenScore = score(next, nextNeuron);
      const hiddenActivation = activation(next, nextNeuron);
      rows.push([
        [T`${partial('L', hiddenScore)}=`],
        [partial('L', hiddenActivation), `da${next}`], [times],
        [partial(hiddenActivation, hiddenScore), `dz${next}`],
      ]);
    }
  }

  if (O === 1 && l < L) {
    const output = score(L + 1, 1);
    rows.push([
      [T`${partial('L', output)}=`],
      [partial('L', T`\hat y`), 'dzout'], [times],
      [partial(T`\hat y`, output), 'dzout'],
    ]);
  } else if (O > 1) {
    const output = score(L + 1, l === L ? neuron : index(L + 1));
    rows.push([
      [T`${partial('L', output)}=\displaystyle\sum_{m=1}^{${O}}\bigg[`],
      [partial('L', T`\hat y_m`), 'dzout'], [times],
      [partial(T`\hat y_m`, output), 'dzout'], [T`\bigg]`],
    ]);
  }
  return rows;
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
  const { L, O, W } = net;
  const rows = [];
  const binaryScore = cache.z[L][0];
  const p = cache.yhat[0];
  const complement = Activations.sigmoid(-binaryScore);
  const lossSlope = grads.t[0] === 1 ? -1 / p : 1 / complement;
  const sigmoidSlope = p * complement;
  const showBinaryFactors = O === 1 && p > 0 && p < 1 && Number.isFinite(lossSlope);

  for (let h = L; h >= l; h--) {
    const destinations = h === l ? [j] : grads.dz[h].map((_, k) => k);
    for (const k of destinations) {
      const delta = partial('L', score(h + 1, k + 1));
      const result = number(grads.dz[h][k]);
      if (h === L) {
        const probability = cache.yhat[k], target = grads.t[k];
        let calculation = T`${number(probability)}-${target}`;
        if (O === 1) {
          calculation = showBinaryFactors ? product(lossSlope, sigmoidSlope)
            : T`${target === 1 ? '-' : ''}\sigma(${number(target === 1 ? -binaryScore : binaryScore)})`;
        } else if (target === 1 && probability === 1 && grads.dz[h][k] !== 0) {
          const otherProbabilities = cache.yhat.filter((_, m) => m !== k);
          calculation = T`-\big(${otherProbabilities.map(number).join(' + ')}\big)`;
        }
        rows.push([T`${delta}\approx${calculation}\approx${result}`, 'dzout', { l: h, j: k }]);
      } else {
        const contributions = W[h + 1].map((weights, next) => product(grads.dz[h + 1][next], weights[k]));
        const incoming = partial('L', activation(h + 1, k + 1));
        rows.push(
          [T`${incoming}\approx${contributions.join(' + ')}\approx${number(grads.da[h][k])}`, `da${h + 1}`, { l: h, j: k }],
          [T`${delta}\approx${product(grads.da[h][k], grads.dadz[h][k])}\approx${result}`, `dz${h + 1}`, { l: h, j: k }],
        );
      }
    }
  }

  const gradient = partial('L', weight(l + 1, j + 1, i + 1));
  const source = net.input(cache, l)[i];
  const gradientFactors = l < L ? [grads.da[l][j], grads.dadz[l][j]] :
    showBinaryFactors ? [lossSlope, sigmoidSlope] : [grads.dz[l][j]];
  const calculation = [...gradientFactors, source].map(factor).join(T`\times`);
  rows.push([T`${gradient}\approx${calculation}\approx${number(grads.dW[l][j][i])}`, `dW${l + 1}`]);
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
        // Keep the math in the accessible name and announce where the button leads.
        element.setAttribute('aria-description', `Open walkthrough step: ${labels.get(step)}`);
        element.addEventListener('click', () => {
          destinationStep = { step, weight: { ...selectedWeight, ...selection }, hasRowSelection: Boolean(selection) };
          dialog.close();
        });
      }
      tex(element, math, false);
      if (step && !element.dataset.mathTip) element.title = labels.get(step);
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
    note.textContent = 'Read each product from left to right, from loss toward the selected weight. The variable in the bottom of one fraction is at the top of the next. Each sum adds paths through the neurons or probabilities it lists. Select a fraction to open its walkthrough step.';
    gradient.replaceChildren(...backwardRows(net, selectedWeight).map(terms => {
      const row = document.createElement('div');
      row.className = 'full-gradient-row';
      renderTerms(row, terms);
      return row;
    }));
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
