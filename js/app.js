/* App state, UI wiring, and the training loop. */
import { wireCalculusDialog } from './calculus.js';
import { wireFullEquation } from './full-equation.js';
import { renderTrace, renderDetails } from './walkthrough-view.js';
(function () {
  'use strict';

  const $ = s => document.querySelector(s);
  const EXT = Datasets.EXTENT;
  const GRID_R = 64;
  const MIN_HIDDEN = 1;
  const MAX_HIDDEN = 8;
  const MIN_LAYERS = 1;
  const MAX_LAYERS = 4;
  const CLASS_NAMES = ['orange', 'blue', 'green'];
  const LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
  const SUBS = '₀₁₂₃₄₅₆₇₈₉';
  const sub = n => String(n).split('').map(d => SUBS[+d]).join('');
  const SUP = n => `⁽${String(n).split('').map(d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join('')}⁾`;
  const fmt = (v, d = 3) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toFixed(d).replace('-', '−');

  const state = {
    dsId: 'circle', noise: 0.1, seed: 1, initSeed: 1,
    H: 1, L: 1, actId: 'relu', lr: 0.03, optimizer: 'adam',
    playing: false, epoch: 0, showTest: true,
    equationDisplay: 'both', step: 0, stepFocus: null,
  };

  let data, train, test, net, grid, pal, sample, cur, steps = [];
  let history = { train: [], test: [] };
  let lastEqKey = '', lastStepId = '', stepListSig = '', choicesSig = '';
  const traceSelection = new Map();

  /* ---------------- KaTeX helper ---------------- */
  function tex(el, s, display = true) {
    if (window.katex) {
      try {
        katex.render(s, el, {
          displayMode: display, throwOnError: false, strict: 'ignore',
          trust: ctx => ctx.command === '\\htmlClass',
        });
        return;
      } catch (e) { /* fall through */ }
    }
    el.textContent = s;
    el.classList.add('eq-fallback');
  }

  /* ---------------- visual components ---------------- */
  const boundary = new BoundaryViz($('#boundary'), {
    ext: EXT,
    onPick: p => { sample = p; renderWalk(true); drawBoundary(); },
  });
  const netViz = new NetworkViz($('#network'), {
    info: nodeInfo,
    onBias: openBias,
    onWeight: weight => showFullEquation('bwd', weight),
  });

  /* ---------------- dataset grid ---------------- */
  const thumbData = {};
  function buildDatasetGrid() {
    const gridEl = $('#dsGrid');
    gridEl.innerHTML = '';
    for (const d of Datasets.list) {
      const b = document.createElement('button');
      b.className = 'ds-item';
      b.dataset.id = d.id;
      b.setAttribute('role', 'radio');
      b.title = `${d.name} · ${d.classes} classes`;
      const dots = Array.from({ length: d.classes }, (_, k) => `<i style="background:var(--c${k})"></i>`).join('');
      b.innerHTML = `<div class="ds-thumb"><canvas></canvas><span class="ds-badge">${dots}</span></div><span>${d.name}</span>`;
      b.addEventListener('click', () => {
        if (state.dsId === d.id) return;
        state.dsId = d.id;
        loadData(true);
      });
      gridEl.appendChild(b);
      thumbData[d.id] = Datasets.generate(d.id, { n: 220, noise: 0.05, seed: 7 }).points;
    }
    drawThumbs();
  }
  function drawThumbs() {
    document.querySelectorAll('.ds-item').forEach(b => {
      Draw.thumb(b.querySelector('canvas'), thumbData[b.dataset.id], pal, EXT);
      const on = b.dataset.id === state.dsId;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', on);
    });
  }

  /* ---------------- activation dialog ---------------- */
  function buildActPills() {
    const wrap = $('#actPills');
    wrap.innerHTML = '';
    for (const a of Activations.list) {
      const b = document.createElement('button');
      b.className = 'pill';
      b.dataset.id = a.id;
      b.setAttribute('role', 'radio');
      b.textContent = a.name;
      b.addEventListener('click', () => {
        if (state.actId === a.id) return;
        state.actId = a.id;
        updateActivation();
        resetNet();
      });
      wrap.appendChild(b);
    }
  }
  function updateActivation() {
    const act = Activations.get(state.actId);
    $('#activationName').textContent = act.name;
    if (!$('#activationDialog').open) return;
    document.querySelectorAll('#actPills .pill').forEach(b => {
      const on = b.dataset.id === act.id;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', on);
    });
    Draw.activation($('#actPlot'), act, pal);
    const t = $('#actTex');
    t.innerHTML = '<div></div><div></div>';
    tex(t.children[0], `f(z) = ${act.texF}`);
    tex(t.children[1], `f'(z) = \\frac{\\partial a}{\\partial z} = ${act.texDf}`);
    $('#actBlurb').textContent = act.blurb;
  }

  /* ---------------- data & network lifecycle ---------------- */
  function loadData(datasetChanged) {
    data = Datasets.generate(state.dsId, { n: 300, noise: state.noise, seed: state.seed });
    train = data.points.filter(p => !p.test);
    test = data.points.filter(p => p.test);
    $('#dsInfo').textContent = `${train.length} train · ${test.length} test`;
    $('#legendClasses').innerHTML = Array.from({ length: data.classes }, (_, k) =>
      `<span><i style="background:var(--c${k})"></i>${CLASS_NAMES[k]}</span>`).join('');
    drawThumbs();
    sample = randomTrainPoint();
    if (datasetChanged || !net) {
      resetNet();
    } else {
      recordMetrics(true);
      refreshVisuals();
      renderWalk(true);
    }
  }

  function resetNet() {
    const act = Activations.get(state.actId);
    net = new NN.Net({
      hidden: state.H, layers: state.L, classes: data.classes, act,
      seed: state.initSeed * 7919 + state.H * 31 + (state.L - 1) * 104729,
    });
    state.epoch = 0;
    history = { train: [], test: [] };
    netViz.build({ H: net.H, L: net.L, O: net.O, K: net.K, actName: act.name, outMode: net.outMode });
    $('#hVal').textContent = state.H;
    $('#hMinus').disabled = state.H <= MIN_HIDDEN;
    $('#hPlus').disabled = state.H >= MAX_HIDDEN;
    $('#lVal').textContent = state.L;
    $('#lMinus').disabled = state.L <= MIN_LAYERS;
    $('#lPlus').disabled = state.L >= MAX_LAYERS;
    recordMetrics(true);
    refreshVisuals();
    renderWalk(true);
  }

  function randomTrainPoint() {
    const pool = train.length ? train : data.points;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  /* ---------------- training ---------------- */
  function recordMetrics(resetOnly) {
    const tr = net.evaluate(train), te = net.evaluate(test);
    if (!resetOnly || history.train.length === 0) {
      history.train.push(tr.loss);
      history.test.push(te.loss);
      if (history.train.length > 500) { history.train.shift(); history.test.shift(); }
    }
    $('#mTrainLoss').textContent = fmt(tr.loss);
    $('#mTestLoss').textContent = fmt(te.loss);
    $('#mTrainAcc').textContent = `${(tr.acc * 100).toFixed(1)}%`;
    $('#mTestAcc').textContent = `${(te.acc * 100).toFixed(1)}%`;
  }

  function trainOneEpoch() {
    net.trainEpoch(train, { lr: state.lr, optimizer: state.optimizer, batchSize: 10 });
    state.epoch++;
    recordMetrics();
  }

  function drawBoundary() {
    boundary.update({ grid, points: data.points, selected: sample, showTest: state.showTest, pal });
  }

  function refreshVisuals() {
    grid = net.evalGrid(GRID_R, EXT);
    drawBoundary();
    netViz.updateWeights(net);
    netViz.drawNeurons(grid, pal);
    Draw.sparkline($('#spark'), [
      { data: history.train, color: Palette.css(pal.text, 0.85) },
      { data: history.test, color: Palette.css(pal.muted, 0.9), dash: true },
    ], pal);
    $('#epochVal').textContent = state.epoch.toLocaleString();
  }

  function setPlaying(on) {
    state.playing = on;
    $('#playBtn').classList.toggle('playing', on);
    $('#playBtn').setAttribute('aria-label', on ? 'Pause training' : 'Start training');
  }

  let frame = 0;
  function tick() {
    if (state.playing) {
      trainOneEpoch();
      refreshVisuals();
      if (++frame % 8 === 0) renderWalk(false);
    }
    requestAnimationFrame(tick);
  }

  /* ---------------- walkthrough ---------------- */
  function computeSample() {
    const c = net.forward(sample.x, sample.y);
    const g = net.backward(c, sample.label);
    cur = { c, g };
    steps = Equations.build({
      net, cache: c, grads: g, label: sample.label,
      lr: state.lr, optimizer: state.optimizer, classNames: CLASS_NAMES,
    });
  }

  function renderTimeline() {
    const sig = JSON.stringify(steps.map(({ phase, band, label }) => [phase, band, label]));
    const list = $('#stepList');
    if (sig !== stepListSig) {
      const focusedIndex = list.contains(document.activeElement) ? document.activeElement.dataset.i : null;
      stepListSig = sig;
      const grid = document.createElement('div');
      grid.className = 'timeline-grid';
      const titles = [
        'Input / update',
        ...Array.from({ length: net.L }, (_, i) => net.L === 1 ? 'Hidden layer' : `Hidden ${i + 1}`),
        'Output path',
        'Loss / error',
      ];
      const positions = [];
      let column = 2; // The first column holds the pass labels.
      titles.forEach((title, band) => {
        const indices = phase => steps.flatMap((step, i) => step.phase === phase && step.band === band ? [i] : []);
        const forward = indices('fwd');
        const backward = indices('bwd');
        const width = Math.max(forward.length, backward.length);
        forward.forEach((i, offset) => { positions[i] = column + offset; });
        backward.forEach((i, offset) => { positions[i] = column + width - 1 - offset; });

        const shade = document.createElement('div');
        shade.className = 'timeline-band';
        shade.dataset.region = band === 0 ? 'endpoints' : band <= net.L ? 'hidden' : band === net.L + 1 ? 'output' : 'loss';
        shade.style.gridColumn = `${column} / span ${width}`;
        shade.textContent = title;
        grid.append(shade);
        column += width;
      });
      grid.style.setProperty('--slots', column - 2);

      for (const phase of ['fwd', 'bwd']) {
        const row = document.createElement('div');
        row.className = `timeline-row ${phase}`;
        const label = document.createElement('span');
        label.className = 'timeline-label';
        label.innerHTML = phase === 'fwd'
          ? 'Forward pass <span aria-hidden="true">→</span>'
          : 'Backward pass <span aria-hidden="true">←</span>';
        row.append(label);
        steps.forEach((step, i) => {
          if (step.phase !== phase) return;
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'step-item';
          button.dataset.i = i;
          button.style.gridColumn = positions[i];
          button.textContent = i + 1;
          button.title = step.label;
          button.setAttribute('aria-label', `Step ${i + 1}. ${step.label}`);
          button.addEventListener('click', () => goStep(i));
          row.append(button);
        });
        grid.append(row);
      }
      list.replaceChildren(grid);
      $('#stepPicker').replaceChildren(...steps.map((step, i) => new Option(`${i + 1}. ${step.label}`, i)));
      if (focusedIndex != null) list.querySelectorAll('.step-item')[focusedIndex]?.focus();
    }
    $('#stepPicker').value = state.step;
    list.querySelectorAll('.step-item').forEach((button, i) => {
      if (i === state.step) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
  }

  function nodeValues(step) {
    const formatValue = x => x !== 0 && Math.abs(x) < 0.001 ? x.toExponential(2).replace('-', '−') : fmt(x);
    const out = {};
    const id = (key, i) => (key === 'in' ? `in${i}` : key === 'out' ? `o${i}` : key === 'loss' ? 'loss' : `${key}-${i}`);
    for (const key in step.vals) {
      const v = step.vals[key];
      v.arr.forEach((x, i) => {
        const node = id(key, i);
        if (!state.stepFocus || step.hl.nodeIds.includes(node)) {
          out[node] = { text: formatValue(x), label: v.label, kind: v.kind };
        }
      });
    }
    return out;
  }

  function renderChoices(selectors, isBias) {
    const shown = selectors.filter(({ key, options }) => options.length > 1 && !(isBias && key.startsWith('source-')));
    const sig = JSON.stringify(shown);
    const root = $('#traceChoices');
    if (sig !== choicesSig) {
      choicesSig = sig;
      const fields = new Map([...root.children].map(field => [field.lastChild.dataset.key, field]));
      shown.forEach(({ key, label, options }, i) => {
        let field = fields.get(key);
        if (!field) {
          field = document.createElement('label');
          const select = document.createElement('select');
          select.dataset.key = key;
          select.addEventListener('change', () => {
            traceSelection.set(key, Number(select.value));
            renderWalk(false);
          });
          field.append(label, select);
        }
        const select = field.lastChild;
        const optionKey = JSON.stringify(options);
        if (select.dataset.options !== optionKey) {
          select.dataset.options = optionKey;
          select.replaceChildren(...options.map((option, index) => new Option(option, index)));
        }
        fields.delete(key);
        const current = root.children[i];
        if (current !== field) root.insertBefore(field, current || null);
      });
      for (const field of fields.values()) field.remove();
    }
    root.querySelectorAll('select').forEach((select, i) => { select.value = choiceIndex(shown[i]); });
  }

  function choiceIndex({ key, options, defaultIndex = 0 }) {
    return Math.min(traceSelection.get(key) ?? defaultIndex, options.length - 1);
  }

  function renderWalk(force) {
    if (!sample || !net) return;
    computeSample();
    if (state.step >= steps.length) state.step = steps.length - 1;
    renderTimeline();
    const s = steps[state.step];

    const primary = s.render(...s.selectors.map(choiceIndex));
    const subchoices = primary.selectors?.map(choiceIndex);
    const guide = subchoices ? primary.render(...subchoices) : primary;
    renderChoices([...s.selectors, ...(primary.selectors || [])], s.id === 'update' && subchoices[1] === 1);
    const progress = `${s.phase === 'fwd' ? 'Forward pass' : 'Backward pass'} · ${state.step + 1} of ${steps.length} · ${s.label}`;
    if ($('#stepPosition').textContent !== progress) $('#stepPosition').textContent = progress;
    if ($('#stepTitle').textContent !== guide.title) $('#stepTitle').textContent = guide.title;
    if ($('#stepNote').textContent !== guide.intro) $('#stepNote').textContent = guide.intro;

    const key = JSON.stringify([s.id, guide, effectiveTheme()]);
    if (force || key !== lastEqKey) {
      lastEqKey = key;
      if (lastStepId !== s.id) $('#traceDetails').replaceChildren();
      renderTrace($('#stepEq'), guide, tex);
      renderDetails($('#traceDetails'), guide.details, tex);
      lastStepId = s.id;
    }
    $('#prevStep').disabled = state.step === 0;
    $('#nextStep').disabled = state.step === steps.length - 1;

    netViz.highlight(state.stepFocus ? guide.hl : null);
    netViz.setValues(nodeValues(guide));
    renderSampleChip();
  }

  function showFullEquation(pass, weight) {
    setPlaying(false);
    renderWalk(false);
    openFullEquation({ model: net, cache: cur.c, grads: cur.g, pass, weight });
  }

  function renderSampleChip() {
    const pred = net.predict(cur.c);
    const ok = pred === sample.label;
    $('#sampleChip').innerHTML =
      `<i style="background:var(--c${sample.label})"></i>` +
      `<span>Sample #${sample.id}${sample.test ? ' (test)' : ''}</span>` +
      `<span class="mono">(${fmt(sample.x, 2)}, ${fmt(sample.y, 2)})</span>` +
      `<span>True: ${CLASS_NAMES[sample.label]}</span>` +
      `<span class="${ok ? 'ok' : 'bad'}">Predicted: ${CLASS_NAMES[pred]}</span>`;
  }

  function setStepFocus(enabled) {
    state.stepFocus = enabled;
    const button = $('#stepFocus');
    button.setAttribute('aria-pressed', enabled);
    button.querySelector('.focus-state').textContent = enabled ? 'On' : 'Off';
  }

  function goStep(i) {
    state.step = Math.max(0, Math.min(steps.length - 1, i));
    if (state.stepFocus === null) {
      setStepFocus(true);
    }
    renderWalk(true);
    const list = $('#stepList');
    const selected = list.querySelector('[aria-current="step"]');
    const toolbar = $('.walk-toolbar');
    const title = $('#stepPosition');
    if (title.getBoundingClientRect().top < toolbar.getBoundingClientRect().bottom) {
      $('.walkthrough').scrollIntoView({ block: 'start' });
    }
    const viewport = list.getBoundingClientRect();
    if (viewport.width === 0) return;
    const button = selected.getBoundingClientRect();
    if (button.left < viewport.left) list.scrollLeft += button.left - viewport.left - 8;
    else if (button.right > viewport.right) list.scrollLeft += button.right - viewport.right + 8;
  }

  function biasDestination(l, j) {
    return l === net.L ? `Output neuron ${j + 1}` : `Hidden layer ${l + 1} · neuron ${j + 1}`;
  }

  function openBias(l, j, trigger) {
    const dialog = $('#biasDialog');
    const bias = fmt(net.b[l][j]);
    $('#biasDestination').textContent = biasDestination(l, j);
    $('#biasValue').textContent = `b = ${bias}`;
    dialog.addEventListener('close', () => trigger.focus(), { once: true });
    dialog.showModal();
  }

  /* ---------------- tooltips ---------------- */
  function nodeInfo(id) {
    if (!cur) return null;
    const { c, g } = cur;
    const f = v => fmt(v, 3);
    const subt = 'for the selected sample';
    if (id.startsWith('in')) {
      const i = +id.slice(2);
      return { title: `x${sub(i + 1)} · input`, sub: subt, rows: [['value', f(c.x[i])]] };
    }
    let m = id.match(/^h(\d+)-(\d+)$/);
    if (m) {
      const l = +m[1], j = +m[2];
      const name = net.L === 1 ? `h${sub(j + 1)} · hidden` : `h${sub(l + 1)},${sub(j + 1)} · hidden layer ${l + 1}`;
      return {
        title: `${name} (${net.act.name})`, sub: subt,
        rows: [['bias b', f(net.b[l][j])], ['z', f(c.z[l][j])], ['a = f(z)', f(c.a[l][j])], ["f′(z)", f(g.dadz[l][j])], ['∂L/∂a', f(g.da[l][j])], ['δ = ∂L/∂z', f(g.dz[l][j])]],
      };
    }
    if (id.startsWith('o')) {
      const k = +id.slice(1), Lo = net.L;
      const name = net.O === 1 ? `P(${CLASS_NAMES[1]})` : CLASS_NAMES[k];
      return {
        title: `${net.O === 1 ? 'ŷ' : `ŷ${sub(k + 1)}`} · output (${net.outMode})`, sub: `${name}, ${subt}`,
        rows: [['bias b', f(net.b[Lo][k])], ['z', f(c.z[Lo][k])], ['ŷ', f(c.yhat[k])], ['target y', String(g.t[k])], ['δ = ŷ − y', f(g.dz[Lo][k])]],
      };
    }
    if (id === 'loss') {
      return {
        title: `Loss · ${net.outMode === 'sigmoid' ? 'binary cross-entropy' : 'cross-entropy'}`, sub: subt,
        rows: [['L', f(net.loss(c, sample.label))], ['true class', CLASS_NAMES[sample.label]], ['predicted', CLASS_NAMES[net.predict(c)]]],
      };
    }
    m = id.match(/^e(\d+)-(\d+)-(\d+)$/);
    if (m) {
      const l = +m[1], j = +m[2], i = +m[3];
      const hName = (layer, n) => (net.L === 1 ? `h${sub(n + 1)}` : `h${sub(layer + 1)},${sub(n + 1)}`);
      const from = l === 0 ? `x${sub(i + 1)}` : hName(l - 1, i);
      const to = l === net.L ? `ŷ${net.O > 1 ? sub(j + 1) : ''}` : hName(l, j);
      return { title: `w${SUP(l + 1)} ${j + 1},${i + 1} : ${from} → ${to}`, sub: subt, rows: [['weight', f(net.W[l][j][i])], ['∂L/∂w', f(g.dW[l][j][i])]] };
    }
    return null;
  }

  /* ---------------- theme ---------------- */
  function effectiveTheme() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t) return t;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function onThemeChange() {
    pal = Palette.read();
    drawThumbs();
    updateActivation();
    refreshVisuals();
    renderWalk(false);
  }

  function wireGuide() {
    const guide = $('#firstRunGuide');
    const prompt = $('#guidePrompt');
    const next = $('#guideNext');
    const open = $('#openGuide');
    const advanced = document.querySelectorAll('[data-advanced-controls]');
    const stages = [
      { step: 'yhat', prompt: 'Start with the selected sample. See how the network turns its scores into a prediction.', action: 'See prediction' },
      { step: 'loss', prompt: 'Compare the prediction with the sample’s true class.', action: 'Inspect loss' },
      { step: 'update', prompt: 'Follow how this sample’s gradient would change one weight.', action: 'Preview weight update' },
      { prompt: 'Try one training epoch and see how the prediction changes.', action: 'Train one epoch' },
    ];
    let stage = 0;

    function showStage() {
      prompt.textContent = stages[stage].prompt;
      next.textContent = stages[stage].action;
    }

    function start() {
      stage = 0;
      guide.hidden = false;
      advanced.forEach(control => { control.hidden = true; });
      showStage();
    }

    function finish(focus) {
      guide.hidden = true;
      advanced.forEach(control => { control.hidden = false; });
      try { localStorage.setItem('nnx-guide-seen', '1'); } catch { /* storage unavailable */ }
      focus.focus();
    }

    open.addEventListener('click', () => {
      start();
      guide.scrollIntoView({ block: 'center' });
      next.focus({ preventScroll: true });
    });
    $('#guideSkip').addEventListener('click', () => finish(open));
    next.addEventListener('click', () => {
      setPlaying(false);
      const { step } = stages[stage];
      if (!step) {
        $('#stepBtn').click();
        goStep(steps.findIndex(item => item.id === 'yhat'));
        finish($('#stepTitle'));
        return;
      }
      goStep(steps.findIndex(item => item.id === step));
      stage++;
      showStage();
    });

    let seen = false;
    try { seen = localStorage.getItem('nnx-guide-seen') === '1'; } catch { /* storage unavailable */ }
    if (!seen && !location.hash) start();
  }

  /* ---------------- controls ---------------- */
  function wireControls() {
    const lrSel = $('#lrSel');
    lrSel.innerHTML = LRS.map(v => `<option value="${v}">${v}</option>`).join('');
    lrSel.value = String(state.lr);
    lrSel.addEventListener('change', () => { state.lr = +lrSel.value; renderWalk(false); });

    const optSel = $('#optSel');
    optSel.value = state.optimizer;
    optSel.addEventListener('change', () => {
      state.optimizer = optSel.value;
      // nudge to a sensible learning rate for each optimizer
      if (state.optimizer === 'sgd' && state.lr < 0.1) state.lr = 0.3;
      if (state.optimizer === 'adam' && state.lr > 0.1) state.lr = 0.03;
      lrSel.value = String(state.lr);
      renderWalk(false);
    });

    $('#playBtn').addEventListener('click', () => setPlaying(!state.playing));
    $('#stepBtn').addEventListener('click', () => { setPlaying(false); trainOneEpoch(); refreshVisuals(); renderWalk(false); });
    $('#resetBtn').addEventListener('click', () => { state.initSeed++; resetNet(); });

    $('#hMinus').addEventListener('click', () => { if (state.H > MIN_HIDDEN) { state.H--; resetNet(); } });
    $('#hPlus').addEventListener('click', () => { if (state.H < MAX_HIDDEN) { state.H++; resetNet(); } });
    $('#lMinus').addEventListener('click', () => { if (state.L > MIN_LAYERS) { state.L--; resetNet(); } });
    $('#lPlus').addEventListener('click', () => { if (state.L < MAX_LAYERS) { state.L++; resetNet(); } });

    const noise = $('#noise');
    noise.addEventListener('input', () => {
      state.noise = +noise.value / 100;
      $('#noiseVal').textContent = `${noise.value}%`;
      loadData(false);
    });
    $('#regenBtn').addEventListener('click', () => { state.seed++; loadData(false); });
    $('#showTest').addEventListener('change', e => { state.showTest = e.target.checked; drawBoundary(); });

    $('#randSample').addEventListener('click', () => {
      const pool = data.points.filter(p => state.showTest || !p.test);
      let p;
      do { p = pool[Math.floor(Math.random() * pool.length)]; } while (pool.length > 1 && p === sample);
      sample = p;
      renderWalk(true);
      drawBoundary();
    });

    $('#stepPicker').addEventListener('change', e => goStep(Number(e.target.value)));
    $('#stepFocus').addEventListener('click', () => {
      setStepFocus(!state.stepFocus);
      renderWalk(false);
    });
    $('#prevStep').addEventListener('click', () => goStep(state.step - 1));
    $('#nextStep').addEventListener('click', () => goStep(state.step + 1));
    $('#fullEquationChip').addEventListener('click', () => showFullEquation(steps[state.step].phase));
    const equationDisplay = $('#equationDisplay');
    const applyEquationDisplay = () => {
      $('.equation-body').dataset.equationDisplay = equationDisplay.value;
    };
    equationDisplay.value = state.equationDisplay;
    applyEquationDisplay();
    equationDisplay.addEventListener('change', applyEquationDisplay);
    wireCalculusDialog();

    const activationChip = $('#activationChip');
    const activationDialog = $('#activationDialog');
    activationChip.addEventListener('click', () => {
      activationDialog.showModal();
      updateActivation();
    });
    activationDialog.addEventListener('close', () => activationChip.focus());

    $('#themeToggle').addEventListener('click', () => {
      const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('nnx-theme', next); } catch (e) { /* ignore */ }
      onThemeChange();
    });
    if (window.matchMedia) {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      const h = () => { if (!document.documentElement.getAttribute('data-theme')) onThemeChange(); };
      if (mq.addEventListener) mq.addEventListener('change', h); else if (mq.addListener) mq.addListener(h);
    }

    document.addEventListener('keydown', e => {
      if (e.target.closest('dialog') || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target.closest('.equation-body') && (e.key.startsWith('Arrow') || e.key === ' ')) return;
      const timelineStep = e.target.closest('.step-item');
      if (timelineStep && ['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'].includes(e.key)) {
        const i = +timelineStep.dataset.i;
        const forward = steps[i].phase === 'fwd';
        const buttons = [...$('#stepList').querySelectorAll('.step-item')];
        const paired = buttons.find(button => steps[+button.dataset.i].phase !== steps[i].phase &&
          button.style.gridColumn === timelineStep.style.gridColumn);
        const next = {
          ArrowRight: i + (forward ? 1 : -1),
          ArrowLeft: i + (forward ? -1 : 1),
          ArrowDown: forward ? paired?.dataset.i : null,
          ArrowUp: forward ? null : paired?.dataset.i,
        }[e.key];
        e.preventDefault();
        if (next == null) return;
        goStep(+next);
        buttons[state.step].focus({ preventScroll: true });
        return;
      }
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === ' ' && tag !== 'BUTTON') { e.preventDefault(); setPlaying(!state.playing); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); goStep(state.step + 1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); goStep(state.step - 1); }
      else if (e.key === 'r' || e.key === 'R') { state.initSeed++; resetNet(); }
    });

    let rt;
    window.addEventListener('resize', () => {
      clearTimeout(rt);
      rt = setTimeout(() => { drawThumbs(); updateActivation(); refreshVisuals(); }, 120);
    });
  }

  /* ---------------- boot ---------------- */
  /** Optional deep-link state, e.g. #ds=spiral3&act=relu&h=6&step=10&mode=symbolic&epochs=200 */
  function readHash() {
    const q = new URLSearchParams(location.hash.slice(1));
    if (Datasets.list.some(d => d.id === q.get('ds'))) state.dsId = q.get('ds');
    if (Activations.list.some(a => a.id === q.get('act'))) state.actId = q.get('act');
    const h = parseInt(q.get('h'), 10);
    if (h >= MIN_HIDDEN && h <= MAX_HIDDEN) state.H = h;
    const layers = parseInt(q.get('layers'), 10);
    if (layers >= MIN_LAYERS && layers <= MAX_LAYERS) state.L = layers;
    const step = Number(q.get('step'));
    if (Number.isInteger(step) && step >= 1) state.step = step - 1; // clamped once steps are built
    const mode = q.get('mode');
    if (mode === 'symbolic') state.equationDisplay = 'symbolic';
    else if (mode === 'formatted' || mode === 'sample') state.equationDisplay = 'sample';
    return Math.min(2000, Math.max(0, parseInt(q.get('epochs'), 10) || 0));
  }

  function init() {
    const pretrain = readHash();
    pal = Palette.read();
    buildDatasetGrid();
    buildActPills();
    updateActivation();
    wireControls();
    loadData(true);
    if (pretrain) {
      for (let e = 0; e < pretrain; e++) trainOneEpoch();
      refreshVisuals();
      renderWalk(true);
    }
    wireGuide();
    requestAnimationFrame(tick);
    // Re-render once web fonts are in so canvas labels use them.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { updateActivation(); refreshVisuals(); });
  }

  const openFullEquation = wireFullEquation({
    tex,
    getSteps: () => steps,
    goToStep(id, { l, i, j }) {
      const index = steps.findIndex(step => step.id === id);
      const layerKey = l === net.L ? 'out' : l;
      if (id === `dW${l + 1}`) {
        traceSelection.set(`source-${layerKey}`, i);
        traceSelection.set(`neuron-${layerKey}`, j);
      } else if (id === `dz${l + 1}` || id === `da${l + 1}` || (id === 'dzout' && l === net.L)) {
        traceSelection.set(`neuron-${layerKey}`, j);
      }
      goStep(index);
      $('#stepTitle').focus();
    },
  });

  window.App = { state, get net() { return net; } };
  init();
})();
