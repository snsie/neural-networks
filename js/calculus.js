const loss = prediction => (prediction - 3) ** 2;
const lossSlope = prediction => 2 * (prediction - 3);
const number = value => String(Math.round(value * 100) / 100).replace('-', '−');

const lessons = {
  derivative: { title: 'Derivative' },
  partial: {
    title: 'Partial derivative',
    prompt: 'Start with weight 1 and bias 0. Which small increase would lower the loss more? Move one slider and watch how each graph holds the other setting fixed.',
    idea: 'A partial derivative measures the slope for one setting while the other settings stay fixed.',
    notation: '∂L/∂w = 4(p − 3) · ∂L/∂b = 2(p − 3), where p = 2w + b.',
    controls: [
      ['Weight', 0.5, 2, 0.05, 1],
      ['Bias', -1, 1, 0.05, 0],
    ],
  },
  chain: {
    title: 'Chain rule',
    prompt: 'Nudge the weight and follow the dots. If raising the weight raises the prediction, what happens to the loss? Check what changes when the prediction passes 3.',
    idea: 'The chain rule multiplies the slopes along a path. Backpropagation uses this idea to work backward from the loss to the weights.',
    notation: '∂L/∂w = (dL/dp) × (∂p/∂w).',
    controls: [['Weight', 0.5, 2, 0.05, 1]],
  },
};

function plot({ xLabel, yLabel, domain, yMax, fn, x, slope, caption }) {
  const [min, max] = domain;
  const xPos = v => 30 + (v - min) / (max - min) * 226;
  const yPos = v => 118 - v / yMax * 103;
  const point = (a, b) => `${xPos(a)},${yPos(b)}`;
  const y = fn(x);
  const a = Math.max(min, x - (max - min) * 0.13);
  const b = Math.min(max, x + (max - min) * 0.13);
  const curve = Array.from({ length: 61 }, (_, i) => {
    const v = min + i / 60 * (max - min);
    return point(v, fn(v));
  }).join(' ');
  return `
    <figure class="calculus-plot">
      <figcaption><strong>${yLabel} vs. ${xLabel.toLowerCase()}</strong><span>${caption}</span></figcaption>
      <svg viewBox="0 0 270 157" role="img" aria-label="${yLabel} versus ${xLabel.toLowerCase()}. At ${number(x)}, ${yLabel.toLowerCase()} is ${number(y)}. Local slope is ${number(slope)}.">
        <path class="calculus-axis" d="M30 15V118H256" />
        <text x="25" y="20" text-anchor="end">${yMax}</text>
        <text x="25" y="118" text-anchor="end">0</text>
        <text x="30" y="134" text-anchor="middle">${number(min)}</text>
        <text x="256" y="134" text-anchor="middle">${number(max)}</text>
        <text x="143" y="153" text-anchor="middle">${xLabel}</text>
        <svg x="30" y="15" width="226" height="103" viewBox="30 15 226 103" overflow="hidden">
          <polyline class="calculus-curve" points="${curve}" />
          <polyline class="calculus-slope" points="${point(a, y + slope * (a - x))} ${point(b, y + slope * (b - x))}" />
          <circle class="calculus-point" cx="${xPos(x)}" cy="${yPos(y)}" r="4" />
        </svg>
      </svg>
    </figure>`;
}

function direction(slope) {
  if (slope < -1e-9) return 'lowers the loss';
  if (slope > 1e-9) return 'raises the loss';
  return 'has almost no effect on the loss at first';
}

let derivativeModule = null;

function mountCalculusLesson(host, id) {
  if (id === 'derivative') {
    host.innerHTML = '<p class="calculus-example">Loading the lesson…</p>';
    derivativeModule ??= import('./calculus-derivative.js');
    derivativeModule.then(
      ({ mountDerivativeLesson }) => { if (host.dataset.lesson === id) mountDerivativeLesson(host); },
      () => { if (host.dataset.lesson === id) host.innerHTML = '<p class="calculus-example">This lesson couldn’t load. Check your connection and reopen it.</p>'; },
    );
    return;
  }
  const lesson = lessons[id];
  host.innerHTML = `
    <p class="calculus-example">One neuron with input 2 and target 3. The prediction is twice the weight plus the bias. Loss is the squared distance from the target.</p>
    <p>${lesson.prompt}</p>
    <div class="calculus-controls">
      ${lesson.controls.map(([label, min, max, step, value]) => `
        <label><span>${label} <output>${number(value)}</output></span>
          <input type="range" min="${min}" max="${max}" step="${step}" value="${value}">
        </label>`).join('')}
    </div>
    <div class="calculus-plots"></div>
    <p class="calculus-result" role="status"></p>
    ${id === 'chain' ? '<details class="calculus-rates"><summary>Show how the slopes combine</summary><p></p></details>' : ''}
    <p>${lesson.idea}</p>
    <details class="calculus-rates">
      <summary>Show notation</summary>
      <p class="calculus-notation mono">${lesson.notation}</p>
    </details>`;

  const inputs = [...host.querySelectorAll('input')];
  const plots = host.querySelector('.calculus-plots');
  const result = host.querySelector('.calculus-result');

  function draw() {
    const [first, second] = inputs.map(input => {
      const value = input.valueAsNumber;
      input.closest('label').querySelector('output').value = number(value);
      return value;
    });
    const weight = first;
    const bias = second ?? 0;
    const prediction = 2 * weight + bias;
    const weightSlope = 2 * lossSlope(prediction);
    if (id === 'partial') {
      plots.innerHTML = plot({
        xLabel: 'Weight', yLabel: 'Loss', domain: [0.5, 2], yMax: 10,
        fn: w => loss(2 * w + bias), x: weight, slope: weightSlope,
        caption: `Bias held at ${number(bias)} · Slope ${number(weightSlope)}`,
      }) + plot({
        xLabel: 'Bias', yLabel: 'Loss', domain: [-1, 1], yMax: 10,
        fn: b => loss(2 * weight + b), x: bias, slope: lossSlope(prediction),
        caption: `Weight held at ${number(weight)} · Slope ${number(lossSlope(prediction))}`,
      });
      result.textContent = `Increasing the weight a little ${direction(weightSlope)} at this setting.`;
      return;
    }

    const predictionSlope = lossSlope(prediction);
    plots.innerHTML = plot({
      xLabel: 'Weight', yLabel: 'Prediction', domain: [0.5, 2], yMax: 5,
      fn: w => 2 * w, x: weight, slope: 2, caption: 'Bias held at 0 · Slope +2',
    }) + plot({
      xLabel: 'Prediction', yLabel: 'Loss', domain: [0.5, 5], yMax: 7,
      fn: loss, x: prediction, slope: predictionSlope,
      caption: `Slope ${number(predictionSlope)}`,
    });
    result.textContent = `Weight ${number(weight)} gives prediction ${number(prediction)} and loss ${number(loss(prediction))}.`;
    host.querySelector('.calculus-rates p').textContent =
      `2 × (${number(predictionSlope)}) = ${number(weightSlope)}. That’s the slope of loss versus weight.`;
  }

  inputs.forEach(input => input.addEventListener('input', draw));
  draw();
}

export function wireCalculusDialog() {
  const chip = document.querySelector('#openCalculus');
  const dialog = document.querySelector('#calculusDialog');
  const title = dialog.querySelector('#calculusStepTitle');
  const host = dialog.querySelector('#calculusLesson');
  const previous = dialog.querySelector('#calculusPrev');
  const next = dialog.querySelector('#calculusNext');
  const chips = dialog.querySelector('#calculusChips');
  const ids = Object.keys(lessons);
  let index = 0;

  chips.innerHTML = ids.map((id, i) =>
    `<button type="button" class="pill calculus-step-chip" data-index="${i}"><span aria-hidden="true">${i + 1}</span> ${lessons[id].title}</button>`).join('');
  chips.addEventListener('click', event => {
    const chip = event.target.closest('button[data-index]');
    if (!chip || +chip.dataset.index === index) return;
    index = +chip.dataset.index;
    showLesson();
  });

  function showLesson() {
    const id = ids[index];
    title.textContent = lessons[id].title;
    host.dataset.lesson = id;
    chips.querySelectorAll('button').forEach((chip, i) => {
      chip.classList.toggle('on', i === index);
      if (i === index) chip.setAttribute('aria-current', 'step');
      else chip.removeAttribute('aria-current');
    });
    mountCalculusLesson(host, id);
    previous.disabled = index === 0;
    next.textContent = index === ids.length - 1 ? 'Done' : 'Next';
    dialog.scrollTop = 0;
    title.focus();
  }

  function openLesson(start) {
    index = start;
    dialog.showModal();
    showLesson();
  }

  chip.addEventListener('click', () => openLesson(0));
  dialog.addEventListener('close', () => chip.focus());
  previous.addEventListener('click', () => {
    index--;
    showLesson();
  });
  next.addEventListener('click', () => {
    if (index === ids.length - 1) {
      dialog.close();
      return;
    }
    index++;
    showLesson();
  });

  const linkedLesson = ids.indexOf(new URLSearchParams(location.hash.slice(1)).get('step'));
  if (linkedLesson >= 0) openLesson(linkedLesson);
}
