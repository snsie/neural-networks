function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function addMath(parent, source, tex, display = false) {
  const el = element(display ? 'div' : 'span', display ? 'trace-math' : '');
  tex(el, source, display);
  parent.append(el);
}

function renderSection({ label, rule, worked }, tex) {
  const el = element('section', 'trace-calculation');
  el.append(element('h4', '', label));
  const lines = element('div', 'trace-lines');
  for (const [heading, source] of [['Rule', rule], ['This sample', worked]]) {
    if (!source) continue;
    const line = element('div', 'trace-line');
    line.append(element('span', 'trace-caption', heading));
    addMath(line, source, tex, true);
    lines.append(line);
  }
  el.append(lines);
  return el;
}

function renderTable({ headings, rows, active }, tex) {
  const wrapper = element('div', 'trace-table-wrap');
  const table = element('table', 'trace-table');
  const head = table.createTHead().insertRow();
  for (const heading of headings) {
    const cell = element('th', '', heading);
    cell.scope = 'col';
    head.append(cell);
  }
  const body = table.createTBody();
  rows.forEach((values, i) => {
    const row = body.insertRow();
    if (i === active) row.classList.add('trace-active-row');
    values.forEach((value, j) => {
      const cell = element(j === 0 ? 'th' : 'td');
      if (j === 0) cell.scope = 'row';
      if (typeof value === 'string') cell.textContent = value;
      else addMath(cell, value.math, tex);
      if (j === 0 && i === active) cell.append(element('span', 'walk-visually-hidden', ' (selected row)'));
      row.append(cell);
    });
  });
  wrapper.append(table);
  return wrapper;
}

function renderPlot({ act, z, a, slope, rate, rateMath }, tex) {
  const figure = element('figure', 'trace-plot');
  const canvas = element('canvas');
  canvas.width = 520;
  canvas.height = 234;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', slope
    ? `${act.name} curve with the selected score, its activation, and the local slope marked.`
    : `${act.name} curve with the selected score and corresponding activation marked.`);
  const caption = element('figcaption', '', slope ? 'Local slope at the selected score' : 'Score z → activation a');
  if (slope) {
    const value = element('span', 'trace-plot-value');
    addMath(value, String.raw`f'(z)\approx${rateMath}`, tex);
    caption.append(value);
  }
  figure.append(canvas, caption);

  const ctx = canvas.getContext('2d');
  const style = getComputedStyle(document.documentElement);
  const color = name => style.getPropertyValue(name).trim();
  const left = 40, right = 500, top = 18, bottom = 199;
  const minX = Math.min(-4, z * 1.12), maxX = Math.max(4, z * 1.12);
  const curve = Array.from({ length: 161 }, (_, i) => {
    const x = minX + (maxX - minX) * i / 160;
    return [x, act.f(x)];
  });
  const values = curve.map(([, y]) => y);
  const padY = Math.max(0.12, (Math.max(...values) - Math.min(...values)) * 0.09);
  const minY = Math.min(0, ...values) - padY, maxY = Math.max(0, ...values) + padY;
  const px = x => left + (x - minX) * (right - left) / (maxX - minX);
  const py = y => bottom - (y - minY) * (bottom - top) / (maxY - minY);
  const draw = (points, stroke, width, dash = []) => {
    ctx.beginPath();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    points.forEach(([x, y], i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)));
    ctx.stroke();
    ctx.setLineDash([]);
  };
  draw([[minX, 0], [maxX, 0]], color('--faint'), 1);
  draw([[0, minY], [0, maxY]], color('--faint'), 1);
  draw(curve, color('--accent'), 3);
  draw([[z, 0], [z, a], [0, a]], color('--faint'), 1.5, [5, 5]);
  if (slope) {
    const span = (maxX - minX) * 0.15;
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, right - left, bottom - top);
    ctx.clip();
    draw([[z - span, a - span * rate], [z + span, a + span * rate]], color('--bwd'), 3, [7, 4]);
    ctx.restore();
  }
  ctx.beginPath();
  ctx.arc(px(z), py(a), 6, 0, 2 * Math.PI);
  ctx.fillStyle = color(slope ? '--bwd' : '--accent');
  ctx.fill();
  ctx.font = '16px system-ui';
  ctx.fillStyle = color('--muted');
  ctx.fillText('score z', right - 48, bottom + 25);
  ctx.fillText('a', px(0) + 7, top + 4);
  return figure;
}

export function renderTrace(el, guide, tex) {
  const fragment = document.createDocumentFragment();
  if (guide.inputs?.length) {
    const inputs = element('dl', 'trace-inputs');
    for (const { label, math, text } of guide.inputs) {
      const input = element('div', 'trace-input');
      input.append(element('dt', '', label));
      const content = element('dd', '', text);
      if (math) addMath(content, math, tex);
      input.append(content);
      inputs.append(input);
    }
    fragment.append(inputs);
  }
  const work = element('div', `trace-work${guide.plot ? ' trace-with-plot' : ''}`);
  const calculations = element('div', 'trace-calculations');
  for (const section of guide.sections) calculations.append(renderSection(section, tex));
  for (const table of guide.tables || []) calculations.append(renderTable(table, tex));
  for (const section of guide.after || []) calculations.append(renderSection(section, tex));
  work.append(calculations);
  if (guide.plot) work.append(renderPlot(guide.plot, tex));
  fragment.append(work);

  const { label, math, text } = guide.result;
  const result = element('div', 'trace-result');
  result.append(element('h4', '', label));
  if (math) addMath(result, math, tex);
  result.append(element('p', '', text));
  fragment.append(result);
  el.replaceChildren(fragment);
}

const panelContent = new WeakMap();

function fillDetail(panel) {
  if (!panel.open) return;
  const { math, text, tex } = panelContent.get(panel);
  const content = panel.lastChild;
  const key = JSON.stringify([math, text]);
  if (content.dataset.key === key) return;
  content.dataset.key = key;
  content.replaceChildren();
  if (text) content.append(element('p', '', text));
  for (const source of [math].flat()) addMath(content, source, tex, true);
}

export function renderDetails(el, details = [], tex, openByDefault) {
  const panels = Array.from(el.children);
  if (panels.length !== details.length || panels.some((panel, i) => panel.firstChild.textContent !== details[i].title)) {
    el.replaceChildren(...details.map(({ title }) => {
      const panel = element('details', 'trace-detail');
      panel.open = openByDefault;
      panel.append(element('summary', '', title), element('div', 'trace-detail-content'));
      panel.addEventListener('toggle', () => fillDetail(panel));
      return panel;
    }));
  }
  details.forEach(({ math, text }, i) => {
    const panel = el.children[i];
    panelContent.set(panel, { math, text, tex });
    fillDetail(panel);
  });
}
