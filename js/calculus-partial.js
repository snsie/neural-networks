/* Partial derivative lesson: an editable L(w, b), its two partial derivatives,
   a contour map of L with the downhill gradient, and two slices through the
   selected point (L vs w with b fixed, L vs b with w fixed). */
import { analyse, editorHTML, lessonHelpHTML, wireEditor, renderTex, fmt, texNum } from './calculus-math.js';
import { createFunctionPlot, valueRange } from './calculus-plot.js';
import { slopeSentence, tangentTex } from './calculus-derivative.js';
import { createSurfacePlot, VIEWS } from './calculus-surface.js';

const W_RANGE = [-1, 3];
const B_RANGE = [-2, 2];
const GRID = 160;      // heat map resolution
const COARSE = 80;     // grid for contour lines and the slice plots' vertical range
const LEVELS = 10;     // contour bands

// Map geometry in SVG units.
const MW = 360, MH = 350;
const M = { l: 44, r: 10, t: 10, b: 34 };
const PW = MW - M.l - M.r, PH = MH - M.t - M.b;
const ARROW = 48;

const hexToRgb = hex => {
  let h = hex.trim().replace('#', '');
  if (h.length === 3) h = [...h].map(c => c + c).join('');
  const n = parseInt(h, 16) || 0;
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

function sensitivity(dw, db) {
  const aw = Math.abs(dw), ab = Math.abs(db);
  if (aw < 5e-3 && ab < 5e-3) return 'Both slopes are zero: L is flat here in every direction.';
  if (aw < 5e-3) return 'Here L only responds to b.';
  if (ab < 5e-3) return 'Here L only responds to w.';
  const ratio = aw / ab;
  if (Math.abs(ratio - 1) < 0.05) return 'L is about equally sensitive to w and b here.';
  return ratio > 1
    ? `L is ${fmt(ratio, 1)}× as sensitive to w as to b here.`
    : `L is ${fmt(1 / ratio, 1)}× as sensitive to b as to w here.`;
}

export function mount(host) {
  host.innerHTML = `
    ${lessonHelpHTML('Partial derivative', `
      <p>When a function has several inputs, a <strong>partial derivative</strong> is its slope along one input while the others stay fixed. Here <var>L</var>(<var>w</var>, <var>b</var>) = (2<var>w</var> + <var>b</var> − 3)² is the squared error of one neuron with input 2, weight <var>w</var>, bias <var>b</var> and target 3. Edit <var>L</var> to try any function of <var>w</var> and <var>b</var>.</p>
      <p>The map and the 3D surface show the same <var>L</var>: the map from above, the surface from any angle. Click the map or the surface to choose a point. The two blue curves on the surface are the slices graphed below, each holding one input fixed.</p>
      <p>Training uses every partial derivative at once. Gradient descent nudges each weight against its own slope, which moves the point downhill on this map.</p>
    `)}
    ${editorHTML({ lhs: 'L(w, b)', value: '(2w + b - 3)^2', label: 'Function of w and b', examples: ['w^2 + b^2', 'w^2 - b^2', 'sin(w) cos(b)', '(w b - 1)^2'] })}
    <div class="calc-math" aria-live="polite">
      <div class="calc-eq-f"></div>
      <div class="calc-eq-df"></div>
      <div class="calc-eq-df"></div>
    </div>
    <div class="calc-partial-grid">
      <figure class="calc-map">
        <figcaption><strong>Contour map of L</strong><span>Stronger color means higher L. The arrow points downhill, along −∇L.</span></figcaption>
        <svg viewBox="0 0 ${MW} ${MH}" tabindex="0" role="slider" aria-roledescription="2D slider" aria-label="Point (w, b). Left and right arrows change w; up and down arrows change b."></svg>
      </figure>
      <figure class="calc-surface">
        <figcaption><strong>Surface of L</strong><span>Drag to rotate. Click the surface to choose a point.</span></figcaption>
        <div class="calc-surface-host"></div>
        <div class="calc-surface-tools">
          <div class="calc-views" role="group" aria-label="Viewing angle">
            ${Object.entries(VIEWS).map(([id, v]) => `<button type="button" class="calc-example" data-view="${id}">${v.label}</button>`).join('')}
          </div>
          <label class="calc-toggle"><input type="checkbox" checked> Tangent plane</label>
        </div>
        <div class="calc-legend calc-surface-legend" aria-hidden="true">
          <span class="lg lg-f">slices</span>
          <span class="lg lg-tan">tangent lines</span>
          <span class="lg lg-plane">tangent plane</span>
          <span class="lg lg-arrow">−∇L</span>
        </div>
      </figure>
    </div>
    <div class="calc-slices">
      <div class="calc-slice-w"></div>
      <div class="calc-slice-b"></div>
    </div>
    <div class="calc-readout" role="status">
      <p class="calc-values"></p>
      <p class="calc-meaning calc-meaning-w"></p>
      <p class="calc-meaning calc-meaning-b"></p>
      <p class="calc-gradient"><span class="calc-tex"></span></p>
      <p class="calc-plane"><span>Tangent plane:</span> <span class="calc-tex"></span></p>
      <p class="calc-meaning calc-meaning-g"></p>
    </div>`;

  const mathBox = host.querySelector('.calc-math');
  const svg = host.querySelector('.calc-map svg');
  const values = host.querySelector('.calc-values');
  const meaningW = host.querySelector('.calc-meaning-w');
  const meaningB = host.querySelector('.calc-meaning-b');
  const gradient = host.querySelector('.calc-gradient');
  const meaningG = host.querySelector('.calc-meaning-g');
  const plane = host.querySelector('.calc-plane');

  let model = null;
  let w = 1, b = 0;
  let heatURL = '';
  let contourPath = '';
  let rangeW = [-1, 1], rangeB = [-1, 1];
  let rankFn = () => 0, rangeL = [-1, 1];

  const sx = v => M.l + ((v - W_RANGE[0]) / (W_RANGE[1] - W_RANGE[0])) * PW;
  const sy = v => M.t + PH - ((v - B_RANGE[0]) / (B_RANGE[1] - B_RANGE[0])) * PH;
  const round = v => Math.round(v * 100) / 100;
  const clampW = v => round(Math.min(W_RANGE[1], Math.max(W_RANGE[0], v)));
  const clampB = v => round(Math.min(B_RANGE[1], Math.max(B_RANGE[0], v)));

  const plotW = createFunctionPlot(host.querySelector('.calc-slice-w'), {
    xName: 'w', fName: 'L', dName: '∂L/∂w', caption: '', width: 420, height: 230, compact: true,
    onSelect: v => { w = v; refresh(); },
  });
  const plotB = createFunctionPlot(host.querySelector('.calc-slice-b'), {
    xName: 'b', fName: 'L', dName: '∂L/∂b', caption: '', width: 420, height: 230, compact: true,
    onSelect: v => { b = v; refresh(); },
  });

  function buildHeat() {
    const cs = getComputedStyle(document.documentElement);
    const low = hexToRgb(cs.getPropertyValue('--surface'));
    const high = mix(low, hexToRgb(cs.getPropertyValue('--accent')), 0.78);
    const missing = hexToRgb(cs.getPropertyValue('--surface-3'));

    const vals = new Float64Array(GRID * GRID);
    for (let j = 0; j < GRID; j++) {
      const bv = B_RANGE[1] - ((j + 0.5) / GRID) * (B_RANGE[1] - B_RANGE[0]);
      for (let i = 0; i < GRID; i++) {
        vals[j * GRID + i] = model.fn(W_RANGE[0] + ((i + 0.5) / GRID) * (W_RANGE[1] - W_RANGE[0]), bv);
      }
    }
    // Shade by rank rather than raw value, so the picture stays readable for any function.
    const sorted = Array.from(vals).filter(Number.isFinite).sort((x, y) => x - y);
    const rank = v => {
      let lo = 0, hi = sorted.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < v) lo = mid + 1; else hi = mid; }
      return sorted.length > 1 ? lo / (sorted.length - 1) : 0;
    };

    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = GRID;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(GRID, GRID);
    vals.forEach((v, k) => {
      const c = Number.isFinite(v) ? mix(low, high, rank(v)) : missing;
      img.data.set([c[0], c[1], c[2], 255], k * 4);
    });
    ctx.putImageData(img, 0, 0);
    heatURL = canvas.toDataURL();
    rankFn = rank;

    // Contour lines at evenly spaced ranks, traced with marching squares on a coarse grid.
    const node = [];
    for (let j = 0; j <= COARSE; j++) {
      const row = [];
      const bv = B_RANGE[0] + (j / COARSE) * (B_RANGE[1] - B_RANGE[0]);
      for (let i = 0; i <= COARSE; i++) row.push(model.fn(W_RANGE[0] + (i / COARSE) * (W_RANGE[1] - W_RANGE[0]), bv));
      node.push(row);
    }
    const levels = sorted.length
      ? Array.from({ length: LEVELS - 1 }, (_, k) => sorted[Math.floor(((k + 1) / LEVELS) * (sorted.length - 1))])
      : [];
    const gx = i => M.l + (i / COARSE) * PW;
    const gy = j => M.t + PH - (j / COARSE) * PH;
    let d = '';
    for (const c of new Set(levels)) {
      for (let j = 0; j < COARSE; j++) {
        for (let i = 0; i < COARSE; i++) {
          const v = [node[j][i], node[j][i + 1], node[j + 1][i + 1], node[j + 1][i]];
          if (!v.every(Number.isFinite)) continue;
          // Corners: 0 bottom-left, 1 bottom-right, 2 top-right, 3 top-left (in w, b).
          const corner = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
          const pts = [];
          for (let e = 0; e < 4; e++) {
            const a = v[e], b2 = v[(e + 1) % 4];
            if ((a > c) !== (b2 > c)) {
              const t = (c - a) / (b2 - a);
              const [x0, y0] = corner[e], [x1, y1] = corner[(e + 1) % 4];
              pts.push([gx(x0 + (x1 - x0) * t), gy(y0 + (y1 - y0) * t)]);
            }
          }
          for (let p = 0; p + 1 < pts.length; p += 2) {
            d += `M${pts[p][0].toFixed(1)},${pts[p][1].toFixed(1)}L${pts[p + 1][0].toFixed(1)},${pts[p + 1][1].toFixed(1)}`;
          }
        }
      }
    }
    contourPath = d;

    // Fix each slice's vertical range over the whole map, so the curves move as the
    // other input changes instead of the axes rescaling under them.
    const Ls = [], Lw = [], Lb = [];
    for (let j = 0; j <= COARSE; j += 2) {
      const bv = B_RANGE[0] + (j / COARSE) * (B_RANGE[1] - B_RANGE[0]);
      for (let i = 0; i <= COARSE; i += 2) {
        const wv = W_RANGE[0] + (i / COARSE) * (W_RANGE[1] - W_RANGE[0]);
        Ls.push(node[j][i]);
        Lw.push(model.d.w.fn(wv, bv));
        Lb.push(model.d.b.fn(wv, bv));
      }
    }
    rangeW = valueRange(Ls, Lw);
    rangeB = valueRange(Ls, Lb);
    rangeL = valueRange(Ls);
  }

  function drawMap(dw, db) {
    let ticks = '';
    for (let v = W_RANGE[0]; v <= W_RANGE[1] + 1e-9; v += 1) {
      ticks += `<line class="calc-map-tick" x1="${sx(v)}" y1="${M.t + PH}" x2="${sx(v)}" y2="${M.t + PH + 4}"/><text x="${sx(v)}" y="${MH - 12}" text-anchor="middle">${fmt(v)}</text>`;
    }
    for (let v = B_RANGE[0]; v <= B_RANGE[1] + 1e-9; v += 1) {
      ticks += `<line class="calc-map-tick" x1="${M.l - 4}" y1="${sy(v)}" x2="${M.l}" y2="${sy(v)}"/><text x="${M.l - 8}" y="${sy(v) + 4}" text-anchor="end">${fmt(v)}</text>`;
    }
    const X = sx(w), Y = sy(b);
    let arrow = '';
    // −∇L in screen space (screen y points down, so the b component flips sign).
    const gx = -dw * (PW / (W_RANGE[1] - W_RANGE[0]));
    const gy = db * (PH / (B_RANGE[1] - B_RANGE[0]));
    const len = Math.hypot(gx, gy);
    if (Number.isFinite(len) && len > 1e-6) {
      const ex = X + (gx / len) * ARROW, ey = Y + (gy / len) * ARROW;
      arrow = `<line class="calc-grad" x1="${X.toFixed(1)}" y1="${Y.toFixed(1)}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" marker-end="url(#calcArrow)"/>`;
    }
    svg.innerHTML = `
      <defs>
        <clipPath id="calcMapClip"><rect x="${M.l}" y="${M.t}" width="${PW}" height="${PH}"/></clipPath>
        <marker id="calcArrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="calc-grad-head"/></marker>
      </defs>
      <image href="${heatURL}" x="${M.l}" y="${M.t}" width="${PW}" height="${PH}" preserveAspectRatio="none"/>
      <path class="calc-contour" d="${contourPath}"/>
      <rect class="calc-map-frame" x="${M.l}" y="${M.t}" width="${PW}" height="${PH}"/>
      ${ticks}
      <text class="calc-axis-name" x="${M.l + PW - 6}" y="${M.t + PH - 8}" text-anchor="end">w</text>
      <text class="calc-axis-name" x="${M.l + 8}" y="${M.t + 18}">b</text>
      <g clip-path="url(#calcMapClip)">
        <line class="calc-slice-line" x1="${M.l}" y1="${Y.toFixed(1)}" x2="${M.l + PW}" y2="${Y.toFixed(1)}"/>
        <line class="calc-slice-line" x1="${X.toFixed(1)}" y1="${M.t}" x2="${X.toFixed(1)}" y2="${M.t + PH}"/>
        ${arrow}
        <circle class="calc-pt-f" cx="${X.toFixed(1)}" cy="${Y.toFixed(1)}" r="6"/>
      </g>`;
    svg.setAttribute('aria-valuetext', `w = ${fmt(w)}, b = ${fmt(b)}, L = ${fmt(model.fn(w, b))}`);
  }

  function refresh() {
    if (!model) return;
    const L = model.fn(w, b), dw = model.d.w.fn(w, b), db = model.d.b.fn(w, b);
    drawMap(dw, db);
    plotW.setFunctions(v => model.fn(v, b), v => model.d.w.fn(v, b), { xRange: W_RANGE, yRange: rangeW });
    plotW.setX(w);
    plotW.setCaption(`<strong>L vs w</strong><span>b held at ${fmt(b)} · slope ${fmt(dw)}</span>`);
    plotB.setFunctions(v => model.fn(w, v), v => model.d.b.fn(w, v), { xRange: B_RANGE, yRange: rangeB });
    plotB.setX(b);
    plotB.setCaption(`<strong>L vs b</strong><span>w held at ${fmt(w)} · slope ${fmt(db)}</span>`);
    surface.setPoint({ w, b, L, dw, db });

    values.innerHTML = `At (<var>w</var>, <var>b</var>) = (<strong>${fmt(w)}</strong>, <strong>${fmt(b)}</strong>): <span class="calc-val-f"><var>L</var> = <strong>${fmt(L)}</strong></span>, <span class="calc-val-df">∂<var>L</var>/∂<var>w</var> = <strong>${fmt(dw)}</strong></span>, <span class="calc-val-df">∂<var>L</var>/∂<var>b</var> = <strong>${fmt(db)}</strong></span>`;
    const defined = [L, dw, db].every(Number.isFinite);
    gradient.hidden = !defined;
    plane.hidden = !defined;
    meaningB.hidden = !defined;
    if (!defined) {
      meaningW.textContent = 'L or one of its slopes is undefined here. Pick a point where the map is shaded.';
      meaningG.textContent = '';
      return;
    }
    meaningW.textContent = `With b fixed: ${slopeSentence(dw, 'w', 'L').replace(/^The slope/, 'the slope')}`;
    meaningB.textContent = `With w fixed: ${slopeSentence(db, 'b', 'L').replace(/^The slope/, 'the slope')}`;
    renderTex(gradient.querySelector('.calc-tex'),
      `\\nabla L = \\left(\\tfrac{\\partial L}{\\partial w},\\ \\tfrac{\\partial L}{\\partial b}\\right) = (${texNum(dw)},\\ ${texNum(db)})`);
    // Tangent plane: L₀ + ∂L/∂w (w − w₀) + ∂L/∂b (b − b₀), written with tidy signs.
    const term = (name, at, m) => (Math.abs(m) < 5e-3 ? '' : tangentTex('', name, at, 0, m).replace(/^\s*=\s*/, ''));
    const wTerm = term('w', w, dw);
    const bTerm = term('b', b, db);
    const terms = [Math.abs(L) < 5e-3 ? '' : texNum(L), wTerm.trim(), bTerm.trim()].filter(Boolean);
    const joined = terms.map((t, i) => (i && !/^-/.test(t) ? `+ ${t}` : t)).join(' ');
    renderTex(plane.querySelector('.calc-tex'), `L \\approx ${joined || '0'}`);
    meaningG.textContent = Math.hypot(dw, db) < 5e-3
      ? 'The gradient is zero, so there is no downhill direction: this is a minimum, maximum or saddle point.'
      : `${sensitivity(dw, db)} The arrow shows −∇L = (${fmt(-dw)}, ${fmt(-db)}), the direction that lowers L fastest.`;
  }

  const surface = createSurfacePlot(host.querySelector('.calc-surface-host'), {
    wRange: W_RANGE, bRange: B_RANGE,
    onSelect: (nw, nb) => { w = clampW(nw); b = clampB(nb); refresh(); },
  });
  host.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => surface.setView(button.dataset.view)));
  host.querySelector('.calc-toggle input').addEventListener('change', event => surface.showTangent(event.target.checked));

  // Map interaction.
  const fromPointer = event => {
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    w = clampW(W_RANGE[0] + ((pt.x - M.l) / PW) * (W_RANGE[1] - W_RANGE[0]));
    b = clampB(B_RANGE[0] + ((M.t + PH - pt.y) / PH) * (B_RANGE[1] - B_RANGE[0]));
    refresh();
  };
  svg.addEventListener('pointerdown', event => {
    if (!model) return;
    svg.classList.add('pointer');
    svg.setPointerCapture(event.pointerId);
    fromPointer(event);
  });
  svg.addEventListener('pointermove', event => {
    if (model && svg.hasPointerCapture(event.pointerId)) fromPointer(event);
  });
  svg.addEventListener('blur', () => svg.classList.remove('pointer'));
  svg.addEventListener('keydown', event => {
    if (!model) return;
    const step = event.shiftKey ? 0.5 : 0.05;
    if (event.key === 'ArrowRight') w = clampW(w + step);
    else if (event.key === 'ArrowLeft') w = clampW(w - step);
    else if (event.key === 'ArrowUp') b = clampB(b + step);
    else if (event.key === 'ArrowDown') b = clampB(b - step);
    else return;
    event.preventDefault();
    svg.classList.remove('pointer');
    refresh();
  });

  // Redraw the shading when the theme changes while the lesson is open.
  const themeWatch = new MutationObserver(() => {
    if (!host.contains(svg)) { themeWatch.disconnect(); return; }
    if (model) { buildHeat(); surface.refreshColors(); refresh(); }
  });
  themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  const editor = wireEditor(host.querySelector('.calc-editor'), text => {
    const result = analyse(text, ['w', 'b']);
    if (result.error) return result.error;
    model = result;
    renderTex(mathBox.children[0], `L(w, b) = ${model.tex}`);
    renderTex(mathBox.children[1], `\\dfrac{\\partial L}{\\partial w} = ${model.d.w.tex}`);
    renderTex(mathBox.children[2], `\\dfrac{\\partial L}{\\partial b} = ${model.d.b.tex}`);
    buildHeat();
    surface.setModel({ fn: model.fn, rank: rankFn, zRange: rangeL });
    refresh();
    return null;
  }, mathBox);
  editor.run();
}
