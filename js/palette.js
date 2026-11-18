/* Reads the CSS color tokens so canvas drawing follows the active theme. */
(function () {
  'use strict';

  function hexToRgb(h) {
    h = (h || '').trim().replace('#', '');
    if (h.length === 3) h = h.split('').map(c => c + c).join('');
    const n = parseInt(h, 16) || 0;
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function read() {
    const cs = getComputedStyle(document.documentElement);
    const g = name => hexToRgb(cs.getPropertyValue(name));
    return {
      c: [g('--c0'), g('--c1'), g('--c2')],
      surface: g('--surface'),
      surface2: g('--surface-2'),
      text: g('--text'),
      muted: g('--muted'),
      faint: g('--faint'),
      border: g('--border'),
      pos: g('--pos'),
      neg: g('--neg'),
      fwd: g('--fwd'),
      bwd: g('--bwd'),
    };
  }

  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const css = (c, a = 1) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;

  window.Palette = { read, mix, css, hexToRgb };
})();
