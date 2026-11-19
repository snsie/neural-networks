/* Hidden-layer activation functions: math, derivatives, and their LaTeX. */
(function () {
  'use strict';

  // Number formatting shared with the equation builder.
  function f3(v) {
    if (!isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
    if (Math.abs(v) >= 1e4) {
      const [m, e] = v.toExponential(2).split('e');
      return `${m}\\times10^{${+e}}`;
    }
    return (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);
  }
  function number(v) {
    if (v !== 0 && Math.abs(v) < 0.001) {
      const [m, e] = v.toExponential(2).split('e');
      return String.raw`${m}\times10^{${+e}}`;
    }
    return v > 0.9995 && v < 1 ? String(v) : f3(v);
  }

  // Parenthesize negative operands in products and powers.
  const P = v => v < 0 ? `(${number(v)})` : number(v);

  const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  const ALPHA = 0.1;

  // texF / texDf are right-hand sides in terms of z.
  // subF / subDf return intermediate LaTeX strings with the number substituted in.
  const LIST = [
    {
      id: 'relu', name: 'ReLU', init: 'he',
      f: z => (z > 0 ? z : 0),
      df: z => (z > 0 ? 1 : 0),
      texF: '\\max(0,\\, z)',
      texDf: '\\begin{cases} 1 & z > 0 \\\\ 0 & z \\le 0 \\end{cases}',
      subF: z => [`\\max(0,\\, ${number(z)})`],
      subDf: z => [`\\mathbf{1}\\big[\\,${number(z)} > 0\\,\\big]`],
      blurb: 'Passes positive inputs straight through and zeroes out negatives. It is cheap and does not saturate for z > 0, but a neuron stuck at z ≤ 0 gets zero gradient and can “die”.',
      gradNote: 'ReLU’s derivative is a switch: 1 when the neuron was active (z > 0) and 0 otherwise. Inactive neurons block the gradient completely for this sample.',
    },
    {
      id: 'leaky', name: 'Leaky ReLU', init: 'he',
      f: z => (z > 0 ? z : ALPHA * z),
      df: z => (z > 0 ? 1 : ALPHA),
      texF: '\\max(0.1\\,z,\\; z)',
      texDf: '\\begin{cases} 1 & z > 0 \\\\ 0.1 & z \\le 0 \\end{cases}',
      subF: z => [`\\max(0.1 \\cdot ${P(z)},\\; ${number(z)})`],
      subDf: z => [z > 0 ? '1' : '0.1'],
      blurb: 'Like ReLU, but negative inputs leak through with slope α = 0.1. Every neuron keeps a nonzero gradient, so none of them can die.',
      gradNote: 'The derivative is 1 for active neurons and α = 0.1 otherwise, so even “off” neurons still pass a small gradient back.',
    },
    {
      id: 'tanh', name: 'Tanh', init: 'xavier',
      f: z => Math.tanh(z),
      df: z => 1 - Math.tanh(z) ** 2,
      texF: '\\tanh(z)',
      texDf: '1-\\tanh^2(z)',
      subF: z => [`\\tanh(${number(z)})`],
      subDf: z => [`1-\\tanh^2(${number(z)})`, `1-${P(Math.tanh(z))}^2`],
      blurb: 'A zero-centered S-curve with outputs in (−1, 1). It is smooth and well behaved for small z, but saturates (flattens out) for large |z|.',
      gradNote: 'tanh′ peaks at 1 when z = 0 and falls toward 0 as |z| grows. Saturated neurons pass back very little gradient.',
    },
    {
      id: 'sigmoid', name: 'Sigmoid', init: 'xavier',
      f: sigmoid,
      df: z => { const s = sigmoid(z); return s * (1 - s); },
      texF: '\\frac{1}{1+e^{-z}}',
      texDf: '\\sigma(z)\\,\\big(1-\\sigma(z)\\big)',
      subF: z => [`\\frac{1}{1+e^{-${P(z)}}}`],
      subDf: z => {
        const s = sigmoid(z);
        return [`\\sigma(${number(z)})\\,\\big(1-\\sigma(${number(z)})\\big)`, `${number(s)} \\cdot ${number(1 - s)}`];
      },
      blurb: 'Squashes z into (0, 1). It is historically important, but its outputs are not zero-centered and its slope is never more than 0.25, so gradients shrink layer by layer.',
      gradNote: 'σ′(z) = σ(z)(1 − σ(z)) is at most 0.25, so every pass through a sigmoid shrinks the gradient by at least 4×.',
    },
    {
      id: 'elu', name: 'ELU', init: 'he',
      f: z => (z > 0 ? z : Math.exp(z) - 1),
      df: z => (z > 0 ? 1 : Math.exp(z)),
      texF: '\\begin{cases} z & z > 0 \\\\ e^{z}-1 & z \\le 0 \\end{cases}',
      texDf: '\\begin{cases} 1 & z > 0 \\\\ e^{z} & z \\le 0 \\end{cases}',
      subF: z => [z > 0 ? number(z) : `e^{${number(z)}}-1`],
      subDf: z => [z > 0 ? '1' : `e^{${number(z)}}`],
      blurb: 'Matches ReLU for positive z, with a smooth exponential curve toward −1 for negative z. Outputs are closer to zero-mean and the gradient never fully vanishes.',
      gradNote: 'For z > 0 the derivative is 1. For z ≤ 0 it is eᶻ, which shrinks smoothly but never reaches exactly 0.',
    },
    {
      id: 'linear', name: 'Linear', init: 'xavier',
      f: z => z,
      df: () => 1,
      texF: 'z',
      texDf: '1',
      subF: z => [number(z)],
      subDf: () => ['1'],
      blurb: 'No nonlinearity at all. Stacked linear layers collapse into one linear map, so the network can only draw a straight decision boundary. Try it on Circle or XOR.',
      gradNote: 'With a linear activation ∂a/∂z = 1 everywhere, so the gradient passes through unchanged. Without a nonlinearity, though, the whole network is still just a linear model.',
    },
  ];

  window.Activations = {
    list: LIST,
    get: id => LIST.find(a => a.id === id) || LIST[0],
    f3, number, P, sigmoid,
  };
})();
