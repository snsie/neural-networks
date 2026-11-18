/* Builds the forward/backward walkthrough as KaTeX strings.
 * mode 'symbolic'  -> general notation: f(z), ∂a/∂z, σ(z), softmax(z) ...
 * mode 'formatted' -> the selected functions written out, with this sample's numbers plugged in. */
(function () {
  'use strict';
  const { f3, P } = Activations;

  const cls = (c, s) => `\\htmlClass{${c}}{${s}}`;
  const val = v => cls('eq-a', f3(v));
  const valP = v => cls('eq-a', P(v));
  const par = v => cls('eq-w', f3(v));
  const parP = v => cls('eq-w', P(v));
  const grd = v => cls('eq-g', f3(v));
  const grdP = v => cls('eq-g', P(v));

  const col = arr => `\\begin{bmatrix}${arr.join(' \\\\ ')}\\end{bmatrix}`;
  const mat = rows => `\\begin{bmatrix}${rows.map(r => r.join(' & ')).join(' \\\\ ')}\\end{bmatrix}`;
  const aligned = rows => `\\begin{aligned}${rows.join(' \\\\[3pt] ')}\\end{aligned}`;
  const signed = (v, wrap) => (v < 0 && Math.abs(v) >= 5e-4 ? `- ${wrap(-v)}` : `+ ${wrap(v)}`);
  /** "expr = expr = result", dropping intermediate steps that just repeat the result. */
  const chain = (parts, resultTex, resultVal) => [...parts.filter(p => p !== f3(resultVal)), resultTex].join(' = ');

  const S1 = '^{(1)}', S2 = '^{(2)}';
  const dfrac = (a, b) => `\\frac{\\partial ${a}}{\\partial ${b}}`;

  function build(ctx) {
    const { net, cache: c, grads: g, label, mode, lr, optimizer, classNames } = ctx;
    const act = net.act, H = net.H, O = net.O, sig = net.outMode === 'sigmoid';
    const F = mode === 'formatted';
    const steps = [];
    const n1 = i => i + 1;
    const cname = k => `\\text{${classNames[k]}}`;

    /* ================= FORWARD ================= */

    // 1. input
    steps.push({
      id: 'input', phase: 'fwd', title: 'Input sample', short: 'x, y',
      blocks: F ? [
        `\\mathbf{x} = ${col([val(c.x[0]), val(c.x[1])])}`,
        sig ? `y = ${label} \\quad (${cname(label)})`
            : `\\mathbf{y} = ${col(g.t.map(String))} \\quad (\\text{one-hot for } ${cname(label)})`,
      ] : [
        `\\mathbf{x} = \\begin{bmatrix} x_1 \\\\ x_2 \\end{bmatrix}`,
        sig ? `y \\in \\{0,\\,1\\}` : `\\mathbf{y} \\in \\{0,1\\}^{${O}} \\quad \\text{(one-hot class label)}`,
      ],
      note: sig
        ? `Each example is a point (x₁, x₂). With two classes, one number is enough for the label: y = 1 means ${classNames[1]} and y = 0 means ${classNames[0]}.`
        : `Each example is a point (x₁, x₂). With ${O} classes, the label is one-hot encoded: a 1 in the true class’s slot and 0 everywhere else.`,
      hl: { nodes: ['in'], edges: [], dir: 'fwd' },
      vals: { in: { arr: c.x, label: 'x', kind: 'fwd' } },
    });

    // 2. hidden pre-activation
    {
      const rows = [];
      for (let j = 0; j < H; j++) {
        rows.push(`z${S1}_{${n1(j)}} &= ${parP(net.W1[j][0])} \\cdot ${valP(c.x[0])} + ${parP(net.W1[j][1])} \\cdot ${valP(c.x[1])} ${signed(net.b1[j], par)} = ${val(c.z1[j])}`);
      }
      steps.push({
        id: 'z1', phase: 'fwd', title: 'Hidden layer: weighted sum', short: 'z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾',
        blocks: F ? [
          `\\mathbf{z}${S1} = ${mat(net.W1.map(r => r.map(par)))}\\,${col(c.x.map(val))} + ${col(net.b1.map(par))} = ${col(c.z1.map(val))}`,
          aligned(rows),
        ] : [
          `\\mathbf{z}${S1} = W${S1}\\,\\mathbf{x} + \\mathbf{b}${S1}`,
          `z${S1}_j = w${S1}_{j1}\\,x_1 + w${S1}_{j2}\\,x_2 + b${S1}_j, \\qquad j = 1,\\dots,${H}`,
          `W${S1} \\in \\mathbb{R}^{${H}\\times 2}, \\qquad \\mathbf{b}${S1} \\in \\mathbb{R}^{${H}}`,
        ],
        note: `Each hidden neuron takes a weighted sum of the two inputs and adds its bias. Row j of W⁽¹⁾ holds neuron j’s two input weights. The result z is called the pre-activation.`,
        hl: { nodes: ['in', 'hid'], edges: ['e1'], dir: 'fwd' },
        vals: { in: { arr: c.x, label: 'x', kind: 'fwd' }, hid: { arr: c.z1, label: 'z', kind: 'fwd' } },
      });
    }

    // 3. hidden activation
    {
      const rows = [];
      for (let j = 0; j < H; j++) rows.push(`a${S1}_{${n1(j)}} &= ${chain(act.subF(c.z1[j]), val(c.a1[j]), c.a1[j])}`);
      steps.push({
        id: 'a1', phase: 'fwd', title: `Hidden layer: ${act.name} activation`, short: 'a⁽¹⁾ = f(z⁽¹⁾)',
        blocks: F ? [
          `f(z) = ${act.texF}`,
          aligned(rows),
        ] : [
          `\\mathbf{a}${S1} = f\\big(\\mathbf{z}${S1}\\big)`,
          `a${S1}_j = f\\big(z${S1}_j\\big) \\qquad \\text{(applied to each neuron separately)}`,
        ],
        note: `${act.name} is applied to each neuron on its own. This nonlinearity is what lets the network bend its decision boundary. ${act.blurb}`,
        hl: { nodes: ['hid'], edges: [], dir: 'fwd' },
        vals: { hid: { arr: c.a1, label: 'a', kind: 'fwd' } },
      });
    }

    // 4. output pre-activation
    {
      const rows = [];
      for (let k = 0; k < O; k++) {
        const terms = c.a1.map((a, j) => `${parP(net.W2[k][j])} \\cdot ${valP(a)}`).join(' + ');
        rows.push(`z${S2}_{${n1(k)}} &= ${terms} ${signed(net.b2[k], par)} = ${val(c.z2[k])}`);
      }
      steps.push({
        id: 'z2', phase: 'fwd', title: 'Output layer: weighted sum', short: 'z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾',
        blocks: F ? [
          `\\mathbf{z}${S2} = ${mat(net.W2.map(r => r.map(par)))}\\,${col(c.a1.map(val))} + ${col(net.b2.map(par))} = ${col(c.z2.map(val))}`,
          aligned(rows),
        ] : [
          `\\mathbf{z}${S2} = W${S2}\\,\\mathbf{a}${S1} + \\mathbf{b}${S2}`,
          `z${S2}_k = \\sum_{j=1}^{${H}} w${S2}_{kj}\\,a${S1}_j + b${S2}_k, \\qquad k = ${O === 1 ? '1' : `1,\\dots,${O}`}`,
          `W${S2} \\in \\mathbb{R}^{${O}\\times ${H}}`,
        ],
        note: sig
          ? 'With two classes the output layer has a single neuron. It combines all hidden activations into one score, often called a logit.'
          : `With ${O} classes there is one output neuron per class. Each one combines all hidden activations into a score (logit) for its class.`,
        hl: { nodes: ['hid', 'out'], edges: ['e2'], dir: 'fwd' },
        vals: { hid: { arr: c.a1, label: 'a', kind: 'fwd' }, out: { arr: c.z2, label: 'z', kind: 'fwd' } },
      });
    }

    // 5. prediction
    if (sig) {
      const p = c.yhat[0];
      steps.push({
        id: 'yhat', phase: 'fwd', title: 'Prediction: sigmoid', short: 'ŷ = σ(z⁽²⁾)',
        blocks: F ? [
          `\\sigma(z) = \\frac{1}{1+e^{-z}}`,
          `\\hat{y} = \\frac{1}{1+e^{-${P(c.z2[0])}}} = ${val(p)}`,
          `P(${cname(1)} \\mid \\mathbf{x}) = ${val(p)}, \\qquad P(${cname(0)} \\mid \\mathbf{x}) = 1-\\hat{y} = ${val(1 - p)}`,
        ] : [
          `\\hat{y} = \\sigma\\big(z${S2}\\big)`,
          `\\hat{y} = P(y = 1 \\mid \\mathbf{x})`,
        ],
        note: 'For a two-class dataset the output activation is the sigmoid. It squashes the logit into (0, 1) so it can be read as the probability of class 1.',
        hl: { nodes: ['out'], edges: [], dir: 'fwd' },
        vals: { out: { arr: c.yhat, label: 'ŷ', kind: 'fwd' } },
      });
    } else {
      const ex = c.z2.map(Math.exp);
      const sum = ex.reduce((a, b) => a + b, 0);
      const rows = c.z2.map((z, k) => `\\hat{y}_{${n1(k)}} &= \\frac{e^{${f3(z)}}}{${f3(sum)}} = \\frac{${f3(ex[k])}}{${f3(sum)}} = ${val(c.yhat[k])} \\quad (${cname(k)})`);
      steps.push({
        id: 'yhat', phase: 'fwd', title: 'Prediction: softmax', short: 'ŷ = softmax(z⁽²⁾)',
        blocks: F ? [
          `\\hat{y}_k = \\frac{e^{z${S2}_k}}{\\sum_{m=1}^{${O}} e^{z${S2}_m}}, \\qquad \\sum_{m} e^{z_m} = ${c.z2.map(z => `e^{${f3(z)}}`).join(' + ')} = ${f3(sum)}`,
          aligned(rows),
        ] : [
          `\\hat{\\mathbf{y}} = \\operatorname{softmax}\\big(\\mathbf{z}${S2}\\big)`,
          `\\hat{y}_k = \\operatorname{softmax}\\big(\\mathbf{z}${S2}\\big)_k = P(\\text{class } k \\mid \\mathbf{x})`,
        ],
        note: `For a ${O}-class dataset the output activation is softmax. It exponentiates every logit and normalizes, so the outputs are positive and sum to 1, giving a probability for each class.`,
        hl: { nodes: ['out'], edges: [], dir: 'fwd' },
        vals: { out: { arr: c.yhat, label: 'ŷ', kind: 'fwd' } },
      });
    }

    // 6. loss
    const L = net.loss(c, label);
    if (sig) {
      const p = c.yhat[0];
      steps.push({
        id: 'loss', phase: 'fwd', title: 'Loss: binary cross-entropy', short: 'L(ŷ, y)',
        blocks: F ? [
          `L = -\\big[\\, ${label} \\cdot \\log(${val(p)}) + ${1 - label} \\cdot \\log(${val(1 - p)}) \\,\\big]`,
          `L = ${label === 1 ? `-\\log(${f3(p)})` : `-\\log(${f3(1 - p)})`} = ${val(L)}`,
        ] : [
          `L = -\\big[\\, y \\log \\hat{y} + (1-y) \\log(1-\\hat{y}) \\,\\big]`,
        ],
        note: 'Cross-entropy is small when the network gives the true class a high probability, and it grows quickly as that probability approaches 0. Only one of the two terms is active for any given sample.',
        hl: { nodes: ['out', 'loss'], edges: ['el'], dir: 'fwd' },
        vals: { out: { arr: c.yhat, label: 'ŷ', kind: 'fwd' }, loss: { arr: [L], label: 'L', kind: 'fwd' } },
      });
    } else {
      const terms = c.yhat.map((p, k) => `${g.t[k]} \\cdot \\log(${val(p)})`).join(' + ');
      steps.push({
        id: 'loss', phase: 'fwd', title: 'Loss: categorical cross-entropy', short: 'L(ŷ, y)',
        blocks: F ? [
          `L = -\\big[\\, ${terms} \\,\\big]`,
          `L = -\\log(${f3(c.yhat[label])}) = ${val(L)}`,
        ] : [
          `L = -\\sum_{k=1}^{${O}} y_k \\log \\hat{y}_k`,
          `L = -\\log \\hat{y}_{c} \\qquad \\text{where } c \\text{ is the true class}`,
        ],
        note: 'Because y is one-hot, only the true class’s term survives. The loss is the negative log of the probability the network gave the correct answer.',
        hl: { nodes: ['out', 'loss'], edges: ['el'], dir: 'fwd' },
        vals: { out: { arr: c.yhat, label: 'ŷ', kind: 'fwd' }, loss: { arr: [L], label: 'L', kind: 'fwd' } },
      });
    }

    /* ================= BACKWARD ================= */

    // 7. output error
    if (sig) {
      const p = c.yhat[0];
      const dLdy = -label / p + (1 - label) / (1 - p);
      steps.push({
        id: 'dz2', phase: 'bwd', title: 'Output error', short: 'δ⁽²⁾ = ŷ − y',
        blocks: F ? [
          `${dfrac('L', '\\hat{y}')} = -\\frac{y}{\\hat{y}} + \\frac{1-y}{1-\\hat{y}} = -\\frac{${label}}{${f3(p)}} + \\frac{${1 - label}}{${f3(1 - p)}} = ${grd(dLdy)}`,
          `${dfrac('\\hat{y}', `z${S2}`)} = \\sigma'(z${S2}) = \\hat{y}\\,(1-\\hat{y}) = ${f3(p)} \\cdot ${f3(1 - p)} = ${val(p * (1 - p))}`,
          `\\delta${S2} = ${grdP(dLdy)} \\cdot ${valP(p * (1 - p))} = \\hat{y} - y = ${f3(p)} - ${label} = ${grd(g.dz2[0])}`,
        ] : [
          `\\delta${S2} = ${dfrac('L', `z${S2}`)} = ${dfrac('L', '\\hat{y}')} \\cdot ${dfrac('\\hat{y}', `z${S2}`)}`,
          `\\delta${S2} = \\hat{y} - y`,
        ],
        note: 'Backpropagation starts at the loss and applies the chain rule. For sigmoid with binary cross-entropy, the two derivatives cancel to give the simple error ŷ − y. This pairing is the reason the dataset’s class count picks the output activation.',
        hl: { nodes: ['loss', 'out'], edges: ['el'], dir: 'bwd' },
        vals: { out: { arr: g.dz2, label: 'δ', kind: 'bwd' }, loss: { arr: [L], label: 'L', kind: 'fwd' } },
      });
    } else {
      const rows = c.yhat.map((p, k) => `\\delta${S2}_{${n1(k)}} &= ${f3(p)} - ${g.t[k]} = ${grd(g.dz2[k])}`);
      steps.push({
        id: 'dz2', phase: 'bwd', title: 'Output error', short: 'δ⁽²⁾ = ŷ − y',
        blocks: F ? [
          `${dfrac('\\hat{y}_m', 'z_k')} = \\hat{y}_m\\big([m{=}k] - \\hat{y}_k\\big), \\qquad ${dfrac('L', '\\hat{y}_m')} = -\\frac{y_m}{\\hat{y}_m}`,
          `${dfrac('L', 'z_k')} = \\sum_m -\\frac{y_m}{\\hat{y}_m}\\,\\hat{y}_m\\big([m{=}k] - \\hat{y}_k\\big) = \\hat{y}_k \\textstyle\\sum_m y_m - y_k = \\hat{y}_k - y_k`,
          aligned(rows),
        ] : [
          `\\boldsymbol{\\delta}${S2} = ${dfrac('L', `\\mathbf{z}${S2}`)} = \\Big(${dfrac('\\hat{\\mathbf{y}}', `\\mathbf{z}${S2}`)}\\Big)^{\\!\\top} ${dfrac('L', '\\hat{\\mathbf{y}}')}`,
          `\\boldsymbol{\\delta}${S2} = \\hat{\\mathbf{y}} - \\mathbf{y}`,
        ],
        note: 'Backpropagation starts at the loss and applies the chain rule. For softmax with cross-entropy, the softmax Jacobian and the log-loss derivative combine to give ŷ − y. The output for the true class gets a negative error that pushes its score up, and every other class gets a positive error that pushes its score down.',
        hl: { nodes: ['loss', 'out'], edges: ['el'], dir: 'bwd' },
        vals: { out: { arr: g.dz2, label: 'δ', kind: 'bwd' }, loss: { arr: [L], label: 'L', kind: 'fwd' } },
      });
    }

    // 8. output-layer gradients
    {
      const rows = c.a1.map((a, j) => `${dfrac('L', `w${S2}_{1${n1(j)}}`)} &= ${grdP(g.dz2[0])} \\cdot ${valP(a)} = ${grd(g.dW2[0][j])}`);
      steps.push({
        id: 'dW2', phase: 'bwd', title: 'Output-layer gradients', short: '∂L/∂W⁽²⁾ = δ⁽²⁾a⁽¹⁾ᵀ',
        blocks: F ? [
          `${dfrac('L', `W${S2}`)} = ${col(g.dz2.map(grd))}\\,${mat([c.a1.map(val)])} = ${mat(g.dW2.map(r => r.map(grd)))}`,
          `${dfrac('L', `\\mathbf{b}${S2}`)} = ${col(g.db2.map(grd))}`,
          aligned(rows),
        ] : [
          `${dfrac('L', `w${S2}_{kj}`)} = ${dfrac('L', `z${S2}_k`)} \\cdot ${dfrac(`z${S2}_k`, `w${S2}_{kj}`)} = \\delta${S2}_k\\, a${S1}_j`,
          `${dfrac('L', `W${S2}`)} = \\boldsymbol{\\delta}${S2}\\,\\mathbf{a}^{(1)\\top}, \\qquad ${dfrac('L', `\\mathbf{b}${S2}`)} = \\boldsymbol{\\delta}${S2}`,
        ],
        note: `A weight’s gradient is the error at its destination times the activation at its source. Large activations feeding a large error get the biggest updates.${O > 1 ? ' The rows written out below are for output neuron 1.' : ''}`,
        hl: { nodes: ['hid', 'out'], edges: ['e2'], dir: 'bwd' },
        vals: { hid: { arr: c.a1, label: 'a', kind: 'fwd' }, out: { arr: g.dz2, label: 'δ', kind: 'bwd' } },
      });
    }

    // 9. backprop into the hidden layer
    {
      const rows = [];
      for (let j = 0; j < H; j++) {
        const terms = g.dz2.map((d, k) => `${parP(net.W2[k][j])} \\cdot ${grdP(d)}`).join(' + ');
        rows.push(`${dfrac('L', `a${S1}_{${n1(j)}}`)} &= ${terms} = ${grd(g.da1[j])}`);
      }
      const W2T = Array.from({ length: H }, (_, j) => net.W2.map(r => par(r[j])));
      steps.push({
        id: 'da1', phase: 'bwd', title: 'Send the error back to the hidden layer', short: '∂L/∂a⁽¹⁾ = W⁽²⁾ᵀδ⁽²⁾',
        blocks: F ? [
          `${dfrac('L', `\\mathbf{a}${S1}`)} = W^{(2)\\top}\\boldsymbol{\\delta}${S2} = ${mat(W2T)}\\,${col(g.dz2.map(grd))} = ${col(g.da1.map(grd))}`,
          aligned(rows),
        ] : [
          `${dfrac('L', `a${S1}_j`)} = \\sum_{k=1}^{${O}} ${dfrac('L', `z${S2}_k`)} \\cdot ${dfrac(`z${S2}_k`, `a${S1}_j`)} = \\sum_{k} \\delta${S2}_k\\, w${S2}_{kj}`,
          `${dfrac('L', `\\mathbf{a}${S1}`)} = W^{(2)\\top}\\boldsymbol{\\delta}${S2}`,
        ],
        note: 'Each hidden neuron’s share of the blame is the output error sent back along the same weights used in the forward pass. Multiplying by the transpose of W⁽²⁾ runs those connections in reverse.',
        hl: { nodes: ['hid', 'out'], edges: ['e2'], dir: 'bwd' },
        vals: { hid: { arr: g.da1, label: '∂a', kind: 'bwd' }, out: { arr: g.dz2, label: 'δ', kind: 'bwd' } },
      });
    }

    // 10. through the activation (the key step for the activation toggle)
    {
      const rows = [];
      for (let j = 0; j < H; j++) {
        rows.push(
          `${dfrac(`a${S1}_{${n1(j)}}`, `z${S1}_{${n1(j)}}`)} &= ${chain(act.subDf(c.z1[j]), val(g.dadz[j]), g.dadz[j])}` +
          ` &\\qquad \\delta${S1}_{${n1(j)}} &= ${grdP(g.da1[j])} \\cdot ${valP(g.dadz[j])} = ${grd(g.dz1[j])}`);
      }
      steps.push({
        id: 'dz1', phase: 'bwd', title: `Back through the ${act.name} activation`, short: 'δ⁽¹⁾ = ∂L/∂a ⊙ ∂a/∂z',
        blocks: F ? [
          `${dfrac('a', 'z')} = f'(z) = ${act.texDf}, \\qquad \\delta${S1}_j = ${dfrac('L', `a${S1}_j`)} \\cdot f'\\big(z${S1}_j\\big)`,
          aligned(rows),
        ] : [
          `\\boldsymbol{\\delta}${S1} = ${dfrac('L', `\\mathbf{z}${S1}`)} = ${dfrac('L', `\\mathbf{a}${S1}`)} \\odot ${dfrac(`\\mathbf{a}${S1}`, `\\mathbf{z}${S1}`)}`,
          `\\delta${S1}_j = ${dfrac('L', `a${S1}_j`)} \\cdot ${dfrac(`a${S1}_j`, `z${S1}_j`)}`,
        ],
        note: `The error has to pass back through the activation function, so it gets multiplied by the slope of f at each neuron’s z. ${act.gradNote}`,
        hl: { nodes: ['hid'], edges: [], dir: 'bwd' },
        vals: { hid: { arr: g.dz1, label: 'δ', kind: 'bwd' } },
      });
    }

    // 11. hidden-layer gradients
    {
      const rows = [];
      for (let j = 0; j < H; j++) {
        rows.push(`${dfrac('L', `w${S1}_{${n1(j)}1}`)} &= ${grdP(g.dz1[j])} \\cdot ${valP(c.x[0])} = ${grd(g.dW1[j][0])} & ${dfrac('L', `w${S1}_{${n1(j)}2}`)} &= ${grdP(g.dz1[j])} \\cdot ${valP(c.x[1])} = ${grd(g.dW1[j][1])}`);
      }
      steps.push({
        id: 'dW1', phase: 'bwd', title: 'Hidden-layer gradients', short: '∂L/∂W⁽¹⁾ = δ⁽¹⁾xᵀ',
        blocks: F ? [
          `${dfrac('L', `W${S1}`)} = ${col(g.dz1.map(grd))}\\,${mat([[val(c.x[0]), val(c.x[1])]])} = ${mat(g.dW1.map(r => r.map(grd)))}, \\qquad ${dfrac('L', `\\mathbf{b}${S1}`)} = ${col(g.db1.map(grd))}`,
          aligned(rows.map(r => r.replace(' & \\frac', ' &\\qquad \\frac'))),
        ] : [
          `${dfrac('L', `w${S1}_{ji}`)} = ${dfrac('L', `z${S1}_j`)} \\cdot ${dfrac(`z${S1}_j`, `w${S1}_{ji}`)} = \\delta${S1}_j\\, x_i`,
          `${dfrac('L', `W${S1}`)} = \\boldsymbol{\\delta}${S1}\\,\\mathbf{x}^{\\top}, \\qquad ${dfrac('L', `\\mathbf{b}${S1}`)} = \\boldsymbol{\\delta}${S1}`,
        ],
        note: 'The first layer follows the same pattern as the output layer: error at the destination times the value at the source. Here the sources are the raw inputs x₁ and x₂.',
        hl: { nodes: ['in', 'hid'], edges: ['e1'], dir: 'bwd' },
        vals: { in: { arr: c.x, label: 'x', kind: 'fwd' }, hid: { arr: g.dz1, label: 'δ', kind: 'bwd' } },
      });
    }

    // 12. update
    {
      const upd = (P0, G0) => P0.map((r, i) => (Array.isArray(r) ? r.map((w, j) => par(w - lr * G0[i][j])) : par(r - lr * G0[i])));
      const lrs = String(lr);
      steps.push({
        id: 'update', phase: 'bwd', title: 'Gradient descent update', short: 'θ ← θ − η ∂L/∂θ',
        blocks: F ? [
          `\\eta = ${lrs} \\qquad \\text{(one SGD step on this sample)}`,
          `W${S2} \\leftarrow ${mat(net.W2.map(r => r.map(par)))} - ${lrs}\\,${mat(g.dW2.map(r => r.map(grd)))} = ${mat(upd(net.W2, g.dW2))}`,
          `\\mathbf{b}${S2} \\leftarrow ${col(net.b2.map(par))} - ${lrs}\\,${col(g.db2.map(grd))} = ${col(upd(net.b2, g.db2))}`,
          `W${S1} \\leftarrow ${mat(net.W1.map(r => r.map(par)))} - ${lrs}\\,${mat(g.dW1.map(r => r.map(grd)))} = ${mat(upd(net.W1, g.dW1))}`,
          `\\mathbf{b}${S1} \\leftarrow ${col(net.b1.map(par))} - ${lrs}\\,${col(g.db1.map(grd))} = ${col(upd(net.b1, g.db1))}`,
        ] : [
          `\\theta \\leftarrow \\theta - \\eta\\, ${dfrac('L', '\\theta')}, \\qquad \\theta \\in \\{\\, W${S1},\\, \\mathbf{b}${S1},\\, W${S2},\\, \\mathbf{b}${S2} \\,\\}`,
          optimizer === 'adam'
            ? `\\text{Adam:}\\quad \\mathbf{m} \\leftarrow \\beta_1 \\mathbf{m} + (1-\\beta_1)\\nabla_\\theta L, \\quad \\mathbf{v} \\leftarrow \\beta_2 \\mathbf{v} + (1-\\beta_2)(\\nabla_\\theta L)^2, \\quad \\theta \\leftarrow \\theta - \\eta\\,\\frac{\\hat{\\mathbf{m}}}{\\sqrt{\\hat{\\mathbf{v}}}+\\epsilon}`
            : `\\text{SGD with mini-batches: } \\nabla_\\theta L = \\frac{1}{B}\\sum_{b=1}^{B} \\nabla_\\theta L_b`,
        ],
        note: optimizer === 'adam'
          ? 'Each parameter takes a small step downhill on the loss. The formatted view shows a plain SGD step for this one sample. During training the app averages gradients over mini-batches of 10 and uses Adam, which scales each parameter’s step using running averages of its gradients.'
          : 'Each parameter takes a small step downhill on the loss. The formatted view shows the step for this one sample. During training the app averages these gradients over mini-batches of 10 samples before stepping.',
        hl: { nodes: ['in', 'hid', 'out'], edges: ['e1', 'e2'], dir: 'bwd' },
        vals: {},
      });
    }

    return steps;
  }

  window.Equations = { build };
})();
