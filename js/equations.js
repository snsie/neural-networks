/* Builds the forward/backward walkthrough as KaTeX strings.
 * mode 'symbolic'  -> general notation: f(z), ∂a/∂z, σ(z), softmax(z) ...
 * mode 'formatted' -> the selected functions written out, with this sample's numbers plugged in.
 *
 * Layers are indexed l = 0..L in code (L = number of hidden layers, l = L is the output),
 * and written with superscripts (1)..(L+1) in the math. */
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
  const transpose = M => M[0].map((_, i) => M.map(r => r[i]));

  const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  const usup = n => `⁽${String(n).split('').map(d => SUP[+d]).join('')}⁾`;
  const S = n => `^{(${n})}`;
  const dfrac = (a, b) => `\\frac{\\partial ${a}}{\\partial ${b}}`;

  function build(ctx) {
    const { net, cache: c, grads: g, label, mode, lr, optimizer, classNames } = ctx;
    const act = net.act, H = net.H, O = net.O, L = net.L, sig = net.outMode === 'sigmoid';
    const F = mode === 'formatted';
    const steps = [];
    const n1 = i => i + 1;
    const cname = k => `\\text{${classNames[k]}}`;
    const So = S(L + 1);

    /* ---- per-layer naming helpers (l = weight-layer index 0..L) ---- */
    const layerName = l => (l === L ? 'Output layer' : L === 1 ? 'Hidden layer' : `Hidden layer ${l + 1}`);
    const layerLower = l => layerName(l).replace(/^./, ch => ch.toLowerCase());
    const nodeKey = l => (l === L ? 'out' : `h${l}`);            // nodes produced by layer l
    const srcKey = l => (l === 0 ? 'in' : `h${l - 1}`);           // nodes feeding layer l
    const inVec = l => (l === 0 ? '\\mathbf{x}' : `\\mathbf{a}${S(l)}`);
    const inElJ = l => (l === 0 ? 'x_i' : `a${S(l)}_i`);
    const inText = l => (l === 0 ? 'x' : `a${usup(l)}`);
    const inDim = l => (l === 0 ? 2 : H);
    const inVals = l => net.input(c, l);
    const srcVals = l => ({ arr: inVals(l), label: l === 0 ? 'x' : 'a', kind: 'fwd' });

    /* ================= FORWARD ================= */

    // input
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

    // hidden layers: weighted sum + activation
    for (let l = 0; l < L; l++) {
      const s = S(l + 1), inp = inVals(l), Wl = net.W[l], bl = net.b[l], key = nodeKey(l);
      const rows = [];
      for (let j = 0; j < H; j++) {
        const terms = inp.map((v, i) => `${parP(Wl[j][i])} \\cdot ${valP(v)}`).join(' + ');
        rows.push(`z${s}_{${n1(j)}} &= ${terms} ${signed(bl[j], par)} = ${val(c.z[l][j])}`);
      }
      const first = l === 0;
      steps.push({
        id: `z${l + 1}`, phase: 'fwd', title: `${layerName(l)}: weighted sum`,
        short: `z${usup(l + 1)} = W${usup(l + 1)}${inText(l)} + b${usup(l + 1)}`,
        blocks: F ? [
          `\\mathbf{z}${s} = ${mat(Wl.map(r => r.map(par)))}\\,${col(inp.map(val))} + ${col(bl.map(par))} = ${col(c.z[l].map(val))}`,
          aligned(rows),
        ] : [
          `\\mathbf{z}${s} = W${s}\\,${inVec(l)} + \\mathbf{b}${s}`,
          first
            ? `z${s}_j = w${s}_{j1}\\,x_1 + w${s}_{j2}\\,x_2 + b${s}_j, \\qquad j = 1,\\dots,${H}`
            : `z${s}_j = \\sum_{i=1}^{${H}} w${s}_{ji}\\,a${S(l)}_i + b${s}_j, \\qquad j = 1,\\dots,${H}`,
          `W${s} \\in \\mathbb{R}^{${H}\\times ${inDim(l)}}, \\qquad \\mathbf{b}${s} \\in \\mathbb{R}^{${H}}`,
        ],
        note: first
          ? `Each neuron in the ${layerLower(l)} takes a weighted sum of the two inputs and adds its bias. Row j of W⁽¹⁾ holds neuron j’s two input weights. The result z is called the pre-activation.`
          : `This layer works exactly like the one before it, except its inputs are the previous layer’s activations a${usup(l)} instead of the raw inputs. Stacking layers lets the network build more complex features out of simpler ones.`,
        hl: { nodes: [srcKey(l), key], edges: [`e${l}`], dir: 'fwd' },
        vals: { [srcKey(l)]: srcVals(l), [key]: { arr: c.z[l], label: 'z', kind: 'fwd' } },
      });

      const arows = [];
      for (let j = 0; j < H; j++) arows.push(`a${s}_{${n1(j)}} &= ${chain(act.subF(c.z[l][j]), val(c.a[l][j]), c.a[l][j])}`);
      steps.push({
        id: `a${l + 1}`, phase: 'fwd', title: `${layerName(l)}: ${act.name} activation`,
        short: `a${usup(l + 1)} = f(z${usup(l + 1)})`,
        blocks: F ? [
          `f(z) = ${act.texF}`,
          aligned(arows),
        ] : [
          `\\mathbf{a}${s} = f\\big(\\mathbf{z}${s}\\big)`,
          `a${s}_j = f\\big(z${s}_j\\big) \\qquad \\text{(applied to each neuron separately)}`,
        ],
        note: first
          ? `${act.name} is applied to each neuron on its own. This nonlinearity is what lets the network bend its decision boundary. ${act.blurb}`
          : `Every hidden layer uses the same activation, ${act.name}. Without a nonlinearity between them, two stacked layers would collapse into a single linear layer.`,
        hl: { nodes: [key], edges: [], dir: 'fwd' },
        vals: { [key]: { arr: c.a[l], label: 'a', kind: 'fwd' } },
      });
    }

    // output pre-activation
    {
      const aL = inVals(L), WL = net.W[L], bL = net.b[L], zo = c.z[L];
      const rows = [];
      for (let k = 0; k < O; k++) {
        const terms = aL.map((a, j) => `${parP(WL[k][j])} \\cdot ${valP(a)}`).join(' + ');
        rows.push(`z${So}_{${n1(k)}} &= ${terms} ${signed(bL[k], par)} = ${val(zo[k])}`);
      }
      steps.push({
        id: 'zout', phase: 'fwd', title: 'Output layer: weighted sum', short: `z${usup(L + 1)} = W${usup(L + 1)}a${usup(L)} + b${usup(L + 1)}`,
        blocks: F ? [
          `\\mathbf{z}${So} = ${mat(WL.map(r => r.map(par)))}\\,${col(aL.map(val))} + ${col(bL.map(par))} = ${col(zo.map(val))}`,
          aligned(rows),
        ] : [
          `\\mathbf{z}${So} = W${So}\\,\\mathbf{a}${S(L)} + \\mathbf{b}${So}`,
          `z${So}_k = \\sum_{j=1}^{${H}} w${So}_{kj}\\,a${S(L)}_j + b${So}_k, \\qquad k = ${O === 1 ? '1' : `1,\\dots,${O}`}`,
          `W${So} \\in \\mathbb{R}^{${O}\\times ${H}}`,
        ],
        note: sig
          ? `With two classes the output layer has a single neuron. It combines all of the last hidden layer’s activations into one score, often called a logit.`
          : `With ${O} classes there is one output neuron per class. Each one combines all of the last hidden layer’s activations into a score (logit) for its class.`,
        hl: { nodes: [srcKey(L), 'out'], edges: [`e${L}`], dir: 'fwd' },
        vals: { [srcKey(L)]: srcVals(L), out: { arr: zo, label: 'z', kind: 'fwd' } },
      });
    }

    // prediction
    const zo = c.z[L];
    if (sig) {
      const p = c.yhat[0];
      steps.push({
        id: 'yhat', phase: 'fwd', title: 'Prediction: sigmoid', short: `ŷ = σ(z${usup(L + 1)})`,
        blocks: F ? [
          `\\sigma(z) = \\frac{1}{1+e^{-z}}`,
          `\\hat{y} = \\frac{1}{1+e^{-${P(zo[0])}}} = ${val(p)}`,
          `P(${cname(1)} \\mid \\mathbf{x}) = ${val(p)}, \\qquad P(${cname(0)} \\mid \\mathbf{x}) = 1-\\hat{y} = ${val(1 - p)}`,
        ] : [
          `\\hat{y} = \\sigma\\big(z${So}\\big)`,
          `\\hat{y} = P(y = 1 \\mid \\mathbf{x})`,
        ],
        note: 'For a two-class dataset the output activation is the sigmoid. It squashes the logit into (0, 1) so it can be read as the probability of class 1.',
        hl: { nodes: ['out'], edges: [], dir: 'fwd' },
        vals: { out: { arr: c.yhat, label: 'ŷ', kind: 'fwd' } },
      });
    } else {
      const ex = zo.map(Math.exp);
      const sum = ex.reduce((a, b) => a + b, 0);
      const rows = zo.map((z, k) => `\\hat{y}_{${n1(k)}} &= \\frac{e^{${f3(z)}}}{${f3(sum)}} = \\frac{${f3(ex[k])}}{${f3(sum)}} = ${val(c.yhat[k])} \\quad (${cname(k)})`);
      steps.push({
        id: 'yhat', phase: 'fwd', title: 'Prediction: softmax', short: `ŷ = softmax(z${usup(L + 1)})`,
        blocks: F ? [
          `\\hat{y}_k = \\frac{e^{z${So}_k}}{\\sum_{m=1}^{${O}} e^{z${So}_m}}, \\qquad \\sum_{m} e^{z_m} = ${zo.map(z => `e^{${f3(z)}}`).join(' + ')} = ${f3(sum)}`,
          aligned(rows),
        ] : [
          `\\hat{\\mathbf{y}} = \\operatorname{softmax}\\big(\\mathbf{z}${So}\\big)`,
          `\\hat{y}_k = \\operatorname{softmax}\\big(\\mathbf{z}${So}\\big)_k = P(\\text{class } k \\mid \\mathbf{x})`,
        ],
        note: `For a ${O}-class dataset the output activation is softmax. It exponentiates every logit and normalizes, so the outputs are positive and sum to 1, giving a probability for each class.`,
        hl: { nodes: ['out'], edges: [], dir: 'fwd' },
        vals: { out: { arr: c.yhat, label: 'ŷ', kind: 'fwd' } },
      });
    }

    // loss
    const Lval = net.loss(c, label);
    if (sig) {
      const p = c.yhat[0];
      steps.push({
        id: 'loss', phase: 'fwd', title: 'Loss: binary cross-entropy', short: 'L(ŷ, y)',
        blocks: F ? [
          `L = -\\big[\\, ${label} \\cdot \\log(${val(p)}) + ${1 - label} \\cdot \\log(${val(1 - p)}) \\,\\big]`,
          `L = ${label === 1 ? `-\\log(${f3(p)})` : `-\\log(${f3(1 - p)})`} = ${val(Lval)}`,
        ] : [
          `L = -\\big[\\, y \\log \\hat{y} + (1-y) \\log(1-\\hat{y}) \\,\\big]`,
        ],
        note: 'Cross-entropy is small when the network gives the true class a high probability, and it grows quickly as that probability approaches 0. Only one of the two terms is active for any given sample.',
        hl: { nodes: ['out', 'loss'], edges: ['el'], dir: 'fwd' },
        vals: { out: { arr: c.yhat, label: 'ŷ', kind: 'fwd' }, loss: { arr: [Lval], label: 'L', kind: 'fwd' } },
      });
    } else {
      const terms = c.yhat.map((p, k) => `${g.t[k]} \\cdot \\log(${val(p)})`).join(' + ');
      steps.push({
        id: 'loss', phase: 'fwd', title: 'Loss: categorical cross-entropy', short: 'L(ŷ, y)',
        blocks: F ? [
          `L = -\\big[\\, ${terms} \\,\\big]`,
          `L = -\\log(${f3(c.yhat[label])}) = ${val(Lval)}`,
        ] : [
          `L = -\\sum_{k=1}^{${O}} y_k \\log \\hat{y}_k`,
          `L = -\\log \\hat{y}_{c} \\qquad \\text{where } c \\text{ is the true class}`,
        ],
        note: 'Because y is one-hot, only the true class’s term survives. The loss is the negative log of the probability the network gave the correct answer.',
        hl: { nodes: ['out', 'loss'], edges: ['el'], dir: 'fwd' },
        vals: { out: { arr: c.yhat, label: 'ŷ', kind: 'fwd' }, loss: { arr: [Lval], label: 'L', kind: 'fwd' } },
      });
    }

    /* ================= BACKWARD ================= */
    const dzo = g.dz[L];

    // output error
    if (sig) {
      const p = c.yhat[0];
      const dLdy = -label / p + (1 - label) / (1 - p);
      steps.push({
        id: 'dzout', phase: 'bwd', title: 'Output error', short: `δ${usup(L + 1)} = ŷ − y`,
        blocks: F ? [
          `${dfrac('L', '\\hat{y}')} = -\\frac{y}{\\hat{y}} + \\frac{1-y}{1-\\hat{y}} = -\\frac{${label}}{${f3(p)}} + \\frac{${1 - label}}{${f3(1 - p)}} = ${grd(dLdy)}`,
          `${dfrac('\\hat{y}', `z${So}`)} = \\sigma'(z${So}) = \\hat{y}\\,(1-\\hat{y}) = ${f3(p)} \\cdot ${f3(1 - p)} = ${val(p * (1 - p))}`,
          `\\delta${So} = ${grdP(dLdy)} \\cdot ${valP(p * (1 - p))} = \\hat{y} - y = ${f3(p)} - ${label} = ${grd(dzo[0])}`,
        ] : [
          `\\delta${So} = ${dfrac('L', `z${So}`)} = ${dfrac('L', '\\hat{y}')} \\cdot ${dfrac('\\hat{y}', `z${So}`)}`,
          `\\delta${So} = \\hat{y} - y`,
        ],
        note: 'Backpropagation starts at the loss and applies the chain rule. For sigmoid with binary cross-entropy, the two derivatives cancel to give the simple error ŷ − y. This pairing is the reason the dataset’s class count picks the output activation.',
        hl: { nodes: ['loss', 'out'], edges: ['el'], dir: 'bwd' },
        vals: { out: { arr: dzo, label: 'δ', kind: 'bwd' }, loss: { arr: [Lval], label: 'L', kind: 'fwd' } },
      });
    } else {
      const rows = c.yhat.map((p, k) => `\\delta${So}_{${n1(k)}} &= ${f3(p)} - ${g.t[k]} = ${grd(dzo[k])}`);
      steps.push({
        id: 'dzout', phase: 'bwd', title: 'Output error', short: `δ${usup(L + 1)} = ŷ − y`,
        blocks: F ? [
          `${dfrac('\\hat{y}_m', 'z_k')} = \\hat{y}_m\\big([m{=}k] - \\hat{y}_k\\big), \\qquad ${dfrac('L', '\\hat{y}_m')} = -\\frac{y_m}{\\hat{y}_m}`,
          `${dfrac('L', 'z_k')} = \\sum_m -\\frac{y_m}{\\hat{y}_m}\\,\\hat{y}_m\\big([m{=}k] - \\hat{y}_k\\big) = \\hat{y}_k \\textstyle\\sum_m y_m - y_k = \\hat{y}_k - y_k`,
          aligned(rows),
        ] : [
          `\\boldsymbol{\\delta}${So} = ${dfrac('L', `\\mathbf{z}${So}`)} = \\Big(${dfrac('\\hat{\\mathbf{y}}', `\\mathbf{z}${So}`)}\\Big)^{\\!\\top} ${dfrac('L', '\\hat{\\mathbf{y}}')}`,
          `\\boldsymbol{\\delta}${So} = \\hat{\\mathbf{y}} - \\mathbf{y}`,
        ],
        note: 'Backpropagation starts at the loss and applies the chain rule. For softmax with cross-entropy, the softmax Jacobian and the log-loss derivative combine to give ŷ − y. The output for the true class gets a negative error that pushes its score up, and every other class gets a positive error that pushes its score down.',
        hl: { nodes: ['loss', 'out'], edges: ['el'], dir: 'bwd' },
        vals: { out: { arr: dzo, label: 'δ', kind: 'bwd' }, loss: { arr: [Lval], label: 'L', kind: 'fwd' } },
      });
    }

    // gradients for layer l's weights
    const gradStep = l => {
      const s = S(l + 1), inp = inVals(l), d = g.dz[l], key = nodeKey(l), src = srcKey(l);
      const isOut = l === L;
      // written-out rows: first destination neuron (or every neuron for the first layer, which has only 2 inputs)
      const rowsFor = j => inp.map((v, i) => `${dfrac('L', `w${s}_{${n1(j)}${n1(i)}}`)} &= ${grdP(d[j])} \\cdot ${valP(v)} = ${grd(g.dW[l][j][i])}`);
      const rows = l === 0
        ? d.map((_, j) => rowsFor(j).join(' &\\qquad '))
        : rowsFor(0);
      const multi = l !== 0 && d.length > 1;
      return {
        id: `dW${l + 1}`, phase: 'bwd', title: isOut ? 'Output-layer gradients' : L === 1 ? 'Hidden-layer gradients' : `${layerName(l)} gradients`,
        short: `∂L/∂W${usup(l + 1)} = δ${usup(l + 1)}${inText(l)}ᵀ`,
        blocks: F ? [
          `${dfrac('L', `W${s}`)} = ${col(d.map(grd))}\\,${mat([inp.map(val)])} = ${mat(g.dW[l].map(r => r.map(grd)))}, \\qquad ${dfrac('L', `\\mathbf{b}${s}`)} = ${col(g.db[l].map(grd))}`,
          aligned(rows),
        ] : [
          `${dfrac('L', `w${s}_{ji}`)} = ${dfrac('L', `z${s}_j`)} \\cdot ${dfrac(`z${s}_j`, `w${s}_{ji}`)} = \\delta${s}_j\\, ${inElJ(l)}`,
          `${dfrac('L', `W${s}`)} = \\boldsymbol{\\delta}${s}\\,${l === 0 ? '\\mathbf{x}^{\\top}' : `\\mathbf{a}^{(${l})\\top}`}, \\qquad ${dfrac('L', `\\mathbf{b}${s}`)} = \\boldsymbol{\\delta}${s}`,
        ],
        note: (isOut
          ? 'A weight’s gradient is the error at its destination times the activation at its source. Large activations feeding a large error get the biggest updates.'
          : l === 0
            ? 'The first layer follows the same pattern as every other layer: error at the destination times the value at the source. Here the sources are the raw inputs x₁ and x₂.'
            : `Same pattern again: the error δ${usup(l + 1)} at each neuron of this layer times the activation a${usup(l)} that fed it.`)
          + (multi ? ` The rows written out below are for ${isOut ? 'output' : 'destination'} neuron 1.` : ''),
        hl: { nodes: [src, key], edges: [`e${l}`], dir: 'bwd' },
        vals: { [src]: srcVals(l), [key]: { arr: d, label: 'δ', kind: 'bwd' } },
      };
    };

    for (let l = L; l >= 1; l--) {
      steps.push(gradStep(l));

      // send the error back into hidden layer h = l - 1
      const h = l - 1, sh = S(l), sl = S(l + 1), hk = `h${h}`, key = nodeKey(l);
      const rows = [];
      for (let j = 0; j < H; j++) {
        const terms = g.dz[l].map((d, k) => `${parP(net.W[l][k][j])} \\cdot ${grdP(d)}`).join(' + ');
        rows.push(`${dfrac('L', `a${sh}_{${n1(j)}}`)} &= ${terms} = ${grd(g.da[h][j])}`);
      }
      const nOut = g.dz[l].length;
      steps.push({
        id: `da${l}`, phase: 'bwd', title: `Send the error back to the ${layerLower(h)}`,
        short: `∂L/∂a${usup(l)} = W${usup(l + 1)}ᵀδ${usup(l + 1)}`,
        blocks: F ? [
          `${dfrac('L', `\\mathbf{a}${sh}`)} = W^{(${l + 1})\\top}\\boldsymbol{\\delta}${sl} = ${mat(transpose(net.W[l]).map(r => r.map(par)))}\\,${col(g.dz[l].map(grd))} = ${col(g.da[h].map(grd))}`,
          aligned(rows),
        ] : [
          `${dfrac('L', `a${sh}_j`)} = \\sum_{k=1}^{${nOut}} ${dfrac('L', `z${sl}_k`)} \\cdot ${dfrac(`z${sl}_k`, `a${sh}_j`)} = \\sum_{k} \\delta${sl}_k\\, w${sl}_{kj}`,
          `${dfrac('L', `\\mathbf{a}${sh}`)} = W^{(${l + 1})\\top}\\boldsymbol{\\delta}${sl}`,
        ],
        note: `Each neuron’s share of the blame is the error from the layer above sent back along the same weights used in the forward pass. Multiplying by the transpose of W${usup(l + 1)} runs those connections in reverse.`,
        hl: { nodes: [hk, key], edges: [`e${l}`], dir: 'bwd' },
        vals: { [hk]: { arr: g.da[h], label: '∂a', kind: 'bwd' }, [key]: { arr: g.dz[l], label: 'δ', kind: 'bwd' } },
      });

      // through the activation
      const arows = [];
      for (let j = 0; j < H; j++) {
        arows.push(
          `${dfrac(`a${sh}_{${n1(j)}}`, `z${sh}_{${n1(j)}}`)} &= ${chain(act.subDf(c.z[h][j]), val(g.dadz[h][j]), g.dadz[h][j])}` +
          ` &\\qquad \\delta${sh}_{${n1(j)}} &= ${grdP(g.da[h][j])} \\cdot ${valP(g.dadz[h][j])} = ${grd(g.dz[h][j])}`);
      }
      steps.push({
        id: `dz${l}`, phase: 'bwd', title: `Back through the ${act.name} activation${L > 1 ? ` (layer ${l})` : ''}`,
        short: `δ${usup(l)} = ∂L/∂a ⊙ ∂a/∂z`,
        blocks: F ? [
          `${dfrac('a', 'z')} = f'(z) = ${act.texDf}, \\qquad \\delta${sh}_j = ${dfrac('L', `a${sh}_j`)} \\cdot f'\\big(z${sh}_j\\big)`,
          aligned(arows),
        ] : [
          `\\boldsymbol{\\delta}${sh} = ${dfrac('L', `\\mathbf{z}${sh}`)} = ${dfrac('L', `\\mathbf{a}${sh}`)} \\odot ${dfrac(`\\mathbf{a}${sh}`, `\\mathbf{z}${sh}`)}`,
          `\\delta${sh}_j = ${dfrac('L', `a${sh}_j`)} \\cdot ${dfrac(`a${sh}_j`, `z${sh}_j`)}`,
        ],
        note: `The error has to pass back through the activation function, so it gets multiplied by the slope of f at each neuron’s z. ${act.gradNote}${L > 1 && h > 0 ? ' This happens once per hidden layer, so in deep networks small slopes can compound and shrink the gradient.' : ''}`,
        hl: { nodes: [hk], edges: [], dir: 'bwd' },
        vals: { [hk]: { arr: g.dz[h], label: 'δ', kind: 'bwd' } },
      });
    }
    steps.push(gradStep(0));

    // update
    {
      const upd = (P0, G0) => P0.map((r, i) => (Array.isArray(r) ? r.map((w, j) => par(w - lr * G0[i][j])) : par(r - lr * G0[i])));
      const lrs = String(lr);
      const blocks = [`\\eta = ${lrs} \\qquad \\text{(one SGD step on this sample)}`];
      for (let l = L; l >= 0; l--) {
        const s = S(l + 1);
        blocks.push(`W${s} \\leftarrow ${mat(net.W[l].map(r => r.map(par)))} - ${lrs}\\,${mat(g.dW[l].map(r => r.map(grd)))} = ${mat(upd(net.W[l], g.dW[l]))}`);
        blocks.push(`\\mathbf{b}${s} \\leftarrow ${col(net.b[l].map(par))} - ${lrs}\\,${col(g.db[l].map(grd))} = ${col(upd(net.b[l], g.db[l]))}`);
      }
      const params = Array.from({ length: L + 1 }, (_, l) => `W${S(l + 1)},\\, \\mathbf{b}${S(l + 1)}`).join(',\\, ');
      steps.push({
        id: 'update', phase: 'bwd', title: 'Gradient descent update', short: 'θ ← θ − η ∂L/∂θ',
        blocks: F ? blocks : [
          `\\theta \\leftarrow \\theta - \\eta\\, ${dfrac('L', '\\theta')}, \\qquad \\theta \\in \\{\\, ${params} \\,\\}`,
          optimizer === 'adam'
            ? `\\text{Adam:}\\quad \\mathbf{m} \\leftarrow \\beta_1 \\mathbf{m} + (1-\\beta_1)\\nabla_\\theta L, \\quad \\mathbf{v} \\leftarrow \\beta_2 \\mathbf{v} + (1-\\beta_2)(\\nabla_\\theta L)^2, \\quad \\theta \\leftarrow \\theta - \\eta\\,\\frac{\\hat{\\mathbf{m}}}{\\sqrt{\\hat{\\mathbf{v}}}+\\epsilon}`
            : `\\text{SGD with mini-batches: } \\nabla_\\theta L = \\frac{1}{B}\\sum_{b=1}^{B} \\nabla_\\theta L_b`,
        ],
        note: optimizer === 'adam'
          ? 'Each parameter takes a small step downhill on the loss. The formatted view shows a plain SGD step for this one sample. During training the app averages gradients over mini-batches of 10 and uses Adam, which scales each parameter’s step using running averages of its gradients.'
          : 'Each parameter takes a small step downhill on the loss. The formatted view shows the step for this one sample. During training the app averages these gradients over mini-batches of 10 samples before stepping.',
        hl: {
          nodes: ['in', ...Array.from({ length: L }, (_, l) => `h${l}`), 'out'],
          edges: Array.from({ length: L + 1 }, (_, l) => `e${l}`), dir: 'bwd',
        },
        vals: {},
      });
    }

    return steps;
  }

  window.Equations = { build };
})();
