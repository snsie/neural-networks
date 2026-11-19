/* Guided calculations for the selected sample. Each step supplies the math and
 * the corresponding network nodes and connections from the same forward/backward pass. */
(function () {
  'use strict';

  const T = String.raw;
  const { number } = Activations;
  const signed = v => v < 0 ? T`\big(${number(v)}\big)` : number(v);
  const S = n => T`^{(${n})}`;
  const d = (a, b) => T`\frac{\partial ${a}}{\partial ${b}}`;
  const matrix = rows => T`\begin{bmatrix}${rows.map(row => row.map(number).join(' & ')).join(T` \\ `)}\end{bmatrix}`;
  const column = values => matrix(values.map(value => [value]));
  const item = (label, math) => ({ label, math });
  const value = (label, v) => item(label, number(v));
  const table = (headings, rows, active = -1) => ({ headings, rows, active });
  const detail = (title, math, text) => ({ title, math, text });
  const section = (label, rule, worked) => ({ label, rule, worked });
  const trace = (title, intro, options) => ({ title, intro, ...options });
  const nodes = (nodeIds, edgeIds = [], dir = 'fwd') => ({ nodeIds, edgeIds, dir });
  const indices = a => a.map((_, i) => i);

  const product = (left, right) => T`${signed(left)}\times${signed(right)}`;
  const expFromShift = (score, max) => T`e^{${number(score)}-${signed(max)}}`;

  function productSum(factors, result) {
    const sumRows = (terms, perLine, lead) => {
      const rows = [];
      for (let i = 0; i < terms.length; i += perLine) {
        rows.push(`${i === 0 ? lead : T`&\quad +`} ${terms.slice(i, i + perLine).join(' + ')}`);
      }
      return rows;
    };
    const rows = [
      ...sumRows(factors.map(([a, b]) => product(a, b)), 2, '&'),
      ...sumRows(factors.map(([a, b]) => signed(a * b)), 4, T`&\approx`),
      T`&\approx ${number(result)}`,
    ];
    return T`\begin{aligned}${rows.join(T` \\ `)}\end{aligned}`;
  }

  function evaluated(lhs, parts, result) {
    const answer = number(result);
    const last = parts.at(-1);
    const chain = last === answer || Number(last) === result ? parts : [...parts, answer];
    return T`${lhs}=${chain.join(T`\approx`)}`;
  }

  function sigmoidFraction(z, exponential) {
    return T`\frac{${z < 0 ? exponential : '1'}}{1+${exponential}}`;
  }

  function sigmoidWork(z, result, negative = false) {
    const expression = sigmoidFraction(z, T`e^{-${number(Math.abs(z))}}`);
    const evaluated = sigmoidFraction(z, number(Math.exp(-Math.abs(z))));
    const sign = negative ? '-' : '';
    return T`${sign}${expression}\approx${sign}${evaluated}\approx${number(result)}`;
  }

  function build({ net, cache: c, grads: g, label, lr, optimizer, classNames }) {
    const { L, O, act } = net;
    const binary = O === 1;
    const so = S(L + 1);
    const layer = l => l === L ? 'Output layer' : L === 1 ? 'Hidden layer' : `Hidden layer ${l + 1}`;
    const group = l => l < 0 ? 'in' : l === L ? 'out' : `h${l}`;
    const node = (l, j) => l < 0 ? `in${j}` : l === L ? `o${j}` : `h${l}-${j}`;
    const edge = (l, j, i) => `e${l}-${j}-${i}`;
    const bias = (l, j) => `bias-${node(l, j)}`;
    const neuronName = (l, j) => l === L && !binary ? classNames[j] : `Neuron ${j + 1}`;
    const sourceName = (l, i) => l === 0 ? `x${i === 0 ? '₁' : '₂'}` : `Neuron ${i + 1}`;
    const choice = (l, defaultIndex = l === L && !binary ? label : 0) => ({
      key: `neuron-${l === L ? 'out' : l}`,
      label: l === L && !binary ? 'Class' : 'Neuron',
      options: indices(c.z[l]).map(j => neuronName(l, j)),
      defaultIndex,
    });
    const sourceChoice = l => ({
      key: `source-${l === L ? 'out' : l}`,
      label: 'Connection from',
      options: indices(net.input(c, l)).map(i => sourceName(l, i)),
    });
    const inputVals = l => ({ arr: net.input(c, l), label: l === 0 ? 'x' : 'a', kind: 'fwd' });
    const display = (arr, label, kind = 'fwd') => ({ arr, label, kind });
    const steps = [];
    const add = (id, phase, band, label, selectors, render) => steps.push({ id, phase, band, label, selectors, render });
    const pred = net.predict(c);
    const probs = net.probs(c);
    const loss = net.loss(c, label);

    add('input', 'fwd', 0, 'Input sample', [], () => trace(
      'What goes into the network?', 'The point’s two coordinates go into the network. Its true class is saved for the loss calculation.', {
        inputs: [value('First coordinate x₁', c.x[0]), value('Second coordinate x₂', c.x[1]), { label: 'True class', text: classNames[label] }],
        sections: [section('Represent the input and target',
          binary ? T`\mathbf{x}=[x_1,x_2]^\top,\qquad y\in\{0,1\}` : T`\mathbf{x}=[x_1,x_2]^\top,\qquad y_k=\mathbf{1}[k=\text{true class}]`,
          binary ? T`\mathbf{x}=${column(c.x)},\qquad y=${label}` : T`\mathbf{x}=${column(c.x)},\qquad \mathbf{y}=${column(g.t)}`)],
        result: { label: 'Network input x', math: column(c.x), text: 'Follow these two values into the first weighted sum.' },
        hl: nodes(['in0', 'in1']), vals: { in: inputVals(0) },
      }));

    function weightedSum(l, j) {
      const inputs = net.input(c, l), weights = net.W[l][j], b = net.b[l][j], z = c.z[l][j];
      const s = S(l + 1);
      const x = l === 0 ? 'x_i' : T`a${S(l)}_i`;
      const a = l === 0 ? T`\mathbf{x}` : T`\mathbf{a}${S(l)}`;
      const factors = inputs.map((x, i) => [weights[i], x]).concat([[b, 1]]);
      const zLabel = l === L ? 'Output score z (logit)' : 'Pre-activation z';
      return trace(l === L ? 'How does the output neuron compute its score?' : 'How does this neuron combine its inputs?',
        'Multiply each incoming value by its connection weight, then add the bias. You can think of the bias as a weight on a constant input of 1. The square marks that bias on the diagram.', {
          inputs: [value('Bias weight b from the bias square', b)],
          sections: [section('Multiply, then add', T`z${s}_j=\sum_i w${s}_{ji}${x}+b${s}_j`)],
          tables: [table(['Source', 'Source value', 'Weight', 'Contribution'], [
            ...inputs.map((x, i) => [sourceName(l, i), value('', x), value('', weights[i]), value('', weights[i] * x)]),
            ['Bias square', item('', '1'), value('', b), value('', b)],
          ])],
          after: [section('Add the contributions', null, productSum(factors, z))],
          result: { label: zLabel, math: number(z), text: l === L ? 'Prediction converts this score into a probability.' : `${act.name} uses this value in the next step.` },
          details: [
            detail('Show all neuron scores', T`\mathbf{z}${s}=${column(c.z[l])}`),
            detail('Show matrix form', [
              T`\mathbf{z}${s}=W${s}${a}+\mathbf{b}${s}`,
              T`${matrix(net.W[l])}\,${column(inputs)} + ${column(net.b[l])}\approx ${column(c.z[l])}`,
            ], 'Each row of W contains the weights feeding one destination neuron.'),
          ],
          hl: nodes([node(l, j), bias(l, j), ...indices(inputs).map(i => node(l - 1, i))], indices(inputs).map(i => edge(l, j, i))),
          vals: { [group(l - 1)]: inputVals(l), [group(l)]: display(c.z[l], 'z') },
        });
    }

    for (let l = 0; l < L; l++) {
      add(`z${l + 1}`, 'fwd', l + 1, `${layer(l)} sum`, [choice(l)], j => weightedSum(l, j));
      add(`a${l + 1}`, 'fwd', l + 1, `${layer(l)} activation`, [choice(l)], j => {
        const z = c.z[l][j], a = c.a[l][j];
        const branch = act.id === 'relu' ? (z > 0 ? ' This score is positive, so ReLU passes it through.' : ' At this score, ReLU outputs zero.') : '';
        return trace(`What does ${act.name} do to the score?`, `Apply the same function to each neuron independently.${branch}`, {
          inputs: [value('Pre-activation z from weighted sum', z)],
          sections: [section('Apply the activation', T`a=f(z)=${act.texF}`, evaluated(T`a=f(${number(z)})`, act.subF(z), a))],
          plot: { z, a, act, slope: false },
          result: { label: 'Activation a', math: number(a), text: 'This becomes an input to the next layer.' },
          details: [detail('Show all neuron activations', T`\mathbf{a}${S(l + 1)}=${column(c.a[l])}`)],
          hl: nodes([node(l, j)]), vals: { [group(l)]: display(c.a[l], 'a') },
        });
      });
    }

    add('zout', 'fwd', L + 1, 'Output score', [choice(L)], j => weightedSum(L, j));
    add('yhat', 'fwd', L + 1, 'Prediction', binary ? [] : [choice(L)], (k = 0) => {
      const logits = c.z[L], p = c.yhat[k];
      const shift = Math.max(...logits), terms = logits.map(z => Math.exp(z - shift));
      const total = terms.reduce((sum, x) => sum + x, 0);
      const denominator = logits.map(z => expFromShift(z, shift)).join(' + ');
      const probabilityTable = table(binary ? ['Class', 'Probability'] : ['Class', 'Score z', 'Probability'],
        classNames.slice(0, net.K).map((name, i) => [i === pred ? `${name} (predicted)` : name,
          ...(binary ? [] : [value('', logits[i])]), value('', probs[i])]), binary ? pred : k);
      return trace('How do scores become probabilities?',
        binary ? 'Sigmoid turns the single score into the probability of class 1. The other class gets the remaining probability.'
          : 'Softmax converts all class scores together. Each class uses the same denominator, so changing one score can affect all probabilities.', {
          inputs: [value(binary ? 'Output score z' : `Score z · ${classNames[k]}`, logits[k])],
          sections: binary ? [section('Apply sigmoid', T`\hat y=\sigma(z)=${logits[0] < 0 ? T`\frac{e^z}{1+e^z}` : T`\frac{1}{1+e^{-z}}`}`, T`\hat y=${sigmoidWork(logits[0], p)}`)] :
            [section('Normalize across classes', T`\hat y_k=\frac{e^{z_k-m}}{D},\quad D=\sum_i e^{z_i-m},\; m=\max_i z_i`, [
              T`D=${denominator}\approx${number(total)}`,
              T`\hat y_{${k + 1}}=\frac{${expFromShift(logits[k], shift)}}{D}\approx\frac{${number(terms[k])}}{${number(total)}}\approx${number(p)}`,
            ])],
          tables: [probabilityTable],
          details: binary ? [] : [detail('Show the shared denominator', [
            T`m=\max_k z_k=${number(shift)}`,
            T`\sum_i e^{z_i-m}=\,${denominator}\approx${terms.map(number).join(' + ')}\approx${number(total)}`,
          ], 'Subtracting the same maximum score from every score keeps the exponentials numerically stable and leaves the probabilities unchanged.')],
          result: { label: 'Predicted class', text: `${classNames[pred]} has the highest probability. The next step uses the probability of the true class.` },
          hl: nodes([node(L, k)]), vals: { out: display(c.yhat, 'ŷ') },
        });
    });

    add('loss', 'fwd', L + 2, 'Sample loss', [], () => {
      const p = probs[label];
      const scores = c.z[L];
      const term = binary ? (label === 1 ? T`\hat y` : T`1-\hat y`) : T`\hat y_{${label + 1}}`;
      const margin = label === 1 ? -scores[0] : scores[0];
      const shift = Math.max(...scores);
      const peak = scores.indexOf(shift);
      const otherScores = scores.filter((_, k) => k !== peak);
      const rest = otherScores.reduce((sum, score) => sum + Math.exp(score - shift), 0);
      return trace('How much loss did this sample produce?', binary ? 'Cross-entropy gets smaller as the true class becomes more likely. We calculate it from the score so it still works when the probability rounds to zero or one.'
        : 'Shift all scores by the maximum m. The maximum contributes 1 because e⁰ = 1. Add the remaining exponentials to get R.', {
        inputs: [{ label: 'True class from the input', text: classNames[label] }, value('True-class probability from prediction', p)],
        sections: [
          section('Write the cross-entropy', T`L=-\log(p_{\text{true}}),\qquad p_{\text{true}}=${term}`),
          binary
            ? section('Calculate from the signed score', T`L=\max(0,s)+\log(1+e^{-|s|}),\quad s=(1-2y)z`,
              [
                T`s=(1-2\times${label})\times${signed(scores[0])}\approx${number(margin)}`,
                T`L=\max(0,${number(margin)})+\log(1+e^{-|${number(margin)}|})`,
                T`\phantom{L}\approx${number(Math.max(0, margin))}+\log(1+${number(Math.exp(-Math.abs(margin)))})\approx${number(loss)}`,
              ])
            : section('Calculate from the shifted scores', T`L=(m-z_{\text{true}})+\log(1+R),\quad R=\sum_{i\ne i_{\max}}e^{z_i-m}`,
              [
                T`R=${otherScores.map(score => expFromShift(score, shift)).join(' + ')}\approx${number(rest)}`,
                T`L=(${number(shift)}-${signed(scores[label])})+\log(1+R)`,
                T`\phantom{L}\approx${number(shift - scores[label])}+\log(1+${number(rest)})\approx${number(loss)}`,
              ]),
        ],
        result: { label: 'Loss for this sample', math: number(loss), text: 'Backpropagation starts here and asks how changing the output score would change this loss.' },
        details: [detail('Show the full cross-entropy formula', binary
          ? T`L=-[y\log\hat y+(1-y)\log(1-\hat y)]`
          : T`L=-\sum_k y_k\log\hat y_k`,
        `Only the term for ${classNames[label]} survives because its target is 1.`)],
        hl: nodes([node(L, binary ? 0 : label), 'loss'], [`l-${binary ? 0 : label}`]),
        vals: { out: display(c.yhat, 'ŷ'), loss: display([loss], 'L') },
      });
    });

    add('dzout', 'bwd', L + 2, 'Output gradient', binary ? [] : [choice(L)], (k = 0) => {
      const p = c.yhat[k], y = g.t[k], delta = g.dz[L][k];
      const lossSlope = y === 1 ? -1 / p : 1 / (1 - p), sigmoidSlope = p * (1 - p);
      const saturated = p === 0 || p === 1 || !Number.isFinite(lossSlope);
      const z = T`z${so}_1`;
      const classScore = T`z${so}_{${k + 1}}`;
      const outputChain = binary
        ? T`${d('L', z)}=${d('L', '\\hat y')}\cdot${d('\\hat y', z)}`
        : T`${d('L', classScore)}=\sum_{m=1}^{${O}}\left[${d('L', '\\hat y_m')}\cdot${d('\\hat y_m', classScore)}\right]`;
      const subtraction = T`${number(p)}-${y}\approx${number(delta)}`;
      const stableBinary = y === 1 ? sigmoidWork(-c.z[L][0], delta, true) : sigmoidWork(c.z[L][0], delta);
      const otherProbabilities = c.yhat.filter((_, i) => i !== k);
      const trueClassSum = T`-\big(${otherProbabilities.map(number).join(' + ')}\big)`;
      return trace('How does loss change with the output score?',
        binary ? (saturated ? 'The probability is at numerical saturation. The model uses the combined rule to avoid dividing by a probability at the limit.' : 'Trace the slope from loss to probability, then from probability to the score.')
          : 'Each score can affect all the softmax probabilities. With cross-entropy, the combined gradient for a class simplifies to its probability minus its target.', {
          inputs: [value('Predicted probability from prediction', p), value('Target from input', y)],
          sections: [section('Chain from loss to output score', outputChain), ...(binary && !saturated ? [
            section('1. How does loss respond to the probability?', T`${d('L', '\\hat y')}=-\frac{y}{\hat y}+\frac{1-y}{1-\hat y}`,
              [
                T`-\frac{${y}}{${number(p)}}+\frac{1-${y}}{1-${number(p)}}`,
                T`\approx ${y === 1 ? T`-\frac{1}{${number(p)}}` : T`\frac{1}{${number(1 - p)}}`}\approx${number(lossSlope)}`,
              ]),
            section('2. How does probability respond to the score?', T`${d('\\hat y', z)}=\hat y(1-\hat y)`,
              T`${number(p)}\times(1-${number(p)})\approx${product(p, 1 - p)}\approx${number(sigmoidSlope)}`),
            section('3. Multiply the two slopes', T`\delta${so}=${d('L', z)}`,
              T`\underbrace{${signed(lossSlope)}}_{\text{Step 1}}\times\underbrace{${number(sigmoidSlope)}}_{\text{Step 2}}\approx${number(delta)}`),
          ] : [section(binary ? 'Use the combined rule at saturation' : 'Subtract the target from the probability',
            binary
              ? T`\delta${so}=\hat y-y=${y === 1 ? T`-\sigma(-z${so})` : T`\sigma(z${so})`}`
              : T`\delta${so}_k=\hat y_k-y_k`,
            binary ? stableBinary : y === 0 ? subtraction : [
              ...(p === 1 ? [] : [subtraction]),
              T`\delta${so}_k=-\sum_{i\ne k}\hat y_i\approx ${trueClassSum}\approx${number(delta)}`,
            ])])],
          tables: binary ? [] : [table(['Class', 'Probability', 'Target', 'Gradient δ'],
            c.yhat.map((prob, i) => [i === label ? `${classNames[i]} (true)` : classNames[i],
              value('', prob), item('', String(g.t[i])), value('', g.dz[L][i])]), k)],
          result: { label: 'Output score gradient δ', math: number(delta), text: delta === 0
            ? 'The output score has zero gradient for this sample at the model’s numerical precision.'
            : `A small ${delta < 0 ? 'increase' : 'decrease'} in this score would lower the loss. This gradient feeds the weight and bias calculations.` },
          details: binary ? [detail('Why does this simplify to ŷ − y?', [
            T`\left(-\frac{y}{\hat y}+\frac{1-y}{1-\hat y}\right)\hat y(1-\hat y)`,
            T`=-y(1-\hat y)+(1-y)\hat y=\hat y-y`,
          ], 'The probability factors cancel when the probability is strictly between 0 and 1. For the positive class, the model uses the equivalent −σ(−z) to preserve small gradients.')]
            : [detail('Show the softmax derivation', [
              T`${d('\\hat y_m', 'z_k')}=\hat y_m\big(\mathbf{1}[m=k]-\hat y_k\big)`,
              T`${d('L', 'z_k')}=\sum_m\left(-\frac{y_m}{\hat y_m}\right)\hat y_m\big(\mathbf{1}[m=k]-\hat y_k\big)`,
              T`=\hat y_k\sum_m y_m-y_k=\hat y_k-y_k`,
            ], 'The indicator is 1 when the two class indices match. The target entries sum to 1.')],
          hl: nodes([node(L, k), 'loss'], [`l-${k}`], 'bwd'),
          vals: { out: display(g.dz[L], 'δ', 'bwd'), loss: display([loss], 'L') },
        });
    });

    for (let l = L; l >= 0; l--) {
      const s = S(l + 1);
      add(`dW${l + 1}`, 'bwd', l + 1, `${layer(l)} gradients`, [choice(l), sourceChoice(l)], (j, i) => {
        const inputs = net.input(c, l), delta = g.dz[l][j], x = inputs[i];
        const z = T`z${s}_{${j + 1}}`;
        const w = T`w${s}_{${j + 1},${i + 1}}`;
        const b = T`b${s}_{${j + 1}}`;
        const source = l === 0 ? T`x_{${i + 1}}` : T`a${S(l)}_{${i + 1}}`;
        const a = l === 0 ? T`\mathbf{x}` : T`\mathbf{a}${S(l)}`;
        return trace('How sensitive is loss to this weight and bias?', 'The connection weight scales its source value. The bias weight scales the constant 1.', {
          inputs: [value('Destination score gradient δ', delta), value(`Source value · ${sourceName(l, i)}`, x)],
          sections: [
            section('Chain from loss to the selected weight', [
              T`${d('L', w)}=${d('L', z)}\cdot${d(z, w)}`,
              T`${d('L', z)}=\delta${s}_{${j + 1}},\qquad ${d(z, w)}=${source}`,
            ], T`${d('L', w)}\approx${product(delta, x)}\approx${number(g.dW[l][j][i])}`),
            section('Chain from loss to the bias', [
              T`${d('L', b)}=${d('L', z)}\cdot${d(z, b)}`,
              T`${d(z, b)}=1`,
            ], T`${d('L', b)}\approx${signed(delta)}\times1=${number(g.db[l][j])}`),
          ],
          result: { label: 'Parameter gradients', text: 'These two gradients tell the update step how sensitive the loss is to this weight and bias.' },
          details: [detail('Show all weight and bias gradients', [
            T`${d('L', `W${s}`)}=\boldsymbol{\delta}${s}\left(${a}\right)^\top\approx${matrix(g.dW[l])}`,
            T`${d('L', T`\mathbf{b}${s}`)}=\boldsymbol{\delta}${s}=${column(g.db[l])}`,
          ])],
          hl: nodes([node(l - 1, i), node(l, j), bias(l, j)], [edge(l, j, i)], 'bwd'),
          vals: { [group(l - 1)]: inputVals(l), [group(l)]: display(g.dz[l], 'δ', 'bwd') },
        });
      });
      if (l === 0) continue;
      const h = l - 1, sh = S(l);
      add(`da${l}`, 'bwd', l + 1, `Back to ${layer(h).toLowerCase()}`, [choice(h)], j => {
        const nextGrad = g.dz[l];
        const a = T`a${sh}_{${j + 1}}`;
        const next = nextGrad.length === 1 ? '1' : 'k';
        const z = T`z${s}_{${next}}`;
        const w = T`w${s}_{${next},${j + 1}}`;
        const path = T`${d('L', z)}\cdot${d(z, a)}`;
        const sum = nextGrad.length === 1 ? path : T`\sum_{k=1}^{${nextGrad.length}}\left[${path}\right]`;
        const factors = nextGrad.map((delta, k) => [delta, net.W[l][k][j]]);
        return trace('How does the gradient reach this hidden activation?', 'Trace the outgoing connections backward. Each connected neuron sends back its gradient multiplied by that connection’s weight.', {
          sections: [section(nextGrad.length === 1 ? 'Follow the path through the next neuron' : 'Add one chain per neuron in the next layer', [
            T`${d('L', a)}=${sum}`,
            T`${d('L', z)}=\delta${s}_{${next}},\qquad ${d(z, a)}=${w}`,
          ])],
          tables: [table(['Next neuron', 'Weight', 'Score gradient δ', 'Contribution'],
            nextGrad.map((delta, k) => [neuronName(l, k), value('', net.W[l][k][j]), value('', delta), value('', factors[k][1] * delta)]))],
          after: [section('Sum the weighted contributions', null, productSum(factors, g.da[h][j]))],
          result: { label: 'Activation gradient ∂L/∂a', math: number(g.da[h][j]), text: 'Next, multiply by the activation’s slope to reach this neuron’s score.' },
          details: [detail('Show matrix form', [
            T`${d('L', T`\mathbf{a}${sh}`)}=(W${s})^\top\boldsymbol{\delta}${s}`,
            T`${matrix(net.W[l][0].map((_, i) => net.W[l].map(row => row[i])))}\,${column(nextGrad)}\approx${column(g.da[h])}`,
          ], 'Transposing W groups the connections by the neuron that receives the gradient.')],
          hl: nodes([node(h, j), ...indices(nextGrad).map(k => node(l, k))], indices(nextGrad).map(k => edge(l, k, j)), 'bwd'),
          vals: { [group(h)]: display(g.da[h], '∂L/∂a', 'bwd'), [group(l)]: display(nextGrad, 'δ', 'bwd') },
        });
      });
      add(`dz${l}`, 'bwd', l, `${layer(h)} slope`, [choice(h)], j => {
        const z = c.z[h][j], slope = g.dadz[h][j], incoming = g.da[h][j], delta = g.dz[h][j];
        const score = T`z${sh}_{${j + 1}}`;
        const activation = T`a${sh}_{${j + 1}}`;
        return trace(`How does the gradient pass through ${act.name}?`, 'Evaluate the slope at the same score used in the forward pass, then multiply by the incoming gradient.', {
          inputs: [value('Pre-activation z from the forward pass', z), value('Incoming activation gradient ∂L/∂a', incoming)],
          sections: [
            section('Evaluate the local slope', T`${d(activation, score)}=f'(${score}),\qquad f'(z)=${act.texDf}`, evaluated(T`f'(${number(z)})`, act.subDf(z), slope)),
            section('Multiply the two partial derivatives', T`\delta${sh}_{${j + 1}}=${d('L', score)}=${d('L', activation)}\cdot${d(activation, score)}`, T`${product(incoming, slope)}\approx${number(delta)}`),
          ],
          plot: { z, a: c.a[h][j], act, slope: true, rate: slope, rateMath: number(slope) },
          result: { label: 'Hidden score gradient δ', math: number(delta), text: act.id === 'relu' && z === 0
            ? 'ReLU has a corner here. The app uses a slope of zero at the corner.'
            : slope === 0 ? 'The local slope is zero, so this neuron passes back zero gradient for this sample.'
              : 'The next calculation uses this score gradient for the incoming weights and bias.' },
          details: [detail('Show all hidden score gradients', T`\boldsymbol{\delta}${sh}=${column(g.dz[h])}`, act.gradNote)],
          hl: nodes([node(h, j)], [], 'bwd'), vals: { [group(h)]: display(g.dz[h], 'δ', 'bwd') },
        });
      });
    }

    const layers = c.z.map((_, l) => layer(l));
    add('update', 'bwd', 0, 'Update parameters', [{ key: 'update-layer', label: 'Layer', options: layers, defaultIndex: L }],
      l => ({
        selectors: [choice(l), { key: 'parameter', label: 'Parameter', options: ['Weight', 'Bias'] }, sourceChoice(l)],
        render(j, isBias, i) {
          const before = isBias ? net.b[l][j] : net.W[l][j][i];
          const gradient = isBias ? g.db[l][j] : g.dW[l][j][i];
          const change = -lr * gradient;
          const after = before + change;
          return trace('How would this parameter update?', `This one-sample SGD example uses the selected sample’s gradient and current learning rate. Training uses ${optimizer === 'adam' ? 'Adam' : 'SGD'} with mini-batches.`, {
            inputs: [value('Current parameter', before), value('Gradient for this sample', gradient), value('Learning rate η', lr)],
            sections: [
              section('Calculate the signed change', T`\Delta\theta=-\eta\frac{\partial L}{\partial\theta}`, T`${product(-lr, gradient)}\approx${number(change)}`),
              section('Apply the change to the parameter', T`\theta_{\text{new}}=\theta_{\text{old}}+\Delta\theta`,
                T`${number(before)}+${signed(change)}\approx${number(after)}`),
            ],
            result: { label: 'Example new parameter', math: number(after),
              text: 'The worked example leaves the network parameters in place.' },
            details: [detail(`How training updates parameters with ${optimizer === 'adam' ? 'Adam' : 'SGD'}`, [
              T`g_{\text{batch}}=\frac{1}{B}\sum_{i=1}^{B}g_i`,
              optimizer === 'adam'
                ? T`\theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}`
                : T`\theta\leftarrow\theta-\eta\,g_{\text{batch}}`,
            ], optimizer === 'adam'
              ? 'Training averages gradients over mini-batches of up to 10 training points. Adam uses running first and second moments of those batch gradients.'
              : 'Training averages gradients over mini-batches of up to 10 training points before each SGD update.')],
            hl: nodes(isBias ? [node(l, j), bias(l, j)] : [node(l - 1, i), node(l, j)], isBias ? [] : [edge(l, j, i)], 'bwd'),
            vals: { [group(l)]: display(g.dz[l], 'δ', 'bwd') },
          });
        },
      }));

    return steps;
  }

  window.Equations = { build, number, matrix, column };
})();
