/* Tiny real logistic-regression sentiment model trained in the visitor's browser.
 * The training examples and update rule match inference/demo_sentiment.py.
 * This is a CPU-only educational classifier, not an LLM or distributed worker.
 * It needs no server and does not upload user input.
 */
(function (root, factory) {
  const model = factory();
  if (typeof module === 'object' && module.exports) module.exports = model;
  if (root) root.ApicalBrowserModel = model;
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : null), function () {
  'use strict';

  const EXAMPLES = [
    ['i love this excellent product', 1],
    ['wonderful helpful and amazing', 1],
    ['this is great and useful', 1],
    ['very happy with the service', 1],
    ['fantastic quality and fast', 1],
    ['good reliable experience', 1],
    ['i hate this terrible product', 0],
    ['awful broken and disappointing', 0],
    ['this is bad and useless', 0],
    ['very unhappy with the service', 0],
    ['horrible quality and slow', 0],
    ['poor unreliable experience', 0],
  ];

  function tokenize(text) {
    return (text.toLowerCase().match(/[a-z]+/g) || []);
  }

  const vocabulary = [...new Set(EXAMPLES.flatMap(example => tokenize(example[0])))].sort();
  const lookup = new Map(vocabulary.map((word, index) => [word, index]));

  function features(text) {
    const vector = new Float64Array(vocabulary.length);
    tokenize(text).forEach(word => {
      const index = lookup.get(word);
      if (index !== undefined) vector[index] += 1;
    });
    return vector;
  }

  function dot(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
  }

  function sigmoid(value) {
    return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));
  }

  function train() {
    const examples = EXAMPLES.map(([text, label]) => ({ x: features(text), label }));
    const weights = new Float64Array(vocabulary.length);
    let bias = 0;
    const learningRate = 0.3;
    // Full-batch gradient descent; equivalent to the NumPy training implementation.
    for (let epoch = 0; epoch < 600; epoch++) {
      const gradient = new Float64Array(vocabulary.length);
      let biasGradient = 0;
      for (const sample of examples) {
        const error = sigmoid(dot(sample.x, weights) + bias) - sample.label;
        biasGradient += error;
        for (let i = 0; i < gradient.length; i++) {
          gradient[i] += sample.x[i] * error;
        }
      }
      const factor = learningRate / examples.length;
      for (let i = 0; i < weights.length; i++) weights[i] -= factor * gradient[i];
      bias -= factor * biasGradient;
    }
    return { weights, bias };
  }

  const trained = train();

  function predict(text) {
    if (typeof text !== 'string' || !text.trim()) {
      throw new Error('Enter a non-empty sentence to classify.');
    }
    if (text.length > 2000) {
      throw new Error('The local demonstration accepts at most 2,000 characters.');
    }
    const vector = features(text);
    if (!vector.some(value => value !== 0)) {
      throw new Error('No known words found. Try: "wonderful great" or "awful terrible".');
    }
    const probability = sigmoid(dot(vector, trained.weights) + trained.bias);
    const positive = Math.round(probability * 10000) / 10000;
    return {
      label: probability >= 0.5 ? 'positive' : 'negative',
      confidence: Math.round(Math.max(probability, 1 - probability) * 10000) / 10000,
      positive_probability: positive,
      model_type: 'browser_trained_logistic_regression',
      execution: 'visitor_browser_cpu',
      note: 'Trained from 12 examples. Educational only; not validated for production accuracy.',
    };
  }

  return Object.freeze({
    predict, vocabularySize: vocabulary.length, trainingExamples: EXAMPLES.length,
  });
});
