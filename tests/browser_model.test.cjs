/* Regression tests for real in-browser model; no external test framework. */
const assert = require('node:assert/strict');
const source = require('../docs/static/js/browser_model.js');
const local = require('../ui/static/js/browser_model.js');

for (const model of [source, local]) {
  assert.equal(model.trainingExamples, 12);
  assert.ok(model.vocabularySize > 20);
  const positive = model.predict('wonderful great excellent');
  const negative = model.predict('awful broken disappointing');
  assert.equal(positive.label, 'positive');
  assert.equal(negative.label, 'negative');
  assert.ok(positive.confidence > 0.5);
  assert.ok(negative.confidence > 0.5);
  assert.equal(positive.execution, 'visitor_browser_cpu');
  assert.equal(model.predict('wonderful great excellent').positive_probability, positive.positive_probability);
  assert.throws(() => model.predict(''), /non-empty/);
  assert.throws(() => model.predict('zzzzzzzz'), /No known words/);
  assert.throws(() => model.predict('good'.repeat(1000)), /2,000/);
}
console.log('Browser sentiment model tests passed (both dashboard copies)');
