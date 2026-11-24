/* Smoke test the GitHub Pages dashboard without browser libraries or network. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { performance } = require('node:perf_hooks');
const path = require('node:path');
const classifier = require('../docs/static/js/browser_model.js');

async function main() {
  const elements = new Map();
  function getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      innerHTML: '', textContent: '', value: '', style: {},
    });
    return elements.get(id);
  }
  const context = vm.createContext({
    console,
    window: { ApicalBrowserModel: classifier, innerWidth: 1200 },
    location: { hostname: 'shikhapallavi123.github.io', protocol: 'https:' },
    document: {
      getElementById,
      addEventListener() {},
      querySelectorAll() { return []; },
    },
    performance,
    setTimeout() {}, // Avoid canvas rendering in this minimal DOM mock.
    setInterval() {},
    clearInterval() {},
  });
  const script = fs.readFileSync(path.join(__dirname, '../docs/static/js/app.js'), 'utf8');
  vm.runInContext(script, context);

  await vm.runInContext('refreshData()', context);
  assert.equal(vm.runInContext('state.browserMode', context), true);
  assert.equal(vm.runInContext('state.models.length', context), 1);
  assert.equal(vm.runInContext('state.workers.length', context), 1);
  assert.equal(vm.runInContext('state.metrics.total_requests', context), 0);
  assert.match(getElementById('content').innerHTML, /LIVE LOCAL MODEL/);
  assert.match(getElementById('sidebar-status').textContent, /your browser/);

  getElementById('inf-model').value = 'browser-sentiment-v1';
  getElementById('inf-input').value = 'wonderful great excellent';
  await vm.runInContext('submitInference()', context);
  assert.match(getElementById('inf-result').innerHTML, /positive/);
  assert.equal(vm.runInContext('state.metrics.total_requests', context), 1);

  getElementById('inf-input').value = 'zzzzzzzz';
  await vm.runInContext('submitInference()', context);
  assert.match(getElementById('inf-result').textContent, /No known words/);
  assert.equal(vm.runInContext('state.metrics.total_requests', context), 1);

  getElementById('batch-model').value = 'browser-sentiment-v1';
  getElementById('batch-input').value = 'awful broken disappointing\nwonderful great excellent';
  await vm.runInContext('submitBatch()', context);
  assert.match(getElementById('batch-result').innerHTML, /Completed 2 real predictions/);
  assert.equal(vm.runInContext('state.metrics.total_requests', context), 3);

  // UI must never claim remote workers, metrics, or GPU capacity.
  assert.equal(vm.runInContext('state.workers[0].gpu_memory_total', context), 0);
  assert.equal(vm.runInContext('state.workers[0].id', context), 'visitor-browser-cpu');
  vm.runInContext('navigate("dashboard")', context);
  assert.match(getElementById('content').innerHTML, /Inference runs on local CPU/);
  assert.doesNotMatch(getElementById('content').innerHTML, /gpu-worker-01/);
  console.log('GitHub Pages real-inference smoke test passed');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
