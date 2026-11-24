// ApicalAI Dashboard — Full-Stack Client
// =========================================

const API = '';  // same origin

// ── State ──────────────────────────────────────────────
const state = {
  models: [],
  workers: [],
  metrics: {},
  latency: {},
  throughput: {},
  page: 'dashboard',
  refreshInterval: null,
  charts: {},
  demoMode: false,
  browserMode: false,
  localEvents: [],
  browserStartedAt: Date.now(),
  terminalLines: [],
};

// ── API helpers ────────────────────────────────────────
async function api(path, opts = {}) {
  try {
    const r = await fetch(API + path, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    return await r.json();
  } catch (e) {
    console.error('API error:', e);
    return null;
  }
}

// ── Actual local inference on GitHub Pages ─────────────────────
function enableBrowserMode() {
  state.browserMode = true;
  state.demoMode = false;
  state.models = [{
    id: 'browser-sentiment-v1',
    name: 'browser-sentiment-v1',
    framework: 'JavaScript (CPU)',
    parameters: window.ApicalBrowserModel.vocabularySize + 1,
    status: 'loaded',
    version: '1.0',
  }];
  state.workers = [{
    id: 'visitor-browser-cpu', worker_id: 'visitor-browser-cpu',
    status: 'healthy', gpu_type: 'Browser CPU',
    gpu_memory_total: 0, gpu_memory_used: 0,
    current_load: null, tasks_completed: state.localEvents.length,
  }];
  updateBrowserMetrics();
  updateConnectionStatus('browser');
  setCounts();
  renderPage();
}

function setCounts() {
  document.getElementById('model-count').textContent = state.models.length;
  document.getElementById('worker-count').textContent = state.workers.length;
}

function updateConnectionStatus(mode) {
  const detail = document.getElementById('connection-status');
  const side = document.getElementById('sidebar-status');
  if (detail) {
    detail.textContent = mode === 'browser' ? 'Local CPU model' : 'Backend connected';
  }
  if (side) {
    side.textContent = mode === 'browser'
      ? 'Inference runs in your browser'
      : 'Connected to the inference API';
  }
}

function percentile(values, percentage) {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const rank = (ordered.length - 1) * percentage / 100;
  const low = Math.floor(rank);
  const high = Math.min(low + 1, ordered.length - 1);
  return ordered[low] + (ordered[high] - ordered[low]) * (rank - low);
}

function updateBrowserMetrics() {
  const now = Date.now();
  const recent = state.localEvents.filter(e => now - e.timestampMs < 60000);
  const latencies = state.localEvents.map(e => e.latencyMs);
  const recentLatencies = recent.map(e => e.latencyMs);
  const secondsActive = Math.max(1, Math.min(60, (now - state.browserStartedAt) / 1000));
  const buckets = new Map();
  for (const event of recent) {
    const second = Math.floor(event.timestampMs / 1000);
    if (!buckets.has(second)) buckets.set(second, []);
    buckets.get(second).push(event.latencyMs);
  }
  state.metrics = {
    total_requests: state.localEvents.length,
    requests_per_second: Number((recent.length / secondsActive).toFixed(3)),
    avg_latency_ms: latencies.length
      ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0,
    uptime_seconds: (now - state.browserStartedAt) / 1000,
    models_loaded: 1,
    active_workers: 1,
  };
  state.latency = {
    p50: percentile(recentLatencies, 50),
    p95: percentile(recentLatencies, 95),
    p99: percentile(recentLatencies, 99),
    history: [...buckets.entries()].sort((a, b) => a[0] - b[0]).map(([second, values]) => ({
      timestamp: second, p50: percentile(values, 50),
      p95: percentile(values, 95), p99: percentile(values, 99),
    })),
  };
  state.throughput = {
    history: [...buckets.entries()].sort((a, b) => a[0] - b[0]).map(([second, values]) => ({
      timestamp: second, requests_per_second: values.length,
    })),
  };
  if (state.workers[0]) state.workers[0].tasks_completed = state.localEvents.length;
}

function recordBrowserPrediction(latencyMs) {
  state.localEvents.push({ timestampMs: Date.now(), latencyMs });
  updateBrowserMetrics();
}

// Do not claim GitHub Pages hosts a Python API or a GPU cluster.
async function refreshData() {
  const staticPage = location.hostname.endsWith('.github.io') || location.protocol === 'file:';
  if (staticPage) {
    enableBrowserMode();
    return;
  }
  const [models, workers, metrics, latency, throughput] = await Promise.all([
    api('/api/models'),
    api('/api/workers'),
    api('/api/metrics'),
    api('/api/metrics/latency'),
    api('/api/metrics/throughput'),
  ]);
  if (!models || !Array.isArray(models.models) || !workers || !Array.isArray(workers.workers) || !metrics) {
    enableBrowserMode();
    return;
  }
  state.browserMode = false;
  state.demoMode = false;
  state.models = models.models;
  state.workers = workers.workers;
  state.metrics = metrics;
  state.latency = latency && Array.isArray(latency.history)
    ? { ...((metrics && metrics.percentiles) || {}), ...latency }
    : { history: [], p50: 0, p95: 0, p99: 0 };
  state.throughput = throughput && Array.isArray(throughput.history) ? throughput : { history: [] };
  updateConnectionStatus('backend');
  setCounts();
  if (state.page !== 'inference') renderPage();
}

// ── Router ─────────────────────────────────────────────
function navigate(page) {
  state.page = page;
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === page);
  });
  renderPage();
}

function renderPage() {
  const c = document.getElementById('content');
  const pages = {
    dashboard: renderDashboard,
    models: renderModels,
    workers: renderWorkers,
    inference: renderInference,
    agents: renderAgents,
    architecture: renderArchitecture,
    topology: renderTopology,
    terminal: renderTerminal,
  };
  const render = pages[state.page] || renderDashboard;
  c.innerHTML = render();
  afterRender();
}

// ── Dashboard ──────────────────────────────────────────
function renderDashboard() {
  const m = state.metrics;
  const l = state.latency;
  const activeWorkers = state.workers.filter(w => w.status === 'healthy' || w.status === 'busy').length;
  const totalGPU = state.workers.reduce((s, w) => s + (w.gpu_memory_total || 0), 0);
  const usedGPU = state.workers.reduce((s, w) => s + (w.gpu_memory_used || 0), 0);
  const gpuPct = totalGPU ? Math.round((usedGPU / totalGPU) * 100) : 0;

  return `
    ${state.browserMode ? '<div style="padding:12px; margin-bottom:12px; background:var(--bg-input); border:1px solid var(--border); border-radius:8px;">LIVE LOCAL MODEL — Sentiment predictions run on your device, not on a remote AI server. Metrics below reflect only this browser session; no GPU cluster is connected. <button class="btn btn-primary btn-sm" style="margin-left:12px;" onclick="navigate(&quot;inference&quot;)">Try a real prediction</button></div>' : '<div style="padding:8px; margin-bottom:12px; color:var(--green);">LIVE — Connected to ApicalAI API</div>'}
    <h1 style="font-size:24px; font-weight:700; margin-bottom:4px;">Dashboard</h1>
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">Real-time inference engine monitoring</p>

    <div class="stats-grid">
      <div class="stat-card accent">
        <div class="stat-label">Total Requests</div>
        <div class="stat-value">${fmt(m.total_requests || 0)}</div>
        <div class="stat-change up">↑ ${m.requests_per_second || 0}/s</div>
      </div>
      <div class="stat-card green">
        <div class="stat-label">Active Workers</div>
        <div class="stat-value">${activeWorkers}/${state.workers.length}</div>
        <div class="stat-change up">↑ All healthy</div>
      </div>
      <div class="stat-card purple">
        <div class="stat-label">Models Loaded</div>
        <div class="stat-value">${state.models.filter(m => m.status === 'loaded').length}</div>
        <div class="stat-change">${state.models.length} registered</div>
      </div>
      <div class="stat-card amber">
        <div class="stat-label">P95 Latency</div>
        <div class="stat-value">${(l.p95 || 0).toFixed(1)}ms</div>
        <div class="stat-change">p99: ${(l.p99 || 0).toFixed(1)}ms</div>
      </div>
      <div class="stat-card red">
        <div class="stat-label">${state.browserMode ? "GPU" : "GPU Utilization"}</div>
        <div class="stat-value">${state.browserMode ? "Not used" : gpuPct + "%"}</div>
        <div class="stat-change">${state.browserMode ? "Inference runs on local CPU" : fmtBytes(usedGPU) + " / " + fmtBytes(totalGPU)}</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <span class="card-title">📈 Latency Distribution</span>
          <span style="font-size:11px; color:var(--text-muted);">Last 60s</span>
        </div>
        <div class="card-body">
          <div class="chart-container"><canvas id="latencyChart"></canvas></div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <span class="card-title">📊 Throughput</span>
          <span style="font-size:11px; color:var(--text-muted);">Requests/sec</span>
        </div>
        <div class="card-body">
          <div class="chart-container"><canvas id="throughputChart"></canvas></div>
        </div>
      </div>
    </div>

    <div class="grid-auto">
      <div class="card">
        <div class="card-header">
          <span class="card-title">⚡ Worker Status</span>
        </div>
        <div class="card-body" style="padding:0;">
          <table class="table">
            <thead><tr>
              <th>Worker</th><th>Status</th><th>GPU</th><th>Load</th><th>Tasks</th>
            </tr></thead>
            <tbody>
              ${state.workers.map(w => `
                <tr>
                  <td style="font-weight:600;">${w.id}</td>
                  <td><span class="badge ${w.status}">${w.status}</span></td>
                  <td>${w.gpu_type || 'CPU'}</td>
                  <td>
                    <div class="progress" style="width:80px;">
                      <div class="progress-fill ${w.current_load > 80 ? 'red' : w.current_load > 50 ? 'amber' : 'green'}"
                           style="width:${w.current_load || 0}%"></div>
                    </div>
                  </td>
                  <td>${w.tasks_completed || 0}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <span class="card-title">🧠 Model Registry</span>
        </div>
        <div class="card-body">
          ${state.models.slice(0, 5).map(m => `
            <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid var(--border);">
              <div>
                <div style="font-weight:600; font-size:13px;">${m.name}</div>
                <div style="font-size:11px; color:var(--text-muted);">${m.framework} · ${m.version || '1.0'}</div>
              </div>
              <span class="badge ${m.status === 'loaded' ? 'loaded' : 'model'}">${m.status}</span>
            </div>
          `).join('')}
        </div>
      </div>
    </div>
  `;
}

// ── Models ─────────────────────────────────────────────
function renderModels() {
  return `
    <h1 style="font-size:24px; font-weight:700; margin-bottom:4px;">Model Registry</h1>
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">${state.browserMode ? 'This is the trained CPU model available in your browser. Custom deployments require a backend.' : 'Manage and deploy AI models across the connected backend.'}</p>
    <div class="card">
      <div class="card-header">
        <span class="card-title">🧠 Registered Models</span>
        ${state.browserMode ? "" : '<button class="btn btn-primary btn-sm" onclick="showLoadModel()">+ Deploy Model</button>'}
      </div>
      <div class="card-body" style="padding:0;">
        <table class="table">
          <thead><tr>
            <th>Model</th><th>Framework</th><th>Parameters</th><th>Status</th><th>Latency (p50)</th><th>Actions</th>
          </tr></thead>
          <tbody>
            ${state.models.map(m => `
              <tr>
                <td>
                  <div style="font-weight:600;">${m.name}</div>
                  <div style="font-size:11px; color:var(--text-muted);">${m.id}</div>
                </td>
                <td><span class="badge model">${m.framework}</span></td>
                <td>${fmtParams(m.parameters)}</td>
                <td><span class="badge ${m.status === 'loaded' ? 'loaded' : m.status === 'loading' ? 'busy' : 'offline'}">${m.status}</span></td>
                <td>${m.avg_latency ? m.avg_latency.toFixed(1) + 'ms' : '—'}</td>
                <td>
                  <button class="btn btn-secondary btn-sm" onclick="runInference('${m.id}')">Run</button>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// ── Workers ────────────────────────────────────────────
function renderWorkers() {
  return `
    <h1 style="font-size:24px; font-weight:700; margin-bottom:4px;">Worker Fleet</h1>
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">${state.browserMode ? 'Your browser is the only active compute worker. No remote cluster or GPU is connected.' : 'Backend-reported worker nodes.'}</p>
    <div class="stats-grid">
      ${state.workers.map(w => `
        <div class="stat-card ${w.status === 'healthy' ? 'green' : w.status === 'busy' ? 'amber' : 'red'}">
          <div style="display:flex; justify-content:space-between; align-items:start;">
            <div class="stat-label">${w.id}</div>
            <span class="badge ${w.status}"><span class="badge-dot"></span> ${w.status}</span>
          </div>
          <div style="margin-top:12px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
              <span style="color:var(--text-muted);">GPU</span>
              <span>${w.gpu_type || 'CPU'}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
              <span style="color:var(--text-muted);">Memory</span>
              <span>${fmtBytes(w.gpu_memory_used || 0)} / ${fmtBytes(w.gpu_memory_total || 0)}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
              <span style="color:var(--text-muted);">Load</span>
              <span>${w.current_load || 0}%</span>
            </div>
            <div class="progress" style="margin-top:8px;">
              <div class="progress-fill ${w.current_load > 80 ? 'red' : w.current_load > 50 ? 'amber' : 'green'}"
                   style="width:${w.current_load || 0}%"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:8px;">
              <span style="color:var(--text-muted);">Tasks</span>
              <span>${w.tasks_completed || 0} completed</span>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

// ── Inference ──────────────────────────────────────────
function renderInference() {
  return `
    <h1 style="font-size:24px; font-weight:700; margin-bottom:4px;">Inference Playground</h1>
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">${state.browserMode ? 'Real sentiment classification runs privately in this browser. Try: wonderful great excellent or awful broken disappointing. This is a small trained demo, not a chatbot.' : 'Run model inference through the connected backend API.'}</p>
    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-title">🔮 Single Inference</span></div>
        <div class="card-body">
          <div style="margin-bottom:16px;">
            <label style="font-size:12px; color:var(--text-muted); display:block; margin-bottom:6px;">Model</label>
            <select id="inf-model" style="width:100%; padding:8px 12px; background:var(--bg-input); border:1px solid var(--border); border-radius:var(--radius-sm); color:var(--text); font-family:var(--font); font-size:13px;">
              ${state.models.map(m => `<option value="${m.id}">${m.name}</option>`).join('')}
            </select>
          </div>
          <div style="margin-bottom:16px;">
            <label style="font-size:12px; color:var(--text-muted); display:block; margin-bottom:6px;">Input</label>
            <textarea id="inf-input" rows="4" placeholder="Enter your prompt or data..."
              style="width:100%; padding:10px 12px; background:var(--bg-input); border:1px solid var(--border); border-radius:var(--radius-sm); color:var(--text); font-family:var(--font); font-size:13px; resize:vertical;"></textarea>
          </div>
          <button class="btn btn-primary" onclick="submitInference()">⚡ Run Inference</button>
          <div id="inf-result" style="margin-top:16px;"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">📦 Batch Processing</span></div>
        <div class="card-body">
          <div style="margin-bottom:16px;">
            <label style="font-size:12px; color:var(--text-muted); display:block; margin-bottom:6px;">Model</label>
            <select id="batch-model" style="width:100%; padding:8px 12px; background:var(--bg-input); border:1px solid var(--border); border-radius:var(--radius-sm); color:var(--text); font-family:var(--font); font-size:13px;">
              ${state.models.map(m => `<option value="${m.id}">${m.name}</option>`).join('')}
            </select>
          </div>
          <div style="margin-bottom:16px;">
            <label style="font-size:12px; color:var(--text-muted); display:block; margin-bottom:6px;">Inputs (one per line)</label>
            <textarea id="batch-input" rows="6" placeholder="Input 1\nInput 2\nInput 3..."
              style="width:100%; padding:10px 12px; background:var(--bg-input); border:1px solid var(--border); border-radius:var(--radius-sm); color:var(--text); font-family:var(--font); font-size:13px; resize:vertical;"></textarea>
          </div>
          <button class="btn btn-primary" onclick="submitBatch()">🚀 Run Batch</button>
          <div id="batch-result" style="margin-top:16px;"></div>
        </div>
      </div>
    </div>
  `;
}

// ── Agents ─────────────────────────────────────────────
function renderAgents() {
  return `
    <h1 style="font-size:24px; font-weight:700; margin-bottom:4px;">Agentic AI</h1>
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">ReAct agents with tool use, planning, and multi-step reasoning</p>
    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-title">🤖 Agent Runner</span></div>
        <div class="card-body">
          <div style="margin-bottom:16px;">
            <label style="font-size:12px; color:var(--text-muted); display:block; margin-bottom:6px;">Task Description</label>
            <textarea id="agent-task" rows="4" placeholder="Describe the task for the agent..."
              style="width:100%; padding:10px 12px; background:var(--bg-input); border:1px solid var(--border); border-radius:var(--radius-sm); color:var(--text); font-family:var(--font); font-size:13px; resize:vertical;"></textarea>
          </div>
          <div style="margin-bottom:16px;">
            <label style="font-size:12px; color:var(--text-muted); display:block; margin-bottom:6px;">Max Steps</label>
            <input id="agent-steps" type="number" value="10" min="1" max="50"
              style="width:100px; padding:8px 12px; background:var(--bg-input); border:1px solid var(--border); border-radius:var(--radius-sm); color:var(--text); font-family:var(--font); font-size:13px;">
          </div>
          <button class="btn btn-primary" onclick="runAgent()">🧠 Execute Agent</button>
          <div id="agent-result" style="margin-top:16px;"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">🛠️ Available Tools</span></div>
        <div class="card-body">
          ${['calculator', 'web_search', 'code_executor', 'file_reader', 'data_analyzer'].map(tool => `
            <div style="display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid var(--border);">
              <div style="width:36px; height:36px; border-radius:8px; background:var(--accent-glow); display:flex; align-items:center; justify-content:center;">🔧</div>
              <div>
                <div style="font-weight:600; font-size:13px;">${tool}</div>
                <div style="font-size:11px; color:var(--text-muted);">Built-in tool for agent reasoning</div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:20px;">
      <div class="card-header"><span class="card-title">🧭 Reasoning Trace</span></div>
      <div class="card-body">
        <div id="reasoning-trace" class="terminal">
          <div class="terminal-line"><span class="terminal-prompt">agent&gt;</span> <span class="terminal-output">Waiting for task...</span></div>
        </div>
      </div>
    </div>
  `;
}

// ── Architecture ───────────────────────────────────────
function renderArchitecture() {
  const layers = [
    { name: 'Layer 7 — Application', color: 'var(--accent)', components: 'REST API · WebSocket · Dashboard UI · CLI · SDK' },
    { name: 'Layer 6 — Agentic AI', color: 'var(--purple)', components: 'ReAct Agent · Planner · Tool Registry · Episodic Memory · Multi-Agent Orchestrator' },
    { name: 'Layer 5 — Generative AI', color: 'var(--pink)', components: 'Text Generator · Image Diffusion · Word2Vec Embeddings · Sampling Strategies' },
    { name: 'Layer 4 — Deep Learning', color: 'var(--coral)', components: 'CNN/ResNet · RNN/LSTM/GRU · Transformer/GPT · VAE · GAN/WGAN' },
    { name: 'Layer 3 — ML Algorithms', color: 'var(--amber)', components: 'Linear/Logistic Regression · Decision Tree · KNN · SVM · Random Forest' },
    { name: 'Layer 2 — Neural Framework', color: 'var(--green)', components: 'Tensor Engine · Autograd · Layers · Optimizers · Loss Functions' },
    { name: 'Layer 1 — Distributed Systems', color: 'var(--red)', components: 'Raft Consensus · Service Discovery · Circuit Breaker · Model Replication' },
    { name: 'Layer 0 — OS Primitives', color: 'var(--text-muted)', components: 'Memory Pool Allocator · Task Scheduler · IPC Message Bus · System Profiler' },
  ];

  return `
    <h1 style="font-size:24px; font-weight:700; margin-bottom:4px;">System Architecture</h1>
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">ApicalAI 8-layer stack — from OS primitives to agentic AI</p>
    <div class="card">
      <div class="card-body">
        ${layers.map(l => `
          <div class="arch-layer" style="border-left-color:${l.color};">
            <div class="layer-name" style="color:${l.color};">${l.name}</div>
            <div class="layer-components">${l.components}</div>
          </div>
        `).join('')}
      </div>
    </div>
    <div class="grid-2" style="margin-top:20px;">
      <div class="card">
        <div class="card-header"><span class="card-title">📂 Project Structure</span></div>
        <div class="card-body">
          <div class="code-block">
apicalai/
├── core/           # OS-level primitives
│   ├── memory/     # Pool allocator, mmap
│   ├── scheduler/  # FIFO, Priority, MLFQ
│   ├── ipc/        # Message bus, shared memory
│   └── profiler/   # CPU/GPU/latency profiling
├── neural/         # Custom neural framework
│   ├── tensor.py   # Tensor + autograd engine
│   ├── layers.py   # Linear, Conv2D, Attention
│   └── optimizers/ # SGD, Adam, AdamW
├── ml/             # Classical ML algorithms
├── deep/           # CNN, RNN, Transformer, VAE, GAN
├── generative/     # Text gen, diffusion, embeddings
├── agentic/        # ReAct, planner, orchestrator
├── distributed/    # Raft, discovery, replication
├── inference/      # Batching, load balancer, pipeline
├── api/            # REST API (pure stdlib)
└── ui/             # Full-stack dashboard
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">🔧 Tech Stack</span></div>
        <div class="card-body">
          ${[
            ['Language', 'Python 3.10+ (zero external deps for core)'],
            ['Math Engine', 'NumPy (only external dependency)'],
            ['API Server', 'Pure Python http.server'],
            ['Frontend', 'Vanilla HTML/CSS/JS (no framework)'],
            ['Consensus', 'Raft protocol (custom implementation)'],
            ['Scheduling', 'Multi-Level Feedback Queue'],
            ['Memory', 'Custom pool allocator with mmap'],
            ['Deployment', 'Docker + GitHub Actions CI/CD'],
          ].map(([k, v]) => `
            <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border); font-size:13px;">
              <span style="color:var(--text-muted);">${k}</span>
              <span style="text-align:right;">${v}</span>
            </div>
          `).join('')}
        </div>
      </div>
    </div>
  `;
}

// ── Topology ───────────────────────────────────────────
function renderTopology() {
  const nodes = state.workers.length ? state.workers : [
    { id: 'master', status: 'healthy' },
    { id: 'worker-1', status: 'healthy' },
    { id: 'worker-2', status: 'busy' },
    { id: 'worker-3', status: 'healthy' },
  ];
  const cx = 300, cy = 180, r = 130;

  return `
    <h1 style="font-size:24px; font-weight:700; margin-bottom:4px;">Cluster Topology</h1>
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">Live view of the distributed inference cluster</p>
    <div class="card">
      <div class="card-body">
        <div class="topology-view" style="display:flex; justify-content:center; padding:40px 0;">
          <svg width="600" height="380" viewBox="0 0 600 380">
            <!-- Connections -->
            ${nodes.map((_, i) => {
              const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
              const x = cx + r * Math.cos(angle);
              const y = cy + r * Math.sin(angle);
              return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="rgba(196,86,122,0.2)" stroke-width="1" stroke-dasharray="4"/>`;
            }).join('')}
            <!-- Center node -->
            <circle cx="${cx}" cy="${cy}" r="30" fill="rgba(196,86,122,0.08)" stroke="var(--accent)" stroke-width="2"/>
            <text x="${cx}" y="${cy + 4}" text-anchor="middle" fill="var(--accent)" font-size="11" font-weight="600">MASTER</text>
            <!-- Worker nodes -->
            ${nodes.map((n, i) => {
              const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
              const x = cx + r * Math.cos(angle);
              const y = cy + r * Math.sin(angle);
              const color = n.status === 'healthy' ? 'var(--green)' : n.status === 'busy' ? 'var(--amber)' : 'var(--red)';
              return `
                <circle cx="${x}" cy="${y}" r="24" fill="#ffffff" stroke="${color}" stroke-width="2"/>
                <text x="${x}" y="${y - 4}" text-anchor="middle" fill="var(--text)" font-size="9" font-weight="600">${n.id}</text>
                <text x="${x}" y="${y + 10}" text-anchor="middle" fill="${color}" font-size="8">${n.status}</text>
              `;
            }).join('')}
          </svg>
        </div>
      </div>
    </div>
  `;
}

// ── Terminal ───────────────────────────────────────────
function renderTerminal() {
  return `
    <h1 style="font-size:24px; font-weight:700; margin-bottom:4px;">System Terminal</h1>
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">Interactive system console</p>
    <div class="card">
      <div class="card-body" style="padding:0;">
        <div class="terminal" id="sys-terminal" style="min-height:400px; max-height:none;">
          <div class="terminal-line"><span class="terminal-success">ApicalAI Inference Engine v1.0</span></div>
          <div class="terminal-line"><span class="terminal-output">Type 'help' for available commands</span></div>
          <div class="terminal-line"><span class="terminal-output">─────────────────────────────────</span></div>
          ${state.terminalLines.map(l => `<div class="terminal-line">${l}</div>`).join('')}
        </div>
        <div style="display:flex; border-top:1px solid var(--border); background:#1e161a;">
          <span style="padding:10px 12px; color:var(--accent); font-family:var(--mono); font-size:12px;">apicalai&gt;</span>
          <input id="term-input" type="text" placeholder="Enter command..."
            style="flex:1; padding:10px 0; background:transparent; border:none; color:var(--text); font-family:var(--mono); font-size:12px; outline:none;"
            onkeydown="if(event.key==='Enter')execCommand()">
        </div>
      </div>
    </div>
  `;
}

// ── Actions ────────────────────────────────────────────
async function submitInference() {
  const model = document.getElementById('inf-model').value;
  const input = document.getElementById('inf-input').value;
  const el = document.getElementById('inf-result');
  el.innerHTML = '<div class="badge busy">Processing...</div>';
  let r;
  if (state.browserMode) {
    if (model !== 'browser-sentiment-v1') {
      el.textContent = 'This browser cannot execute the selected model.';
      return;
    }
    try {
      const started = performance.now();
      const output = window.ApicalBrowserModel.predict(input);
      const latency_ms = performance.now() - started;
      recordBrowserPrediction(latency_ms);
      r = { output, latency_ms, worker_id: 'visitor-browser-cpu' };
    } catch (error) {
      el.textContent = error.message;
      return;
    }
  } else {
    r = await api('/api/inference', {
      method: 'POST',
      body: JSON.stringify({ model: model, input: { text: input } }),
    });
  }
  if (!r || !r.output) {
    el.textContent = (r && r.error) || 'Inference is unavailable. Start the API and deploy a model.';
    return;
  }
  el.innerHTML = `
    <div class="code-block" style="margin-top:8px;">
      <div style="color:var(--green); margin-bottom:8px;">✓ Inference complete in ${r.latency_ms?.toFixed(1) || '?'}ms</div>
      <div>${escapeHtml(JSON.stringify(r.output, null, 2))}</div>
    </div>`;
}

async function submitBatch() {
  const model = document.getElementById('batch-model').value;
  const inputs = document.getElementById('batch-input').value.split('\n').map(t => t.trim()).filter(Boolean);
  const el = document.getElementById('batch-result');
  el.textContent = 'Processing batch...';
  if (inputs.length < 1 || inputs.length > 32) {
    el.textContent = 'Enter 1 to 32 non-empty lines.';
    return;
  }
  let r;
  if (state.browserMode) {
    if (model !== 'browser-sentiment-v1') {
      el.textContent = 'This browser model is unavailable.';
      return;
    }
    const prepared = [];
    try {
      for (const input of inputs) {
        const started = performance.now();
        const output = window.ApicalBrowserModel.predict(input);
        prepared.push({ output, latency_ms: performance.now() - started });
      }
    } catch (error) {
      el.textContent = error.message;
      return;
    }
    prepared.forEach(item => recordBrowserPrediction(item.latency_ms));
    r = { results: prepared, execution_mode: 'sequential_browser_cpu' };
  } else {
    r = await api('/api/batch', {
      method: 'POST',
      body: JSON.stringify({ requests: inputs.map(t => ({ model: model, input: { text: t } })) }),
    });
  }
  if (!r || !Array.isArray(r.results)) {
    el.textContent = (r && r.error) || 'Batch inference unavailable';
    return;
  }
  el.innerHTML = '<div class="code-block" style="margin-top:8px;">'
    + '<div style="color:var(--green);margin-bottom:8px;">'
    + 'Completed ' + r.results.length + ' real predictions'
    + (r.execution_mode ? ' · ' + escapeHtml(r.execution_mode) : '') + '</div>'
    + r.results.map((result, index) => '<div style="margin-bottom:6px;">[' + index + '] '
      + escapeHtml(JSON.stringify(result.output)) + '</div>').join('')
    + '</div>';
}

async function runAgent() {
  const task = document.getElementById('agent-task').value;
  const maxSteps = document.getElementById('agent-steps').value;
  const el = document.getElementById('agent-result');
  const trace = document.getElementById('reasoning-trace');
  el.textContent = 'Checking agent availability...';
  if (state.browserMode) {
    el.textContent = 'Agent execution requires a deployed backend. This page currently runs only a local sentiment classifier.';
    return;
  }
  trace.textContent = 'Agent execution trace is only available after a real backend is connected.';
  const r = await api('/api/agents/run', {
    method: 'POST',
    body: JSON.stringify({ goal: task, max_steps: parseInt(maxSteps) }),
  });
  if (!r || r.error || !r.result) {
    el.textContent = (r && r.error) || 'Agent backend unavailable.';
    return;
  }
  el.textContent = JSON.stringify(r.result, null, 2);
  trace.textContent = JSON.stringify(r.steps || [], null, 2);
}

function execCommand() {
  const input = document.getElementById('term-input');
  const cmd = input.value.trim();
  if (!cmd) return;
  input.value = '';

  const term = document.getElementById('sys-terminal');
  term.innerHTML += `<div class="terminal-line"><span class="terminal-prompt">apicalai&gt;</span> <span class="terminal-output">${cmd}</span></div>`;

  const commands = {
    help: () => `<span class="terminal-success">Available commands:</span>
  status   — System health overview
  models   — List loaded models
  workers  — Show worker fleet
  metrics  — Current performance metrics
  arch     — Architecture summary
  clear    — Clear terminal`,
    status: () => `<span class="terminal-success">${state.browserMode ? 'Local model ready (no remote server)' : 'Backend connected'}</span>
  Workers: ${state.workers.length} active
  Models: ${state.models.length} registered
  Uptime: ${state.demoMode ? 'sample preview' : Math.floor((state.metrics.uptime_seconds || 0) / 60) + 'm'}`,
    models: () => state.models.map(m => `  <span class="terminal-success">●</span> ${m.name} [${m.status}]`).join('\n'),
    workers: () => state.workers.map(w => `  <span class="${w.status === 'healthy' ? 'terminal-success' : 'terminal-error'}">●</span> ${w.id} — ${w.status} (${w.current_load == null ? 'load not measured' : w.current_load + '% load'})`).join('\n'),
    metrics: () => `  Requests: ${fmt(state.metrics.total_requests || 0)}
  RPS: ${state.metrics.requests_per_second || 0}
  P50: ${(state.latency.p50 || 0).toFixed(1)}ms
  P95: ${(state.latency.p95 || 0).toFixed(1)}ms
  P99: ${(state.latency.p99 || 0).toFixed(1)}ms`,
    arch: () => `<span class="terminal-success">ApicalAI Architecture — 8 Layers</span>
  L7 Application    │ REST API, Dashboard, CLI
  L6 Agentic AI     │ ReAct, Planner, Orchestrator
  L5 Generative     │ Text Gen, Diffusion, Embeddings
  L4 Deep Learning  │ CNN, RNN, Transformer, VAE, GAN
  L3 ML Algorithms  │ Regression, Trees, SVM, KNN
  L2 Neural Framework│ Tensor, Autograd, Layers
  L1 Distributed    │ Raft, Discovery, Replication
  L0 OS Primitives  │ Memory, Scheduler, IPC, Profiler`,
    clear: () => { term.innerHTML = ''; return null; },
  };

  const handler = commands[cmd.toLowerCase()];
  if (handler) {
    const output = handler();
    if (output) term.innerHTML += `<div class="terminal-line" style="white-space:pre;"><span class="terminal-output">${output}</span></div>`;
  } else {
    term.innerHTML += `<div class="terminal-line"><span class="terminal-error">Unknown command: ${cmd}. Type 'help' for available commands.</span></div>`;
  }
  term.scrollTop = term.scrollHeight;
  state.terminalLines.push(`<span class="terminal-prompt">apicalai&gt;</span> <span class="terminal-output">${cmd}</span>`);
}

function showLoadModel() {
  alert('Model deployment wizard — coming soon!');
}
function runInference(modelId) {
  navigate('inference');
  setTimeout(() => {
    const sel = document.getElementById('inf-model');
    if (sel) sel.value = modelId;
  }, 100);
}

// ── Charts (simple canvas) ─────────────────────────────
function drawChart(canvasId, data, color, label) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.scale(dpr, dpr);

  const w = rect.width, h = rect.height;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  if (!data || data.length === 0) {
    ctx.fillStyle = 'rgba(44,31,38,0.3)';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('No measured data yet', rect.width / 2, rect.height / 2);
    return;
  }
  const max = Math.max(1, ...data) * 1.2;
  const min = 0;

  // Grid
  ctx.strokeStyle = 'rgba(232,220,225,0.8)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = 'rgba(44,31,38,0.35)';
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(Math.round(max - (max / 4) * i), pad.left - 8, y + 4);
  }

  // Line
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  data.forEach((v, i) => {
    const x = pad.left + (plotW / Math.max(1, data.length - 1)) * i;
    const y = pad.top + plotH - ((v - min) / (max - min)) * plotH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill
  const lastX = pad.left + plotW;
  const lastY = pad.top + plotH - ((data[data.length - 1] - min) / (max - min)) * plotH;
  ctx.lineTo(lastX, pad.top + plotH);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
  grad.addColorStop(0, color.replace(')', ',0.2)').replace('rgb', 'rgba'));
  grad.addColorStop(1, color.replace(')', ',0.01)').replace('rgb', 'rgba'));
  ctx.fillStyle = grad;
  ctx.fill();

  // Label
  ctx.fillStyle = 'rgba(44,31,38,0.35)';
  ctx.font = '10px Inter';
  ctx.textAlign = 'center';
  ctx.fillText(label, w / 2, h - 4);
}

// ── Utilities ──────────────────────────────────────────
function escapeHtml(text) {
  return String(text).replace(/[&<>\"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', "'": '&#39;' })[ch]);
}

function fmt(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function fmtBytes(b) {
  if (b >= 1e9) return (b / 1e9).toFixed(1) + 'GB';
  if (b >= 1e6) return (b / 1e6).toFixed(0) + 'MB';
  return b + 'B';
}

function fmtParams(p) {
  if (!p) return '—';
  if (p >= 1e9) return (p / 1e9).toFixed(1) + 'B';
  if (p >= 1e6) return (p / 1e6).toFixed(0) + 'M';
  return fmt(p);
}

function afterRender() {
  if (state.page === 'dashboard') {
    setTimeout(() => {
      drawChart('latencyChart', (state.latency.history || []).map(p => p.p95), 'rgb(196, 86, 122)', 'Latency (ms)');
      drawChart('throughputChart', (state.throughput.history || []).map(p => p.requests_per_second), 'rgb(155, 123, 184)', 'Throughput (req/s)');
    }, 50);
  }
}

// ── Clock ──────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', () => navigate(el.dataset.page));
  });

  // Mobile menu
  const toggle = document.getElementById('menu-toggle');
  if (window.innerWidth <= 768) toggle.style.display = 'block';
  toggle.addEventListener('click', () => {
    document.getElementById('sidebar').classList.toggle('open');
  });

  updateClock();
  setInterval(updateClock, 1000);
  refreshData();
  if (!(location.hostname.endsWith('.github.io') || location.protocol === 'file:')) {
    state.refreshInterval = setInterval(refreshData, 5000);
  }
});
