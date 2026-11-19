// NeuroFlux Dashboard — Full-Stack Client
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

// ── Mock data for static/demo deployment ──────────────
const MOCK = {
  models: [
    { id: 'gpt-neuroflux-7b', name: 'gpt-neuroflux-7b', framework: 'NeuroFlux', parameters: 7e9, status: 'loaded', avg_latency: 42.3, version: '2.1' },
    { id: 'resnet50-classifier', name: 'resnet50-classifier', framework: 'NeuroFlux', parameters: 26e6, status: 'loaded', avg_latency: 8.7, version: '1.4' },
    { id: 'bert-sentiment', name: 'bert-sentiment', framework: 'NeuroFlux', parameters: 110e6, status: 'loaded', avg_latency: 15.2, version: '3.0' },
    { id: 'vae-imagegen', name: 'vae-imagegen', framework: 'NeuroFlux', parameters: 15e6, status: 'loaded', avg_latency: 23.1, version: '1.2' },
    { id: 'agent-planner-v2', name: 'agent-planner-v2', framework: 'NeuroFlux', parameters: 350e6, status: 'loaded', avg_latency: 67.5, version: '2.0' },
  ],
  workers: [
    { id: 'gpu-worker-01', status: 'healthy', gpu_type: 'A100-80GB', gpu_memory_total: 85899345920, gpu_memory_used: 42949672960, current_load: 62, tasks_completed: 14823 },
    { id: 'gpu-worker-02', status: 'healthy', gpu_type: 'A100-80GB', gpu_memory_total: 85899345920, gpu_memory_used: 34359738368, current_load: 45, tasks_completed: 12104 },
    { id: 'gpu-worker-03', status: 'busy', gpu_type: 'V100-32GB', gpu_memory_total: 34359738368, gpu_memory_used: 30064771072, current_load: 88, tasks_completed: 9847 },
    { id: 'gpu-worker-04', status: 'healthy', gpu_type: 'V100-32GB', gpu_memory_total: 34359738368, gpu_memory_used: 17179869184, current_load: 38, tasks_completed: 11290 },
    { id: 'cpu-worker-05', status: 'healthy', gpu_type: 'CPU-only', gpu_memory_total: 68719476736, gpu_memory_used: 21474836480, current_load: 29, tasks_completed: 6531 },
    { id: 'gpu-worker-06', status: 'healthy', gpu_type: 'A10G-24GB', gpu_memory_total: 25769803776, gpu_memory_used: 15461882266, current_load: 55, tasks_completed: 8912 },
  ],
  metrics: { total_requests: 83472, requests_per_second: 142, avg_latency_ms: 23.4, error_rate: 0.02 },
  latency: { p50: 18.3, p95: 45.7, p99: 89.2 },
  throughput: { current: 142, peak: 312, avg: 127 },
};

async function refreshData() {
  const [models, workers, metrics, latency, throughput] = await Promise.all([
    api('/api/models'),
    api('/api/workers'),
    api('/api/metrics'),
    api('/api/metrics/latency'),
    api('/api/metrics/throughput'),
  ]);
  // Use API data if available, otherwise fall back to mock data (for static/demo deploy)
  state.models = (models && models.models) ? models.models : MOCK.models;
  state.workers = (workers && workers.workers) ? workers.workers : MOCK.workers;
  state.metrics = metrics || MOCK.metrics;
  state.latency = latency || MOCK.latency;
  state.throughput = throughput || MOCK.throughput;
  document.getElementById('model-count').textContent = state.models.length;
  document.getElementById('worker-count').textContent = state.workers.length;
  renderPage();
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
        <div class="stat-label">GPU Utilization</div>
        <div class="stat-value">${gpuPct}%</div>
        <div class="stat-change">${fmtBytes(usedGPU)} / ${fmtBytes(totalGPU)}</div>
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
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">Manage and deploy AI models across the cluster</p>
    <div class="card">
      <div class="card-header">
        <span class="card-title">🧠 Registered Models</span>
        <button class="btn btn-primary btn-sm" onclick="showLoadModel()">+ Deploy Model</button>
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
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">GPU/CPU worker nodes in the inference cluster</p>
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
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">Run AI inference through the distributed engine</p>
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
    <p style="color:var(--text-muted); margin-bottom:24px; font-size:13px;">NeuroFlux 8-layer stack — from OS primitives to agentic AI</p>
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
neuroflux/
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
              return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="rgba(34,211,238,0.2)" stroke-width="1" stroke-dasharray="4"/>`;
            }).join('')}
            <!-- Center node -->
            <circle cx="${cx}" cy="${cy}" r="30" fill="rgba(34,211,238,0.1)" stroke="var(--accent)" stroke-width="2"/>
            <text x="${cx}" y="${cy + 4}" text-anchor="middle" fill="var(--accent)" font-size="11" font-weight="600">MASTER</text>
            <!-- Worker nodes -->
            ${nodes.map((n, i) => {
              const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
              const x = cx + r * Math.cos(angle);
              const y = cy + r * Math.sin(angle);
              const color = n.status === 'healthy' ? 'var(--green)' : n.status === 'busy' ? 'var(--amber)' : 'var(--red)';
              return `
                <circle cx="${x}" cy="${y}" r="24" fill="rgba(17,24,39,0.9)" stroke="${color}" stroke-width="2"/>
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
          <div class="terminal-line"><span class="terminal-success">NeuroFlux Inference Engine v1.0</span></div>
          <div class="terminal-line"><span class="terminal-output">Type 'help' for available commands</span></div>
          <div class="terminal-line"><span class="terminal-output">─────────────────────────────────</span></div>
          ${state.terminalLines.map(l => `<div class="terminal-line">${l}</div>`).join('')}
        </div>
        <div style="display:flex; border-top:1px solid var(--border); background:#000;">
          <span style="padding:10px 12px; color:var(--accent); font-family:var(--mono); font-size:12px;">neuroflux&gt;</span>
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
  let r = await api('/api/inference', {
    method: 'POST',
    body: JSON.stringify({ model_id: model, input: { text: input } }),
  });
  // Mock response for static demo
  if (!r || !r.output) {
    const latency = (Math.random() * 40 + 10).toFixed(1);
    r = { output: { text: `[Demo] Inference result for "${input.slice(0,50)}..." — Generated by ${model}`, confidence: (Math.random() * 0.3 + 0.7).toFixed(3), tokens: Math.floor(Math.random() * 100 + 20) }, latency_ms: parseFloat(latency) };
  }
  el.innerHTML = `
    <div class="code-block" style="margin-top:8px;">
      <div style="color:var(--green); margin-bottom:8px;">✓ Inference complete in ${r.latency_ms?.toFixed(1) || '?'}ms</div>
      <div>${JSON.stringify(r.output, null, 2)}</div>
    </div>`;
}

async function submitBatch() {
  const model = document.getElementById('batch-model').value;
  const inputs = document.getElementById('batch-input').value.split('\n').filter(Boolean);
  const el = document.getElementById('batch-result');
  el.innerHTML = '<div class="badge busy">Processing batch...</div>';
  const r = await api('/api/batch', {
    method: 'POST',
    body: JSON.stringify({ model_id: model, inputs: inputs.map(t => ({ text: t })) }),
  });
  if (r && r.results) {
    el.innerHTML = `
      <div class="code-block" style="margin-top:8px;">
        <div style="color:var(--green); margin-bottom:8px;">✓ Batch complete — ${r.results.length} results in ${r.total_latency_ms?.toFixed(1) || '?'}ms</div>
        ${r.results.map((res, i) => `<div style="margin-bottom:4px;"><span style="color:var(--accent);">[${i}]</span> ${JSON.stringify(res.output)}</div>`).join('')}
      </div>`;
  }
}

async function runAgent() {
  const task = document.getElementById('agent-task').value;
  const maxSteps = document.getElementById('agent-steps').value;
  const el = document.getElementById('agent-result');
  const trace = document.getElementById('reasoning-trace');
  el.innerHTML = '<div class="badge busy">Agent thinking...</div>';
  trace.innerHTML += `<div class="terminal-line"><span class="terminal-prompt">agent&gt;</span> <span class="terminal-output">Task: ${task}</span></div>`;
  let r = await api('/api/agents/run', {
    method: 'POST',
    body: JSON.stringify({ task, max_steps: parseInt(maxSteps) }),
  });
  // Mock response for static demo
  if (!r || !r.result) {
    r = {
      result: `Task completed: "${task.slice(0,60)}"`,
      steps: [
        { type: 'thought', content: `Analyzing task: "${task.slice(0,80)}"` },
        { type: 'action', content: 'Using tool: web_search("' + task.split(' ').slice(0,3).join(' ') + '")' },
        { type: 'observation', content: 'Found 3 relevant results. Processing data...' },
        { type: 'thought', content: 'Synthesizing findings and formulating response.' },
        { type: 'action', content: 'Using tool: data_analyzer(results)' },
        { type: 'result', content: 'Task completed successfully with high confidence.' },
      ],
    };
  }
  el.innerHTML = `<div class="code-block" style="margin-top:8px; color:var(--green);">✓ ${r.result}</div>`;
  if (r.steps) {
    r.steps.forEach(s => {
      trace.innerHTML += `<div class="terminal-line"><span class="terminal-prompt">  ${s.type}&gt;</span> <span class="terminal-output">${s.content}</span></div>`;
    });
  }
}

function execCommand() {
  const input = document.getElementById('term-input');
  const cmd = input.value.trim();
  if (!cmd) return;
  input.value = '';

  const term = document.getElementById('sys-terminal');
  term.innerHTML += `<div class="terminal-line"><span class="terminal-prompt">neuroflux&gt;</span> <span class="terminal-output">${cmd}</span></div>`;

  const commands = {
    help: () => `<span class="terminal-success">Available commands:</span>
  status   — System health overview
  models   — List loaded models
  workers  — Show worker fleet
  metrics  — Current performance metrics
  arch     — Architecture summary
  clear    — Clear terminal`,
    status: () => `<span class="terminal-success">System Status: ONLINE</span>
  Workers: ${state.workers.length} active
  Models: ${state.models.length} registered
  Uptime: ${Math.floor(Math.random() * 24)}h ${Math.floor(Math.random() * 60)}m`,
    models: () => state.models.map(m => `  <span class="terminal-success">●</span> ${m.name} [${m.status}]`).join('\n'),
    workers: () => state.workers.map(w => `  <span class="${w.status === 'healthy' ? 'terminal-success' : 'terminal-error'}">●</span> ${w.id} — ${w.status} (${w.current_load}% load)`).join('\n'),
    metrics: () => `  Requests: ${fmt(state.metrics.total_requests || 0)}
  RPS: ${state.metrics.requests_per_second || 0}
  P50: ${(state.latency.p50 || 0).toFixed(1)}ms
  P95: ${(state.latency.p95 || 0).toFixed(1)}ms
  P99: ${(state.latency.p99 || 0).toFixed(1)}ms`,
    arch: () => `<span class="terminal-success">NeuroFlux Architecture — 8 Layers</span>
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
  state.terminalLines.push(`<span class="terminal-prompt">neuroflux&gt;</span> <span class="terminal-output">${cmd}</span>`);
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
    data = Array.from({ length: 30 }, () => Math.random() * 100 + 20);
  }
  const max = Math.max(...data) * 1.2;
  const min = 0;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
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
    const x = pad.left + (plotW / (data.length - 1)) * i;
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
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '10px Inter';
  ctx.textAlign = 'center';
  ctx.fillText(label, w / 2, h - 4);
}

// ── Utilities ──────────────────────────────────────────
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
      drawChart('latencyChart', null, 'rgb(34, 211, 238)', 'Latency (ms)');
      drawChart('throughputChart', null, 'rgb(167, 139, 250)', 'Throughput (req/s)');
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
  state.refreshInterval = setInterval(refreshData, 5000);
});
