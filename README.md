# ⚡ ApicalAI — Distributed AI Inference Engine

> A production-grade, full-stack distributed AI inference engine built **entirely from scratch** — from OS-level memory allocators to agentic AI orchestration. Zero ML framework dependencies. Pure Python + NumPy.

> **Development status (November 2026):** The public GitHub Pages site is a **static demo with illustrative data**, not a live GPU cluster. The backend currently serves a small real NumPy sentiment classifier and optionally connects to a locally configured Ollama pretrained language model. Distributed scheduling, live GPU measurements, and production readiness are not yet validated. See [Real Inference MVP](docs/REAL_INFERENCE_MVP.md) for exact working features, setup, limitations and testing.\n\n[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-ApicalAI_Dashboard-00d4aa?style=for-the-badge)](https://shikhapallavi123.github.io/distributed-ai-platform/)

[![CI/CD](https://github.com/ShikhaPallavi123/distributed-ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/ShikhaPallavi123/distributed-ai-platform/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

---

## 🏗️ Architecture — 8 Layers, Built From Scratch

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 7 — Application                                      │
│  REST API · WebSocket · Full-Stack Dashboard · CLI           │
├─────────────────────────────────────────────────────────────┤
│  Layer 6 — Agentic AI                                       │
│  ReAct Agent · Planner · Tool Registry · Multi-Agent Orch.  │
├─────────────────────────────────────────────────────────────┤
│  Layer 5 — Generative AI                                    │
│  Text Generator · Diffusion · Word2Vec · Semantic Search    │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 — Deep Learning                                    │
│  CNN/ResNet · RNN/LSTM/GRU · Transformer/GPT · VAE · GAN   │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — Classical ML                                     │
│  Linear/Logistic Reg. · Decision Tree · KNN · SVM · Forest  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Neural Network Framework                         │
│  Tensor Engine · Autograd · Layers · Optimizers · Losses    │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — Distributed Systems                              │
│  Raft Consensus · Service Discovery · Circuit Breaker       │
├─────────────────────────────────────────────────────────────┤
│  Layer 0 — OS Primitives                                    │
│  Memory Pool Allocator · Task Scheduler · IPC · Profiler    │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 What Makes This Rare

Most AI projects use PyTorch/TensorFlow as a black box. **ApicalAI builds everything from the ground up:**

- **Custom Tensor Engine** with automatic differentiation (like a mini-PyTorch)
- **OS-level primitives** — memory pool allocator, MLFQ scheduler, IPC message bus
- **Full neural network framework** — not wrapping any ML library
- **6 classical ML algorithms** implemented from mathematical foundations
- **5 deep learning architectures** (CNN, RNN, Transformer, VAE, GAN) from scratch
- **Agentic AI** — ReAct agents, hierarchical planning, tree-of-thought reasoning
- **Distributed systems** — Raft consensus, service discovery, circuit breakers
- **Production inference engine** — dynamic batching, load balancing, model registry
- **Full-stack dashboard** — real-time monitoring with zero framework dependencies
- **Pure Python API** — no Flask, no FastAPI, just `http.server`

**Total: 50+ modules, 15,000+ lines of code, 1 external dependency (NumPy)**

---

## 📂 Project Structure

```
apicalai/
├── core/                    # Layer 0 — OS Primitives
│   ├── memory/
│   │   └── pool_allocator.py    # Memory pool with first-fit/best-fit, mmap, coalescing
│   ├── scheduler/
│   │   └── task_scheduler.py    # FIFO, Priority, Round-Robin, SJF, MLFQ scheduling
│   ├── ipc/
│   │   └── message_bus.py       # Pub/sub, request/reply, shared memory ring buffer
│   └── profiler/
│       └── system_profiler.py   # CPU/memory sampling, flame graph, latency stats
│
├── neural/                  # Layer 2 — Neural Network Framework
│   ├── tensor.py                # Tensor class with autograd computation graph
│   ├── autograd.py              # 15 gradient functions, topological sort backward
│   ├── layers.py                # Linear, Conv2D, BatchNorm, LayerNorm, Attention
│   ├── activations.py           # ReLU, GELU, Sigmoid, Tanh, Softmax, SiLU
│   ├── optimizers.py            # SGD (momentum), Adam, AdamW, LR schedulers
│   ├── losses.py                # MSE, CrossEntropy, BCE, Huber, KLDiv
│   └── model.py                 # Model base with save/load, summary
│
├── ml/                      # Layer 3 — Classical ML
│   ├── linear_regression.py     # Normal equation + gradient descent
│   ├── logistic_regression.py   # Binary + multiclass (one-vs-rest)
│   ├── decision_tree.py         # Gini/entropy, pruning, feature importance
│   ├── knn.py                   # K-nearest neighbors, multiple distance metrics
│   ├── svm.py                   # Support Vector Machine (simplified SMO)
│   └── ensemble.py              # Random Forest, Gradient Boosting, Bagging
│
├── deep/                    # Layer 4 — Deep Learning
│   ├── cnn.py                   # CNN + ResNet with skip connections
│   ├── rnn.py                   # RNN, LSTM (gated), GRU cells
│   ├── transformer.py           # Full Transformer + GPT with generation
│   ├── vae.py                   # Variational Autoencoder (ELBO loss)
│   └── gan.py                   # GAN + WGAN with Wasserstein distance
│
├── generative/              # Layer 5 — Generative AI
│   ├── text_generator.py        # Transformer text gen, beam search, sampling
│   ├── image_generator.py       # Simple diffusion model + Image VAE
│   └── embedding.py             # Word2Vec, semantic search, vector index
│
├── agentic/                 # Layer 6 — Agentic AI
│   ├── agent.py                 # ReAct agent with thought/action/observe loop
│   ├── tools.py                 # Tool registry + 5 built-in tools
│   ├── planner.py               # Hierarchical task planner with DAG ordering
│   ├── memory.py                # Short-term, long-term, episodic, working memory
│   ├── orchestrator.py          # Multi-agent orchestration, debate protocol
│   └── reasoning.py             # Chain-of-thought, tree-of-thought, reflection
│
├── distributed/             # Layer 1 — Distributed Systems
│   ├── consensus.py             # Raft leader election + log replication
│   ├── discovery.py             # Service registry with gossip protocol
│   ├── fault_tolerance.py       # Circuit breaker, retry, bulkhead patterns
│   ├── replication.py           # Data/model/pipeline parallel sharding
│   ├── node.py                  # Cluster node with heartbeat
│   └── cluster.py               # Ring/mesh topology, consistent hashing
│
├── inference/               # Inference Engine
│   ├── engine.py                # Core inference orchestrator
│   ├── batching.py              # Dynamic + continuous batching
│   ├── load_balancer.py         # 6 load balancing strategies
│   ├── model_registry.py        # Versioned model registry with A/B testing
│   ├── pipeline.py              # Sequential + parallel inference pipelines
│   ├── coordinator.py           # Worker health tracking + auto-scaling
│   ├── task_queue.py            # Priority queue with DLQ + rate limiting
│   └── worker.py                # Inference worker with GPU memory tracking
│
├── api/                     # Layer 7 — REST API
│   └── app.py                   # Pure Python HTTP server, 15+ endpoints
│
├── ui/                      # Full-Stack Dashboard
│   ├── templates/index.html     # SPA with 8 pages
│   └── static/
│       ├── css/dashboard.css    # Dark theme design system
│       └── js/app.js            # Client-side routing, charts, real-time updates
│
├── tests/                   # Comprehensive test suite
├── docs/                    # Documentation
├── examples/                # Usage examples
├── Dockerfile               # Production Docker image
├── docker-compose.yml       # Multi-container deployment
└── .github/workflows/ci.yml # GitHub Actions CI/CD
```

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone
git clone https://github.com/ShikhaPallavi123/distributed-ai-platform.git
cd distributed-ai-platform

# Install (just numpy!)
pip install -r requirements.txt

# Start the server
python -m api.app

# Open dashboard: http://localhost:8080
```

### Run with Docker

```bash
docker-compose up -d
# Dashboard at http://localhost:8080
```

### Run Tests

```bash
python -m unittest discover tests/ -v
```

---

## 💡 Code Examples

### Neural Network Training (From Scratch)

```python
from neural.tensor import Tensor
from neural.layers import Linear, Sequential
from neural.activations import ReLU
from neural.losses import MSELoss
from neural.optimizers import Adam

# Build a network
model = Sequential([
    Linear(2, 16),
    ReLU(),
    Linear(16, 1),
])

optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = MSELoss()

# Training loop
for epoch in range(100):
    pred = model(x_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Agentic AI — ReAct Agent

```python
from agentic.agent import ReactAgent
from agentic.tools import create_default_registry

agent = ReactAgent(
    tools=create_default_registry(),
    max_steps=10,
)
result = agent.run("Calculate the compound interest on $1000 at 5% for 10 years")
print(result.answer)
```

### Distributed Inference

```python
from inference.engine import InferenceEngine
from inference.load_balancer import LoadBalancer

engine = InferenceEngine()
engine.load_model("my-model", model_path="./models/model.npz")

result = engine.infer("my-model", input_data={"text": "Hello, world!"})
```

### OS-Level Memory Allocator

```python
from core.memory.pool_allocator import TensorMemoryManager

mem = TensorMemoryManager(pool_size=1024 * 1024)  # 1MB pool
handle = mem.allocate_tensor((32, 128))  # Allocate tensor memory
mem.free_tensor(handle)
mem.defragment()  # Coalesce free blocks
```

---

## 📊 Live Dashboard

> **[👉 Try the Live Demo](https://shikhapallavi123.github.io/distributed-ai-platform/)** — no installation needed!

The full-stack dashboard includes:

- **Real-time metrics** — requests/sec, latency percentiles, GPU utilization
- **Worker fleet** — health status, load, task completion
- **Model registry** — deploy, monitor, and manage AI models
- **Inference playground** — run single/batch inference from the UI
- **Agentic AI console** — execute agents with reasoning trace
- **Cluster topology** — visual map of distributed nodes
- **Architecture viewer** — interactive 8-layer stack diagram
- **System terminal** — built-in command console

---

## 🧪 Testing

```bash
# Run all tests
python -m unittest discover tests/ -v

# Test specific module
python -m unittest tests.test_neural -v
python -m unittest tests.test_ml -v
python -m unittest tests.test_agentic -v
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Core Language | Python 3.10+ | Readability + scientific ecosystem |
| Math Engine | NumPy | Only external dependency — fast array ops |
| API Server | `http.server` (stdlib) | Zero-dependency REST API |
| Frontend | Vanilla HTML/CSS/JS | No React/Vue — pure web platform |
| Consensus | Raft (custom) | Industry-standard distributed consensus |
| Scheduling | MLFQ | OS-level multi-level feedback queue |
| Memory | Pool allocator + mmap | Custom memory management |
| Deployment | Docker + GitHub Actions | Production-ready CI/CD |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🌟 Star This Repo

If you found this project interesting or educational, please ⭐ star the repository!

Built with 🧠 from scratch — every tensor operation, every gradient, every consensus round.
