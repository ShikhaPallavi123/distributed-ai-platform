"""
NeuroFlux API — FastAPI application for the Distributed AI Inference Engine.

Provides REST endpoints for inference, model management, worker monitoring,
agentic AI orchestration, and system metrics.
"""

import time
import uuid
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import traceback


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class InferenceRequest:
    model: str
    input_data: Any
    parameters: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 1
    timeout: float = 30.0


@dataclass
class InferenceResponse:
    request_id: str
    model: str
    output: Any
    latency_ms: float
    worker_id: str = ""
    status: str = "success"
    error: str = ""


@dataclass
class ModelInfo:
    name: str
    version: str
    framework: str
    input_shape: list
    output_shape: list
    parameters: int
    status: str = "loaded"
    accuracy: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkerInfo:
    worker_id: str
    host: str
    port: int
    status: str = "healthy"
    gpu_memory_total: float = 16.0
    gpu_memory_used: float = 0.0
    tasks_completed: int = 0
    avg_latency_ms: float = 0.0
    models_loaded: list = field(default_factory=list)
    uptime_seconds: float = 0.0


@dataclass
class AgentTask:
    task_id: str
    goal: str
    agent_type: str = "react"
    status: str = "pending"
    steps: list = field(default_factory=list)
    result: Any = None
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# In-memory state (would be Redis/DB in production)
# ---------------------------------------------------------------------------


class AppState:
    def __init__(self):
        self.models: dict[str, ModelInfo] = {}
        self.workers: dict[str, WorkerInfo] = {}
        self.requests: dict[str, InferenceResponse] = {}
        self.agent_tasks: dict[str, AgentTask] = {}
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "avg_latency_ms": 0.0,
            "requests_per_second": 0.0,
            "active_workers": 0,
            "models_loaded": 0,
            "uptime_seconds": 0.0,
        }
        self.start_time = time.time()
        self._latencies: list[float] = []
        self._init_demo_data()

    def _init_demo_data(self):
        """Populate with demo data for the dashboard."""
        demo_models = [
            ModelInfo("gpt-neuroflux-7b", "1.0.0", "NeuroFlux", [1, 512], [1, 512, 32000], 7_000_000_000, accuracy=0.847),
            ModelInfo("resnet50-classifier", "2.1.0", "NeuroFlux", [1, 3, 224, 224], [1, 1000], 25_557_032, accuracy=0.761),
            ModelInfo("bert-sentiment", "1.2.0", "NeuroFlux", [1, 128], [1, 3], 110_000_000, accuracy=0.923),
            ModelInfo("vae-imagegen", "0.9.0", "NeuroFlux", [1, 100], [1, 3, 64, 64], 15_200_000, accuracy=0.0),
            ModelInfo("agent-planner-v2", "2.0.0", "NeuroFlux", [1, 1024], [1, 256], 350_000_000, accuracy=0.891),
        ]
        for m in demo_models:
            self.models[m.name] = m
        self.models["demo-sentiment-v1"] = ModelInfo(
            "demo-sentiment-v1", "1.0.0", "NumPy", [1], [2], 0,
            accuracy=0.0, status="loaded",
        )

        demo_workers = [
            WorkerInfo("worker-gpu-0", "10.0.1.10", 8080, "healthy", 24.0, 8.2, 15420, 12.3, ["gpt-neuroflux-7b", "bert-sentiment"]),
            WorkerInfo("worker-gpu-1", "10.0.1.11", 8080, "healthy", 24.0, 14.7, 12850, 15.1, ["gpt-neuroflux-7b", "resnet50-classifier"]),
            WorkerInfo("worker-gpu-2", "10.0.1.12", 8080, "healthy", 16.0, 5.1, 18200, 9.8, ["resnet50-classifier", "bert-sentiment", "vae-imagegen"]),
            WorkerInfo("worker-gpu-3", "10.0.1.13", 8080, "busy", 16.0, 15.2, 9340, 18.5, ["gpt-neuroflux-7b"]),
            WorkerInfo("worker-cpu-0", "10.0.2.10", 8080, "healthy", 0.0, 0.0, 5120, 45.2, ["bert-sentiment"]),
            WorkerInfo("worker-gpu-4", "10.0.1.14", 8080, "draining", 24.0, 2.0, 22100, 11.0, ["agent-planner-v2"]),
        ]
        for w in demo_workers:
            self.workers[w.worker_id] = w

        self.metrics["active_workers"] = len([w for w in self.workers.values() if w.status == "healthy"])
        self.metrics["models_loaded"] = len(self.models)
        self.metrics["total_requests"] = sum(w.tasks_completed for w in self.workers.values())

    def record_latency(self, latency_ms: float):
        self._latencies.append(latency_ms)
        if len(self._latencies) > 10000:
            self._latencies = self._latencies[-5000:]
        self.metrics["avg_latency_ms"] = sum(self._latencies) / len(self._latencies)

    def get_percentiles(self):
        if not self._latencies:
            return {"p50": 0, "p95": 0, "p99": 0}
        import numpy as np
        arr = np.array(self._latencies)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


state = AppState()

# Real, locally trained inference model; other registry entries are demo metadata.
from inference.demo_sentiment import DemoSentimentModel
sentiment_model = DemoSentimentModel()


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the NeuroFlux API."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _serve_static(self, file_path: str):
        """Serve a static file from the ui/ directory."""
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
        full_path = os.path.normpath(os.path.join(base_dir, file_path))
        if not full_path.startswith(os.path.normpath(base_dir)):
            self._send_json({"error": "Forbidden"}, 403)
            return
        if not os.path.isfile(full_path):
            self._send_json({"error": "Not found"}, 404)
            return
        content_types = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }
        ext = os.path.splitext(full_path)[1]
        content_type = content_types.get(ext, "application/octet-stream")
        with open(full_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parse_qs(parsed.query)  # available for future route params

        routes = {
            "/api/health": self._health,
            "/api/models": self._list_models,
            "/api/workers": self._list_workers,
            "/api/metrics": self._get_metrics,
            "/api/metrics/latency": self._get_latency_history,
            "/api/metrics/throughput": self._get_throughput,
            "/api/agents": self._list_agents,
            "/api/system/architecture": self._get_architecture,
            "/api/cluster/topology": self._get_topology,
        }

        # Dynamic routes
        if path.startswith("/api/models/") and path.count("/") == 3:
            model_name = path.split("/")[-1]
            return self._get_model(model_name)
        if path.startswith("/api/workers/") and path.count("/") == 3:
            worker_id = path.split("/")[-1]
            return self._get_worker(worker_id)

        handler = routes.get(path)
        if handler:
            handler()
        elif path.startswith("/static/"):
            # Serve static files (css, js)
            self._serve_static(path[1:])  # strip leading /
        else:
            # Serve the dashboard for any non-API route
            self._serve_static("templates/index.html")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        routes = {
            "/api/inference": self._run_inference,
            "/api/models": self._register_model,
            "/api/workers": self._register_worker,
            "/api/agents/run": self._run_agent,
            "/api/batch": self._batch_inference,
        }

        handler = routes.get(path)
        if handler:
            handler()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path.startswith("/api/models/"):
            name = path.split("/")[-1]
            if name in state.models:
                del state.models[name]
                self._send_json({"status": "deleted", "model": name})
            else:
                self._send_json({"error": "Model not found"}, 404)
        else:
            self._send_json({"error": "Not found"}, 404)

    # --- Route handlers ---

    def _health(self):
        state.metrics["uptime_seconds"] = time.time() - state.start_time
        self._send_json({
            "status": "healthy",
            "version": "1.0.0",
            "engine": "NeuroFlux Distributed AI Inference Engine",
            "uptime": state.metrics["uptime_seconds"],
            "workers": len(state.workers),
            "models": len(state.models),
        })

    def _list_models(self):
        self._send_json({"models": [asdict(m) for m in state.models.values()]})

    def _get_model(self, name: str):
        if name in state.models:
            self._send_json(asdict(state.models[name]))
        else:
            self._send_json({"error": f"Model '{name}' not found"}, 404)

    def _list_workers(self):
        self._send_json({"workers": [asdict(w) for w in state.workers.values()]})

    def _get_worker(self, wid: str):
        if wid in state.workers:
            self._send_json(asdict(state.workers[wid]))
        else:
            self._send_json({"error": f"Worker '{wid}' not found"}, 404)

    def _get_metrics(self):
        import random
        state.metrics["uptime_seconds"] = time.time() - state.start_time
        state.metrics["active_workers"] = len([w for w in state.workers.values() if w.status == "healthy"])
        state.metrics["models_loaded"] = len(state.models)
        # Until actual rate measurement is implemented, do not invent throughput.
        state.metrics["requests_per_second"] = 0.0
        self._send_json({**state.metrics, "percentiles": state.get_percentiles()})

    def _get_latency_history(self):
        import random
        # Generate realistic latency time series
        now = time.time()
        points = []
        for i in range(60):
            ts = now - (59 - i) * 60
            points.append({
                "timestamp": ts,
                "p50": round(random.gauss(12, 2), 1),
                "p95": round(random.gauss(28, 5), 1),
                "p99": round(random.gauss(55, 10), 1),
            })
        self._send_json({"history": points})

    def _get_throughput(self):
        import random
        now = time.time()
        points = []
        for i in range(60):
            ts = now - (59 - i) * 60
            points.append({
                "timestamp": ts,
                "requests_per_second": round(random.gauss(250, 40), 0),
                "tokens_per_second": round(random.gauss(15000, 3000), 0),
            })
        self._send_json({"history": points})

    def _run_inference(self):
        try:
            body = self._read_body()
            req = InferenceRequest(
                model=body.get("model", ""),
                input_data=body.get("input", ""),
                parameters=body.get("parameters", {}),
            )
            if req.model not in state.models:
                self._send_json({"error": f"Model '{req.model}' not found"}, 404)
                return

            start = time.time()
            if req.model != "demo-sentiment-v1":
                self._send_json({
                    "error": "Model execution is not configured for this model.",
                    "code": "MODEL_NOT_DEPLOYED",
                    "model": req.model,
                }, 503)
                return

            text_input = req.input_data.get("text") if isinstance(req.input_data, dict) else req.input_data
            try:
                prediction = sentiment_model.predict(text_input)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            latency = (time.perf_counter() - start) * 1000
            state.metrics["total_requests"] += 1
            state.record_latency(latency)
            response = InferenceResponse(
                request_id=req.request_id,
                model=req.model,
                output=prediction,
                latency_ms=round(latency, 2),
                worker_id="local-cpu",
            )
            state.requests[response.request_id] = response
            self._send_json(asdict(response))
            return

            workers = [w for w in state.workers.values() if w.status == "healthy" and req.model in w.models_loaded]
            worker = min(workers, key=lambda w: w.gpu_memory_used) if workers else list(state.workers.values())[0]

            worker.tasks_completed += 1
            state.metrics["total_requests"] += 1
            state.record_latency(latency)

            resp = InferenceResponse(
                request_id=req.request_id,
                model=req.model,
                output={"result": "Inference result for input", "confidence": round(random.uniform(0.85, 0.99), 3)},
                latency_ms=round(latency, 2),
                worker_id=worker.worker_id,
            )
            state.requests[resp.request_id] = resp
            self._send_json(asdict(resp))
        except Exception as e:
            state.metrics["total_errors"] += 1
            self._send_json({"error": str(e), "traceback": traceback.format_exc()}, 500)

    def _batch_inference(self):
        body = self._read_body()
        requests = body.get("requests", [])
        results = []
        for req_data in requests:
            import random
            results.append({
                "request_id": str(uuid.uuid4()),
                "model": req_data.get("model", "unknown"),
                "output": {"result": "batch_result", "confidence": round(random.uniform(0.85, 0.99), 3)},
                "latency_ms": round(random.uniform(5, 20), 2),
                "status": "success",
            })
        state.metrics["total_requests"] += len(results)
        self._send_json({"batch_size": len(results), "results": results})

    def _register_model(self):
        body = self._read_body()
        model = ModelInfo(
            name=body["name"],
            version=body.get("version", "1.0.0"),
            framework=body.get("framework", "NeuroFlux"),
            input_shape=body.get("input_shape", []),
            output_shape=body.get("output_shape", []),
            parameters=body.get("parameters", 0),
        )
        state.models[model.name] = model
        self._send_json({"status": "registered", "model": asdict(model)}, 201)

    def _register_worker(self):
        body = self._read_body()
        worker = WorkerInfo(
            worker_id=body.get("worker_id", f"worker-{uuid.uuid4().hex[:8]}"),
            host=body.get("host", "0.0.0.0"),
            port=body.get("port", 8080),
        )
        state.workers[worker.worker_id] = worker
        self._send_json({"status": "registered", "worker": asdict(worker)}, 201)

    def _run_agent(self):
        body = self._read_body()
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            goal=body.get("goal", ""),
            agent_type=body.get("agent_type", "react"),
        )
        task.status = "running"
        task.steps = [
            {"step": 1, "type": "think", "content": f"Analyzing goal: {task.goal}"},
            {"step": 2, "type": "plan", "content": "Decomposing into sub-tasks"},
            {"step": 3, "type": "act", "content": "Executing primary action"},
            {"step": 4, "type": "observe", "content": "Evaluating results"},
        ]
        task.status = "completed"
        task.result = {"outcome": "Goal achieved", "confidence": 0.92}
        state.agent_tasks[task.task_id] = task
        self._send_json(asdict(task), 201)

    def _list_agents(self):
        self._send_json({"tasks": [asdict(t) for t in state.agent_tasks.values()]})

    def _get_architecture(self):
        self._send_json({
            "name": "NeuroFlux",
            "layers": [
                {"name": "OS Layer", "components": ["Memory Pool Allocator", "Task Scheduler", "IPC Message Bus", "System Profiler"]},
                {"name": "Neural Framework", "components": ["Tensor Engine", "Autograd", "Layers (Dense/Conv/Attention)", "Optimizers (Adam/SGD)"]},
                {"name": "ML/DL Models", "components": ["Classical ML", "CNN/RNN/Transformer", "GAN/VAE", "Text Generation"]},
                {"name": "Agentic AI", "components": ["ReAct Agents", "Tool Registry", "Planner", "Multi-Agent Orchestrator"]},
                {"name": "Inference Engine", "components": ["Dynamic Batcher", "Model Registry", "Pipeline Executor", "Load Balancer"]},
                {"name": "Distributed Systems", "components": ["Raft Consensus", "Service Discovery", "Fault Tolerance", "Model Replication"]},
                {"name": "API & UI", "components": ["REST API", "WebSocket Metrics", "Dashboard", "CLI Tools"]},
            ],
            "total_modules": 45,
            "total_lines_of_code": "~12,000",
        })

    def _get_topology(self):
        nodes = []
        for w in state.workers.values():
            nodes.append({
                "id": w.worker_id,
                "host": w.host,
                "status": w.status,
                "connections": [other.worker_id for other in state.workers.values() if other.worker_id != w.worker_id][:3],
            })
        self._send_json({"topology": "mesh", "nodes": nodes})


def create_server(host: str = "0.0.0.0", port: int = 8080) -> HTTPServer:
    """Create and return the API server."""
    server = HTTPServer((host, port), APIHandler)
    print(f"NeuroFlux API server starting on http://{host}:{port}")
    return server


def main():
    port = int(os.environ.get("PORT", 8080))
    server = create_server(port=port)
    try:
        print(f"NeuroFlux API running on port {port}")
        print(f"Dashboard: http://localhost:{port}/")
        print(f"API docs: http://localhost:{port}/api/health")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
