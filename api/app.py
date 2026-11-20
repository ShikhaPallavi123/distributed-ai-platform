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
import hmac
from inference.demo_sentiment import DemoSentimentModel
from inference.ollama_adapter import OllamaAdapter, OllamaBackendError


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
        self._completed_events: list[tuple[float, float]] = []  # (wall time, latency in ms)
        self._init_demo_data()

    def _init_demo_data(self):
        """Register the only executable model and its real local CPU worker."""
        self.models["demo-sentiment-v1"] = ModelInfo(
            "demo-sentiment-v1", "1.0.0", "NumPy", [1], [2],
            0, accuracy=0.0, status="loaded",
        )
        self.workers["local-cpu"] = WorkerInfo(
            worker_id="local-cpu", host="127.0.0.1", port=8080,
            status="healthy", gpu_memory_total=0.0, gpu_memory_used=0.0,
            models_loaded=["demo-sentiment-v1"],
        )
        self.metrics["active_workers"] = 1
        self.metrics["models_loaded"] = 1

    def record_latency(self, latency_ms: float):
        """Record the latency of an *executed*, successful model request."""
        self._latencies.append(latency_ms)
        if len(self._latencies) > 10000:
            self._latencies = self._latencies[-5000:]
        self._completed_events.append((time.time(), latency_ms))
        if len(self._completed_events) > 10000:
            self._completed_events = self._completed_events[-5000:]
        self.metrics["avg_latency_ms"] = sum(self._latencies) / len(self._latencies)

    def requests_per_second(self, window_seconds: int = 60) -> float:
        """Measured throughput, averaged over up to the last 60 seconds."""
        now = time.time()
        recent = sum(
            1 for timestamp, _ in self._completed_events
            if now - window_seconds < timestamp <= now
        )
        elapsed = min(window_seconds, max(1.0, now - self.start_time))
        return round(recent / elapsed, 3)

    def get_latency_history(self, window_seconds: int = 60) -> list[dict]:
        """Return measured latency percentiles for seconds with observations."""
        import numpy as np
        now = time.time()
        buckets: dict[int, list[float]] = {}
        for timestamp, latency in self._completed_events:
            if now - window_seconds < timestamp <= now:
                buckets.setdefault(int(timestamp), []).append(latency)
        return [
            {
                "timestamp": second,
                "p50": round(float(np.percentile(values, 50)), 3),
                "p95": round(float(np.percentile(values, 95)), 3),
                "p99": round(float(np.percentile(values, 99)), 3),
            }
            for second, values in sorted(buckets.items())
        ]

    def get_throughput_history(self, window_seconds: int = 60) -> list[dict]:
        """Actual successful requests per second; never generated randomly."""
        now = time.time()
        counts: dict[int, int] = {}
        for timestamp, _ in self._completed_events:
            if now - window_seconds < timestamp <= now:
                second = int(timestamp)
                counts[second] = counts.get(second, 0) + 1
        return [
            {"timestamp": second, "requests_per_second": count}
            for second, count in sorted(counts.items())
        ]

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

# Real trained model; optional external language model (no fake model registrations).
sentiment_model = DemoSentimentModel()
_ollama_model = os.environ.get("NEUROFLUX_OLLAMA_MODEL", "").strip()
ollama_client = (
    OllamaAdapter(
        _ollama_model,
        base_url=os.environ.get("NEUROFLUX_OLLAMA_URL", "http://127.0.0.1:11434"),
    )
    if _ollama_model else None
)
if ollama_client:
    state.models["ollama-local"] = ModelInfo(
        "ollama-local", "1.0.0", "Ollama", [1], [1], 0,
        status="configured",
    )
    state.workers["ollama-backend"] = WorkerInfo(
        worker_id="ollama-backend", host="configured-ollama",
        port=11434, status="configured",
        gpu_memory_total=0.0, gpu_memory_used=0.0,
        models_loaded=["ollama-local"],
    )


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

    def _require_admin(self) -> bool:
        """Protect administrative metadata writes until user auth exists."""
        token = os.environ.get("NEUROFLUX_ADMIN_TOKEN", "")
        provided = self.headers.get("Authorization", "")
        if not token or not hmac.compare_digest(provided, "Bearer " + token):
            self._send_json({
                "error": "Admin changes are disabled without a valid NEUROFLUX_ADMIN_TOKEN",
            }, 403)
            return False
        return True

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
            if path in ("/api/models", "/api/workers") and not self._require_admin():
                return
            handler()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path.startswith("/api/models/"):
            if not self._require_admin():
                return
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
        self._send_json({"models": [{"id": m.name, **asdict(m)} for m in state.models.values()]})

    def _get_model(self, name: str):
        if name in state.models:
            self._send_json(asdict(state.models[name]))
        else:
            self._send_json({"error": f"Model '{name}' not found"}, 404)

    def _list_workers(self):
        self._send_json({"workers": [{"id": w.worker_id, "gpu_type": "CPU" if not w.gpu_memory_total else "Unknown", "current_load": 0, **asdict(w)} for w in state.workers.values()]})

    def _get_worker(self, wid: str):
        if wid in state.workers:
            self._send_json(asdict(state.workers[wid]))
        else:
            self._send_json({"error": f"Worker '{wid}' not found"}, 404)

    def _get_metrics(self):
        state.metrics["uptime_seconds"] = time.time() - state.start_time
        state.metrics["active_workers"] = len([w for w in state.workers.values() if w.status == "healthy"])
        state.metrics["models_loaded"] = len([m for m in state.models.values() if m.status == "loaded"])
        state.metrics["requests_per_second"] = state.requests_per_second()
        self._send_json({**state.metrics, "percentiles": state.get_percentiles()})

    def _get_latency_history(self):
        self._send_json({"history": state.get_latency_history(), "source": "measured"})

    def _get_throughput(self):
        self._send_json({"history": state.get_throughput_history(), "source": "measured"})

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

            start = time.perf_counter()
            text_input = req.input_data.get("text") if isinstance(req.input_data, dict) else req.input_data

            if req.model == "demo-sentiment-v1":
                worker_id = "local-cpu"
                try:
                    prediction = sentiment_model.predict(text_input)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, 400)
                    return
            elif req.model == "ollama-local" and ollama_client is not None:
                worker_id = "ollama-backend"
                try:
                    prediction = ollama_client.generate(text_input)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, 400)
                    return
                except OllamaBackendError as exc:
                    state.metrics["total_errors"] += 1
                    self._send_json({"error": str(exc), "code": "MODEL_UNAVAILABLE"}, 503)
                    return
            else:
                self._send_json({
                    "error": "No executable backend is configured for this model.",
                    "code": "MODEL_NOT_DEPLOYED",
                    "model": req.model,
                }, 503)
                return

            latency = (time.perf_counter() - start) * 1000
            state.metrics["total_requests"] += 1
            state.record_latency(latency)
            state.workers[worker_id].tasks_completed += 1
            response = InferenceResponse(
                request_id=req.request_id,
                model=req.model,
                output=prediction,
                latency_ms=round(latency, 2),
                worker_id=worker_id,
            )
            state.requests[response.request_id] = response
            self._send_json(asdict(response))

        except Exception:
            state.metrics["total_errors"] += 1
            traceback.print_exc()
            self._send_json({"error": "Internal inference error"}, 500)

    def _batch_inference(self):
        """Execute a bounded batch using the real NumPy classifier.

        The MVP runs items sequentially; this is *not* dynamic GPU batching.
        Validate all inputs before recording metrics or returning success.
        """
        try:
            body = self._read_body()
            requests = body.get("requests")
            if not isinstance(requests, list) or not 1 <= len(requests) <= 32:
                self._send_json({"error": "requests must contain 1 to 32 items"}, 400)
                return

            validated = []
            for item in requests:
                if not isinstance(item, dict):
                    self._send_json({"error": "Every request must be an object"}, 400)
                    return
                model_name = item.get("model")
                if model_name not in state.models:
                    self._send_json({"error": "Model not found: " + str(model_name)}, 404)
                    return
                if model_name != "demo-sentiment-v1":
                    self._send_json({
                        "error": "Batch execution is currently supported only for demo-sentiment-v1",
                        "code": "BATCH_MODEL_UNSUPPORTED",
                    }, 501)
                    return
                data = item.get("input")
                text_input = data.get("text") if isinstance(data, dict) else data
                if not isinstance(text_input, str):
                    self._send_json({"error": "input.text must be a string"}, 400)
                    return
                validated.append(text_input)

            total_start = time.perf_counter()
            prepared = []
            for text_input in validated:
                start = time.perf_counter()
                try:
                    prediction = sentiment_model.predict(text_input)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, 400)
                    return
                prepared.append((prediction, (time.perf_counter() - start) * 1000))

            results = []
            for prediction, latency in prepared:
                req_id = str(uuid.uuid4())
                response = InferenceResponse(
                    request_id=req_id,
                    model="demo-sentiment-v1",
                    output=prediction,
                    latency_ms=round(latency, 2),
                    worker_id="local-cpu",
                )
                state.requests[req_id] = response
                state.metrics["total_requests"] += 1
                state.record_latency(latency)
                state.workers["local-cpu"].tasks_completed += 1
                results.append(asdict(response))
            self._send_json({
                "batch_size": len(results),
                "results": results,
                "total_latency_ms": round((time.perf_counter() - total_start) * 1000, 2),
                "execution_mode": "sequential_cpu",
            })
        except Exception:
            state.metrics["total_errors"] += 1
            traceback.print_exc()
            self._send_json({"error": "Internal batch inference error"}, 500)

    def _register_model(self):
        body = self._read_body()
        model = ModelInfo(
            name=body["name"],
            version=body.get("version", "1.0.0"),
            framework=body.get("framework", "NeuroFlux"),
            input_shape=body.get("input_shape", []),
            output_shape=body.get("output_shape", []),
            parameters=body.get("parameters", 0),
            status="registered",
        )
        state.models[model.name] = model
        self._send_json({"status": "registered", "model": asdict(model)}, 201)

    def _register_worker(self):
        body = self._read_body()
        worker = WorkerInfo(
            worker_id=body.get("worker_id", f"worker-{uuid.uuid4().hex[:8]}"),
            host=body.get("host", "0.0.0.0"),
            port=body.get("port", 8080),
            status="unverified",
            gpu_memory_total=0.0,
        )
        state.workers[worker.worker_id] = worker
        self._send_json({"status": "registered", "worker": asdict(worker)}, 201)

    def _run_agent(self):
        """Do not report synthetic agent steps as successful execution."""
        self._send_json({
            "error": "Agent execution has not been connected to a real backend.",
            "code": "AGENT_NOT_DEPLOYED",
        }, 501)

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
