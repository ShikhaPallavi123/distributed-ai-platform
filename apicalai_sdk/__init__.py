"""Small standard-library client for trusted ApicalAI deployments."""
import json
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, origin, api_key=None, timeout=120):
        parts = urlsplit(origin)
        loopback = parts.hostname in ("localhost", "127.0.0.1", "::1")
        if parts.scheme != "https" and not (parts.scheme == "http" and loopback):
            raise ValueError("Use HTTPS or loopback HTTP")
        if not parts.hostname or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
            raise ValueError("Supply an origin without credentials or path")
        self.origin = origin.rstrip("/")
        self.key = api_key
        self.timeout = timeout
        self._opener = build_opener(_NoRedirect())

    def _request(self, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        body = json.dumps(payload).encode() if payload is not None else None
        with self._opener.open(Request(self.origin + path, data=body, headers=headers), timeout=self.timeout) as response:
            return json.load(response)

    def models(self):
        return self._request("/api/models")

    def infer(self, model, text):
        return self._request("/api/inference", {"model": model, "input": {"text": text}})

    def chat(self, messages):
        return self._request("/api/chat", {"messages": messages})

    def retrieve(self, question, documents):
        return self._request("/api/retrieve", {"question": question, "documents": documents})

    def route(self, text, min_quality=.5, max_latency_ms=None):
        payload = {"text": text, "min_quality": min_quality}
        if max_latency_ms is not None:
            payload["max_latency_ms"] = max_latency_ms
        return self._request("/api/route", payload)
