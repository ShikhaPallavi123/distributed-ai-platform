"""Trusted llama.cpp or vLLM provider; no request-controlled endpoints."""
import json
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener
from inference.ollama_adapter import OllamaBackendError


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class CompatibleAdapter:
    def __init__(self, model, origin, key=None, max_tokens=128, timeout=90):
        parsed = urlsplit(origin)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("provide a trusted HTTP(S) origin")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("non-loopback endpoints must use HTTPS or an SSH tunnel")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model required")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 2048:
            raise ValueError("max_tokens must be 1–2048")
        self.model = model
        self.origin = origin.rstrip("/")
        self.key = key
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 8000:
            raise ValueError("text must contain 1–8000 characters")
        payload = {"model": self.model, "messages": [{"role": "user", "content": text}],
                   "max_tokens": self.max_tokens, "temperature": 0, "stream": False}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = Request(self.origin + "/v1/chat/completions", data=json.dumps(payload).encode(),
                      headers=headers)
        try:
            with build_opener(NoRedirect()).open(req, timeout=self.timeout) as response:
                raw = response.read(1048577)
            if len(raw) > 1048576:
                raise ValueError("oversized response")
            data = json.loads(raw)
            choice = data["choices"][0]
            answer = choice["message"]["content"]
            if not isinstance(answer, str):
                raise ValueError("invalid answer")
            usage = data.get("usage") or {}
            return {"text": answer, "provider_model": self.model,
                    "model_type": "openai_compatible",
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "generated_tokens": usage.get("completion_tokens"),
                    "finish_reason": choice.get("finish_reason")}
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise OllamaBackendError("Configured LLM provider unavailable or invalid response") from exc
