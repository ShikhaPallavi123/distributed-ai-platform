"""Optional adapter for a *real* local Ollama language model.

Configure with APICALAI_OLLAMA_MODEL after pulling a model into Ollama.
Only standard-library networking is used; no model or GPU is fabricated.
"""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaBackendError(RuntimeError):
    """The configured Ollama server or model could not serve a request."""


class OllamaAdapter:
    def __init__(self, model, base_url="http://127.0.0.1:11434", timeout=90):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A non-empty Ollama model name is required")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must use http or https")
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("input.text must be a non-empty string")
        if len(prompt) > 8000:
            raise ValueError("input.text must contain no more than 8000 characters")

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 256},
        }).encode("utf-8")
        request = Request(
            self.base_url + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise OllamaBackendError(
                "Ollama returned HTTP " + str(exc.code)
                + "; verify the configured model has been downloaded"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaBackendError(
                "Cannot reach the configured Ollama server; check that it is running"
            ) from exc
        except (ValueError, TypeError) as exc:
            raise OllamaBackendError("Ollama returned invalid JSON") from exc

        if not isinstance(result, dict) or not isinstance(result.get("response"), str):
            raise OllamaBackendError("Ollama response did not contain generated text")
        return {
            "text": result["response"],
            "model_type": "ollama_pretrained_llm",
            "provider_model": self.model,
            "prompt_tokens": result.get("prompt_eval_count"),
            "generated_tokens": result.get("eval_count"),
            "total_duration_ns": result.get("total_duration"),
        }

    def chat(self, messages, stream=False):
        """Real Ollama chat protocol; caller owns conversation history."""
        if not isinstance(messages, list) or not 1 <= len(messages) <= 24:
            raise ValueError("provide 1–24 messages")
        total = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                raise ValueError("only user and assistant messages are accepted")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("message content required")
            total += len(content)
        if total > 16000 or messages[-1]["role"] != "user":
            raise ValueError("history must end with user and contain at most 16000 characters")
        payload = {"model": self.model, "messages": [
            {"role": m["role"], "content": m["content"]} for m in messages
        ], "stream": stream, "options": {"num_predict": 256}}
        request = Request(self.base_url + "/api/chat", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        if stream:
            return self._chat_chunks(request)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            content = result["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("invalid response content")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise OllamaBackendError("Chat backend failed") from exc
        return {"text": content, "provider_model": self.model,
                "prompt_tokens": result.get("prompt_eval_count"),
                "generated_tokens": result.get("eval_count")}

    def _chat_chunks(self, request):
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for line in response:
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaBackendError("Chat backend returned an error")
                    content = data.get("message", {}).get("content", "")
                    if not isinstance(content, str):
                        raise ValueError("invalid chunk")
                    if content:
                        yield {"text": content, "done": False}
                    if data.get("done") is True:
                        yield {"done": True, "generated_tokens": data.get("eval_count")}
                        return
            raise OllamaBackendError("Chat stream ended without completion")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise OllamaBackendError("Chat stream failed") from exc

    def embed(self, texts):
        """Actual Ollama embedding API for a separately configured embedding model."""
        if not isinstance(texts, list) or not 1 <= len(texts) <= 16:
            raise ValueError("provide 1–16 embedding inputs")
        if any(not isinstance(t, str) or not t.strip() or len(t) > 2000 for t in texts):
            raise ValueError("embedding input must contain 1–2000 characters")
        payload = json.dumps({"model": self.model, "input": texts, "truncate": False}).encode()
        request = Request(self.base_url + "/api/embed", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            vectors = result["embeddings"]
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                raise ValueError("invalid embeddings")
            return vectors
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise OllamaBackendError("Embedding backend unavailable") from exc
