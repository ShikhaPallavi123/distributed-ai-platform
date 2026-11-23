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
