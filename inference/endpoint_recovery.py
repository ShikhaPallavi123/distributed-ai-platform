"""Circuit breaker for trusted model endpoints; automatically retries after cooldown."""
import time
from threading import Lock
from inference.ollama_adapter import OllamaBackendError


class RecoveringEndpoint:
    def __init__(self, predict, failures=2, cooldown=30, clock=time.monotonic):
        if not callable(predict) or failures < 1 or cooldown <= 0:
            raise ValueError("invalid circuit settings")
        self.predict = predict
        self.threshold = failures
        self.cooldown = cooldown
        self.clock = clock
        self.failures = 0
        self.retry_at = 0.0
        self.probing = False
        self.lock = Lock()

    def __call__(self, text):
        with self.lock:
            if self.failures >= self.threshold:
                if self.clock() < self.retry_at or self.probing:
                    raise OllamaBackendError("Endpoint temporarily unavailable (circuit open)")
                self.probing = True
        try:
            result = self.predict(text)
        except OllamaBackendError:
            with self.lock:
                self.failures += 1
                if self.failures >= self.threshold:
                    self.retry_at = self.clock() + self.cooldown
                self.probing = False
            raise
        except BaseException:
            with self.lock:
                self.probing = False
            raise
        with self.lock:
            self.failures = 0
            self.retry_at = 0.0
            self.probing = False
        return result
