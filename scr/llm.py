import os
import json
import urllib.request


class LLMClient:
    """
    Minimal LLM client with 3 modes:

    - dummy: deterministic placeholders so the whole pipeline runs without any model.
      export LLM_MODE=dummy

    - ollama: call a local Ollama server (LLaMA/Mistral/etc.).
      export LLM_MODE=ollama
      export OLLAMA_MODEL=llama3.1
      export OLLAMA_URL=http://localhost:11434/api/generate

    - http: call a custom HTTP endpoint that accepts {"prompt": "..."} and returns {"text": "..."}.
      export LLM_MODE=http
      export LLM_HTTP_URL=http://localhost:8000/generate
    """

    def __init__(self):
        self.mode = os.getenv("LLM_MODE", "dummy").lower().strip()
        self.timeout = int(os.getenv("LLM_HTTP_TIMEOUT", "120"))

        # HTTP mode
        self.http_url = os.getenv("LLM_HTTP_URL", "").strip()

        # Ollama mode
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").strip()
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1").strip()

    def generate(self, prompt: str) -> str:
        if self.mode == "dummy":
            return self._dummy(prompt)

        if self.mode == "http":
            if not self.http_url:
                raise ValueError("LLM_HTTP_URL is required when LLM_MODE=http")
            payload = json.dumps({"prompt": prompt}).encode("utf-8")
            req = urllib.request.Request(
                self.http_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data.get("text", "")).strip()

        if self.mode == "ollama":
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.ollama_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            return str(out.get("response", "")).strip()

        raise ValueError(f"Unsupported LLM_MODE: {self.mode}")

    def _dummy(self, prompt: str) -> str:
        # If a JSON schema is expected, return minimal valid JSON.
        if "Return ONLY valid JSON" in prompt or "Return ONLY JSON" in prompt:
            if '"rewritten_queries"' in prompt:
                return '{"rewritten_queries":["PSC scDRS pipeline inputs outputs","scDRS thresholds p-value z-score"],"entities_of_interest":["PSC","scDRS"]}'
            if '"entities"' in prompt and '"relations"' in prompt:
                return '{"entities":[{"type":"Disease","name":"PSC"},{"type":"Method","name":"scDRS"}],"relations":[{"source":"scDRS","relation":"REQUIRES_INPUT","target":"GWAS summary statistics","evidence_chunk_id":"dummy_0000"}]}'
            return '{"supported": true, "unsupported_claims": [], "notes": "dummy mode"}'
        return "Not found in provided documents."
