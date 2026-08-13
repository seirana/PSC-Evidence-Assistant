import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict


NOT_FOUND_MESSAGE = "Not found in provided documents."


class LLMClient:
    """
    Small interface between our Python application and an LLM.

    Supported modes
    ---------------

    1. dummy
       No real LLM is used.
       Useful for testing the application structure.

    2. ollama
       Sends prompts to a locally running Ollama model.

    3. http
       Sends prompts to a custom HTTP endpoint.

    The rest of our application does not need to know
    which mode is being used.

    It simply calls:

        llm.generate(prompt)

    and receives a string.
    """

    SUPPORTED_MODES = {
        "dummy",
        "ollama",
        "http",
    }

    def __init__(self):
        """
        Read LLM settings from environment variables.

        Examples:

        LLM_MODE=dummy

        or:

        LLM_MODE=ollama
        OLLAMA_MODEL=llama3.1

        or:

        LLM_MODE=http
        LLM_HTTP_URL=http://localhost:8000/generate
        """

        # -----------------------------------------------------
        # General LLM settings
        # -----------------------------------------------------

        self.mode = os.getenv(
            "LLM_MODE",
            "dummy",
        ).strip().lower()

        self.timeout = int(
            os.getenv(
                "LLM_HTTP_TIMEOUT",
                "120",
            )
        )

        self.temperature = float(
            os.getenv(
                "LLM_TEMPERATURE",
                "0.0",
            )
        )

        # -----------------------------------------------------
        # Ollama settings
        # -----------------------------------------------------

        self.ollama_url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/generate",
        ).strip()

        self.ollama_model = os.getenv(
            "OLLAMA_MODEL",
            "llama3.1",
        ).strip()

        # -----------------------------------------------------
        # Custom HTTP settings
        # -----------------------------------------------------

        self.http_url = os.getenv(
            "LLM_HTTP_URL",
            "",
        ).strip()

        # -----------------------------------------------------
        # Validate configuration
        # -----------------------------------------------------

        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported LLM_MODE: {self.mode}. "
                f"Expected one of: "
                f"{sorted(self.SUPPORTED_MODES)}"
            )

        if self.timeout <= 0:
            raise ValueError(
                "LLM_HTTP_TIMEOUT must be greater than 0"
            )

        if self.mode == "ollama":

            if not self.ollama_url:
                raise ValueError(
                    "OLLAMA_URL cannot be empty "
                    "when LLM_MODE=ollama"
                )

            if not self.ollama_model:
                raise ValueError(
                    "OLLAMA_MODEL cannot be empty "
                    "when LLM_MODE=ollama"
                )

        if self.mode == "http":

            if not self.http_url:
                raise ValueError(
                    "LLM_HTTP_URL is required "
                    "when LLM_MODE=http"
                )

    # =========================================================
    # PUBLIC FUNCTION
    # =========================================================

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Send a prompt to the configured LLM.

        This is the most important function in this file.

        The rest of the project simply does:

            answer = llm.generate(prompt)

        It does not need to know whether the model is:
        - dummy
        - Ollama
        - custom HTTP
        """

        if not prompt or not prompt.strip():
            return ""

        if self.mode == "dummy":
            return self._generate_dummy(
                prompt
            )

        if self.mode == "ollama":
            return self._generate_ollama(
                prompt
            )

        if self.mode == "http":
            return self._generate_http(
                prompt
            )

        # This should never happen because __init__
        # already validates the mode.
        raise RuntimeError(
            f"Unexpected LLM mode: {self.mode}"
        )

    # =========================================================
    # OLLAMA
    # =========================================================

    def _generate_ollama(
        self,
        prompt: str,
    ) -> str:
        """
        Send a prompt to a local Ollama server.
        """

        payload: Dict[str, Any] = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        # Some of our prompts request JSON:
        #
        # query rewriting
        # graph extraction
        # grounding verification
        #
        # Tell Ollama that JSON output is expected.
        if self._expects_json(prompt):
            payload["format"] = "json"

        response = self._post_json(
            url=self.ollama_url,
            payload=payload,
        )

        text = response.get(
            "response"
        )

        if not isinstance(text, str):
            raise RuntimeError(
                "Ollama returned no valid "
                "'response' field."
            )

        text = text.strip()

        if not text:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return text

    # =========================================================
    # CUSTOM HTTP MODEL
    # =========================================================

    def _generate_http(
        self,
        prompt: str,
    ) -> str:
        """
        Send the prompt to a custom HTTP server.

        We expect the server to accept:

            {
                "prompt": "..."
            }

        and return:

            {
                "text": "..."
            }
        """

        payload = {
            "prompt": prompt,
        }

        response = self._post_json(
            url=self.http_url,
            payload=payload,
        )

        text = response.get(
            "text"
        )

        if not isinstance(text, str):
            raise RuntimeError(
                "Custom LLM server returned no valid "
                "'text' field."
            )

        text = text.strip()

        if not text:
            raise RuntimeError(
                "Custom LLM server returned "
                "an empty response."
            )

        return text

    # =========================================================
    # HTTP HELPER
    # =========================================================

    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Send a JSON POST request and return the JSON response.

        Both Ollama mode and custom HTTP mode use this function.
        """

        encoded_payload = json.dumps(
            payload
        ).encode(
            "utf-8"
        )

        request = urllib.request.Request(
            url=url,
            data=encoded_payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                raw_response = (
                    response.read().decode(
                        "utf-8"
                    )
                )

        except urllib.error.HTTPError as error:

            try:
                error_body = (
                    error.read().decode(
                        "utf-8",
                        errors="ignore",
                    )
                )
            except Exception:
                error_body = ""

            raise RuntimeError(
                f"LLM server returned HTTP "
                f"{error.code}: {error_body}"
            ) from error

        except urllib.error.URLError as error:

            raise RuntimeError(
                f"Could not connect to LLM server "
                f"at {url}: {error.reason}"
            ) from error

        except TimeoutError as error:

            raise RuntimeError(
                f"LLM request timed out after "
                f"{self.timeout} seconds."
            ) from error

        try:

            parsed = json.loads(
                raw_response
            )

        except json.JSONDecodeError as error:

            raise RuntimeError(
                "LLM server did not return valid JSON."
            ) from error

        if not isinstance(parsed, dict):
            raise RuntimeError(
                "LLM server response must be "
                "a JSON object."
            )

        return parsed

    # =========================================================
    # DUMMY MODE
    # =========================================================

    def _generate_dummy(
        self,
        prompt: str,
    ) -> str:
        """
        Fake responses for testing the Python pipeline.

        IMPORTANT:

        Dummy mode is NOT an LLM.

        It should never invent scientific information.

        Its purpose is only to let us test that:
        - agent.py works
        - rag.py works
        - prompts are passed around correctly
        - JSON parsing works
        """

        # -----------------------------------------------------
        # Query rewrite prompt
        # -----------------------------------------------------

        if '"rewritten_queries"' in prompt:

            question = self._extract_user_question(
                prompt
            )

            if not question:
                question = ""

            return json.dumps(
                {
                    "rewritten_queries": [
                        question
                    ],
                    "entities_of_interest": [],
                }
            )

        # -----------------------------------------------------
        # Graph extraction prompt
        # -----------------------------------------------------

        if (
            '"entities"' in prompt
            and '"relations"' in prompt
        ):

            # Do NOT make up graph facts in dummy mode.
            return json.dumps(
                {
                    "entities": [],
                    "relations": [],
                }
            )

        # -----------------------------------------------------
        # Grounding verification prompt
        # -----------------------------------------------------

        if (
            '"supported"' in prompt
            and '"unsupported_claims"' in prompt
        ):

            return json.dumps(
                {
                    "supported": True,
                    "unsupported_claims": [],
                    "notes": "Dummy-mode verification.",
                }
            )

        # -----------------------------------------------------
        # Answer generation
        # -----------------------------------------------------

        # Dummy mode cannot actually understand the retrieved
        # scientific context, so it must never generate a
        # scientific answer.
        return NOT_FOUND_MESSAGE

    # =========================================================
    # SMALL HELPERS
    # =========================================================

    @staticmethod
    def _expects_json(
        prompt: str,
    ) -> bool:
        """
        Detect whether one of our prompts expects JSON.
        """

        markers = (
            "Return ONLY valid JSON",
            "Return ONLY JSON",
        )

        return any(
            marker in prompt
            for marker in markers
        )

    @staticmethod
    def _extract_user_question(
        prompt: str,
    ) -> str:
        """
        Extract the question from the query-rewrite prompt.

        Our prompts.py uses:

            User question:
            <question>
        """

        marker = "User question:"

        if marker not in prompt:
            return ""

        question = prompt.split(
            marker,
            1,
        )[1]

        return question.strip()
