import json
from typing import Any, Optional


def _remove_markdown_fence(text: str) -> str:
    """
    Remove Markdown code fences that an LLM may add.

    Example:

        ```json
        {"supported": true}
        ```

    becomes:

        {"supported": true}
    """

    text = text.strip()

    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()

        # Remove opening fence:
        # ``` or ```json
        if lines:
            lines = lines[1:]

        # Remove closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()

    return text


def _extract_json_from_text(text: str) -> Optional[Any]:
    """
    Try to find a JSON object or array inside extra text.

    Example:

        Here is the result:
        {"supported": true}

    Even though this is not pure JSON, we try to recover
    the JSON object.
    """

    decoder = json.JSONDecoder()

    # Look for possible beginnings of JSON structures.
    possible_starts = []

    for i, character in enumerate(text):
        if character in "{[":
            possible_starts.append(i)

    for start in possible_starts:
        try:
            value, _ = decoder.raw_decode(
                text[start:]
            )
            return value

        except json.JSONDecodeError:
            continue

    return None


def parse_json_safely(
    text: str,
) -> Optional[Any]:
    """
    Convert LLM-generated JSON text into a Python object.

    Returns:
        Parsed Python object if successful.

        None if the output cannot be parsed.

    This function handles:
    - normal JSON,
    - leading/trailing whitespace,
    - Markdown ```json fences,
    - small amounts of text before JSON.
    """

    if not isinstance(text, str):
        return None

    text = text.strip()

    if not text:
        return None

    # Sometimes text may begin with a Unicode BOM.
    text = text.lstrip("\ufeff")

    # ---------------------------------------------------------
    # STEP 1:
    # Remove Markdown fences if the model added them.
    # ---------------------------------------------------------

    cleaned = _remove_markdown_fence(
        text
    )

    # ---------------------------------------------------------
    # STEP 2:
    # First try normal JSON parsing.
    # ---------------------------------------------------------

    try:
        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:
        pass

    # ---------------------------------------------------------
    # STEP 3:
    # The LLM may have written something like:
    #
    # Here is the requested JSON:
    # {"supported": true}
    #
    # Try to recover the JSON portion.
    # ---------------------------------------------------------

    return _extract_json_from_text(
        cleaned
    )


def is_valid_json(
    text: str,
) -> bool:
    """
    Check whether an LLM response contains parseable JSON.

    Unlike checking:

        parse_json_safely(text) is not None

    this function also correctly handles valid JSON such as:

        null
    """

    if not isinstance(text, str):
        return False

    text = text.strip()

    if not text:
        return False

    text = text.lstrip("\ufeff")
    cleaned = _remove_markdown_fence(text)

    try:
        json.loads(cleaned)
        return True

    except json.JSONDecodeError:
        pass

    recovered = _extract_json_from_text(
        cleaned
    )

    return recovered is not None


def json_validity_score(
    texts: list[str],
) -> float:
    """
    Calculate what fraction of LLM responses contain
    valid or recoverable JSON.

    Example:

        [
            '{"a": 1}',       # valid
            '{"b": 2}',       # valid
            'hello',          # invalid
            '```json\\n{"c": 3}\\n```'  # recoverable
        ]

    gives:

        3 / 4 = 0.75
    """

    if not texts:
        return 0.0

    valid_count = sum(
        1
        for text in texts
        if is_valid_json(text)
    )

    return valid_count / len(texts)
