import json


def parse_json_safely(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def json_validity_score(texts: list[str]) -> float:
    if not texts:
        return 0.0
    ok = 0
    for t in texts:
        try:
            json.loads(t)
            ok += 1
        except Exception:
            pass
    return ok / len(texts)
