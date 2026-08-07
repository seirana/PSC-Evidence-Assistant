import hashlib
from pathlib import Path
from typing import List


def stable_id(text: str, n: int = 10) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:n]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def list_corpus_files(corpus_dir: Path) -> List[Path]:
    exts = {".txt", ".md"}
    return sorted([p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
