from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from .utils import read_text_file, stable_id, list_corpus_files


@dataclass
class Chunk:
    doc_id: str
    doc_path: str
    chunk_id: str
    text: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def ingest_corpus(corpus_dir: Path, chunk_size_chars: int, chunk_overlap_chars: int) -> List[Chunk]:
    files = list_corpus_files(corpus_dir)
    chunks: List[Chunk] = []

    for fp in files:
        raw = read_text_file(fp)
        doc_id = stable_id(str(fp.resolve()))
        doc_chunks = chunk_text(raw, chunk_size_chars, chunk_overlap_chars)
        for i, c in enumerate(doc_chunks):
            chunk_id = f"{doc_id}_{i:04d}"
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    doc_path=str(fp),
                    chunk_id=chunk_id,
                    text=c,
                )
            )
    return chunks


def chunks_to_records(chunks: List[Chunk]) -> List[Dict]:
    return [
        {
            "doc_id": c.doc_id,
            "doc_path": c.doc_path,
            "chunk_id": c.chunk_id,
            "text": c.text,
        }
        for c in chunks
    ]
