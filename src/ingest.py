from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from .utils import (
    read_document,
    stable_id,
    list_corpus_files,
)


@dataclass
class Chunk:
    """
    One piece of text extracted from a document.
    """

    doc_id: str
    doc_path: str
    chunk_id: str
    text: str


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Split a long text into smaller overlapping chunks.

    Example:

    chunk_size = 1000
    overlap = 200

    Chunk 1:
        characters 0 - 1000

    Chunk 2:
        characters 800 - 1800

    Chunk 3:
        characters 1600 - 2600

    The overlap helps prevent information from being lost
    when an important sentence lies close to a chunk boundary.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # We reached the end of the document
        if end == text_length:
            break

        # Move forward, but keep some overlap
        start = end - overlap

    return chunks


def ingest_corpus(
    corpus_dir: Path,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> List[Chunk]:
    """
    Read every supported file from corpus_dir,
    convert it to plain text,
    split the text into chunks,
    and return all chunks from all documents.
    """

    files = list_corpus_files(corpus_dir)

    chunks: List[Chunk] = []

    for file_path in files:

        # 1. Read the document
        raw_text = read_document(file_path)

        # Ignore documents from which no text could be extracted
        if not raw_text.strip():
            continue

        # 2. Give this document a unique ID
        doc_id = stable_id(
            str(file_path.resolve())
        )

        # 3. Split the document into chunks
        document_chunks = chunk_text(
            raw_text,
            chunk_size_chars,
            chunk_overlap_chars,
        )

        # 4. Store every chunk together with information
        #    about which document it came from
        for i, chunk_text_value in enumerate(document_chunks):

            chunk_id = f"{doc_id}_{i:04d}"

            chunk = Chunk(
                doc_id=doc_id,
                doc_path=str(file_path),
                chunk_id=chunk_id,
                text=chunk_text_value,
            )

            chunks.append(chunk)

    return chunks


def chunks_to_records(
    chunks: List[Chunk],
) -> List[Dict]:
    """
    Convert Chunk objects into normal Python dictionaries.

    Some later parts of the application are easier to work
    with when the data is represented as dictionaries.
    """

    return [
        {
            "doc_id": chunk.doc_id,
            "doc_path": chunk.doc_path,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
