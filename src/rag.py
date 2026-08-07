from dataclasses import dataclass
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    doc_path: str
    chunk_id: str
    score: float
    text: str


class TfidfRAG:
    def __init__(self, records: List[Dict]):
        self.records = records
        self.texts = [r["text"] for r in records]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 6) -> List[RetrievedChunk]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        idxs = sims.argsort()[::-1][:top_k]

        out: List[RetrievedChunk] = []
        for i in idxs:
            r = self.records[i]
            out.append(
                RetrievedChunk(
                    doc_path=r["doc_path"],
                    chunk_id=r["chunk_id"],
                    score=float(sims[i]),
                    text=r["text"],
                )
            )
        return out


def format_context(retrieved: List[RetrievedChunk], max_chars: int = 9000) -> Tuple[str, List[Dict]]:
    blocks = []
    citations = []
    total = 0
    for r in retrieved:
        header = f"[SOURCE doc={r.doc_path} chunk={r.chunk_id} score={r.score:.4f}]"
        block = header + "\n" + r.text.strip() + "\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
        citations.append({"doc_path": r.doc_path, "chunk_id": r.chunk_id, "score": r.score})
    return "\n".join(blocks).strip(), citations
