from dataclasses import dataclass
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    """
    One chunk returned by the retrieval system.
    """

    doc_path: str
    chunk_id: str
    score: float
    text: str


class TfidfRAG:
    """
    Simple retrieval system based on TF-IDF and cosine similarity.

    It receives text chunks created by ingest.py and allows us
    to search for chunks that are relevant to a user's question.
    """

    def __init__(self, records: List[Dict]):

        self.records = records

        # Extract only the text from every chunk.
        self.texts = [
            record["text"]
            for record in records
            if record.get("text", "").strip()
        ]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
        )

        # matrix will contain the TF-IDF representation
        # of all document chunks.
        self.matrix = None

        # Avoid crashing if the corpus is empty.
        if self.texts:
            try:
                self.matrix = self.vectorizer.fit_transform(
                    self.texts
                )
            except ValueError:
                # This can happen if no useful vocabulary
                # can be extracted from the documents.
                self.matrix = None

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        min_score: float = 0.05,
    ) -> List[RetrievedChunk]:
        """
        Find chunks that are relevant to the user's question.

        Parameters
        ----------
        query:
            The user's question.

        top_k:
            Maximum number of chunks to return.

        min_score:
            Minimum cosine similarity required for a chunk
            to be considered relevant.

        Returns
        -------
        List[RetrievedChunk]

        If nothing is relevant enough, an empty list is returned.
        """

        # Empty question -> no retrieval
        if not query.strip():
            return []

        # Empty/unusable corpus -> no retrieval
        if not self.records or self.matrix is None:
            return []

        if top_k <= 0:
            return []

        if min_score < 0:
            raise ValueError("min_score must be >= 0")

        # Convert the question to a TF-IDF vector
        query_vector = self.vectorizer.transform([query])

        # If none of the words in the question exist
        # in our corpus vocabulary, there is no evidence.
        if query_vector.nnz == 0:
            return []

        # Compare question with every document chunk
        similarities = cosine_similarity(
            query_vector,
            self.matrix,
        ).ravel()

        # Sort from most similar to least similar
        ranked_indices = similarities.argsort()[::-1]

        retrieved: List[RetrievedChunk] = []

        for index in ranked_indices:

            score = float(similarities[index])

            # Because results are sorted from highest to lowest,
            # once we go below the threshold we can stop.
            if score < min_score:
                break

            record = self.records[index]

            retrieved.append(
                RetrievedChunk(
                    doc_path=record["doc_path"],
                    chunk_id=record["chunk_id"],
                    score=score,
                    text=record["text"],
                )
            )

            # Stop when we have enough chunks
            if len(retrieved) >= top_k:
                break

        return retrieved


def format_context(
    retrieved: List[RetrievedChunk],
    max_chars: int = 9000,
) -> Tuple[str, List[Dict]]:
    """
    Convert retrieved chunks into text that can be given to the LLM.

    It also returns citation information so we know exactly
    which document/chunk was used.
    """

    # No evidence was retrieved.
    if not retrieved:
        return "", []

    blocks = []
    citations = []
    total_chars = 0

    for chunk in retrieved:

        header = (
            f"[SOURCE "
            f"doc={chunk.doc_path} "
            f"chunk={chunk.chunk_id} "
            f"score={chunk.score:.4f}]"
        )

        block = (
            header
            + "\n"
            + chunk.text.strip()
            + "\n"
        )

        # Do not make the context too large.
        if total_chars + len(block) > max_chars:
            break

        blocks.append(block)

        total_chars += len(block)

        citations.append(
            {
                "doc_path": chunk.doc_path,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
            }
        )

    context = "\n".join(blocks).strip()

    return context, citations
