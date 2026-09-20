from dataclasses import dataclass
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    """
    A chunk that was retrieved because it is relevant
    to the user's question.
    """

    doc_path: str
    chunk_id: str
    score: float
    text: str


class TfidfRAG:
    """
    Simple RAG retrieval system using:

    1. TF-IDF
    2. cosine similarity

    The class receives chunks created by ingest.py and
    searches those chunks when the user asks a question.
    """

    def __init__(self, records: List[Dict]):
        """
        Build the TF-IDF search index.

        Parameters
        ----------
        records:
            List of dictionaries produced by chunks_to_records().

            Example:

            {
                "doc_id": "...",
                "doc_path": "data/corpus/paper.pdf",
                "chunk_id": "...",
                "text": "..."
            }
        """

        # IMPORTANT:
        # Keep records and texts aligned.
        #
        # If an empty record is removed, it must be removed
        # from both lists. Otherwise indexes could point to
        # the wrong document.
        self.records = [
            record
            for record in records
            if record.get("text", "").strip()
        ]

        self.texts = [
            record["text"]
            for record in self.records
        ]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
        )

        # This will contain the vector representation
        # of all document chunks.
        self.matrix = None

        # Build the TF-IDF index only if we actually
        # have documents.
        if self.texts:
            try:
                self.matrix = self.vectorizer.fit_transform(
                    self.texts
                )

            except ValueError:
                # Example:
                # The corpus exists but contains no usable words.
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
            User question or rewritten search query.

        top_k:
            Maximum number of chunks to return.

        min_score:
            Minimum cosine-similarity score required
            for a chunk to be accepted.

        Returns
        -------
        List[RetrievedChunk]

        If no chunk is relevant enough, this function
        returns an empty list:

            []
        """

        # No useful question
        if not query or not query.strip():
            return []

        # No usable corpus
        if not self.records or self.matrix is None:
            return []

        if top_k <= 0:
            return []

        if min_score < 0:
            raise ValueError(
                "min_score must be >= 0"
            )

        # ---------------------------------------------
        # STEP 1:
        # Convert the user's question into the same
        # TF-IDF vector space as the document chunks.
        # ---------------------------------------------

        query_vector = self.vectorizer.transform(
            [query]
        )

        # nnz = number of non-zero values.
        #
        # If it is 0, none of the useful words from
        # the question exist in our TF-IDF vocabulary.
        #
        # Therefore there is no useful lexical match.
        if query_vector.nnz == 0:
            return []

        # ---------------------------------------------
        # STEP 2:
        # Compare the question with every chunk.
        # ---------------------------------------------

        similarities = cosine_similarity(
            query_vector,
            self.matrix,
        ).ravel()

        # ---------------------------------------------
        # STEP 3:
        # Sort chunks from highest similarity
        # to lowest similarity.
        # ---------------------------------------------

        ranked_indices = similarities.argsort()[::-1]

        retrieved: List[RetrievedChunk] = []

        # ---------------------------------------------
        # STEP 4:
        # Keep only chunks above min_score.
        # ---------------------------------------------

        for index in ranked_indices:

            score = float(
                similarities[index]
            )

            # Results are already sorted from high to low.
            #
            # Therefore, once we reach a score below the
            # threshold, all following results will also
            # be below the threshold.
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

            # We already collected enough evidence.
            if len(retrieved) >= top_k:
                break

        return retrieved


def format_context(
    retrieved: List[RetrievedChunk],
    max_chars: int = 9000,
) -> Tuple[str, List[Dict]]:
    """
    Prepare retrieved chunks for the LLM.

    Returns two things:

    1. context
       Text that will be placed in the LLM prompt.

    2. citations
       Information about where each piece of evidence
       came from.
    """

    if not retrieved:
        return "", []

    if max_chars <= 0:
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

        # Do not send an excessively large context
        # to the LLM.
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

    context = "\n".join(
        blocks
    ).strip()

    return context, citations
