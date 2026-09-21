from dataclasses import dataclass
import re
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

        # Build a glossary from the corpus itself. Examples:
        #
        #   Example Long Name (ELN)
        #   Adaptive Evidence Framework (AEF)
        #
        # This avoids maintaining a hard-coded list of terms or
        # trying to predict the questions users might ask.
        self.glossary = self._extract_glossary(
            self.texts
        )

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

    @staticmethod
    def _extract_glossary(
        texts: List[str],
    ) -> Dict[str, str]:
        """Extract ``Expanded Name (short name)`` pairs."""

        glossary: Dict[str, str] = {}

        # Require title-style words in the expanded form. This
        # intentionally favors headings and formal definitions
        # over arbitrary parenthetical text in prose.
        pair_pattern = re.compile(
            r"(?P<long>[A-Z][A-Za-z0-9-]*"
            r"(?:\s+(?:[A-Z][A-Za-z0-9-]*|of|and|for|"
            r"in|the|to)){1,11})"
            r"\s*\(\s*"
            r"(?P<short>[A-Za-z][A-Za-z0-9-]{1,20})"
            r"\s*\)"
        )

        for text in texts:
            for raw_line in text.splitlines():
                line = " ".join(
                    raw_line.strip().lstrip("#").split()
                )
                line = re.sub(
                    r"^\d+(?:\.\d+)*\s+",
                    "",
                    line,
                )

                for match in pair_pattern.finditer(line):
                    short_name = match.group(
                        "short"
                    ).strip()
                    long_name = match.group(
                        "long"
                    ).strip(" -:;")

                    if short_name.casefold() == long_name.casefold():
                        continue

                    glossary.setdefault(
                        short_name.casefold(),
                        long_name,
                    )

        return glossary

    def expand_query(
        self,
        query: str,
    ) -> str:
        """
        Append corpus-derived expansions for names in a query.

        The original wording is retained. Therefore a question
        about the meaning of an abbreviation remains intact,
        while retrieval also sees its expanded form.
        """

        expanded = " ".join(
            (query or "").split()
        )

        if not expanded:
            return ""

        additions = []

        for short_name, long_name in self.glossary.items():
            if not re.search(
                rf"(?<!\w){re.escape(short_name)}(?!\w)",
                expanded,
                flags=re.IGNORECASE,
            ):
                continue

            if long_name.casefold() in expanded.casefold():
                continue

            additions.append(long_name)

        if additions:
            expanded += " " + " ".join(additions)

        return expanded

    def replace_query_aliases(
        self,
        query: str,
    ) -> str:
        """Replace corpus-defined aliases for definition matching."""

        resolved = " ".join(
            (query or "").split()
        )

        for short_name, long_name in sorted(
            self.glossary.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if long_name.casefold() in resolved.casefold():
                continue

            resolved = re.sub(
                rf"(?<!\w){re.escape(short_name)}(?!\w)",
                long_name,
                resolved,
                flags=re.IGNORECASE,
            )

        return resolved

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

        expanded_query = self.expand_query(
            query
        )

        query_vector = self.vectorizer.transform(
            [expanded_query]
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

    def retrieve_term_definitions(
        self,
        term: str,
        top_k: int = 3,
    ) -> List[RetrievedChunk]:
        """
        Find chunks that explicitly expand or define a term.

        A normal TF-IDF query can miss a heading such as::

            Expanded Method Name (EMN)

        when the same short term appears many times elsewhere
        in the document. This method uses only exact text
        patterns; it does not infer or invent an expansion.
        The returned score remains the term's ordinary TF-IDF
        cosine similarity so citation metadata stays honest.
        """

        cleaned_term = " ".join(
            (term or "").split()
        ).strip(" ?.!,:;")

        if (
            not cleaned_term
            or not self.records
            or self.matrix is None
            or top_k <= 0
        ):
            return []

        escaped = re.escape(
            cleaned_term
        )

        explicit_patterns = (
            # "X stands for ..." and similar prose.
            re.compile(
                rf"(?<!\w){escaped}(?!\w)\s+"
                r"(?:stands\s+for|means|is\s+short\s+for|"
                r"is\s+an?\s+abbreviation\s+for)\b",
                flags=re.IGNORECASE,
            ),
            # "Expanded Name (X)" in a heading or sentence.
            re.compile(
                rf"[^()\n]{{2,120}}\(\s*{escaped}\s*\)",
                flags=re.IGNORECASE,
            ),
            # "X: descriptive title" at the start of a line.
            re.compile(
                rf"^\s*{escaped}\s*:\s*\S.+$",
                flags=(
                    re.IGNORECASE
                    | re.MULTILINE
                ),
            ),
        )

        query_vector = self.vectorizer.transform(
            [cleaned_term]
        )

        if query_vector.nnz == 0:
            similarities = None
        else:
            similarities = cosine_similarity(
                query_vector,
                self.matrix,
            ).ravel()

        matches = []

        for index, record in enumerate(
            self.records
        ):
            text = record.get(
                "text",
                "",
            )

            pattern_rank = None

            for rank, pattern in enumerate(
                explicit_patterns
            ):
                if pattern.search(text):
                    pattern_rank = rank
                    break

            if pattern_rank is None:
                continue

            score = (
                float(similarities[index])
                if similarities is not None
                else 0.0
            )

            matches.append(
                (
                    pattern_rank,
                    -score,
                    RetrievedChunk(
                        doc_path=record["doc_path"],
                        chunk_id=record["chunk_id"],
                        score=score,
                        text=text,
                    ),
                )
            )

        matches.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        return [
            item[2]
            for item in matches[:top_k]
        ]


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
