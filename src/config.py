from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """
    Central configuration for the PSC Evidence Assistant.

    Instead of hard-coding values in many different files,
    we keep important settings here.

    Other files can simply do:

        cfg = Settings()

    and then use:

        cfg.corpus_dir
        cfg.chunk_size_chars
        cfg.top_k
        ...
    """

    # ---------------------------------------------------------
    # PROJECT
    # ---------------------------------------------------------

    # __file__ is:
    #
    # PSC-Evidence-Assistant/src/config.py
    #
    # parents[1] therefore gives:
    #
    # PSC-Evidence-Assistant/
    #
    project_root: Path = Path(__file__).resolve().parents[1]

    # ---------------------------------------------------------
    # RAG / INGESTION SETTINGS
    # ---------------------------------------------------------

    # Number of characters in each chunk.
    chunk_size_chars: int = 1800

    # Number of characters shared between neighboring chunks.
    chunk_overlap_chars: int = 250

    # Maximum number of chunks returned by RAG.
    top_k: int = 6

    # Minimum TF-IDF cosine similarity required
    # for a chunk to be considered relevant.
    #
    # If no chunk reaches this score,
    # RAG returns [].
    min_retrieval_score: float = 0.05

    # ---------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------

    @property
    def corpus_dir(self) -> Path:
        """
        Folder containing the source documents.

        Supported by our updated utils.py:

        .txt
        .md
        .docx
        .pdf
        """
        return self.project_root / "data" / "corpus"

    @property
    def outputs_dir(self) -> Path:
        """
        Folder where generated output files are stored.
        """
        return self.project_root / "outputs"

    @property
    def demos_dir(self) -> Path:
        """
        Folder for saved demo results.
        """
        return self.outputs_dir / "demos"

    # ---------------------------------------------------------
    # KNOWLEDGE GRAPH PATHS
    # ---------------------------------------------------------

    @property
    def graph_path_graphml(self) -> Path:
        """
        Path for saving the knowledge graph in GraphML format.
        """
        return self.outputs_dir / "knowledge_graph.graphml"

    @property
    def graph_path_json(self) -> Path:
        """
        Path for saving the knowledge graph in JSON format.
        """
        return self.outputs_dir / "knowledge_graph.json"

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    def __post_init__(self):
        """
        Check that configuration values make sense.

        This catches mistakes early when the application starts.
        """

        if self.chunk_size_chars <= 0:
            raise ValueError(
                "chunk_size_chars must be greater than 0"
            )

        if self.chunk_overlap_chars < 0:
            raise ValueError(
                "chunk_overlap_chars must be >= 0"
            )

        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError(
                "chunk_overlap_chars must be smaller "
                "than chunk_size_chars"
            )

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if not 0 <= self.min_retrieval_score <= 1:
            raise ValueError(
                "min_retrieval_score must be between 0 and 1"
            )
