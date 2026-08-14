import json
from pathlib import Path
from typing import Tuple

import streamlit as st

from src.config import Settings
from src.utils import list_corpus_files
from src.ingest import ingest_corpus, chunks_to_records
from src.rag import TfidfRAG
from src.llm import LLMClient
from src.graph_kb import KnowledgeGraph
from src.agent import EvidenceAgent


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="PSC Evidence Assistant",
    layout="wide",
)

st.title("PSC Evidence Assistant")

st.caption(
    "Answers are grounded only in documents from data/corpus."
)


# ============================================================
# CONFIGURATION
# ============================================================

cfg = Settings()

# Make sure output folders exist.
cfg.outputs_dir.mkdir(
    parents=True,
    exist_ok=True,
)

cfg.demos_dir.mkdir(
    parents=True,
    exist_ok=True,
)

# Make sure corpus folder exists.
cfg.corpus_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CORPUS SIGNATURE
# ============================================================

def get_corpus_signature(
    corpus_dir: Path,
) -> Tuple:
    """
    Create a small fingerprint describing the current corpus.

    Why do we need this?

    Streamlit caches expensive operations.

    But our files in data/corpus may change:

        paper1.pdf added
        old.docx removed
        notes.txt modified

    The fingerprint changes when the corpus changes.

    That causes Streamlit to rebuild the RAG index.
    """

    files = list_corpus_files(
        corpus_dir
    )

    signature = []

    for file_path in files:

        try:
            stat = file_path.stat()

            signature.append(
                (
                    str(
                        file_path.relative_to(
                            corpus_dir
                        )
                    ),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            )

        except OSError:
            # If a file disappears while we are checking
            # the folder, simply skip it.
            continue

    return tuple(signature)


# ============================================================
# BUILD RAG INDEX
# ============================================================

@st.cache_resource(
    show_spinner="Reading documents and building RAG index..."
)
def build_index(
    corpus_signature: Tuple,
):
    """
    Read all supported documents, create chunks,
    and build the TF-IDF retrieval index.

    corpus_signature is intentionally an argument.

    Even though we do not directly use it inside the
    function, Streamlit uses it as part of the cache key.

    If the corpus changes, this function runs again.
    """

    chunks = ingest_corpus(
        corpus_dir=cfg.corpus_dir,
        chunk_size_chars=cfg.chunk_size_chars,
        chunk_overlap_chars=cfg.chunk_overlap_chars,
    )

    records = chunks_to_records(
        chunks
    )

    rag = TfidfRAG(
        records
    )

    return records, rag


# ============================================================
# CURRENT CORPUS
# ============================================================

corpus_signature = get_corpus_signature(
    cfg.corpus_dir
)

corpus_files = list_corpus_files(
    cfg.corpus_dir
)

records, rag = build_index(
    corpus_signature
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Corpus")

    st.write(
        f"Documents: {len(corpus_files)}"
    )

    st.write(
        f"Chunks: {len(records)}"
    )

    st.caption(
        f"Corpus folder: {cfg.corpus_dir}"
    )

    if corpus_files:

        with st.expander(
            "Corpus files"
        ):

            for file_path in corpus_files:

                try:
                    relative_path = (
                        file_path.relative_to(
                            cfg.corpus_dir
                        )
                    )

                    st.write(
                        f"• {relative_path}"
                    )

                except ValueError:

                    st.write(
                        f"• {file_path.name}"
                    )

    st.divider()

    st.header("RAG settings")

    top_k = st.slider(
        "Maximum retrieved chunks",
        min_value=1,
        max_value=12,
        value=cfg.top_k,
        step=1,
    )

    min_score = st.slider(
        "Minimum retrieval score",
        min_value=0.0,
        max_value=0.50,
        value=float(
            cfg.min_retrieval_score
        ),
        step=0.01,
    )

    st.caption(
        "Chunks scoring below the minimum retrieval "
        "score are ignored."
    )

    st.divider()

    st.header("LLM")


# ============================================================
# LLM
# ============================================================

try:

    llm = LLMClient()

except Exception as error:

    st.error(
        f"Could not initialize LLM: {error}"
    )

    st.stop()


with st.sidebar:

    st.write(
        f"Mode: `{llm.mode}`"
    )

    if llm.mode == "dummy":

        st.info(
            "Dummy mode does not generate real scientific "
            "answers. Set LLM_MODE=ollama to use a local LLM."
        )

    elif llm.mode == "ollama":

        st.write(
            f"Model: `{llm.ollama_model}`"
        )


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

# Streamlit reruns app.py whenever the user interacts
# with the page.
#
# If we simply did:
#
#     kg = KnowledgeGraph()
#
# the graph would be erased on every rerun.
#
# session_state keeps it alive during the current session.

if (
    "knowledge_graph" not in st.session_state
    or
    st.session_state.get(
        "knowledge_graph_corpus_signature"
    ) != corpus_signature
):

    st.session_state.knowledge_graph = (
        KnowledgeGraph()
    )

    st.session_state[
        "knowledge_graph_corpus_signature"
    ] = corpus_signature


kg = st.session_state.knowledge_graph


# ============================================================
# AGENT
# ============================================================

agent = EvidenceAgent(
    rag=rag,
    kg=kg,
    llm=llm,
    top_k=top_k,
    min_score=min_score,
)


# ============================================================
# CORPUS STATUS
# ============================================================

if not corpus_files:

    st.warning(
        "No supported documents were found in data/corpus.\n\n"
        "Add one or more .txt, .md, .docx, or .pdf files."
    )


elif not records:

    st.warning(
        "Documents were found, but no readable text "
        "could be extracted from them."
    )


else:

    st.success(
        f"Corpus ready: "
        f"{len(corpus_files)} document(s), "
        f"{len(records)} chunks."
    )


# ============================================================
# QUESTION
# ============================================================

question = st.text_input(
    "Ask a question about the documents",
    placeholder=(
        "Example: Which cell populations were enriched "
        "for PSC genetic risk?"
    ),
)


# ============================================================
# RUN QUESTION
# ============================================================

run_clicked = st.button(
    "Ask",
    type="primary",
)


if run_clicked:

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    if not records:

        st.warning(
            "The corpus contains no readable text."
        )

        st.stop()

    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            result = agent.answer(
                question
            )

    except Exception as error:

        st.error(
            f"Question processing failed: {error}"
        )

        st.stop()

    # ========================================================
    # ANSWER
    # ========================================================

    st.subheader("Answer")

    st.write(
        result["answer"]
    )


    # ========================================================
    # SOURCES
    # ========================================================

    st.subheader(
        "Retrieved evidence"
    )

    citations = result.get(
        "citations",
        []
    )

    if citations:

        for citation in citations:

            with st.expander(
                (
                    f"{Path(citation['doc_path']).name}"
                    f" — {citation['chunk_id']}"
                )
            ):

                st.write(
                    f"Similarity score: "
                    f"{citation['score']:.4f}"
                )

                st.write(
                    f"Source: {citation['doc_path']}"
                )

    else:

        st.info(
            "No sufficiently relevant evidence was retrieved."
        )


    # ========================================================
    # GROUNDING CHECK
    # ========================================================

    with st.expander(
        "Grounding check"
    ):

        st.json(
            result.get(
                "grounding_check",
                {}
            )
        )


    # ========================================================
    # GRAPH FACTS
    # ========================================================

    with st.expander(
        "Extracted knowledge-graph facts"
    ):

        st.json(
            result.get(
                "graph_facts",
                {
                    "entities": [],
                    "relations": [],
                },
            )
        )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    output_path = (
        cfg.demos_dir
        / "latest_result.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    # ========================================================
    # SAVE KNOWLEDGE GRAPH
    # ========================================================

    kg.save_graphml(
        cfg.graph_path_graphml
    )

    kg.save_json(
        cfg.graph_path_json
    )

    st.caption(
        f"Latest result saved to: {output_path}"
    )
