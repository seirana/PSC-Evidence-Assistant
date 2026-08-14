import json

from src.config import Settings
from src.ingest import ingest_corpus, chunks_to_records
from src.rag import TfidfRAG
from src.graph_kb import KnowledgeGraph
from src.llm import LLMClient
from src.agent import EvidenceAgent


def main():
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    cfg = Settings()

    cfg.outputs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.demos_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.corpus_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Read corpus and create chunks
    # ---------------------------------------------------------

    chunks = ingest_corpus(
        corpus_dir=cfg.corpus_dir,
        chunk_size_chars=cfg.chunk_size_chars,
        chunk_overlap_chars=cfg.chunk_overlap_chars,
    )

    records = chunks_to_records(
        chunks
    )

    print(
        f"Loaded {len(records)} chunks "
        f"from {cfg.corpus_dir}"
    )

    if not records:
        print(
            "No readable documents were found "
            "in data/corpus."
        )
        return

    # ---------------------------------------------------------
    # 2. Build RAG
    # ---------------------------------------------------------

    rag = TfidfRAG(
        records
    )

    # ---------------------------------------------------------
    # 3. Create knowledge graph
    # ---------------------------------------------------------

    kg = KnowledgeGraph()

    # ---------------------------------------------------------
    # 4. Create LLM client
    # ---------------------------------------------------------

    llm = LLMClient()

    print(
        f"LLM mode: {llm.mode}"
    )

    # ---------------------------------------------------------
    # 5. Create evidence agent
    # ---------------------------------------------------------

    agent = EvidenceAgent(
        rag=rag,
        kg=kg,
        llm=llm,
        top_k=cfg.top_k,
        min_score=cfg.min_retrieval_score,
    )

    # ---------------------------------------------------------
    # 6. Demo question
    # ---------------------------------------------------------

    question = (
        "What cell populations were enriched "
        "for PSC genetic risk?"
    )

    print()
    print("Question:")
    print(question)

    # ---------------------------------------------------------
    # 7. Run the complete pipeline
    # ---------------------------------------------------------

    result = agent.answer(
        question
    )

    # ---------------------------------------------------------
    # 8. Save result
    # ---------------------------------------------------------

    output_path = (
        cfg.demos_dir
        / "demo_result.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 9. Save knowledge graph
    # ---------------------------------------------------------

    kg.save_graphml(
        cfg.graph_path_graphml
    )

    kg.save_json(
        cfg.graph_path_json
    )

    # ---------------------------------------------------------
    # 10. Show result
    # ---------------------------------------------------------

    print()
    print("Answer:")
    print(
        result["answer"]
    )

    print()
    print("Retrieved evidence:")

    citations = result.get(
        "citations",
        []
    )

    if not citations:
        print(
            "No relevant evidence retrieved."
        )

    else:
        for citation in citations:
            print(
                f"- {citation['doc_path']} "
                f"| {citation['chunk_id']} "
                f"| score={citation['score']:.4f}"
            )

    print()
    print(
        f"Saved result: {output_path}"
    )


if __name__ == "__main__":
    main()
