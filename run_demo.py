import json

from src.config import Settings
from src.ingest import ingest_corpus, chunks_to_records
from src.rag import TfidfRAG
from src.graph_kb import KnowledgeGraph
from src.llm import LLMClient
from src.agent import EvidenceAgent

cfg = Settings()
cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
cfg.demos_dir.mkdir(parents=True, exist_ok=True)

chunks = ingest_corpus(cfg.corpus_dir, cfg.chunk_size_chars, cfg.chunk_overlap_chars)
records = chunks_to_records(chunks)

rag = TfidfRAG(records)
kg = KnowledgeGraph()
llm = LLMClient()
agent = EvidenceAgent(rag=rag, kg=kg, llm=llm, top_k=cfg.top_k)

question = "What are the inputs and outputs of scDRS in our PSC pipeline?"
result = agent.answer(question)

# Save artifacts
out = cfg.demos_dir / "demo_result.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")

kg.save_graphml(cfg.graph_path_graphml)
kg.save_json(cfg.graph_path_json)

print(f"Saved: {out}")
print(result["answer"])
