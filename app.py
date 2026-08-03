import json
import streamlit as st

from src.config import Settings
from src.ingest import ingest_corpus, chunks_to_records
from src.rag import TfidfRAG
from src.llm import LLMClient
from src.graph_kb import KnowledgeGraph
from src.agent import EvidenceAgent

st.set_page_config(page_title="PSC Evidence Assistant", layout="wide")

cfg = Settings()
cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
cfg.demos_dir.mkdir(parents=True, exist_ok=True)

st.title("PSC Evidence Assistant (RAG + Agent + Knowledge Graph + Fine-tune-ready)")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k retrieval", min_value=3, max_value=12, value=cfg.top_k)
    st.caption("LLM runs in dummy mode unless you set env vars (LLM_MODE=ollama).")

@st.cache_data(show_spinner=True)
def build_index():
    chunks = ingest_corpus(cfg.corpus_dir, cfg.chunk_size_chars, cfg.chunk_overlap_chars)
    records = chunks_to_records(chunks)
    rag = TfidfRAG(records)
    return records, rag

records, rag = build_index()
kg = KnowledgeGraph()
llm = LLMClient()
agent = EvidenceAgent(rag=rag, kg=kg, llm=llm, top_k=top_k)

q = st.text_input(
    "Ask a question",
    value="What are the inputs and outputs of scDRS in our PSC pipeline?",
)

if st.button("Run"):
    res = agent.answer(q)

    st.subheader("Answer")
    st.write(res["answer"])

    st.subheader("Citations (retrieved chunks)")
    st.json(res["citations"])

    st.subheader("Graph facts")
    st.json(res["graph_facts"])

    # Save result
    out_path = cfg.demos_dir / "latest_result.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # Save KG
    kg.save_graphml(cfg.graph_path_graphml)
    kg.save_json(cfg.graph_path_json)

    st.success(f"Saved: {out_path}")
