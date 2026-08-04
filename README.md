# PSC Evidence Assistant

- **RAG** (retrieval-augmented generation) over local project/literature documents
- **Prompt engineering** (strict schemas for query planning + extraction)
- A tool-using **AI agent** (plan → retrieve → extract → graph → answer → verify)
- **Knowledge graph** construction and querying (NetworkX; GraphML/JSON export)
- **Fine-tuning ready** data format (JSONL) to improve schema-compliant extraction

> Default mode runs without any model (dummy). You can plug in **LLaMA via Ollama** in minutes.

---

## Project definition
**Goal:** Build an evidence-grounded assistant for PSC work that can answer questions from * documentation* and produce a small knowledge graph of entities/relations (Disease, Method, Dataset, Parameters, etc.).

### Example question
> *What are the inputs and outputs in our PSC pipeline?*

### Expected solution (behavior)
- Retrieve relevant chunks from your local corpus (RAG)
- Provide an answer with **citations** (chunk IDs)
- Extract entities/relations and update a knowledge graph
- Optionally run a grounding check

---

## One-click data download link (optional)
If you want to download your existing dataset zip from GitHub into this repo (example):

```bash
wget -O data/HumanLiverHealthyscRNAseqData.zip \
  "https://raw.githubusercontent.com/seirana/PSC-scDRS/main/data/HumanLiverHealthyscRNAseqData.zip"
```

> If the file is managed by Git LFS, `wget` may download a pointer file. In that case, use `git lfs pull` in the original repo.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add your own docs into data/corpus/ as .md or .txt

# Run the demo (works in dummy mode)
python run_demo.py
```

Demo output:
- `outputs/demos/demo_result.json`
- `outputs/knowledge_graph.graphml`
- `outputs/knowledge_graph.json`

---

## Use LLaMA (Ollama)

1) Install and start Ollama, then pull a model:
```bash
ollama pull llama3.1
ollama serve
```

2) Run this project with Ollama:
```bash
export LLM_MODE=ollama
export OLLAMA_MODEL=llama3.1
python run_demo.py
```

---

## Streamlit UI

```bash
streamlit run app.py
```

---

## Fine-tuning (optional, later)

The folder `data/fine_tune/` contains JSONL templates for supervised fine-tuning (SFT) to improve strict JSON outputs for entity/relation extraction.

Typical goal: increase **JSON validity** and **schema compliance** for the extractor.

---

## Notes
- This repo is intentionally **model-agnostic**: swap LLaMA/GPT by editing only `src/llm.py`.
- Replace the sample corpus files with your real PSC/scDRS documentation for meaningful answers.
