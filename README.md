# Understanding LLMs Through a Real Project: PSC Evidence Assistant

- **RAG** (retrieval-augmented generation) over local project/literature documents
- **Prompt engineering** (strict schemas for query planning + extraction)
- A tool-using **AI agent** (plan → retrieve → extract → graph → answer → verify)
- **Knowledge graph** construction and querying (NetworkX; GraphML/JSON export)

> Default mode runs without any model (dummy). You can plug in **LLaMA via Ollama** in minutes.

---

## Project definition
**Goal:** Build an evidence-grounded assistant for PSC work that can answer questions from *documentation* and produce a small knowledge graph of entities/relations (Disease, Method, Dataset, Parameters, etc.).

### Example question
> *What are the inputs and outputs of scDRS in our PSC pipeline?*

### Expected solution (behavior)
- Retrieve relevant chunks from your local corpus (RAG)
- Provide an answer with **citations** (chunk IDs)
- Extract entities/relations and update a knowledge graph
- Optionally run a grounding check

---
## Clone the repository

```bash
cd ~
git clone https://github.com/seirana/PSC-Evidence-Assistant.git

```

---


## Quickstart and Run demo

```bash
# Make an environment
cd ~/PSC-Evidence-Assistant

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Add your own docs into data/corpus/ as .md or .txt or .docx, or .pdf

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

## Notes
- This repo is intentionally **model-agnostic**: swap LLaMA/GPT by editing only `src/llm.py`.
- Replace the sample corpus files with your real PSC/scDRS documentation for meaningful answers.
