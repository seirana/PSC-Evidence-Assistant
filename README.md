# PSC Evidence Assistant

An evidence-grounded RAG application for asking questions about a local PSC/document corpus. The system retrieves relevant chunks, answers only from retrieved evidence, attaches chunk citations, optionally extracts a knowledge graph, and applies a grounding check before returning generated answers.

> This repository is an educational/research software project, not a clinical decision system.

## Core behavior

The assistant is designed around one strict rule:

**If the available corpus does not support an answer, return `Not found in provided documents.`**

The application therefore uses deterministic Python gates in addition to prompt instructions.

## Architecture

```text
data/corpus/
    ↓
src/utils.py       read .txt/.md/.docx/.pdf
    ↓
src/ingest.py      split documents into attributed chunks
    ↓
src/rag.py         TF-IDF + cosine-similarity retrieval
    ↓
src/agent.py       planning, retrieval coverage, evidence gates
    ↓
src/prompts.py     answer / graph / verification instructions
    ↓
src/llm.py         dummy, Ollama, or custom HTTP model
    ↓
src/verify.py      resilient JSON parsing
    ↓
src/graph_kb.py    optional evidence-linked knowledge graph
```

## Safety and trust boundaries

Corpus documents and LLM outputs are treated as untrusted input.

- Retrieved documents are evidence, never executable instructions.
- Prompt-injection text inside a document is not supposed to override application instructions.
- Structured LLM output is validated with Pydantic before it is used by the agent.
- Failed grounding verification is fail-closed: the draft answer is suppressed.
- No-evidence cases return a deterministic Python response rather than asking the LLM to improvise.
- The application never evaluates model/document text with `eval()`, `exec()`, or a shell.

See [SECURITY.md](SECURITY.md) for the threat model.

## Supported corpus formats

Place source material under `data/corpus/`.

Supported formats:

- `.txt`
- `.md`
- `.docx`
- text-based `.pdf`

Image-only/scanned PDFs require OCR and are not handled by the current reader.

## Quick start

```bash
git clone https://github.com/seirana/PSC-Evidence-Assistant.git
cd PSC-Evidence-Assistant

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the command-line demo

```bash
python run_demo.py
```

### Run the Streamlit interface

```bash
streamlit run app.py
```

## LLM modes

### Dummy mode

Useful for testing application plumbing. It is not a real scientific model.

```bash
export LLM_MODE=dummy
python run_demo.py
```

### Ollama

```bash
ollama pull llama3.2:3b
ollama serve

export LLM_MODE=ollama
export OLLAMA_MODEL=llama3.2:3b
python run_demo.py
```

Additional settings:

```bash
export LLM_HTTP_TIMEOUT=600
export LLM_TEMPERATURE=0.0
```

### Custom HTTP model

Set:

```bash
export LLM_MODE=http
export LLM_HTTP_URL=http://localhost:8000/generate
```

The current custom endpoint contract is documented in `src/llm.py`.

## Testing and code quality

Install development tools:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Run lint checks:

```bash
python -m ruff check src tests app.py run_demo.py
```

The test suite includes:

- abbreviation and corpus-derived terminology handling;
- compound-question retrieval coverage;
- resilient JSON parsing;
- prompt-injection/fail-closed behavior;
- grounding-verifier failure behavior.

GitHub Actions runs linting and tests on Python 3.11 and 3.12 for pushes and pull requests.

## Docker

Build:

```bash
docker build -t psc-evidence-assistant .
```

Run the Streamlit app:

```bash
docker run --rm -p 8501:8501 \
  -e LLM_MODE=ollama \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  psc-evidence-assistant
```

When Ollama runs outside the container, configure `OLLAMA_URL` for the host environment.

## Configuration

Core RAG settings are centralized in `src/config.py`:

- chunk size;
- chunk overlap;
- retrieval top-k;
- minimum retrieval score;
- corpus/output paths.

LLM deployment settings are environment variables so the same code can run against different model backends.

## Generated outputs

Runtime outputs are written under `outputs/`, including:

- demo JSON results;
- knowledge-graph JSON;
- GraphML export.

Generated outputs are ignored by Git and should be regenerated from source data and code.

## Repository layout

```text
.
├── app.py
├── run_demo.py
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── SECURITY.md
├── data/
│   └── corpus/
├── outputs/
├── src/
│   ├── agent.py
│   ├── config.py
│   ├── graph_kb.py
│   ├── ingest.py
│   ├── llm.py
│   ├── prompts.py
│   ├── rag.py
│   ├── utils.py
│   └── verify.py
└── tests/
```

## Development principle

This project intentionally keeps retrieval, model access, prompting, verification, and UI concerns separate. That makes it easier to test one layer at a time and to replace individual components—such as TF-IDF retrieval or the LLM backend—without rewriting the whole application.
