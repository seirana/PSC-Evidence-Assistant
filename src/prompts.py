from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class QueryPlan(BaseModel):
    rewritten_queries: List[str] = Field(..., min_items=1, max_items=5)
    entities_of_interest: List[str] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    type: Literal[
        "Disease",
        "Gene",
        "CellType",
        "Method",
        "Dataset",
        "Parameter",
        "Output",
        "Input",
        "Pathway",
        "Drug",
        "Other",
    ]
    name: str
    normalized_id: Optional[str] = None


class ExtractedRelation(BaseModel):
    source: str
    relation: Literal[
        "ASSOCIATED_WITH",
        "TARGETS",
        "TREATS",
        "USED_IN",
        "REQUIRES_INPUT",
        "PRODUCES_OUTPUT",
        "PART_OF",
        "MENTIONED_IN",
    ]
    target: str
    evidence_chunk_id: str


class GraphFacts(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)


def prompt_query_rewrite(user_question: str) -> str:
    return f"""
You rewrite the user question into up to 5 retrieval queries.
Return ONLY valid JSON following this schema:
{QueryPlan.model_json_schema()}

User question:
{user_question}
""".strip()


def prompt_answer_with_citations(user_question: str, context: str) -> str:
    return f"""
You are an evidence-grounded assistant. Answer ONLY using the provided CONTEXT.
If information is missing, say: "Not found in provided documents."

Rules:
- Every factual claim must include a citation in parentheses like (chunk_id).
- Use the chunk_id from the SOURCE headers in CONTEXT.
- Keep the answer concise and technical.

User question:
{user_question}

CONTEXT:
{context}
""".strip()


def prompt_extract_graph_facts(context: str) -> str:
    return f"""
Extract biomedical/project entities and relations from the CONTEXT.
Return ONLY valid JSON following this schema:
{GraphFacts.model_json_schema()}

Guidelines:
- Use entity names exactly as in the text when possible.
- Every relation MUST include evidence_chunk_id referring to a chunk in CONTEXT.
- Prefer a small set of high-confidence relations over many uncertain ones.

CONTEXT:
{context}
""".strip()


def prompt_grounding_verify(answer: str, context: str) -> str:
    return f"""
Check whether the ANSWER is fully supported by the CONTEXT.

Return ONLY JSON:
{{
  "supported": true/false,
  "unsupported_claims": ["..."],
  "notes": "short"
}}

ANSWER:
{answer}

CONTEXT:
{context}
""".strip()
