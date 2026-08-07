from typing import Dict, Any

from .rag import TfidfRAG, format_context
from .prompts import (
    prompt_query_rewrite,
    prompt_answer_with_citations,
    prompt_extract_graph_facts,
    prompt_grounding_verify,
)
from .llm import LLMClient
from .verify import parse_json_safely
from .graph_kb import KnowledgeGraph


class EvidenceAgent:
    """Tool-using agent: plan -> retrieve -> extract graph -> answer -> verify."""

    def __init__(self, rag: TfidfRAG, kg: KnowledgeGraph, llm: LLMClient, top_k: int = 6):
        self.rag = rag
        self.kg = kg
        self.llm = llm
        self.top_k = top_k

    def answer(self, user_question: str) -> Dict[str, Any]:
        # 1) Plan retrieval queries
        plan_prompt = prompt_query_rewrite(user_question)
        plan_text = self.llm.generate(plan_prompt)
        plan = parse_json_safely(plan_text) or {"rewritten_queries": [user_question], "entities_of_interest": []}
        queries = plan.get("rewritten_queries") or [user_question]

        # 2) Retrieve evidence (RAG)
        retrieved_all = []
        for q in queries[:3]:
            retrieved_all.extend(self.rag.retrieve(q, top_k=max(2, self.top_k // 2)))

        # Deduplicate by chunk_id keeping best score
        best = {}
        for r in retrieved_all:
            if (r.chunk_id not in best) or (r.score > best[r.chunk_id].score):
                best[r.chunk_id] = r
        retrieved = sorted(best.values(), key=lambda x: x.score, reverse=True)[: self.top_k]

        context, citations = format_context(retrieved)

        # 3) Extract graph facts and update KG
        facts_prompt = prompt_extract_graph_facts(context)
        facts_text = self.llm.generate(facts_prompt)
        facts = parse_json_safely(facts_text) or {"entities": [], "relations": []}

        for e in facts.get("entities", []):
            self.kg.add_entity(e.get("name", ""), e.get("type", "Other"), e.get("normalized_id"))
        for rel in facts.get("relations", []):
            self.kg.add_relation(
                rel.get("source", ""),
                rel.get("relation", "MENTIONED_IN"),
                rel.get("target", ""),
                rel.get("evidence_chunk_id", ""),
            )

        # 4) Generate grounded answer
        ans_prompt = prompt_answer_with_citations(user_question, context)
        answer_text = self.llm.generate(ans_prompt)

        # 5) Verify grounding
        verify_prompt = prompt_grounding_verify(answer_text, context)
        verify_text = self.llm.generate(verify_prompt)
        verify = parse_json_safely(verify_text) or {"supported": True, "unsupported_claims": [], "notes": ""}

        return {
            "question": user_question,
            "plan": plan,
            "plan_raw": plan_text,
            "citations": citations,
            "graph_facts": facts,
            "graph_facts_raw": facts_text,
            "answer": answer_text,
            "grounding_check": verify,
            "grounding_check_raw": verify_text,
        }
