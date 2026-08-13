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


NOT_FOUND_MESSAGE = "Not found in provided documents."


class EvidenceAgent:
    """
    Evidence-grounded agent.

    Workflow:

        user question
            ↓
        rewrite/search planning
            ↓
        RAG retrieval
            ↓
        evidence found?
          /       \
        yes        no
         ↓          ↓
        extract     stop
        facts       ↓
         ↓       "Not found..."
        answer
         ↓
        verify grounding
         ↓
        supported?
         /      \
       yes       no
        ↓         ↓
      answer   "Not found..."
    """

    def __init__(
        self,
        rag: TfidfRAG,
        kg: KnowledgeGraph,
        llm: LLMClient,
        top_k: int = 6,
        min_score: float = 0.05,
    ):
        """
        Parameters
        ----------
        rag:
            Retrieval system.

        kg:
            Knowledge graph.

        llm:
            LLM client.

        top_k:
            Maximum number of chunks used as evidence.

        min_score:
            Minimum retrieval similarity score.
        """

        self.rag = rag
        self.kg = kg
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def _no_evidence_response(
        self,
        user_question: str,
        plan: Dict[str, Any],
        plan_text: str,
        note: str,
    ) -> Dict[str, Any]:
        """
        Create a consistent response when the system cannot
        find enough evidence to answer safely.

        Important:
        This response is produced by Python.

        The LLM is NOT asked to invent an answer.
        """

        return {
            "question": user_question,

            "plan": plan,

            "plan_raw": plan_text,

            "citations": [],

            "graph_facts": {
                "entities": [],
                "relations": [],
            },

            "graph_facts_raw": "",

            "answer": NOT_FOUND_MESSAGE,

            "grounding_check": {
                "supported": True,
                "unsupported_claims": [],
                "notes": note,
            },

            "grounding_check_raw": "",
        }

    def answer(
        self,
        user_question: str,
    ) -> Dict[str, Any]:
        """
        Answer a user's question using only evidence
        retrieved from the corpus.
        """

        # =====================================================
        # 0. Validate the question
        # =====================================================

        if not user_question or not user_question.strip():

            return self._no_evidence_response(
                user_question=user_question,
                plan={
                    "rewritten_queries": [],
                    "entities_of_interest": [],
                },
                plan_text="",
                note="The question was empty.",
            )

        # =====================================================
        # 1. PLAN RETRIEVAL
        #
        # Ask the LLM to create useful search queries.
        #
        # This is NOT the final answer.
        # It only helps retrieval.
        # =====================================================

        plan_prompt = prompt_query_rewrite(
            user_question
        )

        plan_text = self.llm.generate(
            plan_prompt
        )

        plan = parse_json_safely(
            plan_text
        )

        # If query rewriting fails, simply use
        # the original question.
        if not plan:

            plan = {
                "rewritten_queries": [
                    user_question
                ],
                "entities_of_interest": [],
            }

        rewritten_queries = (
            plan.get("rewritten_queries")
            or []
        )

        # =====================================================
        # IMPORTANT SAFETY / ROBUSTNESS CHANGE
        #
        # Always search using the ORIGINAL user question.
        #
        # Rewritten LLM queries may help, but they should not
        # completely replace what the user actually asked.
        # =====================================================

        queries = [
            user_question
        ]

        for query in rewritten_queries:

            if not isinstance(query, str):
                continue

            query = query.strip()

            if not query:
                continue

            if query not in queries:
                queries.append(query)

        # Use a maximum of 3 retrieval queries.
        queries = queries[:3]

        # =====================================================
        # 2. RETRIEVE EVIDENCE
        # =====================================================

        retrieved_all = []

        for query in queries:

            results = self.rag.retrieve(
                query,
                top_k=max(
                    2,
                    self.top_k // 2,
                ),
                min_score=self.min_score,
            )

            retrieved_all.extend(
                results
            )

        # =====================================================
        # 3. DEDUPLICATE CHUNKS
        #
        # The same chunk may have been found by several
        # rewritten queries.
        #
        # Keep only its best score.
        # =====================================================

        best = {}

        for retrieved_chunk in retrieved_all:

            existing = best.get(
                retrieved_chunk.chunk_id
            )

            if (
                existing is None
                or retrieved_chunk.score > existing.score
            ):
                best[
                    retrieved_chunk.chunk_id
                ] = retrieved_chunk

        retrieved = sorted(
            best.values(),
            key=lambda chunk: chunk.score,
            reverse=True,
        )[:self.top_k]

        # =====================================================
        # 4. VERY IMPORTANT:
        # NO EVIDENCE = STOP
        # =====================================================

        if not retrieved:

            return self._no_evidence_response(
                user_question=user_question,
                plan=plan,
                plan_text=plan_text,
                note=(
                    "No retrieved chunk passed "
                    f"the minimum relevance score "
                    f"of {self.min_score}."
                ),
            )

        # =====================================================
        # 5. FORMAT EVIDENCE FOR THE LLM
        # =====================================================

        context, citations = format_context(
            retrieved
        )

        # It is possible, for example because of max_chars,
        # that formatting produces no usable context.
        #
        # Again: do NOT call the answering LLM.
        if not context.strip():

            return self._no_evidence_response(
                user_question=user_question,
                plan=plan,
                plan_text=plan_text,
                note=(
                    "Relevant chunks were retrieved, "
                    "but no usable context could be built."
                ),
            )

        # =====================================================
        # 6. EXTRACT GRAPH FACTS
        #
        # Only now do we allow the LLM to inspect evidence.
        # =====================================================

        facts_prompt = prompt_extract_graph_facts(
            context
        )

        facts_text = self.llm.generate(
            facts_prompt
        )

        facts = parse_json_safely(
            facts_text
        )

        if not facts:

            facts = {
                "entities": [],
                "relations": [],
            }

        # =====================================================
        # 7. UPDATE KNOWLEDGE GRAPH
        # =====================================================

        for entity in facts.get(
            "entities",
            [],
        ):

            self.kg.add_entity(
                entity.get(
                    "name",
                    "",
                ),
                entity.get(
                    "type",
                    "Other",
                ),
                entity.get(
                    "normalized_id"
                ),
            )

        for relation in facts.get(
            "relations",
            [],
        ):

            self.kg.add_relation(
                relation.get(
                    "source",
                    "",
                ),
                relation.get(
                    "relation",
                    "MENTIONED_IN",
                ),
                relation.get(
                    "target",
                    "",
                ),
                relation.get(
                    "evidence_chunk_id",
                    "",
                ),
            )

        # =====================================================
        # 8. GENERATE ANSWER
        #
        # The LLM receives:
        #
        # - the user's question
        # - ONLY the retrieved document context
        # =====================================================

        answer_prompt = prompt_answer_with_citations(
            user_question,
            context,
        )

        answer_text = self.llm.generate(
            answer_prompt
        )

        # =====================================================
        # 9. VERIFY GROUNDING
        #
        # Ask whether the generated answer is actually
        # supported by the retrieved context.
        # =====================================================

        verify_prompt = prompt_grounding_verify(
            answer_text,
            context,
        )

        verify_text = self.llm.generate(
            verify_prompt
        )

        verify = parse_json_safely(
            verify_text
        )

        # =====================================================
        # IMPORTANT SAFETY CHANGE
        #
        # Previous code effectively treated a failed verifier
        # parse as:
        #
        #     supported = True
        #
        # For a strict evidence assistant, that is unsafe.
        #
        # We use the conservative default:
        #
        #     supported = False
        # =====================================================

        if not verify:

            verify = {
                "supported": False,
                "unsupported_claims": [],
                "notes": (
                    "Grounding verification could "
                    "not be parsed."
                ),
            }

        # =====================================================
        # 10. FINAL GATE
        #
        # If the verifier says the answer is not completely
        # supported by the documents, do not show it.
        # =====================================================

        if not verify.get(
            "supported",
            False,
        ):

            answer_text = NOT_FOUND_MESSAGE

        # =====================================================
        # 11. RETURN RESULT
        # =====================================================

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
