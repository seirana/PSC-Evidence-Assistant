import re
from typing import Dict, Any, List, Optional

from .rag import (
    TfidfRAG,
    RetrievedChunk,
    format_context,
)

from .prompts import (
    QueryPlan,
    GraphFacts,
    GroundingCheck,
    prompt_query_rewrite,
    prompt_answer_with_citations,
    prompt_extract_graph_facts,
    prompt_grounding_verify,
)

from .llm import LLMClient
from .verify import parse_json_safely
from .graph_kb import KnowledgeGraph


NOT_FOUND_MESSAGE = "Not found in provided documents."


# These aliases clarify the user's wording for retrieval.
# They do not supply answer content: every answer must still
# come from a retrieved document chunk.
KNOWN_ABBREVIATIONS = {
    "PSC": "Primary Sclerosing Cholangitis",
}


def _expand_known_abbreviations(
    question: str,
) -> str:
    """
    Expand known domain abbreviations before retrieval.

    A short lexical query such as "What is PSC?" otherwise
    gives TF-IDF only the broad token "PSC". That token occurs
    throughout the corpus, including in drug-ranking passages.
    Expanding it lets retrieval and the direct-definition
    matcher search for the disease name the user intended.

    The original question is still preserved in the returned
    result, and all answer text must still come from evidence.
    """

    expanded = " ".join(
        (question or "").split()
    )

    for abbreviation, full_name in (
        KNOWN_ABBREVIATIONS.items()
    ):
        # Do not turn an already explicit phrase such as
        # "Primary Sclerosing Cholangitis (PSC)" into a
        # duplicated phrase.
        if full_name.casefold() in expanded.casefold():
            continue

        expanded = re.sub(
            rf"\b{re.escape(abbreviation)}\b",
            full_name,
            expanded,
            flags=re.IGNORECASE,
        )

    return expanded


def _parse_model_output(
    text: str,
    model_class: Any,
) -> Optional[Dict[str, Any]]:
    """
    Parse and validate one structured LLM response.

    Ollama normally obeys the supplied JSON Schema, but a
    small model can occasionally wrap the requested object
    in a one-item list. We safely unwrap that one case and
    then let Pydantic validate every field.
    """

    value = parse_json_safely(
        text
    )

    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
    ):
        value = value[0]

    if not isinstance(value, dict):
        return None

    try:
        validated = model_class.model_validate(
            value
        )
    except Exception:
        return None

    return validated.model_dump()


def _question_subject(
    question: str,
) -> str:
    """
    Extract the subject from a direct definition question.

    Example:
        "What is Primary Sclerosing Cholangitis?"
        -> "primary sclerosing cholangitis"
    """

    normalized = " ".join(
        _expand_known_abbreviations(
            question
        ).casefold().split()
    ).strip()

    prefixes = (
        "what is ",
        "what are ",
        "define ",
        "describe ",
    )

    for prefix in prefixes:
        if normalized.startswith(prefix):
            return normalized[
                len(prefix):
            ].strip(" ?.!")

    return ""


def _is_direct_answer_chunk(
    question: str,
    chunk: RetrievedChunk,
) -> bool:
    """
    Detect a chunk that directly defines the requested
    subject. This is a ranking hint, not proof of support.
    """

    subject = _question_subject(
        question
    )

    if not subject:
        return False

    text = " ".join(
        chunk.text.casefold().split()
    )

    patterns = (
        f"what is {subject}",
        f"what are {subject}",
        f"{subject} is ",
        f"{subject} are ",
    )

    return any(
        pattern in text
        for pattern in patterns
    )


def _looks_like_references(
    chunk: RetrievedChunk,
) -> bool:
    """Return True for chunks dominated by bibliography."""

    text = chunk.text.casefold()

    link_count = (
        text.count("http://")
        + text.count("https://")
        + text.count("doi:")
    )

    return link_count >= 3


def _order_context_chunks(
    question: str,
    chunks: List[RetrievedChunk],
) -> List[RetrievedChunk]:
    """
    Put direct explanatory evidence first and reference-list
    chunks last. TF-IDF can otherwise rank a bibliography
    highly merely because it repeats the disease name.
    """

    return sorted(
        chunks,
        key=lambda chunk: (
            not _is_direct_answer_chunk(
                question,
                chunk,
            ),
            _looks_like_references(
                chunk
            ),
            -chunk.score,
        ),
    )


def _is_not_found_answer(
    answer: str,
) -> bool:
    """Recognize the project's exact abstention message."""

    if not isinstance(answer, str):
        return True

    normalized = " ".join(
        answer.strip().strip('"').split()
    ).casefold().rstrip(".")

    expected = NOT_FOUND_MESSAGE.casefold().rstrip(
        "."
    )

    return normalized == expected


def _extract_direct_answer(
    question: str,
    chunk: RetrievedChunk,
) -> str:
    """
    Extract a short answer directly from a matching document
    section instead of asking the LLM to paraphrase it.

    This path is used only for direct definition questions
    such as "What is X?" when the retrieved chunk contains a
    matching heading or an explicit "X is ..." statement.
    Because the text comes from one known chunk, Python can
    attach the real chunk ID without relying on the model to
    invent or copy citations.
    """

    subject = _question_subject(
        question
    )

    if not subject:
        return ""

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            chunk.text,
        )
        if paragraph.strip()
    ]

    selected: List[str] = []

    # Prefer text immediately below a matching section
    # heading, for example:
    #
    #   ## What is Primary Sclerosing Cholangitis?
    #   Primary sclerosing cholangitis is ...
    for index, paragraph in enumerate(
        paragraphs
    ):
        heading = " ".join(
            paragraph.lstrip("# ").casefold().split()
        )

        heading = re.sub(
            r"^\d+[.)]?\s*",
            "",
            heading,
        )

        heading_matches = (
            f"what is {subject}" in heading
            or f"what are {subject}" in heading
        )

        if not heading_matches:
            continue

        for candidate in paragraphs[
            index + 1:
        ]:
            if candidate.lstrip().startswith("#"):
                break

            if candidate.strip() in {
                "---",
                "***",
            }:
                continue

            if (
                candidate.count("http://")
                + candidate.count("https://")
                >= 2
            ):
                break

            selected.append(
                candidate
            )

            # Two source paragraphs are enough for a concise
            # definition and prevent unrelated material from
            # being included.
            if len(selected) >= 2:
                break

        break

    # If no matching heading exists, accept one paragraph
    # containing an explicit definition sentence.
    if not selected:
        patterns = (
            f"{subject} is ",
            f"{subject} are ",
        )

        for paragraph in paragraphs:
            normalized = " ".join(
                paragraph.casefold().split()
            )

            if any(
                pattern in normalized
                for pattern in patterns
            ):
                selected = [
                    paragraph
                ]
                break

    cleaned_paragraphs = []

    for paragraph in selected:
        cleaned = paragraph.strip().lstrip(
            "> "
        )
        cleaned = cleaned.replace(
            "**",
            "",
        )
        cleaned = re.sub(
            r"\[(?:\d+(?:\s*,\s*\d+)*)\]",
            "",
            cleaned,
        )
        cleaned = " ".join(
            cleaned.split()
        ).strip()
        cleaned = re.sub(
            r"\s+([,.;:!?])",
            r"\1",
            cleaned,
        )

        if cleaned:
            cleaned_paragraphs.append(
                f"{cleaned} ({chunk.chunk_id})"
            )

    return "\n\n".join(
        cleaned_paragraphs
    )


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

            "answer_raw": NOT_FOUND_MESSAGE,

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

        # Resolve known abbreviations deterministically before
        # asking the LLM to plan retrieval. This prevents a
        # small model from having to guess what PSC means.
        retrieval_question = (
            _expand_known_abbreviations(
                user_question
            )
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
            retrieval_question
        )

        plan_text = self.llm.generate(
            plan_prompt,
            json_schema=(
                QueryPlan.model_json_schema()
            ),
        )

        plan = _parse_model_output(
            plan_text,
            QueryPlan,
        )

        # If rewriting fails, retrieval still uses the
        # original question. Query planning must never stop
        # an otherwise answerable request.
        if plan is None:
            plan = {
                "rewritten_queries": [
                    retrieval_question
                ],
                "entities_of_interest": [],
            }

        rewritten_queries = plan[
            "rewritten_queries"
        ]

        # =====================================================
        # IMPORTANT SAFETY / ROBUSTNESS CHANGE
        #
        # Always search using the ORIGINAL user question.
        #
        # Rewritten LLM queries may help, but they should not
        # completely replace what the user actually asked.
        # =====================================================

        queries = [
            retrieval_question
        ]

        # Also keep the literal wording when it differs. The
        # expanded form is searched first because it carries
        # the user's intended scientific meaning.
        if user_question not in queries:
            queries.append(
                user_question
            )

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
                top_k = self.top_k,
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

        context_chunks = _order_context_chunks(
            retrieval_question,
            retrieved,
        )

        context, citations = format_context(
            context_chunks
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
            facts_prompt,
            json_schema=(
                GraphFacts.model_json_schema()
            ),
        )

        facts = _parse_model_output(
            facts_text,
            GraphFacts,
        )

        # Knowledge-graph extraction is optional. Invalid
        # graph output must not block a supported answer.
        if facts is None:
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

        direct_chunks = [
            chunk
            for chunk in context_chunks
            if _is_direct_answer_chunk(
                retrieval_question,
                chunk,
            )
        ]

        extractive_answer = ""
        extractive_chunk = None

        for chunk in direct_chunks:
            candidate = _extract_direct_answer(
                retrieval_question,
                chunk,
            )

            if candidate:
                extractive_answer = candidate
                extractive_chunk = chunk
                break

        used_exact_extract = (
            extractive_chunk is not None
        )

        if used_exact_extract:
            # The answer consists only of text extracted from
            # one retrieved chunk. Use its real citation and do
            # not let a second LLM reinterpret or reject it.
            answer_text = extractive_answer
            context, citations = format_context(
                [extractive_chunk],
                max_chars=5000,
            )

        else:
            answer_prompt = prompt_answer_with_citations(
                retrieval_question,
                context,
            )

            answer_text = self.llm.generate(
                answer_prompt
            ).strip()

            # If the model abstains despite a direct-match
            # chunk that could not be extracted, retry once
            # using only the focused evidence.
            if (
                _is_not_found_answer(answer_text)
                and direct_chunks
            ):
                focused_context, focused_citations = (
                    format_context(
                        direct_chunks,
                        max_chars=5000,
                    )
                )

                focused_prompt = (
                    prompt_answer_with_citations(
                        retrieval_question,
                        focused_context,
                    )
                )

                focused_answer = self.llm.generate(
                    focused_prompt
                ).strip()

                if not _is_not_found_answer(
                    focused_answer
                ):
                    answer_text = focused_answer
                    context = focused_context
                    citations = focused_citations

        generated_answer_text = answer_text

        # =====================================================
        # 9. VERIFY GROUNDING
        #
        # Ask whether the generated answer is actually
        # supported by the retrieved context.
        # =====================================================

        if used_exact_extract:
            verify_text = ""
            verify = {
                "supported": True,
                "unsupported_claims": [],
                "notes": (
                    "Answer extracted directly from "
                    f"retrieved chunk "
                    f"{extractive_chunk.chunk_id}."
                ),
            }

        elif _is_not_found_answer(
            answer_text
        ):
            verify_text = ""
            verify = {
                "supported": True,
                "unsupported_claims": [],
                "notes": (
                    "The answer generator abstained."
                ),
            }

        else:
            verify_prompt = prompt_grounding_verify(
                answer_text,
                context,
            )

            verify_text = self.llm.generate(
                verify_prompt,
                json_schema=(
                    GroundingCheck.model_json_schema()
                ),
            )

            verify = _parse_model_output(
                verify_text,
                GroundingCheck,
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

        if verify is None:

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

            "answer_raw": generated_answer_text,

            "answer": answer_text,

            "grounding_check": verify,

            "grounding_check_raw": verify_text,
        }
