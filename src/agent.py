import re
from typing import (
    Dict,
    Any,
    List,
    Optional,
)

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


def _normalize_question_for_retrieval(
    question: str,
) -> str:
    """
    Normalize harmless punctuation mistakes for retrieval.

    For example, users sometimes type::

        What does ABC? stand for ...

    The question mark separates the term from "stand for"
    and makes an already short lexical query less reliable.
    Only this specific grammatical pattern is repaired; real
    sentence-ending question marks are left unchanged.
    """

    normalized = " ".join(
        (question or "").split()
    )

    return re.sub(
        r"(?i)(\bwhat\s+does\s+[^?]{1,100})"
        r"\?\s+(stand\s+for\b)",
        r"\1 \2",
        normalized,
    )


def _stand_for_term(
    question: str,
) -> str:
    """Return the term in a "What does X stand for?" query."""

    normalized = _normalize_question_for_retrieval(
        question
    )

    match = re.search(
        r"(?i)\bwhat\s+does\s+(.+?)\s+stand\s+for\b",
        normalized,
    )

    if match is None:
        return ""

    term = match.group(1).strip(
        " ?.!,:;\"'"
    )

    # A very long capture is probably not a name or acronym.
    if not term or len(term.split()) > 8:
        return ""

    return term


def _question_facets(
    question: str,
) -> List[str]:
    """
    Split a compound question into deterministic search facets.

    Query rewriting by the LLM is still used, but retrieval must
    not depend on a small model noticing that two different
    pieces of evidence are required.
    """

    normalized = _normalize_question_for_retrieval(
        question
    )
    facets = [normalized] if normalized else []

    match = re.search(
        r"(?i)\s+and\s+"
        r"(?=(?:how|what|why|where|when|which)\b)",
        normalized,
    )

    if match is None:
        return facets

    first = normalized[:match.start()].strip(
        " ?.!,:;"
    )
    second = normalized[match.end():].strip(
        " ?.!,:;"
    )

    term = _stand_for_term(
        normalized
    )

    if term and second:
        # Resolve the first pronoun in a clause such as
        # "how is it used ..." so TF-IDF sees the entity name.
        second = re.sub(
            r"(?i)\bit\b",
            term,
            second,
            count=1,
        )

        if term.casefold() not in second.casefold():
            second = f"{term} {second}"

    for facet in (first, second):
        if facet and facet not in facets:
            facets.append(facet)

    return facets


def _extract_term_expansion(
    term: str,
    chunk: RetrievedChunk,
) -> str:
    """
    Extract "Expanded Name" from "Expanded Name (TERM)".

    The extraction is intentionally narrow so the code cannot
    manufacture an acronym expansion from general knowledge.
    """

    if not term:
        return ""

    escaped = re.escape(term)

    for line in chunk.text.splitlines():
        cleaned = " ".join(
            line.split()
        ).strip()
        cleaned = re.sub(
            r"^\d+(?:\.\d+)*\s+",
            "",
            cleaned,
        )

        match = re.fullmatch(
            rf"(.{{2,120}}?)\s*\(\s*{escaped}\s*\)\s*",
            cleaned,
            flags=re.IGNORECASE,
        )

        if match is None:
            continue

        expansion = match.group(1).strip(
            " -:;"
        )

        if expansion:
            return expansion

    return ""


def _is_term_expansion_chunk(
    question: str,
    chunk: RetrievedChunk,
) -> bool:
    """Return True when a chunk explicitly expands the term."""

    term = _stand_for_term(
        question
    )

    return bool(
        _extract_term_expansion(
            term,
            chunk,
        )
    )


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
        "What is Example Process?"
        -> "example process"
    """

    normalized = " ".join(
        (question or "").casefold().split()
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
            0
            if _is_term_expansion_chunk(
                question,
                chunk,
            )
            else 1
            if _is_direct_answer_chunk(
                question,
                chunk,
            )
            else 2,
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
        #   ## What is Example Process?
        #   Example Process is ...
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

        # Normalize only harmless punctuation. Terminology is
        # resolved from definitions discovered in the corpus,
        # not from a hard-coded list of anticipated questions.
        retrieval_question = (
            _normalize_question_for_retrieval(
                user_question
            )
        )

        stand_for_term = _stand_for_term(
            retrieval_question
        )

        # Direct "What is X?" matching benefits from replacing
        # a corpus-defined alias such as PSC with its full name.
        # For "What does X stand for?", keep X unchanged so its
        # explicit expansion heading can be recognized.
        matching_question = (
            retrieval_question
            if stand_for_term
            else self.rag.replace_query_aliases(
                retrieval_question
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

        question_facets = _question_facets(
            retrieval_question
        )

        queries = list(
            question_facets
        )

        # Add corpus-derived terminology to each facet. The
        # original facet remains present, so this is query
        # expansion rather than question-specific replacement.
        for facet in question_facets:
            expanded_facet = self.rag.expand_query(
                facet
            )

            if expanded_facet not in queries:
                queries.append(
                    expanded_facet
                )

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

        # Keep the retrieval work bounded while leaving room for
        # the original question, its facets, corpus expansions,
        # and useful LLM rewrites.
        queries = queries[:8]

        # =====================================================
        # 2. RETRIEVE EVIDENCE
        # =====================================================

        retrieved_all = []
        results_by_query = []

        candidate_k = max(
            self.top_k * 3,
            12,
        )

        for query in queries:

            results = self.rag.retrieve(
                query,
                top_k=candidate_k,
                min_score=self.min_score,
            )

            retrieved_all.extend(
                results
            )

            results_by_query.append(
                results
            )

        # A short name can appear throughout a paper, causing
        # its explicit expansion heading to have a low TF-IDF
        # score. Search exact definition patterns separately;
        # this method returns only text actually present in the
        # corpus and never guesses the expansion.
        definition_results = []

        if stand_for_term:
            definition_results = (
                self.rag.retrieve_term_definitions(
                    stand_for_term,
                    top_k=3,
                )
            )

            retrieved_all.extend(
                definition_results
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

        # Preserve evidence coverage across question facets.
        # A single globally sorted list can otherwise devote all
        # available slots to one half of a compound question.
        selected = []
        selected_ids = set()

        def add_selected(
            chunk: RetrievedChunk,
        ) -> None:
            if chunk.chunk_id in selected_ids:
                return

            selected.append(
                best.get(
                    chunk.chunk_id,
                    chunk,
                )
            )
            selected_ids.add(
                chunk.chunk_id
            )

        for chunk in definition_results:
            add_selected(chunk)

        for results in results_by_query:
            if results:
                add_selected(
                    results[0]
                )

        for chunk in _order_context_chunks(
            matching_question,
            list(best.values()),
        ):
            add_selected(chunk)

        retrieved = _order_context_chunks(
            matching_question,
            selected[:self.top_k],
        )

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
            matching_question,
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
                matching_question,
                chunk,
            )
        ]

        extractive_answer = ""
        extractive_chunk = None

        for chunk in direct_chunks:
            candidate = _extract_direct_answer(
                matching_question,
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
            # This is a general direct-definition path. The text
            # is copied from one matching evidence chunk rather
            # than generated from a stored answer.
            answer_text = extractive_answer
            context, citations = format_context(
                [extractive_chunk],
                max_chars=5000,
            )

        else:
            answer_prompt = prompt_answer_with_citations(
                retrieval_question,
                context,
                question_facets=question_facets,
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
                        question_facets=(
                            question_facets
                        ),
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
