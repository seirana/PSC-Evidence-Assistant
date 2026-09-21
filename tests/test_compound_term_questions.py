import json
import unittest

from src.agent import (
    EvidenceAgent,
    _normalize_question_for_retrieval,
    _question_facets,
)
from src.rag import TfidfRAG


class StubLLM:
    """Return an answer only when both evidence chunks arrive."""

    mode = "test"

    def generate(self, prompt, json_schema=None):
        if '"rewritten_queries"' in prompt:
            return json.dumps(
                {
                    "rewritten_queries": [
                        "DEviRank graph neural networks"
                    ],
                    "entities_of_interest": [
                        "DEviRank"
                    ],
                }
            )

        if (
            "Extract biomedical and project-related"
            in prompt
        ):
            return json.dumps(
                {
                    "entities": [],
                    "relations": [],
                }
            )

        if "evidence-grounded scientific assistant" in prompt:
            if not all(
                marker in prompt
                for marker in (
                    "Evidence-Weighted Drug Ranking "
                    "(DEviRank)",
                    "graph neural networks",
                    "complementary strategy",
                )
            ):
                return "Not found in provided documents."

            return (
                "DEviRank stands for Evidence-Weighted Drug "
                "Ranking (paper_0009). It is not used as a "
                "graph neural network; the document presents "
                "it as a complementary PPI-based proximity "
                "framework (paper_0003)."
            )

        if "strict grounding verifier" in prompt:
            return json.dumps(
                {
                    "supported": True,
                    "unsupported_claims": [],
                    "notes": "Both claims have cited evidence.",
                }
            )

        raise AssertionError("Unexpected LLM prompt.")


class StubKnowledgeGraph:
    def add_entity(self, *args, **kwargs):
        return None

    def add_relation(self, *args, **kwargs):
        return None


class TestCompoundTermQuestions(unittest.TestCase):
    def setUp(self):
        records = [
            {
                "doc_id": "paper",
                "doc_path": "DEviRank-Manuscript.pdf",
                "chunk_id": "paper_0009",
                "text": (
                    "3.3 Evidence-Weighted Drug Ranking "
                    "(DEviRank)\nProblem Setup. Let G denote "
                    "a PPI network."
                ),
            },
            {
                "doc_id": "paper",
                "doc_path": "DEviRank-Manuscript.pdf",
                "chunk_id": "paper_0003",
                "text": (
                    "Representation-learning approaches, "
                    "including network embeddings and graph "
                    "neural networks, can exploit "
                    "multirelational data. DEviRank follows a "
                    "complementary strategy. Rather than "
                    "replacing explicit network proximity with "
                    "a learned latent representation, it "
                    "retains an interpretable PPI-based "
                    "proximity framework and extends it through "
                    "statistically defined candidate filtering "
                    "and evidence-weighted ranking. The method "
                    "does not require a disease-specific "
                    "labeled training set."
                ),
            },
            {
                "doc_id": "paper",
                "doc_path": "DEviRank-Manuscript.pdf",
                "chunk_id": "paper_0018",
                "text": (
                    "DEviRank DEviRank DEviRank drug ranks and "
                    "candidate compounds."
                ),
            },
        ]

        self.rag = TfidfRAG(records)

    def test_stray_question_mark_is_repaired(self):
        question = (
            "what does DEviRank? stand for and how is it used "
            "in the context of graph neural networks?"
        )

        normalized = _normalize_question_for_retrieval(
            question
        )

        self.assertIn(
            "DEviRank stand for",
            normalized,
        )
        self.assertNotIn(
            "DEviRank? stand for",
            normalized,
        )

    def test_compound_question_is_split_and_pronoun_resolved(self):
        facets = _question_facets(
            "what does DEviRank? stand for and how is it used "
            "in the context of graph neural networks?"
        )

        self.assertIn(
            "what does DEviRank stand for",
            facets,
        )
        self.assertIn(
            "how is DEviRank used in the context of graph "
            "neural networks",
            facets,
        )

    def test_exact_heading_is_found_outside_normal_top_results(self):
        matches = self.rag.retrieve_term_definitions(
            "DEviRank"
        )

        self.assertEqual(
            matches[0].chunk_id,
            "paper_0009",
        )

    def test_glossary_is_created_from_documents(self):
        self.assertEqual(
            self.rag.glossary.get("devirank"),
            "Evidence-Weighted Drug Ranking",
        )

    def test_agent_answers_both_supported_parts(self):
        agent = EvidenceAgent(
            rag=self.rag,
            kg=StubKnowledgeGraph(),
            llm=StubLLM(),
            top_k=3,
            min_score=0.05,
        )

        result = agent.answer(
            "what does DEviRank? stand for and how is it used "
            "in the context of graph neural networks?"
        )

        self.assertIn(
            "DEviRank stands for Evidence-Weighted Drug "
            "Ranking",
            result["answer"],
        )
        self.assertIn(
            "not used as a graph neural network",
            result["answer"],
        )
        self.assertIn(
            "PPI-based proximity",
            result["answer"],
        )
        citation_ids = {
            citation["chunk_id"]
            for citation in result["citations"]
        }

        self.assertTrue(
            {
                "paper_0009",
                "paper_0003",
            }.issubset(citation_ids),
        )
        self.assertTrue(
            result["grounding_check"]["supported"]
        )


if __name__ == "__main__":
    unittest.main()
