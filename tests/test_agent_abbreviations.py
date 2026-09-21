import json
import unittest

from src.agent import (
    EvidenceAgent,
)
from src.config import Settings
from src.ingest import (
    chunks_to_records,
    ingest_corpus,
)
from src.rag import TfidfRAG


class StubLLM:
    """Return valid planning and graph JSON for agent tests."""

    mode = "test"

    def generate(self, prompt, json_schema=None):
        if '"rewritten_queries"' in prompt:
            return json.dumps(
                {
                    "rewritten_queries": [
                        "Primary Sclerosing Cholangitis"
                    ],
                    "entities_of_interest": [
                        "Primary Sclerosing Cholangitis"
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

        raise AssertionError(
            "A direct definition should not require "
            "LLM answer generation or verification."
        )


class StubKnowledgeGraph:
    def add_entity(self, *args, **kwargs):
        return None

    def add_relation(self, *args, **kwargs):
        return None


class TestAbbreviationHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = Settings()
        chunks = ingest_corpus(
            corpus_dir=settings.corpus_dir,
            chunk_size_chars=settings.chunk_size_chars,
            chunk_overlap_chars=(
                settings.chunk_overlap_chars
            ),
        )
        cls.rag = TfidfRAG(
            chunks_to_records(chunks)
        )

    def make_agent(self):
        return EvidenceAgent(
            rag=self.rag,
            kg=StubKnowledgeGraph(),
            llm=StubLLM(),
            top_k=6,
            min_score=0.05,
        )

    def test_psc_is_discovered_from_the_corpus(self):
        self.assertEqual(
            self.rag.glossary.get("psc"),
            "Primary Sclerosing Cholangitis",
        )

    def test_psc_query_is_expanded_from_the_corpus(self):
        expanded = self.rag.expand_query(
            "What is PSC?"
        )

        self.assertIn(
            "Primary Sclerosing Cholangitis",
            expanded,
        )
        self.assertIn(
            "What is PSC?",
            expanded,
        )

    def test_short_question_returns_the_definition(self):
        result = self.make_agent().answer(
            "What is PSC?"
        )

        answer = result["answer"].casefold()

        self.assertIn(
            "chronic fibro-inflammatory disease",
            answer,
        )
        self.assertNotIn(
            "tenecteplase",
            answer,
        )
        self.assertEqual(
            result["question"],
            "What is PSC?",
        )
        self.assertEqual(
            len(result["citations"]),
            1,
        )
        self.assertIn(
            "Universal Overview of PSC.md",
            result["citations"][0]["doc_path"],
        )


if __name__ == "__main__":
    unittest.main()
