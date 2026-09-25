import unittest

from src.verify import (
    is_valid_json,
    json_validity_score,
    parse_json_safely,
)


class TestJsonVerification(unittest.TestCase):
    def test_parses_plain_json(self):
        self.assertEqual(
            parse_json_safely('{"supported": true}'),
            {"supported": True},
        )

    def test_recovers_fenced_json(self):
        text = '```json\n{"supported": false}\n```'
        self.assertEqual(
            parse_json_safely(text),
            {"supported": False},
        )

    def test_recovers_json_after_explanatory_text(self):
        text = 'Result:\n{"supported": true, "notes": "ok"}'
        self.assertEqual(
            parse_json_safely(text),
            {"supported": True, "notes": "ok"},
        )

    def test_invalid_text_returns_none(self):
        self.assertIsNone(
            parse_json_safely("not json")
        )

    def test_validity_score(self):
        values = [
            '{"a": 1}',
            '```json\n{"b": 2}\n```',
            "invalid",
        ]
        self.assertAlmostEqual(
            json_validity_score(values),
            2 / 3,
        )

    def test_empty_input_is_not_valid(self):
        self.assertFalse(is_valid_json(""))


if __name__ == "__main__":
    unittest.main()
