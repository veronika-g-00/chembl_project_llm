import unittest

from libs.inference import BACE1Predictor
from libs.llm_assistant import (
    SYSTEM_INSTRUCTIONS,
    answer_question,
    build_prediction_context,
)


class FakeClient:
    def __init__(self) -> None:
        self.model = None
        self.messages = None

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        self.model = model
        self.messages = messages
        return "Testowa odpowiedź LLM."


class LLMAssistantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = BACE1Predictor().predict("CCO")

    def test_context_contains_model_predictions(self) -> None:
        context = build_prediction_context(self.result)
        self.assertIn("GNN", context)
        self.assertIn("MLP", context)
        self.assertIn("CHEMBL4822", context)
        self.assertIn(self.result.standardized_smiles, context)

    def test_ollama_receives_instructions_and_context(self) -> None:
        client = FakeClient()
        answer = answer_question(
            "Jak interpretować wynik?",
            self.result,
            client=client,
            model="test-model",
        )
        self.assertEqual(answer, "Testowa odpowiedź LLM.")
        self.assertEqual(client.model, "test-model")
        self.assertEqual(
            client.messages[0],
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        )
        self.assertIn("GNN", client.messages[1]["content"])
        self.assertIn(
            "Jak interpretować wynik?",
            client.messages[1]["content"],
        )


if __name__ == "__main__":
    unittest.main()

