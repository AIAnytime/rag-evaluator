import unittest
from evaluator import RAGEvaluator
import nltk
nltk.download('punkt')
from nltk.tokenize import PunktTokenizer

class TestRAGEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = RAGEvaluator()

    def test_evaluate_all(self):
        question = "What are the causes of climate change?"
        response = "Climate change is caused by human activities."
        reference = "Human activities such as burning fossil fuels cause climate change."
        metrics = self.evaluator.evaluate_all(question, response, reference)
        self.assertIsInstance(metrics, dict)
        self.assertIn("BLEU", metrics)
        self.assertIn("ROUGE-1", metrics)
        self.assertIn("BERT P", metrics)
        self.assertIn("Perplexity", metrics)
        self.assertIn("Diversity", metrics)
        self.assertIn("Racial Bias", metrics)
        self.assertIn("MAUVE", metrics)
        self.assertIn("METEOR", metrics)
        self.assertIn("CHRF", metrics)
        self.assertIn("Flesch Reading Ease", metrics)
        self.assertIn("Flesch-Kincaid Grade", metrics)

if __name__ == "__main__":
    unittest.main()
