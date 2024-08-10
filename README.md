# RAG Evaluator

## Overview

RAG Evaluator is a Python library for evaluating Retrieval-Augmented Generation (RAG) systems. It provides various metrics to evaluate the quality of generated text against reference text.

## Installation

You can install the library using pip:

```bash
pip install rag-evaluator
```

## Usage

Here's how to use the RAG Evaluator library:

```python
from rag_evaluator import RAGEvaluator

# Initialize the evaluator
evaluator = RAGEvaluator()

# Input data
question = "What are the causes of climate change?"
response = "Climate change is caused by human activities."
reference = "Human activities such as burning fossil fuels cause climate change."

# Evaluate the response
metrics = evaluator.evaluate_all(question, response, reference)

# Print the results
print(metrics)
```

## Streamlit Web App

To run the web app:

1. cd into streamlit app folder.
2. Create a virtual env
3. Activate the virtual env
4. Install all dependencies
5. Run the app:
```
streamlit run app.py
```

## Metrics

The RAG Evaluator provides the following metrics:

1. **BLEU** (0-100): Measures the overlap between the generated output and reference text based on n-grams.
   - 0-20: Low similarity, 20-40: Medium-low, 40-60: Medium, 60-80: High, 80-100: Very high

2. **ROUGE-1** (0-1): Measures the overlap of unigrams between the generated output and reference text.
   - 0.0-0.2: Poor overlap, 0.2-0.4: Fair, 0.4-0.6: Good, 0.6-0.8: Very good, 0.8-1.0: Excellent

3. **BERT Score** (0-1): Evaluates the semantic similarity using BERT embeddings (Precision, Recall, F1).
   - 0.0-0.5: Low similarity, 0.5-0.7: Moderate, 0.7-0.8: Good, 0.8-0.9: High, 0.9-1.0: Very high

4. **Perplexity** (1 to ∞, lower is better): Measures how well a language model predicts the text.
   - 1-10: Excellent, 10-50: Good, 50-100: Moderate, 100+: High (potentially nonsensical)

5. **Diversity** (0-1): Measures the uniqueness of bigrams in the generated output.
   - 0.0-0.2: Very low, 0.2-0.4: Low, 0.4-0.6: Moderate, 0.6-0.8: High, 0.8-1.0: Very high

6. **Racial Bias** (0-1): Detects the presence of biased language in the generated output.
   - 0.0-0.2: Low probability, 0.2-0.4: Moderate, 0.4-0.6: High, 0.6-0.8: Very high, 0.8-1.0: Extreme

7. **MOUVE** (0-1): MAUVE captures contextual meaning, coherence, and fluency while measuring both semantic similarity and stylistic alignment .
   -  0.0-0.2 (Poor), 0.2-0.4 (Fair), 0.4-0.6 (Good), 0.6-0.8 (Very good), 0.8-1.0 (Excellent).
     
8. **METEOR** (0-1): Calculates semantic similarity considering synonyms and paraphrases.
   - 0.0-0.2: Poor, 0.2-0.4: Fair, 0.4-0.6: Good, 0.6-0.8: Very good, 0.8-1.0: Excellent

9. **CHRF** (0-1): Computes Character n-gram F-score for fine-grained text similarity.
   - 0.0-0.2: Low, 0.2-0.4: Moderate, 0.4-0.6: Good, 0.6-0.8: High, 0.8-1.0: Very high

10. **Flesch Reading Ease** (0-100): Assesses text readability.
   - 0-30: Very difficult, 30-50: Difficult, 50-60: Fairly difficult, 60-70: Standard, 70-80: Fairly easy, 80-90: Easy, 90-100: Very easy

11. **Flesch-Kincaid Grade** (0-18+): Indicates the U.S. school grade level needed to understand the text.
    - 1-6: Elementary, 7-8: Middle school, 9-12: High school, 13+: College level

## Testing

To run the tests, use the following command:

```
python -m unittest discover -s rag_evaluator -p "test_*.py"
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have any improvements, suggestions, or bug fixes, feel free to create a pull request (PR) or open an issue on GitHub. Please ensure your contributions adhere to the project's coding standards and include appropriate tests.

### How to Contribute

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Run tests to ensure everything is working.
5. Commit your changes and push to your fork.
6. Create a pull request (PR) with a detailed description of your changes.

## Contact

If you have any questions or need further assistance, feel free to reach out via [email](mailto:aianytime07@gmail.com).
