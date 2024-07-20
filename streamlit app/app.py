import streamlit as st
from evaluation_module import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator()

st.title("RAG System Evaluation Dashboard")

st.write("## Input Data")

# Pre-filled input fields for testing
question = st.text_input("Question", "What are the causes of climate change?")
context = st.text_area("Reference Context (top 'k' documents)", """
Climate change is caused by a variety of factors, including natural processes and human activities. Human activities, such as burning fossil fuels, deforestation, and industrial processes, release greenhouse gases into the atmosphere. These gases trap heat from the sun, causing the Earth's temperature to rise. Natural processes, such as volcanic eruptions and variations in solar radiation, also play a role in climate change.
""")
generated_output = st.text_area("LLM Generated Output", """
Climate change is primarily caused by human activities that release greenhouse gases into the atmosphere. These activities include burning fossil fuels for energy, deforestation, and various industrial processes. The increase in greenhouse gases, such as carbon dioxide and methane, traps more heat in the Earth's atmosphere, leading to a rise in global temperatures. Natural factors, like volcanic activity and changes in solar radiation, can also contribute to climate change, but their impact is relatively minor compared to human activities.
""")

if st.button("Evaluate"):
    if question and context and generated_output:
        st.write("### Evaluation Results")

        # Perform evaluations
        metrics = evaluator.evaluate_all(generated_output, context)

        # Display metrics with explanations
        st.write(f"**BLEU Score**: {metrics['BLEU']:.2f}")
        st.write("BLEU measures the overlap between the generated output and reference text based on n-grams. Range: 0-100. Higher scores indicate better match.")

        st.write(f"**ROUGE-1 Score**: {metrics['ROUGE-1']:.2f}")
        st.write("ROUGE-1 measures the overlap of unigrams between the generated output and reference text. Range: 0-1. Higher scores indicate better match.")

        st.write(f"**BERT Precision**: {metrics['BERT P']:.2f}")
        st.write(f"**BERT Recall**: {metrics['BERT R']:.2f}")
        st.write(f"**BERT F1 Score**: {metrics['BERT F1']:.2f}")
        st.write("BERTScore evaluates the semantic similarity between the generated output and reference text using BERT embeddings. Range: 0-1. Higher scores indicate better semantic similarity.")

        st.write(f"**Perplexity**: {metrics['Perplexity']:.2f}")
        st.write("Perplexity measures how well a language model predicts the text. Range: 1 to ∞. Lower values indicate better fluency and coherence.")

        st.write(f"**Diversity**: {metrics['Diversity']:.2f}")
        st.write("Diversity measures the uniqueness of bigrams in the generated output. Range: 0-1. Higher values indicate more diverse and varied output.")

        st.write(f"**Racial Bias**: {metrics['Racial Bias']:.2f}")
        st.write("Racial Bias score indicates the presence of biased language in the generated output. Range: 0-1. Lower scores indicate less bias.")

        st.write(f"**METEOR Score**: {metrics['METEOR']:.2f}")
        st.write("METEOR calculates semantic similarity considering synonyms and paraphrases. Range: 0-1. Higher scores indicate better semantic alignment.")

        st.write(f"**CHRF Score**: {metrics['CHRF']:.2f}")
        st.write("CHRF computes Character n-gram F-score for fine-grained text similarity. Range: 0-1. Higher scores indicate better character-level similarity.")

        st.write(f"**Flesch Reading Ease**: {metrics['Flesch Reading Ease']:.2f}")
        st.write("Flesch Reading Ease assesses text readability. Range: 0-100. Higher scores indicate easier readability.")

        st.write(f"**Flesch-Kincaid Grade**: {metrics['Flesch-Kincaid Grade']:.2f}")
        st.write("Flesch-Kincaid Grade indicates the U.S. school grade level needed to understand the text. Range: 0-18+. Lower scores indicate easier readability.")

    else:
        st.write("Please provide all inputs to evaluate.")