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
        st.write(f"**BLEU Score**: {metrics['BLEU']}")
        st.write("BLEU measures the overlap between the generated output and reference text based on n-grams. Higher scores indicate better match.")

        st.write(f"**ROUGE-1 Score**: {metrics['ROUGE-1']}")
        st.write("ROUGE-1 measures the overlap of unigrams between the generated output and reference text. Higher scores indicate better match.")

        st.write(f"**BERT Precision**: {metrics['BERT P']}")
        st.write(f"**BERT Recall**: {metrics['BERT R']}")
        st.write(f"**BERT F1 Score**: {metrics['BERT F1']}")
        st.write("BERTScore evaluates the semantic similarity between the generated output and reference text using BERT embeddings.")

        st.write(f"**Perplexity**: {metrics['Perplexity']}")
        st.write("Perplexity measures how well a language model predicts the text. Lower values indicate better fluency and coherence.")

        st.write(f"**Diversity**: {metrics['Diversity']}")
        st.write("Diversity measures the uniqueness of bigrams in the generated output. Higher values indicate more diverse and varied output.")

        st.write(f"**Racial Bias**: {metrics['Racial Bias']}")
        st.write("Racial Bias score indicates the presence of biased language in the generated output. Higher scores indicate more bias.")

    else:
        st.write("Please provide all inputs to evaluate.")
