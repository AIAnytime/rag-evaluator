import torch
import mauve
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sklearn.metrics.pairwise import cosine_similarity
from mauve import compute_mauve
import nltk
nltk.download()

class RAGEvaluator:
    def __init__(self):
        self.gpt2_model, self.gpt2_tokenizer = self.load_gpt2_model()
        self.bias_pipeline = pipeline("zero-shot-classification", model="Hate-speech-CNERG/dehatebert-mono-english")

    def load_gpt2_model(self):
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return model, tokenizer

    def evaluate_bleu_rouge(self, candidates, references):
        bleu_score = corpus_bleu(candidates, [references]).score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
        rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        return bleu_score, rouge1

    def evaluate_bert_score(self, candidates, references):
        P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
        return P.mean().item(), R.mean().item(), F1.mean().item()

    def evaluate_perplexity(self, text):
        encodings = self.gpt2_tokenizer(text, return_tensors='pt')
        max_length = self.gpt2_model.config.n_positions
        stride = 512
        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len
            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()

    def evaluate_diversity(self, texts):
        all_tokens = [tok for text in texts for tok in text.split()]
        unique_bigrams = set(ngrams(all_tokens, 2))
        diversity_score = len(unique_bigrams) / len(all_tokens) if all_tokens else 0
        return diversity_score

    def evaluate_racial_bias(self, text):
        results = self.bias_pipeline([text], candidate_labels=["hate speech", "not hate speech"])
        bias_score = results[0]['scores'][results[0]['labels'].index('hate speech')]
        return bias_score

    def evaluate_meteor(self, candidates, references):
        nltk.download('punkt', quiet=True)  
        
        meteor_scores = [
            meteor_score([word_tokenize(ref)], word_tokenize(cand))
            for ref, cand in zip(references, candidates)
        ]
        return sum(meteor_scores) / len(meteor_scores)
    
    def evaluate_chrf(self, candidates, references):
        chrf_scores = [sentence_chrf(ref, cand) for ref, cand in zip(references, candidates)]
        return sum(chrf_scores) / len(chrf_scores)
    
    def evaluate_readability(self, text):
        flesch_ease = flesch_reading_ease(text)
        flesch_grade = flesch_kincaid_grade(text)
        return flesch_ease, flesch_grade
        
    def evaluate_mauve(self,reference_texts, generated_texts):
        out = mauve.compute_mauve(
                                  p_text=reference_texts,  # List of reference texts
                                  q_text=generated_texts,  # List of generated texts
                                  device_id=0,             # GPU device ID; set to -1 for CPU
                                  max_text_length=1024,     # Maximum length of text to truncate
                                  verbose=False            # Whether to print additional information
                                )
        return  out.mauve
        
    def evaluate_all(self, question, response, reference):
        candidates = [response]
        references = [reference]
        bleu, rouge1 = self.evaluate_bleu_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
        perplexity = self.evaluate_perplexity(response)
        diversity = self.evaluate_diversity(candidates)
        racial_bias = self.evaluate_racial_bias(response)
        mauve_score = self.evaluate_mauve(reference, response)
        meteor = self.evaluate_meteor(candidates, references)
        chrf = self.evaluate_chrf(candidates, references)
        flesch_ease, flesch_grade = self.evaluate_readability(response)
        return {
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "BERT P": bert_p,
            "BERT R": bert_r,
            "BERT F1": bert_f1,
            "Perplexity": perplexity,
            "Diversity": diversity,
            "Racial Bias": racial_bias,
            "MAUVE": mauve_score,
            "METEOR": meteor,
            "CHRF": chrf,
            "Flesch Reading Ease": flesch_ease,
            "Flesch-Kincaid Grade": flesch_grade,
        }
