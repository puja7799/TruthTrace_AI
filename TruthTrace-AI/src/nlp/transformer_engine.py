import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedNLPEngine:
    def __init__(self):
        print("Loading Transformer Models (This takes a moment)...")
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  
        
       
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True
        )
        
        
        zero_shot_tokenizer = AutoTokenizer.from_pretrained(
            "cross-encoder/nli-distilroberta-base",
            use_fast=False
        )

        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-distilroberta-base",
            tokenizer=zero_shot_tokenizer
        )
        
    def compute_embedding(self, text: str) -> np.ndarray:
        return self.embedder.encode([text])[0]
        
    def calculate_drift(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return float((1 - sim) / 2)

    def analyze_sentiment(self, text: str) -> float:
        """Returns -1 for extreme negative (panic/fear), 1 for positive."""
        result = self.sentiment_pipe(text)[0]
        score = result['score']
        return -score if result['label'] == 'NEGATIVE' else score

    def detect_exaggeration(self, text: str) -> float:
        """Uses LLM zero-shot to detect panic, exaggeration, or sensationalism."""
        labels = ["factual reporting", "exaggerated panic", "conspiracy", "urgent warning"]
        
        
        result = self.zero_shot(text, candidate_labels=labels, multi_label=True)
        
        scores = dict(zip(result['labels'], result['scores']))
        
        
        risk_score = max(
            scores.get("exaggerated panic", 0), 
            scores.get("conspiracy", 0), 
            scores.get("urgent warning", 0)
        )
        
        return float(risk_score)