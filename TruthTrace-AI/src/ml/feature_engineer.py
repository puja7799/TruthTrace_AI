import pandas as pd
import sys
import os
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_model import Message

from src.graph_model import PropagationGraph

from src.nlp.transformer_engine import AdvancedNLPEngine

class FeatureEngineer:
    """Transforms raw Message chains into an ML-ready numerical feature matrix."""
    
    def __init__(self):
        self.nlp = AdvancedNLPEngine() 
        self.graph = PropagationGraph()
        
    def extract_features(self, messages: List[Message]) -> pd.DataFrame:
        """
        Processes a conversation tree and extracts semantic + structural features.
        Returns a DataFrame where each row is a message.
        """
        self.graph.build_graph(messages)
        
        # Pre-compute root metrics for baseline comparison
        root_msg = messages[0]
        root_emb = self.nlp.compute_embedding(root_msg.text)
        root_sentiment = self.nlp.analyze_sentiment(root_msg.text)
        
        features_list = []
        
        for msg in messages:
            # 1. NLP Features
            emb = self.nlp.compute_embedding(msg.text)
            semantic_drift = self.nlp.calculate_drift(root_emb, emb)
            sentiment = self.nlp.analyze_sentiment(msg.text)
            sentiment_delta = abs(sentiment - root_sentiment)
            exaggeration = self.nlp.detect_exaggeration(msg.text)
            
            # 2. Graph/Structural Features
            depth = self.graph.get_node_depth(msg.id)
            
            # Calculate Information Mutation Rate (Drift per Hop)
            mutation_rate = semantic_drift / depth if depth > 0 else 0.0
            
            # 3. Assemble Feature Row
            features_list.append({
                "msg_id": msg.id,
                "text": msg.text,  # Kept for reference
                "semantic_drift": semantic_drift,
                "sentiment_delta": sentiment_delta,
                "exaggeration_score": exaggeration,
                "graph_depth": depth,
                "mutation_rate": mutation_rate
            })
            
        return pd.DataFrame(features_list)

# --- Quick Local Test ---
if __name__ == "__main__":
    from src.data_pipeline.dataset_parser import TwitterTreeParser
    
    print("Loading mock PHEME thread...")
    root, replies = TwitterTreeParser.get_pheme_mock_data()
    messages = TwitterTreeParser.parse_thread(root, replies)
    
    print("Extracting ML Features...")
    engineer = FeatureEngineer()
    df = engineer.extract_features(messages)
    
    print("\n✅ Extracted Feature Matrix (Ready for XGBoost):")
    
    print(df.drop(columns=['text']).to_markdown(index=False))