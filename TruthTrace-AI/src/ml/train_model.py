import torch
import os
import sys
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.dataset_parser import TwitterTreeParser
from src.ml.feature_engineer import FeatureEngineer

class MisinformationClassifier:
    """Trains and manages the XGBoost model for rumor detection."""
    
    def __init__(self):
        # Initialize XGBoost with balanced hyperparameters
        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        )
        # Define the exact features the model expects (to prevent schema errors later)
        self.feature_cols = [
            'semantic_drift', 
            'sentiment_delta', 
            'exaggeration_score', 
            'graph_depth', 
            'mutation_rate'
        ]

    def train(self, df: pd.DataFrame, target_col: str = 'is_misinformation'):
        """Trains the model and prints evaluation metrics."""
        print("🔧 Preparing data for training...")
        X = df[self.feature_cols]
        y = df[target_col]
        
        # 80/20 Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"🧠 Training XGBoost model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        print("📊 Evaluating model on test set...")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")
        
    def save_model(self, filepath: str = 'data/models/xgb_model.pkl'):
        """Saves the trained model artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"💾 Model saved successfully to {filepath}")

# --- Offline Training Pipeline Execution ---
if __name__ == "__main__":
    print("🚀 Starting Offline ML Training Pipeline...")
    
    # 1. Generate a larger batch of synthetic/mock data for training
    # In a real scenario, you'd loop through thousands of PHEME json files here.
    root, replies = TwitterTreeParser.get_pheme_mock_data()
    messages = TwitterTreeParser.parse_thread(root, replies)
    
    # We duplicate and mutate the data slightly to simulate a dataset of 500 messages
    print("Ingesting and expanding dataset...")
    all_messages = messages * 100 
    
    # 2. Extract Features
    engineer = FeatureEngineer()
    df = engineer.extract_features(all_messages)
    
    # 3. Create Target Variable (y)
    # Mocking the ground truth: High drift + deep propagation = likely misinformation
    np.random.seed(42)
    # A simple heuristic to generate synthetic labels for our mock data
    df['is_misinformation'] = np.where(
        (df['semantic_drift'] > 0.3) | (df['graph_depth'] >= 2) | (df['exaggeration_score'] > 0.2),
        1, 0
    )
    # Add some noise to make the ML model actually have to "learn"
    noise = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    df['is_misinformation'] = np.abs(df['is_misinformation'] - noise) 
    
    # 4. Train and Save
    classifier = MisinformationClassifier()
    classifier.train(df)
    
    # Save to the data/models directory
    classifier.save_model(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/models/xgb_model.pkl')))