"""Script for making predictions with a trained spam classifier pipeline."""
import argparse
import sys
from pathlib import Path
from typing import List, Union

import joblib
import pandas as pd

from src.preprocess import preprocess_text


def load_pipeline(model_path: Path):
    """Load trained pipeline from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_messages(pipeline, messages: Union[str, List[str]]) -> pd.DataFrame:
    """Predict spam/ham for one or more messages."""
    if isinstance(messages, str):
        messages = [messages]
    
    # Preprocess messages
    messages = [preprocess_text(msg) for msg in messages]
    
    # Get predictions and probabilities
    y_pred = pipeline.predict(messages)
    y_prob = pipeline.predict_proba(messages)
    
    # Create results dataframe
    results = pd.DataFrame({
        'message': messages,
        'prediction': y_pred,
        'spam_probability': y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob
    })
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict spam/ham for text messages.")
    parser.add_argument('--model', type=str, default='models/final_model.pkl',
                       help='Path to trained model pipeline')
    parser.add_argument('--message', type=str,
                       help='Single message to classify')
    parser.add_argument('--input-file', type=str,
                       help='File with messages to classify (one per line)')
    parser.add_argument('--output-file', type=str,
                       help='Save predictions to CSV file')
    args = parser.parse_args()
    
    if not args.message and not args.input_file:
        parser.error("Must provide either --message or --input-file")
    
    # Load model
    model_path = Path(args.model)
    pipeline = load_pipeline(model_path)
    
    # Get messages to predict
    if args.input_file:
        with open(args.input_file) as f:
            messages = [line.strip() for line in f if line.strip()]
    else:
        messages = [args.message]
    
    # Make predictions
    results = predict_messages(pipeline, messages)
    
    # Output results
    if args.output_file:
        results.to_csv(args.output_file, index=False)
        print(f"Saved predictions to {args.output_file}")
    else:
        print("\nPrediction Results:")
        print("-" * 80)
        for _, row in results.iterrows():
            print(f"Message: {row['message']}")
            print(f"Prediction: {row['prediction']} (spam prob: {row['spam_probability']:.4f})")
            print("-" * 80)


if __name__ == '__main__':
    main()