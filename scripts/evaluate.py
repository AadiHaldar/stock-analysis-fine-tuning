import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.inference.unified_api import UnifiedStockAnalyst
from src.models.memory_utils import clear_gpu_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_llm_signal(text: str) -> str:
    """
    Parses predicted signal from LLM raw generation string.
    """
    import re
    signal_match = re.search(r"Trading Signal:\s*(Buy|Sell|Hold)", text, re.IGNORECASE)
    if signal_match:
        return signal_match.group(1).capitalize()
    return "Hold"

def main():
    logger.info("Starting evaluation suite...")
    
    test_path = settings.PROCESSED_DATA_DIR / "llama_dataset_test.jsonl"
    
    if not test_path.exists():
        logger.error(f"Test dataset not found at {test_path}. "
                     "Please run 'python scripts/download_data.py' first to build datasets.")
        sys.exit(1)
        
    # 1. Load test data
    logger.info("Loading test datasets...")
    test_records = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_records.append(json.loads(line))
            
    test_df = pd.DataFrame(test_records)
    logger.info(f"Loaded {len(test_df)} test samples.")
    
    # 2. Load unified model
    logger.info("Loading models for evaluation...")
    try:
        analyst = UnifiedStockAnalyst()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)
        
    y_true_sentiment = []
    y_pred_sentiment = []
    
    y_true_signal = []
    y_pred_signal = []
    
    logger.info("Running batch evaluation...")
    # Cap test samples to 100 for fast validation, since running LLM on 500 samples takes a long time
    eval_subset = test_df.sample(min(len(test_df), 100), random_state=42)
    
    for idx, row in eval_subset.iterrows():
        headline = row['headline']
        ticker = row['ticker']
        
        # Assemble indicators dict
        indicators = {
            "close": row['price_close'],
            "rsi": row['rsi'],
            "macd": row['macd'],
            "macd_signal": row['macd_signal'],
            "volume_change": row['volume_change'],
            "volatility": row['volatility']
        }
        
        # Ground truths
        true_signal = row['trading_signal'] # Buy, Sell, Hold
        # Neutral maps to Hold in Sentiment
        true_sentiment = "Neutral" if true_signal == "Hold" else ("Bullish" if true_signal == "Buy" else "Bearish")
        
        # Run inference
        try:
            report = analyst.analyze_event(headline, ticker, indicators)
            
            y_true_sentiment.append(true_sentiment)
            y_pred_sentiment.append(report["sentiment"])
            
            y_true_signal.append(true_signal)
            y_pred_signal.append(report["trading_signal"])
            
        except Exception as e:
            logger.error(f"Error predicting sample {idx}: {e}")
            
    # 3. Compute Metrics
    logger.info("Computing metrics...")
    
    sentiment_report = classification_report(y_true_sentiment, y_pred_sentiment, output_dict=True, zero_division=0)
    signal_report = classification_report(y_true_signal, y_pred_signal, output_dict=True, zero_division=0)
    
    # Save reports to outputs/results/
    with open(settings.RESULTS_DIR / "sentiment_metrics.json", "w") as f:
        json.dump(sentiment_report, f, indent=4)
        
    with open(settings.RESULTS_DIR / "trading_signal_metrics.json", "w") as f:
        json.dump(signal_report, f, indent=4)
        
    # Generate Confusion Matrix Plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sentiment matrix
    cm_sent = confusion_matrix(y_true_sentiment, y_pred_sentiment, labels=["Bullish", "Neutral", "Bearish"])
    sns.heatmap(cm_sent, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Bullish", "Neutral", "Bearish"], 
                yticklabels=["Bullish", "Neutral", "Bearish"], ax=axes[0])
    axes[0].set_title("Sentiment Classification Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")
    
    # Trading Signal matrix
    cm_sig = confusion_matrix(y_true_signal, y_pred_signal, labels=["Buy", "Hold", "Sell"])
    sns.heatmap(cm_sig, annot=True, fmt='d', cmap='Greens', 
                xticklabels=["Buy", "Hold", "Sell"], 
                yticklabels=["Buy", "Hold", "Sell"], ax=axes[1])
    axes[1].set_title("Trading Signal Classification Confusion Matrix")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")
    
    plt.tight_layout()
    plot_path = settings.RESULTS_DIR / "evaluation_confusion_matrices.png"
    plt.savefig(plot_path)
    logger.info(f"Confusion matrix plot saved to {plot_path}")
    
    # Clean up GPU
    clear_gpu_memory()
    
    # Print results to stdout
    print("\n" + "=" * 50)
    print("         EVALUATION REPORT SUMMARY")
    print("=" * 50)
    print("\n--- Sentiment Metrics ---")
    print(f"Accuracy: {sentiment_report['accuracy']:.4f}")
    print(f"Macro F1: {sentiment_report['macro avg']['f1-score']:.4f}")
    
    print("\n--- Trading Signal Metrics ---")
    print(f"Accuracy: {signal_report['accuracy']:.4f}")
    print(f"Macro F1: {signal_report['macro avg']['f1-score']:.4f}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
