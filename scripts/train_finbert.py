import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.data_pipeline.sentiment_data import load_financial_phrasebank
from src.models.finbert_trainer import train_finbert

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing FinBERT training script...")
    
    # 1. Load phrasebank dataset
    processed_path = settings.PROCESSED_DATA_DIR / "phrasebank_processed.csv"
    
    if processed_path.exists():
        logger.info(f"Loading processed sentiment dataset from {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        logger.info("Processed dataset not found. Downloading fresh from HuggingFace...")
        df = load_financial_phrasebank()
        df.to_csv(processed_path, index=False)
        
    # 2. Split dataset into train and validation (85% train, 15% validation)
    logger.info("Splitting dataset into train/validation sets...")
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    
    logger.info(f"Train size: {len(train_df)} | Validation size: {len(val_df)}")
    
    # 3. Trigger training loop
    eval_results = train_finbert(train_df, val_df)
    
    # Print final evaluation metrics
    logger.info("=== FinBERT Evaluation Results ===")
    for k, v in eval_results.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    main()
