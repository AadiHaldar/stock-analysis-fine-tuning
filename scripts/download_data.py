import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.data_pipeline.stock_data import download_multiple_tickers
from src.data_pipeline.technical_indicators import add_technical_indicators
from src.data_pipeline.label_generator import generate_trading_labels
from src.data_pipeline.sentiment_data import load_financial_phrasebank
from src.data_pipeline.news_data import fetch_multiple_tickers_news
from src.data_pipeline.preprocessor import align_news_with_indicators, split_and_create_datasets

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_synthetic_analyst_data(phrasebank_df: pd.DataFrame) -> pd.DataFrame:
    """
    To supplement the yfinance real-time news (which only yields ~80 items),
    this function merges the Financial PhraseBank dataset with realistic simulated
    technical indicators to create a rich 2,000+ sample instruction-tuning dataset.
    """
    logger.info("Generating synthetic analyst training dataset from Financial PhraseBank...")
    synthetic_rows = []
    
    # Common financial tickers to assign randomly
    tickers = settings.DEFAULT_TICKERS
    
    np.random.seed(42)
    
    for idx, row in phrasebank_df.iterrows():
        sentiment = row['sentiment_label'] # Bullish, Bearish, Neutral
        sentence = row['sentence']
        ticker = np.random.choice(tickers)
        
        # Simulate realistic indicator values that align with the sentiment
        if sentiment == "Bullish":
            rsi = np.random.uniform(45.0, 78.0)
            macd = np.random.uniform(0.1, 2.5)
            macd_signal = macd - np.random.uniform(0.05, 0.5) # MACD above signal
            price = np.random.uniform(50.0, 400.0)
            volume_change = np.random.uniform(-0.02, 0.25)
            volatility = np.random.uniform(0.015, 0.045)
            trading_signal = "Buy"
            confidence = np.random.uniform(70.0, 95.0)
        elif sentiment == "Bearish":
            rsi = np.random.uniform(15.0, 55.0)
            macd = np.random.uniform(-2.5, -0.1)
            macd_signal = macd + np.random.uniform(0.05, 0.5) # MACD below signal
            price = np.random.uniform(50.0, 400.0)
            volume_change = np.random.uniform(-0.15, 0.10)
            volatility = np.random.uniform(0.025, 0.060)
            trading_signal = "Sell"
            confidence = np.random.uniform(70.0, 95.0)
        else: # Neutral
            rsi = np.random.uniform(35.0, 65.0)
            macd = np.random.uniform(-0.5, 0.5)
            macd_signal = macd + np.random.uniform(-0.2, 0.2)
            price = np.random.uniform(50.0, 400.0)
            volume_change = np.random.uniform(-0.08, 0.08)
            volatility = np.random.uniform(0.010, 0.030)
            trading_signal = "Hold"
            confidence = np.random.uniform(40.0, 75.0)
            
        synthetic_rows.append({
            'ticker': ticker,
            'news_date': "2026-05-24",
            'headline': sentence,
            'summary': sentence,
            'price_close': price,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'ema_12': price * 1.01,
            'ema_26': price,
            'sma_20': price * 0.99,
            'bb_high': price * 1.05,
            'bb_low': price * 0.95,
            'volume_change': volume_change,
            'volatility': volatility,
            'trading_label': 2 if trading_signal == "Buy" else (0 if trading_signal == "Sell" else 1),
            'trading_signal': trading_signal,
            'confidence': round(confidence, 1)
        })
        
    syn_df = pd.DataFrame(synthetic_rows)
    logger.info(f"Generated {len(syn_df)} synthetic analyst training samples.")
    return syn_df

def main():
    logger.info("Starting master data pipeline download...")
    
    # 1. Download stock prices
    logger.info("Downloading historical price data...")
    raw_stocks = download_multiple_tickers()
    
    # 2. Process stock prices (indicators & signals)
    processed_stocks = {}
    for ticker, df in raw_stocks.items():
        logger.info(f"Processing indicators for {ticker}...")
        df_indicators = add_technical_indicators(df)
        df_labeled = generate_trading_labels(df_indicators)
        processed_stocks[ticker] = df_labeled
        
        # Save processed stock data
        df_labeled.to_csv(settings.PROCESSED_DATA_DIR / f"{ticker}_processed.csv")
    
    # 3. Download news headlines
    logger.info("Fetching real-time news headlines...")
    raw_news = fetch_multiple_tickers_news(settings.DEFAULT_TICKERS)
    news_df = pd.DataFrame(raw_news)
    
    # Save raw news
    news_df.to_csv(settings.RAW_DATA_DIR / "news_raw.csv", index=False)
    
    # 4. Load Financial PhraseBank for FinBERT
    phrasebank_df = load_financial_phrasebank("default")
    phrasebank_df.to_csv(settings.PROCESSED_DATA_DIR / "phrasebank_processed.csv", index=False)
    
    # 5. Build combined dataset for LLM training
    # Match real-time news with technical indicators
    aligned_news_df = pd.DataFrame()
    if not news_df.empty:
        try:
            aligned_news_df = align_news_with_indicators(news_df, processed_stocks)
        except Exception as e:
            logger.error(f"Failed to align news with indicators: {e}")
            
    # Create the augmented training dataset
    synthetic_analyst_df = generate_synthetic_analyst_data(phrasebank_df)
    
    # Combine real-time news with synthetic news
    if not aligned_news_df.empty:
        final_analyst_df = pd.concat([aligned_news_df, synthetic_analyst_df], ignore_index=True)
    else:
        final_analyst_df = synthetic_analyst_df
        
    # Save combined dataset
    final_analyst_df.to_csv(settings.PROCESSED_DATA_DIR / "aligned_market_news.csv", index=False)
    logger.info(f"Total alignment dataset built: {len(final_analyst_df)} samples.")
    
    # 6. Format and split Hugging Face datasets
    logger.info("Formatting datasets for LLM instruction fine-tuning...")
    dataset_dict = split_and_create_datasets(final_analyst_df)
    
    # Save split datasets as JSON lines for easy local loading
    for split_name, dataset in dataset_dict.items():
        output_path = settings.PROCESSED_DATA_DIR / f"llama_dataset_{split_name}.jsonl"
        dataset.to_json(output_path, force_ascii=False)
        logger.info(f"Saved {split_name} split to {output_path}")
        
    logger.info("Data pipeline execution finished successfully!")

if __name__ == "__main__":
    main()
