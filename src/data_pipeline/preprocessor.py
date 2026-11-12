import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Cleans raw text strings: removes HTML tags, removes excess whitespace,
    standardizes quotes.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Replace weird unicode quotes
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    return text

def align_news_with_indicators(news_df: pd.DataFrame, stock_df_dict: dict) -> pd.DataFrame:
    """
    Aligns news items with stock technical indicators on the publication date.
    
    Parameters:
    - news_df: DataFrame with ['ticker', 'title', 'summary', 'published_at']
    - stock_df_dict: Dictionary mapping ticker -> DataFrame of stock data (with technical indicators)
    
    Returns:
    - Combined DataFrame containing news text, indicators, and future labels.
    """
    combined_rows = []
    
    # Parse dates in news
    news_df = news_df.copy()
    # Filter empty published dates
    news_df = news_df.dropna(subset=['published_at'])
    news_df['date'] = pd.to_datetime(news_df['published_at']).dt.date
    news_df = news_df[~pd.isna(news_df['date'])]
    
    for idx, row in news_df.iterrows():
        ticker = row['ticker']
        news_date = row['date']
        
        # Check if news_date is NaT or null
        if pd.isna(news_date):
            continue
            
        if ticker not in stock_df_dict:
            continue
            
        stock_df = stock_df_dict[ticker]
        
        # Find the stock price date closest to the news date (on or after, to avoid future leakage)
        # Convert index to date series
        stock_dates = pd.to_datetime(stock_df.index).date
        
        # Match news date
        match_idx = np.where(stock_dates == news_date)[0]
        
        if len(match_idx) == 0:
            # If no exact match (e.g. weekend news), search for the next trading day
            future_indices = np.where(stock_dates >= news_date)[0]
            if len(future_indices) > 0:
                idx_pos = future_indices[0]
            else:
                continue
        else:
            idx_pos = match_idx[0]
            
        # Get indicators row on trading day
        stock_row = stock_df.iloc[idx_pos]
        
        # Only take if label exists (not nan)
        if 'Trading_Label' in stock_row and not pd.isna(stock_row['Trading_Label']):
            # Clean texts
            title_clean = clean_text(row['title'])
            summary_clean = clean_text(row['summary'])
            
            combined_rows.append({
                'ticker': ticker,
                'news_date': str(news_date),
                'headline': title_clean,
                'summary': summary_clean,
                'price_close': float(stock_row['Close']),
                'rsi': float(stock_row['RSI']),
                'macd': float(stock_row['MACD']),
                'macd_signal': float(stock_row['MACD_Signal']),
                'ema_12': float(stock_row['EMA_12']),
                'ema_26': float(stock_row['EMA_26']),
                'sma_20': float(stock_row['SMA_20']),
                'bb_high': float(stock_row['BB_High']),
                'bb_low': float(stock_row['BB_Low']),
                'volume_change': float(stock_row['Volume_Change']) if not pd.isna(stock_row['Volume_Change']) else 0.0,
                'volatility': float(stock_row['Volatility']) if not pd.isna(stock_row['Volatility']) else 0.0,
                'trading_label': int(stock_row['Trading_Label']),
                'trading_signal': str(stock_row['Trading_Signal']),
                'confidence': float(stock_row['Label_Confidence'])
            })
            
    result_df = pd.DataFrame(combined_rows)
    logger.info(f"Successfully aligned {len(result_df)} news items with technical indicators.")
    return result_df

def format_instruction_prompt(row: Dict[str, Any]) -> str:
    """
    Formats the sample row into LLM training text instruction.
    """
    headline = row['headline']
    ticker = row['ticker']
    close = row['price_close']
    rsi = row['rsi']
    macd = row['macd']
    macd_signal = row['macd_signal']
    volume_change = row['volume_change'] * 100 # percentage
    volatility = row['volatility'] * 100 # percentage
    
    # Analyze state
    rsi_state = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral")
    macd_trend = "Bullish Crossover" if macd > macd_signal else "Bearish Trend"
    
    indicators_summary = (
        f"Stock Price: ${close:.2f}. "
        f"RSI(14): {rsi:.1f} ({rsi_state}). "
        f"MACD: {macd:.3f} vs Signal: {macd_signal:.3f} ({macd_trend}). "
        f"Volume Change: {volume_change:+.1f}%. "
        f"20-day Volatility: {volatility:.1f}%."
    )
    
    instruction = (
        f"### Instruction: Analyze the following financial news and historical technical indicators for {ticker}. "
        f"Determine the sentiment label (Bullish, Bearish, or Neutral), provide a trading signal (Buy, Sell, or Hold), "
        f"assign a confidence level (0-100%), and explain your reasoning.\n\n"
        f"### News Headline: {headline}\n"
        f"### Technical Indicators: {indicators_summary}\n\n"
        f"### Output:\n"
        f"Sentiment: {row['trading_signal'] if row['trading_signal'] != 'Hold' else 'Neutral'}\n"
        f"Confidence: {row['confidence']}%\n"
        f"Trading Signal: {row['trading_signal']}\n"
        f"Reason: News headline indicates dynamic shifting and indicators support {row['trading_signal'].lower()} signal."
    )
    return instruction

def split_and_create_datasets(df: pd.DataFrame, train_size=0.7, val_size=0.15) -> DatasetDict:
    """
    Splits the pandas DataFrame and converts it into Hugging Face DatasetDict.
    """
    # First split train/temp
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=42)
    
    # Split temp into val/test
    val_ratio_adjusted = val_size / (1.0 - train_size)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio_adjusted, random_state=42)
    
    # Format instruction texts for each
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['text'] = train_df.apply(format_instruction_prompt, axis=1)
    val_df['text'] = val_df.apply(format_instruction_prompt, axis=1)
    test_df['text'] = test_df.apply(format_instruction_prompt, axis=1)
    
    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Convert to Hugging Face Dataset
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'validation': Dataset.from_pandas(val_df.reset_index(drop=True)),
        'test': Dataset.from_pandas(test_df.reset_index(drop=True))
    })
    
    return dataset_dict
