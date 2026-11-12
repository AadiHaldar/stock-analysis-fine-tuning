import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_trading_labels(
    df: pd.DataFrame, 
    lookahead_days: int = 5, 
    threshold_pct: float = 0.02
) -> pd.DataFrame:
    """
    Generates Buy/Sell/Hold labels based on future price movements.
    
    Formula:
    Forward Return = (Close_{t + lookahead} - Close_t) / Close_t
    
    Labels:
    - 2 (BUY): Forward Return > threshold_pct
    - 0 (SELL): Forward Return < -threshold_pct
    - 1 (HOLD): -threshold_pct <= Forward Return <= threshold_pct
    
    Also computes the exact confidence target (regression label or scaled return)
    for model output comparison.
    """
    df = df.copy()

    # Calculate forward returns
    # shift(-lookahead_days) gets the Close value lookahead_days in the future
    df['Forward_Close'] = df['Close'].shift(-lookahead_days)
    df['Forward_Return'] = (df['Forward_Close'] - df['Close']) / df['Close']

    # Generate discrete signals
    # 2 = Buy, 1 = Hold, 0 = Sell
    df['Trading_Label'] = 1 # Default is Hold
    df.loc[df['Forward_Return'] > threshold_pct, 'Trading_Label'] = 2
    df.loc[df['Forward_Return'] < -threshold_pct, 'Trading_Label'] = 0

    # Map labels to text for instruction fine-tuning
    label_map = {0: "Sell", 1: "Hold", 2: "Buy"}
    df['Trading_Signal'] = df['Trading_Label'].map(label_map)

    # Calculate the label confidence based on the return magnitude relative to threshold
    # Maximum return is bounded, scaled to 0-100% confidence
    # If return is double the threshold, confidence is 100%
    # If return is at threshold, confidence is 50%
    # If return is 0, confidence is 10% (uncertain)
    scaled_conf = (df['Forward_Return'].abs() / (2 * threshold_pct)) * 100
    df['Label_Confidence'] = scaled_conf.clip(lower=10, upper=98).round(1)

    # Drop rows at the end of the dataframe that don't have lookahead data
    df = df.dropna(subset=['Forward_Close', 'Forward_Return', 'Trading_Label'])
    
    # Drop temp forward columns to keep clean
    df = df.drop(columns=['Forward_Close'])

    logger.info(f"Generated trading labels: BUY={len(df[df['Trading_Label'] == 2])}, "
                f"HOLD={len(df[df['Trading_Label'] == 1])}, "
                f"SELL={len(df[df['Trading_Label'] == 0])}")

    return df
