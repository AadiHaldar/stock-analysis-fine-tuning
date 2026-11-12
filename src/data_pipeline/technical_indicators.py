import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
import logging

logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators for the input stock DataFrame:
    - RSI (14)
    - MACD (12, 26, 9)
    - EMA (12, 26, 50)
    - SMA (20, 50, 200)
    - Bollinger Bands (20, 2 stddev)
    - Volume change (%)
    - Volatility (rolling standard deviation of log returns)
    """
    # Create copy to avoid modifying original df in-place or returning a slice view
    df = df.copy()

    # Ensure Close, Volume, High, Low are 1D arrays/Series (if they have MultiIndex or are floats)
    # Check if df columns are multi-indexed (should not be, but double-check)
    close_col = df['Close']
    volume_col = df['Volume']
    high_col = df['High']
    low_col = df['Low']

    # 1. Momentum: Relative Strength Index (RSI)
    try:
        rsi_ind = RSIIndicator(close=close_col, window=14)
        df['RSI'] = rsi_ind.rsi()
    except Exception as e:
        logger.warning(f"Error computing RSI: {e}")
        df['RSI'] = np.nan

    # 2. Trend: Moving Average Convergence Divergence (MACD)
    try:
        macd_ind = MACD(close=close_col, window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd_ind.macd()
        df['MACD_Signal'] = macd_ind.macd_signal()
        df['MACD_Diff'] = macd_ind.macd_diff()
    except Exception as e:
        logger.warning(f"Error computing MACD: {e}")
        df['MACD'] = df['MACD_Signal'] = df['MACD_Diff'] = np.nan

    # 3. Trend: Exponential Moving Averages (EMA)
    try:
        df['EMA_12'] = EMAIndicator(close=close_col, window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(close=close_col, window=26).ema_indicator()
        df['EMA_50'] = EMAIndicator(close=close_col, window=50).ema_indicator()
    except Exception as e:
        logger.warning(f"Error computing EMA: {e}")
        df['EMA_12'] = df['EMA_26'] = df['EMA_50'] = np.nan

    # 4. Trend: Simple Moving Averages (SMA)
    try:
        df['SMA_20'] = SMAIndicator(close=close_col, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=close_col, window=50).sma_indicator()
        df['SMA_200'] = SMAIndicator(close=close_col, window=200).sma_indicator()
    except Exception as e:
        logger.warning(f"Error computing SMA: {e}")
        df['SMA_20'] = df['SMA_50'] = df['SMA_200'] = np.nan

    # 5. Volatility: Bollinger Bands
    try:
        bb = BollingerBands(close=close_col, window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['BB_Low'] = bb.bollinger_lband()
    except Exception as e:
        logger.warning(f"Error computing Bollinger Bands: {e}")
        df['BB_High'] = df['BB_Mid'] = df['BB_Low'] = np.nan

    # 6. Volume changes
    try:
        # Avoid division by zero by adding a tiny epsilon or replacing zero volume
        volume_safe = volume_col.replace(0, np.nan)
        df['Volume_Change'] = volume_safe.pct_change(fill_method=None)
    except Exception as e:
        logger.warning(f"Error computing Volume Change: {e}")
        df['Volume_Change'] = np.nan

    # 7. Volatility (rolling 20-day standard deviation of percent returns)
    try:
        pct_returns = close_col.pct_change(fill_method=None)
        df['Volatility'] = pct_returns.rolling(window=20).std()
    except Exception as e:
        logger.warning(f"Error computing Volatility: {e}")
        df['Volatility'] = np.nan

    # Drop early rows with NaN values due to lag/moving windows
    df = df.dropna()

    return df
