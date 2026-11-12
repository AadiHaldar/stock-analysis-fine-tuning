import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Optional
import sys

# Add root folder to sys.path to allow imports from config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_ticker_data(
    ticker: str,
    start_date: str = settings.DEFAULT_START_DATE,
    end_date: str = settings.DEFAULT_END_DATE,
    use_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    Downloads historical stock OHLCV data for a given ticker from Yahoo Finance.
    Saves the raw downloaded data as a CSV cache.
    """
    cache_path = settings.RAW_DATA_DIR / f"{ticker}_raw.csv"

    if use_cache and cache_path.exists():
        logger.info(f"Loading cached stock data for {ticker} from {cache_path}")
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Failed to read cache for {ticker}: {e}. Downloading fresh data...")

    logger.info(f"Downloading historical stock data for {ticker} ({start_date} to {end_date})...")
    try:
        # Download data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            logger.warning(f"No data returned for ticker {ticker}")
            return None

        # Clean column names (sometimes MultiIndex if downloaded multiple stocks)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Save to cache
        df.to_csv(cache_path)
        logger.info(f"Successfully downloaded and cached {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        return None

def download_multiple_tickers(
    tickers: List[str] = settings.DEFAULT_TICKERS,
    start_date: str = settings.DEFAULT_START_DATE,
    end_date: str = settings.DEFAULT_END_DATE,
    use_cache: bool = True
) -> dict:
    """
    Downloads historical data for multiple tickers.
    Returns a dictionary of DataFrames.
    """
    results = {}
    for ticker in tickers:
        df = download_ticker_data(ticker, start_date, end_date, use_cache)
        if df is not None:
            results[ticker] = df
    return results

if __name__ == "__main__":
    # Quick test run
    df = download_ticker_data("AAPL", start_date="2025-01-01", end_date="2025-12-31")
    if df is not None:
        print(f"Sample AAPL Head:")
        print(df.head())
