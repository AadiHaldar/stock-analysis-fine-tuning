import yfinance as yf
import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def fetch_ticker_news(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetches the latest news headlines for a ticker using yfinance.
    Formats it into standard dictionary items.
    """
    logger.info(f"Fetching news for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        news_list = t.news
        if not news_list:
            logger.warning(f"No news articles found for {ticker}")
            return []

        formatted_news = []
        for item in news_list:
            # yfinance news returns: uuid, title, publisher, link, providerPublishTime, type, relatedTickers
            pub_time = item.get("providerPublishTime", 0)
            date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M:%S') if pub_time else ""
            
            formatted_news.append({
                "ticker": ticker,
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published_at": date_str,
                "type": item.get("type", "")
            })
        
        logger.info(f"Fetched {len(formatted_news)} news items for {ticker}")
        return formatted_news

    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return []

def fetch_multiple_tickers_news(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Fetches latest news for multiple tickers.
    """
    all_news = []
    for ticker in tickers:
        all_news.extend(fetch_ticker_news(ticker))
    return all_news

if __name__ == "__main__":
    news = fetch_ticker_news("AAPL")
    if news:
        print(f"Sample news title: {news[0]['title']}")
