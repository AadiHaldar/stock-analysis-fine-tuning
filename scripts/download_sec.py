import sys
import os
import re
from pathlib import Path
from html.parser import HTMLParser
import logging

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from sec_edgar_downloader import Downloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
    def handle_data(self, d):
        self.text.append(d)
    def get_data(self):
        return "".join(self.text)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def clean_sec_text(text: str) -> str:
    # Remove multiple newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text

def download_and_process_filings(tickers: list, filing_type: str = "10-K", limit: int = 1):
    # SEC guidelines require a user agent string: "Company Name EmailAddress"
    # We use a standard compliant placeholder or read from environment variable
    user_agent_name = "StockAnalysisLocalAI"
    user_agent_email = "admin@stockanalysis.local"
    
    # We will download to a temp folder inside cache
    download_dir = settings.CACHE_DIR / "sec_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    dl = Downloader(user_agent_name, user_agent_email, download_dir)
    
    target_doc_dir = settings.RAW_DATA_DIR / "documents"
    target_doc_dir.mkdir(parents=True, exist_ok=True)
    
    for ticker in tickers:
        logger.info(f"Downloading {filing_type} for {ticker}...")
        try:
            dl.get(filing_type, ticker, limit=limit)
            
            # Find the downloaded file
            # sec-edgar-downloader saves it in:
            # {download_dir}/sec-edgar-filings/{ticker}/{filing_type}/**/*.html
            sec_filings_dir = download_dir / "sec-edgar-filings" / ticker / filing_type
            
            html_files = list(sec_filings_dir.glob("**/*.html"))
            txt_files = list(sec_filings_dir.glob("**/*.txt"))
            all_files = html_files + txt_files
            
            if not all_files:
                logger.warning(f"No downloaded files found for {ticker} in {sec_filings_dir}")
                continue
                
            # Process the latest file
            latest_file = max(all_files, key=os.path.getmtime)
            logger.info(f"Processing and cleaning downloaded file: {latest_file.name}")
            
            with open(latest_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            # Clean HTML to text
            if latest_file.suffix == ".html":
                text_content = strip_tags(content)
            else:
                text_content = content
                
            clean_text = clean_sec_text(text_content)
            
            # Save clean text to our documents directory
            output_file = target_doc_dir / f"{ticker}_sec_summary.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                # Add a brief header for the RAG system to identify
                f.write(f"{ticker} SEC Form {filing_type} Summary\n")
                # Cap at 100k characters to keep testing performant and avoid huge embeddings files
                f.write(clean_text[:100000])
                
            logger.info(f"Successfully saved cleaned SEC filing to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to download/process filing for {ticker}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download SEC filings and clean them for the RAG system.")
    parser.add_argument("--tickers", type=str, default="AAPL,TSLA", help="Comma-separated tickers (e.g. AAPL,TSLA)")
    parser.add_argument("--type", type=str, default="10-K", help="Filing type (e.g. 10-K, 10-Q)")
    parser.add_argument("--limit", type=int, default=1, help="Number of recent filings to download")
    args = parser.parse_args()
    
    tickers_list = [t.strip().upper() for t in args.tickers.split(",")]
    download_and_process_filings(tickers_list, args.type, args.limit)

if __name__ == "__main__":
    main()
