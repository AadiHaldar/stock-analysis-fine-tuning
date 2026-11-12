import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.inference.sentiment_engine import SentimentEngine
from src.inference.analyst_engine import AnalystEngine
from src.models.memory_utils import clear_gpu_memory

logger = logging.getLogger(__name__)
console = Console()

class UnifiedStockAnalyst:
    """
    Unified API orchestrating FinBERT and Llama 3.2 3B.
    Performs fast sentiment analysis and deep context reasoning.
    """
    def __init__(self, load_bert: bool = True, load_llm: bool = True):
        self.sentiment_engine = None
        self.analyst_engine = None
        
        if load_bert:
            try:
                self.sentiment_engine = SentimentEngine()
            except Exception as e:
                logger.error(f"Failed to load FinBERT engine: {e}")
                
        if load_llm:
            try:
                self.analyst_engine = AnalystEngine()
            except Exception as e:
                logger.error(f"Failed to load Llama Analyst engine: {e}")
                
    def analyze_event(self, headline: str, ticker: str = "SPY", indicators: dict = None) -> dict:
        """
        Orchestrates sentiment and technical analysis.
        """
        result = {
            "ticker": ticker,
            "headline": headline,
            "sentiment": "Neutral",
            "confidence": "50%",
            "trading_signal": "Hold",
            "reasoning": "Model processing failed.",
            "probabilities": {}
        }
        
        # 1. Quick sentiment check via FinBERT
        if self.sentiment_engine:
            try:
                bert_res = self.sentiment_engine.predict(headline)
                result["sentiment"] = bert_res["sentiment"]
                result["confidence"] = f"{bert_res['confidence']}%"
                result["probabilities"] = bert_res["probabilities"]
            except Exception as e:
                logger.error(f"FinBERT prediction error: {e}")
                
        # 2. Detailed context generation via LLM
        if self.analyst_engine:
            try:
                llm_res = self.analyst_engine.analyze(headline, ticker, indicators)
                result["trading_signal"] = llm_res["trading_signal"]
                result["reasoning"] = llm_res["reasoning"]
                
                # If LLM confidence is high, combine them
                # (usually, we trust LLM for overall signal & reasoning, and FinBERT for pure headline sentiment)
                if not self.sentiment_engine:
                    result["sentiment"] = llm_res["sentiment"]
                    result["confidence"] = llm_res["confidence"]
            except Exception as e:
                logger.error(f"LLM analyst prediction error: {e}")
                
        return result

    def print_beautiful_report(self, report: dict):
        """
        Formats and displays the analysis report inside the terminal using rich.
        """
        grid = Table.grid(expand=True)
        grid.add_column(style="bold cyan", justify="right", width=20)
        grid.add_column(style="white", justify="left")
        
        grid.add_row("Ticker Symbol:", f"[bold yellow]{report['ticker']}[/bold yellow]")
        grid.add_row("Event/Headline:", f"[italic]\"{report['headline']}\"[/italic]")
        
        # Color mapping for sentiment
        sent = report['sentiment']
        sent_color = "green" if sent == "Bullish" else ("red" if sent == "Bearish" else "yellow")
        grid.add_row("Sentiment:", f"[{sent_color}]{sent} ({report['confidence']})[/{sent_color}]")
        
        # Color mapping for signal
        sig = report['trading_signal']
        sig_color = "bold green" if sig == "Buy" else ("bold red" if sig == "Sell" else "bold yellow")
        grid.add_row("Trading Signal:", f"[{sig_color}]{sig}[/{sig_color}]")
        
        grid.add_row("Reasoning:", report['reasoning'])
        
        if report['probabilities']:
            probs_str = ", ".join([f"{k}: {v}%" for k, v in report['probabilities'].items()])
            grid.add_row("Probabilities:", f"[dim]{probs_str}[/dim]")
            
        panel = Panel(
            grid,
            title="[bold white]Financial Stock Analysis Report[/bold white]",
            border_style="cyan",
            expand=False,
            padding=(1, 2)
        )
        
        console.print(panel)

if __name__ == "__main__":
    # Test unified client
    analyst = UnifiedStockAnalyst()
    report = analyst.analyze_event(
        headline="Apple reports record earnings and raises dividend payouts",
        ticker="AAPL",
        indicators={"close": 185.20, "rsi": 62.1, "macd": 2.1, "macd_signal": 1.5, "volume_change": 0.08, "volatility": 0.015}
    )
    analyst.print_beautiful_report(report)
