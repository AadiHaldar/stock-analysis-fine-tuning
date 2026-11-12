import sys
from pathlib import Path
import logging

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.unified_api import UnifiedStockAnalyst
from src.models.memory_utils import clear_gpu_memory

logging.basicConfig(level=logging.WARNING) # Suppress warnings for clean stdout
logger = logging.getLogger(__name__)

def main():
    print("Loading local inference engines... (Please wait)")
    
    # Initialize the unified analyst (loads both models on the GPU)
    try:
        analyst = UnifiedStockAnalyst()
    except Exception as e:
        print(f"Failed to load models: {e}")
        return
        
    print("\nLoading complete. Running standard inference examples:")
    print("=" * 60)
    
    # Example 1: Bullish Case (from prompt requirements)
    print("\nRunning Example 1: Tesla Earnings surprise")
    indicators_tsla = {
        "close": 210.50,
        "rsi": 68.5,
        "macd": 2.45,
        "macd_signal": 1.20,
        "volume_change": 0.28,
        "volatility": 0.035
    }
    
    # We will run the exact request input
    report_tsla = analyst.analyze_event(
        headline="Tesla beats earnings estimates and revenue rises",
        ticker="TSLA",
        indicators=indicators_tsla
    )
    analyst.print_beautiful_report(report_tsla)
    
    # Example 2: Bearish Case
    print("\nRunning Example 2: Apple guidance drop")
    indicators_aapl = {
        "close": 172.10,
        "rsi": 28.2,
        "macd": -3.10,
        "macd_signal": -2.00,
        "volume_change": 0.12,
        "volatility": 0.045
    }
    
    report_aapl = analyst.analyze_event(
        headline="Apple stock falls as sales warning hits supplier shares",
        ticker="AAPL",
        indicators=indicators_aapl
    )
    analyst.print_beautiful_report(report_aapl)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Entering Interactive Mode. Press Ctrl+C to exit.")
    print("=" * 60)
    
    try:
        while True:
            ticker = input("\nEnter Stock Ticker (e.g. MSFT): ").strip().upper()
            if not ticker:
                ticker = "SPY"
                
            headline = input("Enter News Headline: ").strip()
            if not headline:
                print("Headline cannot be empty.")
                continue
                
            print("\nAnalyzing... (computing technicals dynamically)")
            
            # Use average neutral indicators for mock custom input
            custom_indicators = {
                "close": 100.0,
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "volume_change": 0.0,
                "volatility": 0.02
            }
            
            report = analyst.analyze_event(
                headline=headline,
                ticker=ticker,
                indicators=custom_indicators
            )
            analyst.print_beautiful_report(report)
            
    except KeyboardInterrupt:
        print("\nExiting interactive inference. Goodbye!")
    finally:
        # Cleanup
        clear_gpu_memory()

if __name__ == "__main__":
    main()
