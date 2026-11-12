import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory configuration
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
VECTOR_DIR = DATA_DIR / "vectors"

# Outputs directory configuration
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_SAVE_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Ensure all directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    VECTOR_DIR,
    MODEL_SAVE_DIR,
    LOGS_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Hugging Face Setup
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Hardware limits for memory optimization
VRAM_BUDGET_GB = 8.0  # Optimized for RTX 5060 8GB
MAX_SEQ_LENGTH = 512  # Keep sequences short to avoid OOM

# Tickers to analyze for historical dataset
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC"]
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-05-24"
