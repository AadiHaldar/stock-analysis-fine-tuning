import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.models.model_selector import check_gpu_and_select_model
from src.models.llm_trainer import train_qlora_llm
from src.models.memory_utils import clear_gpu_memory

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing LLM fine-tuning script...")
    
    # 1. Perform device and model check
    selected_model_id = check_gpu_and_select_model()
    logger.info(f"Model selected for fine-tuning: {selected_model_id}")
    
    # 2. Check dataset existence
    train_path = settings.PROCESSED_DATA_DIR / "llama_dataset_train.jsonl"
    val_path = settings.PROCESSED_DATA_DIR / "llama_dataset_validation.jsonl"
    
    if not train_path.exists() or not val_path.exists():
        logger.error(f"Training datasets not found at {train_path}. "
                     "Please run 'python scripts/download_data.py' first to build the training set.")
        sys.exit(1)
        
    # 3. Launch training
    try:
        train_qlora_llm(str(train_path), str(val_path))
        logger.info("LLM fine-tuning completed successfully!")
    except Exception as e:
        logger.error(f"Training crashed with error: {e}")
        # Make sure memory is cleared on crash
        clear_gpu_memory()
        sys.exit(1)

if __name__ == "__main__":
    main()
