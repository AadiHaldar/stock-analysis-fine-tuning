import torch
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import model_config, settings

logger = logging.getLogger(__name__)

def check_gpu_and_select_model() -> str:
    """
    Checks the system's available GPU memory and returns the optimal model ID.
    Supports fallback if VRAM is extremely limited.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available on this machine. Falling back to CPU/Smaller model.")
        return model_config.LLM_FALLBACK_MODEL_ID
        
    device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_vram_gb = device_properties.total_memory / (1024 ** 3)
    gpu_name = device_properties.name
    
    logger.info(f"Detected GPU: {gpu_name} with {total_vram_gb:.2f} GB total VRAM")
    
    # RTX 5060 has 8GB VRAM.
    if total_vram_gb >= 7.5:
        logger.info("GPU meets RTX 5060 (8GB VRAM) standards. Selecting Llama 3.2 3B.")
        return model_config.LLM_MODEL_ID
    elif total_vram_gb >= 3.5:
        logger.warning(f"VRAM is tight ({total_vram_gb:.2f} GB). Selecting Llama 3.2 3B (requires aggressive memory optimization).")
        return model_config.LLM_MODEL_ID
    else:
        logger.error(f"Extremely low VRAM ({total_vram_gb:.2f} GB). Selecting fallback model Qwen 3B.")
        return model_config.LLM_FALLBACK_MODEL_ID

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_id = check_gpu_and_select_model()
    print(f"Selected Model: {model_id}")
