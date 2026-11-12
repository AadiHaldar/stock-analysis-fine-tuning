# Model configuration constants

# 1. FinBERT (Sentiment Model)
FINBERT_MODEL_ID = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 8
FINBERT_LR = 2e-5
FINBERT_EPOCHS = 3
FINBERT_WEIGHT_DECAY = 0.01

# 2. Llama 3.2 3B (Financial Analyst Model)
# We can use either the official Meta Llama or the Unsloth-optimized version
LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" 
# Fallback model in case the Hugging Face gated access is pending
LLM_FALLBACK_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# QLoRA training parameters
QLORA_R = 16
QLORA_ALPHA = 32
QLORA_DROPOUT = 0.05
QLORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

# LLM training arguments (optimized for RTX 5060 8GB VRAM)
LLM_TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8, # Effective batch size = 16
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "fp16": True, # set to True for older/consumer gpus, bf16 is preferred if supported
    "bf16": False, # set to True during initialization if device supports it
    "optim": "paged_adamw_8bit", # offload states to CPU RAM when required
    "gradient_checkpointing": True, # saves huge memory
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "weight_decay": 0.01,
}

# 3. RAG / Embedding settings
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2" # extremely small (~80MB), fast, low VRAM
FAISS_INDEX_NAME = "financial_rag_index"
