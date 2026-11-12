import os
import sys
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings, model_config
from src.models.memory_utils import VRAMTracker, print_vram_report, clear_gpu_memory

logger = logging.getLogger(__name__)

# Check if unsloth is available
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    logger.info("Unsloth is available. We will use Unsloth for optimized training.")
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.info("Unsloth is not available. Falling back to standard HF + PEFT + bitsandbytes.")

def train_qlora_llm(train_path: str, val_path: str):
    """
    Loads Llama 3.2 3B and trains LoRA adapters using QLoRA.
    Uses Unsloth if installed, otherwise defaults to standard Hugging Face PEFT QLoRA.
    """
    model_id = model_config.LLM_MODEL_ID
    output_dir = settings.MODEL_SAVE_DIR / "llama3"
    
    # Load dataset
    logger.info(f"Loading training data from {train_path} and validation from {val_path}...")
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    
    # Initialize VRAM tracker
    with VRAMTracker("LLM QLoRA Fine-tuning"):
        if UNSLOTH_AVAILABLE:
            # 1. Training with Unsloth (highly optimized)
            logger.info(f"Loading {model_id} via Unsloth...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=settings.MAX_SEQ_LENGTH,
                dtype=None, # auto-detect (float16/bfloat16)
                load_in_4bit=True,
                token=settings.HF_TOKEN
            )
            
            # Apply PEFT configuration using Unsloth
            logger.info("Applying Unsloth LoRA configuration...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=model_config.QLORA_R,
                target_modules=model_config.QLORA_TARGET_MODULES,
                lora_alpha=model_config.QLORA_ALPHA,
                lora_dropout=model_config.QLORA_DROPOUT,
                bias="none",
                use_gradient_checkpointing="unsloth", # optimized checkpointing
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
        else:
            # 2. Training with standard HF Transformers + PEFT
            logger.info(f"Loading {model_id} via standard HuggingFace...")
            
            # Setup BitsAndBytes configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, # float16 is safe for consumer cards
                bnb_4bit_use_double_quant=True,
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=settings.HF_TOKEN)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                token=settings.HF_TOKEN
            )
            
            # Prepare model for training (gradient checkpointing, kbit training)
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            
            # Setup LoRA configuration
            lora_config = LoraConfig(
                r=model_config.QLORA_R,
                lora_alpha=model_config.QLORA_ALPHA,
                target_modules=model_config.QLORA_TARGET_MODULES,
                lora_dropout=model_config.QLORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            
        model.print_trainable_parameters()
        
        # Configure training arguments
        # Set up dynamic bf16 detection
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if use_bf16:
            logger.info("BF16 is supported by the hardware. Enabling BF16.")
            model_config.LLM_TRAINING_ARGS["bf16"] = True
            model_config.LLM_TRAINING_ARGS["fp16"] = False
        else:
            logger.info("BF16 not supported. Using FP16.")
            model_config.LLM_TRAINING_ARGS["bf16"] = False
            model_config.LLM_TRAINING_ARGS["fp16"] = True
            
        # Compile training args
        training_args = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=model_config.LLM_TRAINING_ARGS["per_device_train_batch_size"],
            gradient_accumulation_steps=model_config.LLM_TRAINING_ARGS["gradient_accumulation_steps"],
            learning_rate=model_config.LLM_TRAINING_ARGS["learning_rate"],
            num_train_epochs=model_config.LLM_TRAINING_ARGS["num_train_epochs"],
            fp16=model_config.LLM_TRAINING_ARGS["fp16"],
            bf16=model_config.LLM_TRAINING_ARGS["bf16"],
            optim=model_config.LLM_TRAINING_ARGS["optim"],
            gradient_checkpointing=model_config.LLM_TRAINING_ARGS["gradient_checkpointing"],
            logging_steps=model_config.LLM_TRAINING_ARGS["logging_steps"],
            save_strategy=model_config.LLM_TRAINING_ARGS["save_strategy"],
            eval_strategy=model_config.LLM_TRAINING_ARGS["eval_strategy"],
            load_best_model_at_end=model_config.LLM_TRAINING_ARGS["load_best_model_at_end"],
            metric_for_best_model=model_config.LLM_TRAINING_ARGS["metric_for_best_model"],
            weight_decay=model_config.LLM_TRAINING_ARGS["weight_decay"],
            report_to="none", # disable external logins
            dataset_text_field="text",
            max_length=settings.MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False, # packing can cause issues with structure analysis
        )
        
        # Instantiate SFTTrainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            args=training_args,
        )
        
        logger.info("Starting QLoRA fine-tuning...")
        trainer.train()
        
    # Save the adapter weights and tokenizer
    logger.info(f"Saving fine-tuned LLM adapters to {output_dir}...")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info("LLM adapter training completed successfully!")
    return output_dir
