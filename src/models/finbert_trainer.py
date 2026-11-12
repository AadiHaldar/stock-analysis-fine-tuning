import os
import sys
import numpy as np
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings, model_config
from src.models.memory_utils import VRAMTracker

logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1 score for sequence classification evaluation.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_finbert(train_df, val_df):
    """
    Fine-tunes FinBERT (ProsusAI/finbert) on sequence classification data.
    """
    model_id = model_config.FINBERT_MODEL_ID
    output_dir = settings.MODEL_SAVE_DIR / "finbert"
    
    logger.info(f"Loading FinBERT tokenizer and model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=settings.HF_TOKEN)
    
    # FinBERT natively has 3 labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=3, 
        ignore_mismatched_sizes=True,
        token=settings.HF_TOKEN
    )
    
    # Convert pandas dfs to HF Dataset
    train_dataset = Dataset.from_pandas(train_df[['sentence', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['sentence', 'label']])
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], truncation=True, max_length=settings.MAX_SEQ_LENGTH)
        
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # Collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=model_config.FINBERT_LR,
        per_device_train_batch_size=model_config.FINBERT_BATCH_SIZE,
        per_device_eval_batch_size=model_config.FINBERT_BATCH_SIZE,
        num_train_epochs=model_config.FINBERT_EPOCHS,
        weight_decay=model_config.FINBERT_WEIGHT_DECAY,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none", # disable wandb/tensorboard reporting for lightweight execution
    )
    
    # Instantiate trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Track VRAM during training
    with VRAMTracker("FinBERT Training"):
        logger.info("Starting FinBERT fine-tuning...")
        trainer.train()
        
    # Save the best model
    logger.info(f"Saving fine-tuned FinBERT to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info("FinBERT training completed and model saved successfully!")
    return trainer.evaluate()
