import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings, model_config

import logging
logger = logging.getLogger(__name__)

class SentimentEngine:
    """
    Inference engine using fine-tuned FinBERT to predict market sentiment.
    """
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Check if fine-tuned model exists
            ft_dir = settings.MODEL_SAVE_DIR / "finbert"
            if ft_dir.exists():
                self.model_dir = str(ft_dir)
            else:
                logger.warning(f"Fine-tuned FinBERT not found at {ft_dir}. Using base model.")
                self.model_dir = model_config.FINBERT_MODEL_ID
        else:
            self.model_dir = model_dir
            
        logger.info(f"Loading Sentiment Model from: {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, token=settings.HF_TOKEN)
        
        # Load classification model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, 
            token=settings.HF_TOKEN
        ).to(self.device)
        self.model.eval()
        
        # FinBERT labels: 0 -> Negative (Bearish), 1 -> Neutral, 2 -> Positive (Bullish)
        self.labels = ["Bearish", "Neutral", "Bullish"]
        logger.info("Sentiment Engine loaded successfully.")

    def predict(self, text: str) -> dict:
        """
        Predicts sentiment for a single news headline/sentence.
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=settings.MAX_SEQ_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0)
            
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        return {
            "sentiment": self.labels[pred_idx],
            "confidence": round(confidence * 100, 1),
            "probabilities": {self.labels[i]: round(probs[i].item() * 100, 1) for i in range(3)}
        }

if __name__ == "__main__":
    # Test prediction
    logging.basicConfig(level=logging.INFO)
    engine = SentimentEngine()
    result = engine.predict("Tesla beats earnings estimates and revenue rises")
    print(result)
