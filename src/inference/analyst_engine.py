import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings, model_config
from src.models.model_selector import check_gpu_and_select_model

import logging
logger = logging.getLogger(__name__)

# Check if unsloth is available
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

class AnalystEngine:
    """
    Inference engine using fine-tuned Llama 3.2 3B (with adapters)
    to perform advanced financial analysis and trading recommendations.
    """
    def __init__(self, base_model_id: str = None, adapter_dir: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Determine model and adapter paths
        if base_model_id is None:
            self.base_model_id = check_gpu_and_select_model()
        else:
            self.base_model_id = base_model_id
            
        if adapter_dir is None:
            ft_dir = settings.MODEL_SAVE_DIR / "llama3"
            if ft_dir.exists():
                self.adapter_dir = str(ft_dir)
                self.has_adapters = True
            else:
                logger.warning(f"Fine-tuned adapters not found at {ft_dir}. Running base model.")
                self.adapter_dir = None
                self.has_adapters = False
        else:
            self.adapter_dir = adapter_dir
            self.has_adapters = True

        logger.info(f"Loading LLM Engine (Base: {self.base_model_id}, Adapters: {self.adapter_dir})...")
        
        # 2. Load model & tokenizer
        if UNSLOTH_AVAILABLE:
            logger.info("Loading Causal LM via Unsloth...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.adapter_dir if self.has_adapters else self.base_model_id,
                max_seq_length=settings.MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
                token=settings.HF_TOKEN
            )
            FastLanguageModel.for_inference(model) # enable fast inference mode (2x faster)
            self.model = model
            self.tokenizer = tokenizer
        else:
            logger.info("Loading Causal LM via standard HF PEFT...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, token=settings.HF_TOKEN)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                token=settings.HF_TOKEN
            )
            
            if self.has_adapters:
                logger.info(f"Loading LoRA adapters onto base model...")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
            else:
                self.model = base_model
                
            self.model.eval()

        logger.info("LLM Analyst Engine loaded successfully.")

    def construct_prompt(self, headline: str, ticker: str, indicators: dict = None) -> str:
        """
        Creates the instruction prompt format.
        """
        if indicators is None:
            # Fallback to realistic/generic values if none provided
            indicators = {
                "close": 150.0,
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "volume_change": 0.0,
                "volatility": 0.02
            }
            
        close = indicators.get("close", 150.0)
        rsi = indicators.get("rsi", 50.0)
        macd = indicators.get("macd", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)
        vol_change = indicators.get("volume_change", 0.0) * 100
        volatility = indicators.get("volatility", 0.02) * 100
        
        rsi_state = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral")
        macd_trend = "Bullish Crossover" if macd > macd_signal else "Bearish Trend"
        
        indicators_summary = (
            f"Stock Price: ${close:.2f}. "
            f"RSI(14): {rsi:.1f} ({rsi_state}). "
            f"MACD: {macd:.3f} vs Signal: {macd_signal:.3f} ({macd_trend}). "
            f"Volume Change: {vol_change:+.1f}%. "
            f"20-day Volatility: {volatility:.1f}%."
        )
        
        prompt = (
            f"### Instruction: Analyze the following financial news and historical technical indicators for {ticker}. "
            f"Determine the sentiment label (Bullish, Bearish, or Neutral), provide a trading signal (Buy, Sell, or Hold), "
            f"assign a confidence level (0-100%), and explain your reasoning.\n\n"
            f"### News Headline: {headline}\n"
            f"### Technical Indicators: {indicators_summary}\n\n"
            f"### Output:\n"
        )
        return prompt

    def analyze(self, headline: str, ticker: str = "SPY", indicators: dict = None) -> dict:
        """
        Runs generation and parses structured response.
        """
        prompt = self.construct_prompt(headline, ticker, indicators)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1, # Keep temperature low for deterministic structures
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode only the generated part
        generated_tokens = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Parse fields from the response
        sentiment = "Neutral"
        confidence = "50%"
        signal = "Hold"
        reason = response
        
        sentiment_match = re.search(r"Sentiment:\s*(Bullish|Bearish|Neutral)", response, re.IGNORECASE)
        if sentiment_match:
            sentiment = sentiment_match.group(1).capitalize()
            
        confidence_match = re.search(r"Confidence:\s*(\d+%)", response, re.IGNORECASE)
        if confidence_match:
            confidence = confidence_match.group(1)
            
        signal_match = re.search(r"Trading Signal:\s*(Buy|Sell|Hold)", response, re.IGNORECASE)
        if signal_match:
            signal = signal_match.group(1).capitalize()
            
        reason_match = re.search(r"Reason:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
            
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "trading_signal": signal,
            "reasoning": reason,
            "raw_response": response
        }

if __name__ == "__main__":
    # Test analyst engine
    logging.basicConfig(level=logging.INFO)
    engine = AnalystEngine()
    result = engine.analyze(
        headline="Tesla beats earnings estimates and revenue rises",
        ticker="TSLA",
        indicators={"close": 220.0, "rsi": 65.0, "macd": 1.2, "macd_signal": 0.8, "volume_change": 0.15, "volatility": 0.03}
    )
    print(result)
