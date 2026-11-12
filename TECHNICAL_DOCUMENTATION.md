# Technical Walkthrough: Local Stock-Analysis AI System

A comprehensive, production-grade guide documenting the architecture, implementation, and optimization of a local financial intelligence pipeline. Optimized for consumer hardware with an NVIDIA RTX 5060 GPU (8GB VRAM / Blackwell Architecture / sm_120).

---

## 1. System Architecture Overview

The system orchestrates a hybrid ML workflow combining an encoder-only sequence classifier (FinBERT), a quantized generative LLM (Llama 3.2 3B) fine-tuned via QLoRA, and a CPU-bound Dense Retrieval-Augmented Generation (RAG) system.

```
                                  [ Yahoo Finance API ]
                                            │
                                            ▼
                               [ Historical Stock Prices ]
                                            │
                                            ▼
                              [ Technical Indicators (ta) ]
                           (RSI, MACD, EMA, SMA, BB, Volatility)
                                            │
                                            ▼
                              [ Feature Vector Alignment ] ◄─── [ lmassaron/FinancialPhraseBank ]
                                            │
                                            ▼
                              [ Instruction Dataset splits ]
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
          [ FinBERT Classifier ]                          [ Llama 3.2 3B Causal LM ]
          (Sequence Classification)                        (Quantized QLoRA Alignment)
                    │                                               │
                    ▼                                               ▼
          [ Sentiment Probabilities ]                     [ Analyst Trading Signals ]
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                                 [ Unified API Engine ] ◄─── [ FAISS Vector Index ]
                                            │                 (all-MiniLM-L6-v2 Embeddings)
                                            ▼
                                [ Financial Copilot CLI ]
```

---

## 2. Deep Dive: Data Pipeline & Preprocessing

The pipeline links market pricing series with unstructured sentiment corpus documents.

### A. Market Indicators Pipeline
We fetch stock prices using the `yfinance` library and cache them locally in CSV format to respect API limits. We then calculate mathematical feature sets using the pure Python `ta` library:
- **Relative Strength Index (RSI)**: Measures momentum on a 14-day rolling window to identify overbought (>70) and oversold (<30) conditions.
- **Moving Average Convergence Divergence (MACD)**: Compares 12-day and 26-day Exponential Moving Averages (EMA) with a 9-day signal line.
- **Bollinger Bands**: Evaluates volatility envelopes on a 20-day Simple Moving Average (SMA) offset by $\pm 2$ standard deviations.
- **Volatility**: Captures rolling 20-day historical standard deviation of daily logarithmic returns.

### B. Trading Label Generation
Trading signals are generated programmatically based on forward return outcomes. Given lookahead window $N = 5$ days and change threshold $\theta = 2\%$:
$$\text{Forward Return} = \frac{\text{Close}_{t + N} - \text{Close}_t}{\text{Close}_t}$$

$$\text{Trading Label} = \begin{cases} 
2 \text{ (BUY)} & \text{Forward Return} > \theta \\
0 \text{ (SELL)} & \text{Forward Return} < -\theta \\
1 \text{ (HOLD)} & \text{otherwise}
\end{cases}$$

### C. Corpus Construction & Alignment
- **Sentiment Baseline**: Loaded from the `lmassaron/FinancialPhraseBank` dataset (~3,872 annotated sentences), bypassing deprecated Python dataset loading scripts via modern Parquet file resolution.
- **Instruction Construction**: Since financial news headlines are sparse relative to market dates, we synthesize instruction samples. We match PhraseBank sentiment tags with realistic, mathematically correlated indicator distributions to teach the model to associate sentiment concepts with technical setups (e.g. positive earnings news paired with bullish MACD crossovers and healthy RSI levels).
- **Format**: Dumps into train/validation/test splits formatted as instruction prompts in `llama_dataset_*.jsonl`.

---

## 3. FinBERT Sequence Classifier Fine-Tuning

### A. Model Overview
We utilize `ProsusAI/finbert`, a BERT-base model (110M parameters) pre-trained on Financial Web corpus texts and fine-tuned for sequence classification (3 classes: positive, negative, neutral).

### B. Hyperparameter Configuration
- **Learning Rate**: $2 \times 10^{-5}$ with AdamW optimizer.
- **Batch Size**: 8 (eval batch size 8).
- **Epochs**: 3.
- **Weight Decay**: 0.01.
- **VRAM Fingerprint**: ~1.24 GB allocated during training.

### C. Validation Performance
- **Validation Accuracy**: **85.20%**
- **Weighted F1-Score**: **85.23%**
- **Evaluation Speed**: 232 samples/second.

---

## 4. Llama 3.2 3B QLoRA Parameter Tuning

### A. 4-bit NF4 Quantization
To fit a 3B model (which requires ~6.5 GB VRAM in float16) within 8GB VRAM alongside the Windows desktop environment, we apply **NormalFloat4 (NF4) quantization**:
- Base model weights are loaded in 4-bit.
- **Double Quantization**: Quantizes the quantization constants themselves, saving ~0.37 bits per parameter.
- **Compute Dtype**: `bfloat16` (auto-selected because the RTX 5060 Laptop GPU supports native BF16 operations).

### B. LoRA Parameter Grid
- **LoRA Rank ($r$)**: 16.
- **LoRA Alpha ($\alpha$)**: 32 (standard scaling ratio of 2.0).
- **LoRA Dropout**: 0.05.
- **Target Modules**: Trainable parameters are injected into all projection layers of the attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP blocks (`gate_proj`, `up_proj`, `down_proj`).
- **Trainable parameters**: 24.3 Million parameters (~0.75% of total model size).

### C. VRAM & Hardware Optimizations
- **Gradient Checkpointing**: Discards intermediate activations during the forward pass, recalculating them during the backward pass. This reduces activation VRAM usage by ~60%.
- **Paged AdamW 8-bit Optimizer**: Offloads optimizer states dynamically to system RAM if memory constraints are hit, preventing OOM.
- **Windows Memory Paging Management**: When VRAM approaches the 8GB limit, Windows WDDM drivers page memory to System RAM. This prevents crashes but causes temporary bottlenecks (seen as a temporary increase in step time from ~8s to ~400s as pages are swapped).

---

## 5. Retrieval-Augmented Generation (RAG) Architecture

### A. Document Loader & Processing
Textual financial records (earnings call transcripts, SEC filings, analyst updates) are chunked via a custom overlap parser:
- **Chunk Size**: 500 characters.
- **Overlap**: 100 characters.
- Delimiters are split on sentence boundaries to preserve logical context.

### B. Vector Storage & Retrieval
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (~80MB footprint, 384 dimensions). Run on **CPU** to reserve VRAM for generative inference.
- **Indexing**: FLAT L2 distance metric index constructed using `faiss-cpu`.
- **Retrieval strategy**: Vectorized similarity search queries map search queries to matching passage chunks, supporting source file metadata filtering.

### C. SEC EDGAR Downloader Integration
To make the RAG system capable of retrieving real-world SEC filing data, we integrated a downloader module (`scripts/download_sec.py`):
- **Library**: `sec-edgar-downloader` is used to fetch filings from EDGAR directly.
- **Compliance**: Uses a standardized `User-Agent` to comply with the SEC fair-use policies.
- **HTML Parsing**: Uses Python's built-in `html.parser.HTMLParser` to clean downloaded filing HTML sheets into clean plain text summaries, capping character lengths to prevent excessively large indexes.
- **Folder alignment**: Cleaned txt summaries are outputted directly to `data/raw/documents/{TICKER}_sec_summary.txt` to be automatically discovered by the FAISS builder.

---

## 6. Unified Inference API

The system exposes a consolidated client API class `UnifiedStockAnalyst` that orchestrates:
1. **FinBERT Prediction**: Obtains quick logits mapping to pure text sentiment.
2. **Llama 3.2 3B Generation**: Feeds the context and technical indicators into the generative model to output structured trading recommendations and analyst reasoning.

### CLI Example Representation
```
╔══════════════════════════════════════════════════════════════════════════╗
║                    Financial Stock Analysis Report                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║      Ticker Symbol: TSLA                                                 ║
║     Event/Headline: "Tesla beats earnings estimates and revenue rises"   ║
║          Sentiment: Bullish (92.4%)                                      ║
║     Trading Signal: Buy                                                  ║
║          Reasoning: Positive earnings surprise and revenue growth        ║
║                     supports the momentum. MACD crossover is positive.   ║
║     Probabilities: Bearish: 1.2%, Neutral: 6.4%, Bullish: 92.4%          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 7. Operational Instructions

### Setup Environment
```powershell
# Set up virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.8 wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run Pipeline Tasks
```powershell
# 1. Download market data, news, and phrasebank
.venv\Scripts\python.exe scripts\download_data.py

# 2. Train FinBERT classification model
.venv\Scripts\python.exe scripts\train_finbert.py

# 3. Train Llama 3.2 3B QLoRA model (with Windows UTF8 config)
$env:PYTHONUTF8="1"
.venv\Scripts\python.exe scripts\train_llm.py

# 4. Download real-world SEC filings (Optional)
$env:PYTHONUTF8="1"
.venv\Scripts\python.exe scripts\download_sec.py --tickers AAPL,TSLA --type 10-K

# 5. Build RAG FAISS Vector Index (incorporates downloaded filings)
.venv\Scripts\python.exe scripts\build_rag.py

# 6. Run Unified Inference / Interactive CLI Analyst
.venv\Scripts\python.exe scripts\run_inference.py

# 7. Run complete Evaluation Suite
.venv\Scripts\python.exe scripts\evaluate.py
```
