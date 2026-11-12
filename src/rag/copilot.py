import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.rag.embeddings import FinancialEmbedder
from src.rag.vector_store import LocalFAISSStore
from src.inference.analyst_engine import AnalystEngine
from src.models.memory_utils import clear_gpu_memory

logger = logging.getLogger(__name__)

class FinancialCopilot:
    """
    RAG-powered Financial Copilot capable of answering questions,
    summarizing documents, and explaining stock movements using local contexts.
    """
    def __init__(self, vector_dir: Path = None):
        self.embedder = FinancialEmbedder()
        self.vector_store = LocalFAISSStore.load(vector_dir) if vector_dir else LocalFAISSStore.load()
        self.llm = None # lazy load to preserve GPU memory initially
        
    def query(self, question: str, ticker: str = None, top_k: int = 3) -> str:
        """
        Retrieves matching chunks, formats prompt, and generates LLM answer.
        """
        # 1. Embed query
        query_vec = self.embedder.embed_queries([question])[0]
        
        # 2. Similarity search
        results = self.vector_store.similarity_search(query_vec, k=top_k)
        
        # Filter by ticker in metadata if specified
        if ticker:
            ticker = ticker.upper()
            results = [r for r in results if r["metadata"].get("ticker") == ticker or r["metadata"].get("ticker") == "GLOBAL"]
            # If filtering left us empty, fallback to original list
            if not results:
                results = self.vector_store.similarity_search(query_vec, k=top_k)
                
        # 3. Compile context
        if not results:
            context = "No relevant document chunks found in database."
        else:
            context_parts = []
            for r in results:
                src = r["metadata"].get("source", "Unknown")
                context_parts.append(f"--- Context (Source: {src}): ---\n{r['document']}")
            context = "\n\n".join(context_parts)
            
        # 4. Lazy-load LLM to save memory
        if self.llm is None:
            logger.info("Initializing LLM for Copilot generation...")
            self.llm = AnalystEngine()
            
        # 5. Formulate QA prompt
        prompt = (
            f"### Instruction: You are a professional financial advisor and quantitative research analyst. "
            f"Answer the question below using only the provided context. If the answer cannot be found "
            f"in the context, state that you do not have enough information.\n\n"
            f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Output:\n"
        )
        
        # 6. Generate answer
        inputs = self.llm.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        import torch
        with torch.no_grad():
            outputs = self.llm.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.llm.tokenizer.eos_token_id
            )
                
        generated_tokens = outputs[0][len(inputs.input_ids[0]):]
        answer = self.llm.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return answer

if __name__ == "__main__":
    # Test script run
    logging.basicConfig(level=logging.INFO)
    copilot = FinancialCopilot()
    # If index is empty, it will alert.
    print("Copilot ready.")
