from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import model_config

import logging
logger = logging.getLogger(__name__)

class FinancialEmbedder:
    """
    Sentence embedding wrapper using local sentence-transformers models.
    Default: sentence-transformers/all-MiniLM-L6-v2 (~80MB, fast, low VRAM)
    """
    def __init__(self, model_id: str = model_config.EMBEDDING_MODEL_ID):
        logger.info(f"Loading local Embedding Model: {model_id}...")
        self.model = SentenceTransformer(model_id, device="cpu") # run on CPU to save VRAM for fine-tuning LLM
        logger.info("Embedding Model loaded successfully on CPU.")
        
    def embed_queries(self, queries: list) -> list:
        """
        Generates dense vector embeddings for a list of string queries.
        """
        return self.model.encode(queries, convert_to_numpy=True)
        
    def embed_documents(self, documents: list) -> list:
        """
        Generates dense vector embeddings for a list of document strings.
        """
        return self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        
if __name__ == "__main__":
    embedder = FinancialEmbedder()
    vecs = embedder.embed_queries(["Tesla earnings surprise", "Apple share drop"])
    print(f"Vectors shape: {vecs.shape}")
