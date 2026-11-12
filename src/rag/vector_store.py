import faiss
import numpy as np
import pickle
from pathlib import Path
import sys
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

import logging
logger = logging.getLogger(__name__)

class LocalFAISSStore:
    """
    FAISS-based local vector store with basic metadata and documents persistence.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # FlatL2 index (L2 distance metric)
        self.index = faiss.IndexFlatL2(dimension)
        # Store corresponding document payloads and metadata
        self.documents = []
        self.metadata = []
        
    def add_vectors(self, vectors: np.ndarray, docs: List[str], metas: List[Dict[str, Any]]):
        """
        Inserts new document texts and their dense embeddings into the FAISS index.
        """
        assert vectors.shape[1] == self.dimension, f"Vector dimension {vectors.shape[1]} doesn't match index {self.dimension}"
        
        # Add to FAISS index
        self.index.add(vectors.astype('float32'))
        
        # Store payloads
        self.documents.extend(docs)
        self.metadata.extend(metas)
        logger.info(f"Added {len(docs)} documents. Total index size: {self.index.ntotal}")
        
    def similarity_search(self, query_vector: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs L2 distance nearest neighbor search.
        Returns top k matching documents with metadata and score.
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty vector store.")
            return []
            
        # Reshape to 2D array
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
            
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for rank in range(len(indices[0])):
            idx = indices[0][rank]
            if idx == -1:
                continue
            
            # Map index back to documents payload
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "distance": float(distances[0][rank])
            })
            
        return results

    def save(self, folder_path: Path = settings.VECTOR_DIR):
        """
        Serializes both index and metadata payloads to local disk.
        """
        folder_path.mkdir(parents=True, exist_ok=True)
        index_file = folder_path / "faiss_index.bin"
        payload_file = folder_path / "faiss_payload.pkl"
        
        # Write binary FAISS index
        faiss.write_index(self.index, str(index_file))
        
        # Save documents and metadata
        with open(payload_file, "wb") as f:
            pickle.dump({"docs": self.documents, "metas": self.metadata, "dim": self.dimension}, f)
            
        logger.info(f"FAISS index and metadata saved to {folder_path}")

    @classmethod
    def load(cls, folder_path: Path = settings.VECTOR_DIR) -> 'LocalFAISSStore':
        """
        Deserializes index and metadata payloads from local disk.
        """
        index_file = folder_path / "faiss_index.bin"
        payload_file = folder_path / "faiss_payload.pkl"
        
        if not index_file.exists() or not payload_file.exists():
            logger.warning(f"No FAISS index files found at {folder_path}. Creating new index.")
            return cls()
            
        # Load binary index
        index = faiss.read_index(str(index_file))
        
        # Load payload
        with open(payload_file, "rb") as f:
            payload = pickle.load(f)
            
        store = cls(dimension=payload["dim"])
        store.index = index
        store.documents = payload["docs"]
        store.metadata = payload["metas"]
        
        logger.info(f"Loaded FAISS index containing {store.index.ntotal} items.")
        return store
